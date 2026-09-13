"""Stage 4 of the Wan pipeline: one Wan 2.2 Fun-Control video per control video of a run dir, through the ComfyUI API.

  python experiments/wan_canny/make_wan.py <run_dir> [--controls canny,geoedge,...] [--ref PNG] [--prompt TEXT]
      [--seed 0] [--steps 20] [--cfg 3.5] [--shift 8] [--size 512] [--frames 81] [--force] [--dry_run]

<run_dir> is experiments/wan_canny/runs/<task>_<tag>/ with <prefix>_<control>.mp4 control videos and
<prefix>_ref_sim.png (prefix = folder name, override with --prefix). For every control that has a video this writes
<prefix>_wan_<control>.mp4 and <prefix>_wan_<control>.json (every parameter, prompt id, timing) into the same folder.
The prompt comes from tasks.TASKS[<task>] unless --prompt is given; the reference image is <prefix>_ref_sim.png unless
--ref is given. Existing outputs are skipped unless --force.

ComfyUI: talks to --comfy_url (default http://127.0.0.1:8188). If nothing answers there, starts a server from
--comfy_dir on --comfy_gpu (default GPU 1: Isaac Sim only renders on GPU 0 here and Wan 14B peaks at ~20 GB, so the
two cannot share a GPU) and leaves it running. The graph is workflows/wan22_fun_control_api.json (the ComfyUI
"Wan 2.2 14B Fun Control" template in API form); nodes are found by class_type, so the file can be re-exported from
the UI ("Export (API)") as long as it keeps the same node types. Standard library only, no torch needed.
"""
import argparse
import json
import mimetypes
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime

from tasks import CONTROLS, NEGATIVE_PROMPT, TASKS, task_of

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "workflows", "wan22_fun_control_api.json")
FPS = 16


# ----------------------------------------------------------------------------------------------------- ComfyUI API
def _http_error(path: str, e: urllib.error.HTTPError) -> RuntimeError:
    """ComfyUI answers a rejected request with 4xx and the reason in the JSON body ({"error", "node_errors"});
    urllib raises before that body is read, so unpack it into the exception message."""
    try:
        body = e.read().decode(errors="replace")
    except Exception:  # noqa: BLE001
        body = ""
    msg = f"comfyui {path} -> HTTP {e.code}"
    try:
        info = json.loads(body) if body else {}
    except ValueError:
        return RuntimeError(f"{msg}: {body[:1000]}")
    if isinstance(info, dict) and ("error" in info or "node_errors" in info):
        err = info.get("error") or {}
        lines = [err.get("message", "") + (f" ({err['details']})" if err.get("details") else "")] if err else []
        for nid, ne in (info.get("node_errors") or {}).items():
            for item in ne.get("errors", []):
                lines.append(f"node {nid} {ne.get('class_type', '')}: {item.get('message', '')} {item.get('details', '')}".strip())
        return RuntimeError(f"{msg}: " + "; ".join(line for line in lines if line))
    return RuntimeError(f"{msg}: {body[:1000]}")


def api_get(url: str, path: str, timeout: float = 10):
    try:
        with urllib.request.urlopen(url + path, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise _http_error(path, e) from None


def api_post(url: str, path: str, obj=None, timeout: float = 60):
    data = json.dumps(obj).encode() if obj is not None else b""
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        raise _http_error(path, e) from None
    return json.loads(body) if body else {}


def server_alive(url: str) -> bool:
    try:
        api_get(url, "/system_stats", timeout=5)
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def ensure_server(url: str, comfy_dir: str, comfy_python: str, gpu: int | None, log_path: str, wait_s: int = 240):
    if server_alive(url):
        return
    port = urllib.parse.urlparse(url).port or 8188
    cmd = [comfy_python, "main.py", "--listen", "0.0.0.0", "--port", str(port)]
    if gpu is not None:
        cmd += ["--cuda-device", str(gpu)]
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    print(f"comfyui: not running at {url}, starting: {' '.join(cmd)}  (log: {log_path})", flush=True)
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(cmd, cwd=comfy_dir, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        time.sleep(3)
        if server_alive(url):
            print(f"comfyui: up (pid {proc.pid}, {time.time() - t0:.0f}s)", flush=True)
            return
        if proc.poll() is not None:
            raise RuntimeError(f"comfyui exited with code {proc.returncode}, see {log_path}")
    raise RuntimeError(f"comfyui did not answer at {url} within {wait_s}s, see {log_path}")


def upload(url: str, path: str) -> str:
    """Upload a file into ComfyUI's input folder (overwrite) and return the name to use in LoadImage/LoadVideo."""
    name = os.path.basename(path)
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        payload = f.read()
    boundary = uuid.uuid4().hex
    parts = []
    for key, val in (("overwrite", "true"), ("type", "input")):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{val}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n".encode() + payload + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        url + "/upload/image", data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        raise _http_error("/upload/image", e) from None
    return f"{resp['subfolder']}/{resp['name']}" if resp.get("subfolder") else resp["name"]


def submit(url: str, graph: dict) -> str:
    resp = api_post(url, "/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})
    if resp.get("node_errors"):
        raise RuntimeError("comfyui rejected the prompt: " + json.dumps(resp["node_errors"], ensure_ascii=False)[:2000])
    if "prompt_id" not in resp:
        raise RuntimeError("comfyui rejected the prompt: " + json.dumps(resp, ensure_ascii=False)[:2000])
    return resp["prompt_id"]


def cancel(url: str, prompt_id: str):
    """Drop our own prompt only: delete it if still pending, interrupt it if it is the one running. A bare
    POST /interrupt would stop whatever is running (possibly someone else's web-UI job) and leave ours queued."""
    for path, body in (("/queue", {"delete": [prompt_id]}), ("/interrupt", {"prompt_id": prompt_id})):
        try:
            api_post(url, path, body, timeout=10)
        except Exception as e:  # noqa: BLE001
            print(f"wan: cancel {path} failed: {e}", flush=True)


def wait_done(url: str, prompt_id: str, timeout_s: float, poll_s: float = 5) -> dict:
    """Block until the prompt shows up in /history as finished; return its history entry."""
    t0 = time.time()
    try:
        while True:
            hist = api_get(url, f"/history/{prompt_id}")
            ent = hist.get(prompt_id)
            if ent and (ent.get("status", {}).get("completed") or ent.get("status", {}).get("status_str") in ("success", "error")):
                return ent
            if time.time() - t0 > timeout_s:
                cancel(url, prompt_id)
                raise TimeoutError(f"prompt {prompt_id} not finished after {timeout_s:.0f}s, cancelled")
            time.sleep(poll_s)
    except KeyboardInterrupt:
        cancel(url, prompt_id)
        raise


def output_files(entry: dict) -> list[dict]:
    """All {filename, subfolder, type} produced by SaveVideo-like nodes of a history entry."""
    files = []
    for outs in entry.get("outputs", {}).values():
        for val in outs.values():
            if isinstance(val, list):
                files += [f for f in val if isinstance(f, dict) and f.get("filename") and f.get("type") == "output"]
    return files


def download(url: str, info: dict, dst: str):
    """Fetch an output file through /view into dst (via dst.part, so an aborted transfer never leaves a
    half-written file that the skip logic would take for a finished video)."""
    q = urllib.parse.urlencode({"filename": info["filename"], "subfolder": info.get("subfolder", ""), "type": info.get("type", "output")})
    tmp = dst + ".part"
    try:
        with urllib.request.urlopen(f"{url}/view?{q}", timeout=300) as r, open(tmp, "wb") as f:
            f.write(r.read())
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ----------------------------------------------------------------------------------------------------- graph
def _ids(graph: dict, class_type: str) -> list[str]:
    return [k for k, v in graph.items() if v.get("class_type") == class_type]


def _one(graph: dict, class_type: str) -> str:
    ids = _ids(graph, class_type)
    if len(ids) != 1:
        raise ValueError(f"template must contain exactly one {class_type} node, found {len(ids)}")
    return ids[0]


def build_graph(template: dict, control_file: str, ref_file: str, prompt: str, negative: str, seed: int, steps: int,
                cfg: float, shift: float, size: int, frames: int, out_prefix: str, fps: float = FPS) -> dict:
    g = json.loads(json.dumps(template))  # deep copy
    wan = _one(g, "Wan22FunControlToVideo")
    g[wan]["inputs"].update(width=size, height=size, length=frames)
    g[g[wan]["inputs"]["positive"][0]]["inputs"]["text"] = prompt
    g[g[wan]["inputs"]["negative"][0]]["inputs"]["text"] = negative
    g[_one(g, "LoadVideo")]["inputs"]["file"] = control_file
    g[_one(g, "LoadImage")]["inputs"]["image"] = ref_file
    for k in _ids(g, "ModelSamplingSD3"):
        g[k]["inputs"]["shift"] = shift
    samplers = _ids(g, "KSamplerAdvanced")
    if len(samplers) != 2:
        raise ValueError(f"template must contain two KSamplerAdvanced nodes (high/low noise), found {len(samplers)}")
    if steps < 2:
        raise ValueError("steps must be >= 2: the high-noise and low-noise experts each need at least one step")
    split = steps // 2  # high-noise expert for the first half of the steps, low-noise expert for the rest
    for k in samplers:
        ins = g[k]["inputs"]
        ins.update(steps=steps, cfg=cfg)
        if ins["add_noise"] == "enable":
            ins.update(noise_seed=seed, start_at_step=0, end_at_step=split)
        else:
            ins.update(start_at_step=split, end_at_step=10000)
    g[_one(g, "CreateVideo")]["inputs"]["fps"] = fps
    g[_one(g, "SaveVideo")]["inputs"]["filename_prefix"] = out_prefix
    return g


# ----------------------------------------------------------------------------------------------------- stage
def run(run_dir: str, controls=None, ref: str | None = None, prompt: str | None = None, negative: str = NEGATIVE_PROMPT,
        seed: int = 0, steps: int = 20, cfg: float = 3.5, shift: float = 8.0, size: int = 512, frames: int = 81,
        force: bool = False, dry_run: bool = False, prefix: str | None = None, task: str | None = None,
        comfy_url: str = "http://127.0.0.1:8188", comfy_dir: str = "/workspace/tools/ComfyUI",
        comfy_python: str = "/opt/miniconda3/envs/comfyui/bin/python", comfy_gpu: int | None = 1,
        comfy_log: str = "logs/comfyui.log", template: str = TEMPLATE, timeout_s: float = 1800) -> dict:
    run_dir = os.path.normpath(run_dir)
    prefix = prefix or os.path.basename(run_dir)
    task = task or task_of(prefix)
    prompt = prompt if prompt is not None else TASKS[task]["prompt"]
    ref = ref or os.path.join(run_dir, f"{prefix}_ref_sim.png")
    controls = list(controls or CONTROLS)
    if not os.path.isfile(ref):
        raise FileNotFoundError(f"reference image not found: {ref}")
    with open(template) as f:
        tpl = json.load(f)

    todo = []
    for c in controls:
        src = os.path.join(run_dir, f"{prefix}_{c}.mp4")
        dst = os.path.join(run_dir, f"{prefix}_wan_{c}.mp4")
        if not os.path.isfile(src):
            print(f"wan: {c}: control video missing ({src}), skipped")
        elif os.path.isfile(dst) and not force:
            print(f"wan: {c}: {dst} exists, skipped (--force to redo)")
        else:
            todo.append((c, src, dst))
    print(f"wan: task={task} prefix={prefix} ref={ref} controls={[c for c, _, _ in todo]} seed={seed} steps={steps} "
          f"cfg={cfg} shift={shift} {size}x{size}x{frames}", flush=True)
    if dry_run:
        for c, src, dst in todo:
            g = build_graph(tpl, os.path.basename(src), os.path.basename(ref), prompt, negative, seed, steps, cfg, shift,
                            size, frames, f"video/{prefix}_wan_{c}")
            print(f"--- {c}: {dst}\n" + json.dumps({k: v["inputs"] for k, v in g.items()}, ensure_ascii=False, indent=1)[:3000])
        return {}
    if not todo:
        return {}

    ensure_server(comfy_url, comfy_dir, comfy_python, comfy_gpu, comfy_log)
    ref_name = upload(comfy_url, ref)
    results = {}
    for c, src, dst in todo:
        t0 = time.time()
        control_name = upload(comfy_url, src)
        g = build_graph(tpl, control_name, ref_name, prompt, negative, seed, steps, cfg, shift, size, frames,
                        f"video/{prefix}_wan_{c}")
        pid = submit(comfy_url, g)
        print(f"wan: {c}: submitted {pid}, waiting...", flush=True)
        ent = wait_done(comfy_url, pid, timeout_s)
        status = ent.get("status", {})
        if status.get("status_str") == "error":
            errs = [m for m in status.get("messages", []) if m and m[0] in ("execution_error", "execution_interrupted")]
            msg = "; ".join(f"{m[0]} {m[1].get('node_type')}: {m[1].get('exception_message', '')}".strip() for m in errs) or "unknown error"
            print(f"wan: {c}: FAILED after {time.time() - t0:.0f}s: {msg[:500]}", flush=True)
            results[c] = None
            continue
        files = output_files(ent)
        if not files:
            print(f"wan: {c}: FAILED, no output file in history entry", flush=True)
            results[c] = None
            continue
        download(comfy_url, files[0], dst)
        meta = dict(task=task, prefix=prefix, control=c, control_video=os.path.basename(src), ref_image=ref,
                    prompt=prompt, negative=negative, seed=seed, steps=steps, cfg=cfg, shift=shift, size=size,
                    frames=frames, fps=FPS, template=os.path.relpath(template), comfy_url=comfy_url, prompt_id=pid,
                    comfy_output=files[0], seconds=round(time.time() - t0, 1), finished_at=datetime.now().isoformat(timespec="seconds"))
        with open(os.path.splitext(dst)[0] + ".json", "w") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)
        print(f"wan: {c}: {dst} ({meta['seconds']:.0f}s, comfy output {files[0]['filename']})", flush=True)
        results[c] = dst
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--controls", default=",".join(CONTROLS), help="comma-separated subset of " + ",".join(CONTROLS))
    p.add_argument("--ref", default=None, help="reference image (default <prefix>_ref_sim.png in run_dir)")
    p.add_argument("--prompt", default=None, help="positive prompt (default tasks.TASKS[task]['prompt'])")
    p.add_argument("--negative", default=NEGATIVE_PROMPT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=3.5)
    p.add_argument("--shift", type=float, default=8.0)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--force", action="store_true", help="regenerate existing outputs")
    p.add_argument("--dry_run", action="store_true", help="print the graphs, do not contact ComfyUI")
    p.add_argument("--prefix", default=None, help="file prefix (default: run_dir folder name)")
    p.add_argument("--task", default=None, help="task key in tasks.TASKS (default: derived from the prefix)")
    p.add_argument("--comfy_url", default="http://127.0.0.1:8188")
    p.add_argument("--comfy_dir", default="/workspace/tools/ComfyUI")
    p.add_argument("--comfy_python", default="/opt/miniconda3/envs/comfyui/bin/python")
    p.add_argument("--comfy_gpu", type=int, default=1, help="GPU for an auto-started ComfyUI (-1: let ComfyUI choose)")
    p.add_argument("--comfy_log", default="logs/comfyui.log")
    p.add_argument("--template", default=TEMPLATE)
    p.add_argument("--timeout", type=float, default=1800, help="seconds to wait per video")
    a = p.parse_args()
    out = run(a.run_dir, [c.strip() for c in a.controls.split(",") if c.strip()], a.ref, a.prompt, a.negative, a.seed,
              a.steps, a.cfg, a.shift, a.size, a.frames, a.force, a.dry_run, a.prefix, a.task, a.comfy_url, a.comfy_dir,
              a.comfy_python, None if a.comfy_gpu < 0 else a.comfy_gpu, a.comfy_log, a.template, a.timeout)
    if out and any(v is None for v in out.values()):
        sys.exit(1)
