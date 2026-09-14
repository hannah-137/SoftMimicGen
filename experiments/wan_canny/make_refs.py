"""Stage 4a of the Wan pipeline: photorealistic reference images from the simulator frame with Qwen-Image-Edit-2511
through the ComfyUI API (models: docker/setup_qwen_image_edit.sh).

  python experiments/wan_canny/make_refs.py <run_dir> --n_ref 10 [--seed 0] [--steps 20] [--cfg 4] [--lightning]
      [--prompt TEXT] [--negative TEXT] [--gen_scale 2] [--force] [--dry_run] [--prefix P] [--task T] [--comfy_url URL]

images/<prefix>_ref_sim.png (stage 2, frame 0)  ->  images/<prefix>_ref_ai_01.png ... _<N>.png, one .json each.
N = --n_ref, image k uses seed --seed + k - 1, so the N images are variations of the same instruction. make_wan.py
then makes one Wan video per (reference image, control). Existing images are skipped unless --force; images the
user put there (<prefix>_ref_ai.png or any other suffix) are never touched, both kinds are used by make_wan.py.

Prompt: tasks.REF_PROMPT, deliberately static (no task action words: "folds a towel" made the model fold the towel)
and without colour or material words like the Wan prompt; image k gets the variation sentence REF_VARIATIONS[k-1
mod len] (background, table, robot wear, lighting), so the N images differ in environment while the layout stays fixed.
--prompt overrides the whole instruction for every image.
The edit runs at --gen_scale x the frame size (512x512 -> 1024x1024, the humanoid 512x320 -> 1024x640; Qwen-Image
works around 1 MP) and the result is scaled back to the frame size (cv2 INTER_AREA), so the reference has exactly
the video's size and aspect. Defaults follow the ComfyUI "Qwen-Image-Edit 2511" template: 20 steps, cfg 4,
shift 3.1, euler/simple, CFGNorm. --lightning adds the lightx2v 4-step LoRA (4 steps, cfg 1, several times faster,
a little less detail). Template: workflows/qwen_image_edit_2511_api.json (nodes found by class_type).
ComfyUI: same server and GPU as make_wan.py (auto-started on GPU 1 if nothing answers)."""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2

from make_wan import HERE, _one, download, ensure_server, output_files, png_size, submit, upload, wait_done
from tasks import REF_VARIATIONS, REF_NEGATIVE, REF_PROMPT, SUBDIRS, task_of

TEMPLATE = os.path.join(HERE, "workflows", "qwen_image_edit_2511_api.json")


def gen_size(width: int, height: int, scale: float) -> tuple[int, int]:
    """Edit resolution: the frame size times scale, rounded to multiples of 16 (Qwen-Image latent patch)."""
    return max(16, round(width * scale / 16) * 16), max(16, round(height * scale / 16) * 16)


def build_graph(template: dict, ref_sim_file: str, prompt: str, negative: str, seed: int, steps: int, cfg: float,
                shift: float, width: int, height: int, lightning: bool, out_prefix: str) -> dict:
    g = json.loads(json.dumps(template))  # deep copy
    g[_one(g, "LoadImage")]["inputs"]["image"] = ref_sim_file
    g[_one(g, "ImageScale")]["inputs"].update(width=width, height=height)
    g[_one(g, "ModelSamplingAuraFlow")]["inputs"]["shift"] = shift
    ks = _one(g, "KSampler")
    g[ks]["inputs"].update(seed=seed, steps=steps, cfg=cfg)
    pos, neg = g[ks]["inputs"]["positive"][0], g[ks]["inputs"]["negative"][0]
    for cond, text in ((pos, prompt), (neg, negative)):
        enc = g[cond]["inputs"]["conditioning"][0] if g[cond]["class_type"] != "TextEncodeQwenImageEditPlus" else cond
        g[enc]["inputs"]["prompt"] = text
    lora = _one(g, "LoraLoaderModelOnly")
    if not lightning:  # bypass the LoRA: the sampler takes the LoRA node's input model
        g[ks]["inputs"]["model"] = g[lora]["inputs"]["model"]
        del g[lora]
    g[_one(g, "SaveImage")]["inputs"]["filename_prefix"] = out_prefix
    return g


def run(run_dir: str, n_ref: int = 1, seed: int = 0, steps: int | None = None, cfg: float | None = None,
        shift: float = 3.1, lightning: bool = False, prompt: str | None = None, negative: str = REF_NEGATIVE,
        gen_scale: float = 2.0, force: bool = False, dry_run: bool = False, prefix: str | None = None,
        task: str | None = None, comfy_url: str = "http://127.0.0.1:8188", comfy_dir: str = "/workspace/tools/ComfyUI",
        comfy_python: str = "/opt/miniconda3/envs/comfyui/bin/python", comfy_gpu: int | None = 1,
        comfy_log: str = "logs/comfyui.log", template: str = TEMPLATE, timeout_s: float = 900) -> dict:
    run_dir = os.path.normpath(run_dir)
    prefix = prefix or os.path.basename(run_dir)
    task = task or task_of(prefix)
    steps = steps if steps is not None else (4 if lightning else 20)
    cfg = cfg if cfg is not None else (1.0 if lightning else 4.0)
    d_img = os.path.join(run_dir, SUBDIRS["images"])
    ref_sim = os.path.join(d_img, f"{prefix}_ref_sim.png")
    if not os.path.isfile(ref_sim):
        raise FileNotFoundError(f"simulator frame not found: {ref_sim} (run make_source.py first)")
    width, height = png_size(ref_sim)
    gw, gh = gen_size(width, height, gen_scale)
    with open(template) as f:
        tpl = json.load(f)

    def prompt_of(k: int) -> str:
        return prompt if prompt is not None else REF_PROMPT.format(variation=REF_VARIATIONS[(k - 1) % len(REF_VARIATIONS)])

    todo = []
    for k in range(1, n_ref + 1):
        dst = os.path.join(d_img, f"{prefix}_ref_ai_{k:02d}.png")
        if os.path.isfile(dst) and not force:
            print(f"refs: {os.path.basename(dst)} exists, skipped (--force to redo)")
        else:
            todo.append((k, seed + k - 1, dst))
    print(f"refs: task={task} prefix={prefix} n_ref={n_ref} todo={[k for k, _, _ in todo]} seeds from {seed} "
          f"steps={steps} cfg={cfg} shift={shift} lightning={lightning} edit at {gw}x{gh} -> {width}x{height}", flush=True)
    print(f"refs: prompt: {prompt_of(1)}" + ("" if prompt or len(REF_VARIATIONS) == 1 else f"  (+{len(REF_VARIATIONS) - 1} more variations, cycled)"), flush=True)
    if dry_run:
        for k, sd, dst in todo:
            g = build_graph(tpl, os.path.basename(ref_sim), prompt_of(k), negative, sd, steps, cfg, shift, gw, gh, lightning,
                            f"refs/{prefix}_ref_ai_{k:02d}")
            print(f"--- {k:02d}: {dst}\n" + json.dumps({i: v["inputs"] for i, v in g.items()}, ensure_ascii=False, indent=1)[:3000])
        return {}
    if not todo:
        return {}

    ensure_server(comfy_url, comfy_dir, comfy_python, comfy_gpu, comfy_log)
    ref_sim_name = upload(comfy_url, ref_sim)
    results = {}
    for k, sd, dst in todo:
        t0 = time.time()
        g = build_graph(tpl, ref_sim_name, prompt_of(k), negative, sd, steps, cfg, shift, gw, gh, lightning, f"refs/{prefix}_ref_ai_{k:02d}")
        pid = submit(comfy_url, g)
        print(f"refs: {k:02d}: submitted {pid} (seed {sd}), waiting...", flush=True)
        ent = wait_done(comfy_url, pid, timeout_s)
        status = ent.get("status", {})
        if status.get("status_str") == "error":
            errs = [m for m in status.get("messages", []) if m and m[0] in ("execution_error", "execution_interrupted")]
            msg = "; ".join(f"{m[0]} {m[1].get('node_type')}: {m[1].get('exception_message', '')}".strip() for m in errs) or "unknown error"
            print(f"refs: {k:02d}: FAILED after {time.time() - t0:.0f}s: {msg[:500]}", flush=True)
            results[k] = None
            continue
        files = output_files(ent)
        if not files:
            print(f"refs: {k:02d}: FAILED, no output file in history entry", flush=True)
            results[k] = None
            continue
        tmp = os.path.join(d_img, f".{prefix}_ref_ai_{k:02d}.gen.png")  # full-resolution edit, kept only until resized
        download(comfy_url, files[0], tmp)
        img = cv2.imread(tmp, cv2.IMREAD_COLOR)
        os.remove(tmp)
        if img is None:
            print(f"refs: {k:02d}: FAILED, could not decode {files[0]['filename']}", flush=True)
            results[k] = None
            continue
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst, img)
        meta = dict(task=task, prefix=prefix, index=k, source=os.path.basename(ref_sim), model=tpl[_one(tpl, "UNETLoader")]["inputs"]["unet_name"],
                    lightning=lightning, prompt=prompt_of(k), negative=negative, seed=sd, steps=steps, cfg=cfg, shift=shift,
                    edit_width=gw, edit_height=gh, width=width, height=height, template=os.path.relpath(template),
                    comfy_url=comfy_url, prompt_id=pid, comfy_output=files[0], seconds=round(time.time() - t0, 1),
                    finished_at=datetime.now().isoformat(timespec="seconds"))
        with open(os.path.splitext(dst)[0] + ".json", "w") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)
        print(f"refs: {k:02d}: {dst} ({meta['seconds']:.0f}s)", flush=True)
        results[k] = dst
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--n_ref", type=int, default=1, help="number of reference images (_01 .. _NN)")
    p.add_argument("--seed", type=int, default=0, help="seed of image 01; image k uses seed + k - 1")
    p.add_argument("--steps", type=int, default=None, help="default 20 (4 with --lightning)")
    p.add_argument("--cfg", type=float, default=None, help="default 4.0 (1.0 with --lightning)")
    p.add_argument("--shift", type=float, default=3.1)
    p.add_argument("--lightning", action="store_true", help="lightx2v 4-step Lightning LoRA (fast, a little less detail)")
    p.add_argument("--prompt", default=None, help="edit instruction (default tasks.REF_PROMPT with the task scene)")
    p.add_argument("--negative", default=REF_NEGATIVE)
    p.add_argument("--gen_scale", type=float, default=2.0, help="edit resolution = frame size x this (then scaled back)")
    p.add_argument("--force", action="store_true", help="regenerate existing _ref_ai_NN.png")
    p.add_argument("--dry_run", action="store_true", help="print the graphs, do not contact ComfyUI")
    p.add_argument("--prefix", default=None, help="file prefix (default: run_dir folder name)")
    p.add_argument("--task", default=None, help="task key in tasks.TASKS (default: derived from the prefix)")
    p.add_argument("--comfy_url", default="http://127.0.0.1:8188")
    p.add_argument("--comfy_dir", default="/workspace/tools/ComfyUI")
    p.add_argument("--comfy_python", default="/opt/miniconda3/envs/comfyui/bin/python")
    p.add_argument("--comfy_gpu", type=int, default=1, help="GPU for an auto-started ComfyUI (-1: let ComfyUI choose)")
    p.add_argument("--comfy_log", default="logs/comfyui.log")
    p.add_argument("--template", default=TEMPLATE)
    p.add_argument("--timeout", type=float, default=900, help="seconds to wait per image")
    a = p.parse_args()
    out = run(a.run_dir, a.n_ref, a.seed, a.steps, a.cfg, a.shift, a.lightning, a.prompt, a.negative, a.gen_scale, a.force,
              a.dry_run, a.prefix, a.task, a.comfy_url, a.comfy_dir, a.comfy_python, None if a.comfy_gpu < 0 else a.comfy_gpu,
              a.comfy_log, a.template, a.timeout)
    if out and any(v is None for v in out.values()):
        sys.exit(1)
