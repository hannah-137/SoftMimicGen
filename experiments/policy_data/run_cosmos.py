"""Make real-looking videos from the demo videos with Cosmos3 (video2video with an edge control video).

  python experiments/policy_data/run_cosmos.py <run_dir> --framework <cosmos-framework dir> \
      --checkpoint <Cosmos3 checkpoint dir> --gpus 2,3 [--cp 2] [--demos 0-49] [--max 25] [--port 29511]

Input: <run_dir>/demos/000-049/demo_NNN/demo_NNN_geoedge.mp4 (control video) and the reference images in
<run_dir>/refs/000-049/ (demo_NNN_<tag>.png or demo_NNN.png, checked by checks/check_references.py; 50 demos per
folder, see layout.py). One video per reference image. Cosmos keeps the reference image as frame 0 and follows the
edge video.

In a dataset folder (make_dataset.py) the script chooses the demos itself: only the numbers that status.py shows as
needs_video (the reference image passed its check and is the file that was checked, there is no video yet, no
--replace works on the number, and no other run_cosmos.py has it). The other numbers are skipped with the reason, so
the same command can run again: it makes only what is still missing. --demos (numbers and ranges, e.g. 0-49) limits
the choice, --max N takes the first N. Two jobs can run at the same time on two GPU pairs; each one writes the
numbers it takes into its run_config.json and holds .running.lock in its folder until it ends, so the second job
takes other numbers:
    run_cosmos.py <dataset> --demos 0-49 --max 25 --gpus 0,1 --cp 2 ...
    run_cosmos.py <dataset> --demos 0-49 --max 25 --gpus 2,3 --cp 2 ...
In a run folder (make_demos.sh) the script refuses to start when refs/check_references.csv is missing or lists a
failed image (--skip_check overrides). Without --demos it takes every image that passed.

Output: <run_dir>/cosmos/<checkpoint folder name>_<YYYYMMDD>_<HHMM>/ (with _2, _3, ... when that name exists) with,
per reference, 000-049/<name>.mp4 (the video) and 000-049/<name>.json (the settings the framework used, plus
policy_data: source demo, reference image name and sha1, seed), and run_config.json, run.log, debug.log and
benchmark.json at the top. A video goes into its group folder only when it has all 81 frames. Temporary files
(reference videos, specs, framework folders) are removed after a successful run; after a failure everything stays
for a look.
Stop a job with Ctrl-C (or kill <pid>): the script stops the framework, keeps the finished videos and ends. The
same command then makes the rest. When a job ended without that (for example kill -9), the next run_cosmos.py on
the dataset takes over its finished videos, if the reference image and the demo are still the same.

Settings are the ones of the earlier tests: resolution 480 tier (patched to 1024x512), 81 frames, 16 fps,
35 steps, guidance 3, control guidance 3, shift 5, seed 0. The framework's default negative prompt is used.
The prompt is short on purpose: the reference image gives the colors and materials. --prompt replaces it.
The Cosmos3 checkpoint must fit on the given GPUs: Super fp8 needs 2 GPUs (48 GB) with --cp 2, Nano fp8 needs 1.
The torchrun port is the first free one from 29511 that no running job of the same dataset has taken (--port sets
it; give --port when jobs of other datasets or run folders start at the same moment).
The framework downloads small parts (text encoder, tokenizer) with "uv run hf download". Give --tools <dir> when
uv and its caches are not in the default places: <dir>/bin/uv, <dir>/uv_cache, <dir>/uv_python, <dir>/uv_tools.
Run with any python; the framework's own .venv python is used for torchrun.
"""

import argparse
import contextlib
import csv
import glob
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "checks"))
import check_references  # noqa: E402
import layout  # noqa: E402
import status  # noqa: E402

N_FRAMES = 81  # frames of every video (spec num_frames)
PROMPT = (
    "A Franka robot arm folds a single towel on a table, realistic video. The towel has the same color and texture "
    "on both sides. The left camera is fixed and the right camera moves with the gripper. Split screen: left, the "
    "room camera; right, the camera on the robot gripper."
)


def ref_video(png: str, mp4: str, frames: int, fps: int) -> None:
    """Reference image -> 1024x512 still video, lossless. Cosmos reads the reference as a video."""
    img, w, h, reason = check_references.load_and_resize(png)
    if img is None:
        sys.exit(f"{png}: {reason}")
    tmp = mp4[: -len(".mp4")] + ".png"
    cv2.imwrite(tmp, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-i", tmp, "-frames:v", str(frames), "-r", str(fps),
           "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", mp4]
    subprocess.run(cmd, check=True)
    os.remove(tmp)


def spec(name: str, control: str, ref: str, prompt: str, seed: int, steps: int) -> dict:
    return {
        "model_mode": "video2video", "resolution": "480", "aspect_ratio": "16,9", "num_frames": 81, "fps": 16,
        "shift": 5.0, "num_steps": steps, "seed": seed, "num_video_frames_per_chunk": 81, "num_conditional_frames": 1,
        "share_vision_temporal_positions": True, "negative_metadata_mode": "none", "negative_prompt_keep_metadata": False,
        "guidance": 3.0, "control_guidance": 3.0, "name": name,
        "edge": {"control_path": control, "preset_edge_threshold": "medium"},
        "vision_path": ref, "num_first_chunk_conditional_frames": 1, "prompt": prompt,
    }


def checked_references(run_dir: str, wanted: list | None, skip_check: bool) -> list:
    """Run folder: (demo index, reference name) pairs to run: the images that passed check_references.py."""
    path = f"{run_dir}/refs/check_references.csv"
    if skip_check:
        if wanted is None:
            sys.exit("--skip_check needs --demos")
        pairs = []
        for i in wanted:
            names = sorted(os.path.basename(f)[: -len(".png")]
                           for f in glob.glob(f"{layout.ref_dir(run_dir + '/refs', i)}/{layout.demo_name(i)}*.png"))
            pairs += [(i, n) for n in names]
        return pairs
    if not os.path.isfile(path):
        sys.exit(f"{path} not found: run check_references.py first (or use --skip_check with --demos)")
    rows = list(csv.DictReader(open(path)))
    failed = [r["name"] for r in rows if r["passed"] != "True"]
    passed = [(int(r["demo"]), r["name"]) for r in rows if r["passed"] == "True"]
    if wanted is None:
        if failed:
            sys.exit(f"{len(failed)} reference images failed the check: see {run_dir}/refs/failed_references.txt")
        return passed
    bad = [n for n in failed if int(n.split("_")[1]) in wanted]
    if bad:
        sys.exit(f"references {bad} did not pass the check: see {run_dir}/refs/failed_references.txt")
    return [(i, n) for i, n in passed if i in wanted]


def new_run_folder(run_dir: str, checkpoint: str) -> str:
    """cosmos/<checkpoint>_<YYYYMMDD>_<HHMM>, with _2, _3, ... when that folder exists."""
    base = f"{run_dir}/cosmos/{os.path.basename(os.path.normpath(checkpoint))}_{time.strftime('%Y%m%d_%H%M')}"
    out, k = base, 1
    while True:
        try:
            os.makedirs(out)
            return out
        except FileExistsError:
            k += 1
            out = f"{base}_{k}"


def complete(mp4: str) -> bool:
    """True when the video file has all N_FRAMES frames (a stopped writer leaves a file that cannot be read)."""
    if not os.path.isfile(mp4):
        return False
    cap = cv2.VideoCapture(mp4)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n >= N_FRAMES


def take_over(ds: str, rows: list) -> None:
    """Finished videos of jobs that ended without moving them (for example kill -9): move them into their group
    folders when the number still needs a video and its reference image and source demo are still the same. Call it
    while holding lock(ds)."""
    for cfg_path in sorted(glob.glob(f"{ds}/cosmos/*/run_config.json")):
        out = os.path.dirname(cfg_path)
        if status.lock_holder(out, status.RUNNING_LOCK) is not None:
            continue  # this job still runs
        try:
            cfg = json.load(open(cfg_path))
        except (OSError, ValueError):
            continue
        meta = cfg.get("policy_data", {})
        for idx, name in zip(cfg.get("demos", []), cfg.get("references", [])):
            m, row = meta.get(name), rows[idx] if idx < len(rows) else None
            if not m or not row or row["state"] != "needs_video" or row["note"] or row["reference"] != name:
                continue  # another job makes this number now, or it is not waiting for a video
            ref = f"{layout.ref_dir(ds + '/refs', idx)}/{name}.png"
            if row["source"] != m.get("source") or status.file_sha1(ref) != m.get("reference_sha1"):
                continue  # made for another image or demo: leave it
            if place_video(out, idx, name, m):
                print(f"{name}: took over the finished video of {os.path.basename(out)}", flush=True)
                row["state"] = "needs_review"


def dataset_pairs(ds: str, wanted: list | None, max_n: int | None, args, stack) -> tuple:
    """Dataset: choose the needs_video numbers, make the run folder and claim the numbers, the GPUs and a torchrun
    port, all under the list lock. -> (run folder, pairs, {name: source}). The claim holds until the job ends
    (.running.lock in the folder, also held by the framework process)."""
    with status.lock(ds):
        rows, _, _ = status.collect(ds)
        take_over(ds, rows)
        chosen, sources, taken = [], {}, []
        for r in rows:
            if wanted is not None and r["demo"] not in wanted:
                continue
            if r["state"] == "needs_video" and not r["note"]:
                chosen.append((r["demo"], r["reference"]))
                sources[r["reference"]] = r["source"]
            elif r["state"] == "needs_video":
                taken.append(r)
            elif wanted is not None:
                print(f"{layout.demo_name(r['demo'])}: {r['state']}{', ' + r['note'] if r['note'] else ''}, skipped",
                      flush=True)
        if max_n is not None:
            chosen = chosen[:max_n]
        if not chosen:
            if taken:
                runs = sorted({t["note"][len("cosmos running ("):-1] for t in taken})
                sys.exit(f"all {len(taken)} ready demos are taken by running jobs: {', '.join(runs)}")
            sys.exit("no demo is ready for a video (state needs_video): see status.py")
        busy = {c.get("port") for c in status.running_cosmos(ds).values()}  # a job may not listen on it yet
        out = new_run_folder(ds, args.checkpoint)
        stack.enter_context(status.lock(out, status.RUNNING_LOCK))
        status.write_json(f"{out}/run_config.json", {"demos": [i for i, _ in chosen],
                                                    "references": [n for _, n in chosen], "gpus": args.gpus,
                                                    "port": args.port or free_port(skip=busy)})
    return out, chosen, sources


def place_video(out: str, idx: int, name: str, meta: dict) -> bool:
    """<out>/<name>/vision.mp4 -> <out>/<group>/<name>.mp4 and sample_args.json -> <name>.json with the key
    policy_data added. Only a complete video moves. -> True when it moved."""
    src = f"{out}/{name}"
    if not complete(f"{src}/vision.mp4"):
        return False
    dst = f"{out}/{layout.chunk(idx)}"
    os.makedirs(dst, exist_ok=True)
    info = {}
    if os.path.isfile(f"{src}/sample_args.json"):
        try:
            info = json.load(open(f"{src}/sample_args.json"))
        except ValueError:
            info = {"sample_args": open(f"{src}/sample_args.json").read()}
    if not isinstance(info, dict):
        info = {"sample_args": info}
    info["policy_data"] = meta
    status.write_json(f"{dst}/{name}.json", info)  # the json first: status.py reads it with the video
    os.replace(f"{src}/vision.mp4", f"{dst}/{name}.mp4")
    shutil.rmtree(src)
    return True


def tidy(out: str, pairs: list, meta: dict) -> list:
    """After the run: move every complete video (place_video), remove the temporary files when nothing is missing.
    Returns the names whose video is missing (their folders stay)."""
    missing = [name for idx, name in pairs if not place_video(out, idx, name, meta.get(name, {}))]
    if not missing:
        for d in ("refs", "specs"):
            shutil.rmtree(f"{out}/{d}", ignore_errors=True)
        for f in ("console.log",):
            if os.path.isfile(f"{out}/{f}"):
                os.remove(f"{out}/{f}")
    return missing


def framework_env(fw: str, tools: str | None) -> dict:
    """Environment for the framework's .venv: CUDA libs from the pip packages, like the framework's own setup."""
    env = dict(os.environ)
    if tools:
        tools = os.path.abspath(tools)
        env["PATH"] = f"{tools}/bin:" + env.get("PATH", "")
        env["UV_CACHE_DIR"], env["UV_PYTHON_INSTALL_DIR"] = f"{tools}/uv_cache", f"{tools}/uv_python"
        env["UV_TOOL_DIR"], env["UV_TOOL_BIN_DIR"] = f"{tools}/uv_tools", f"{tools}/bin"
    if not any(os.path.isfile(f"{d}/uv") for d in env["PATH"].split(":") if d):
        sys.exit("uv not found on PATH: give --tools <dir> with bin/uv")
    nv = glob.glob(f"{fw}/.venv/lib/python3.*/site-packages/nvidia")
    if nv:
        libs = [d for d in glob.glob(f"{nv[0]}/*/lib") if os.path.isdir(d)]
        env["LD_LIBRARY_PATH"] = ":".join(libs)
        env["NVRTC_HOME"], env["CURAND_HOME"], env["CUDNN_HOME"] = f"{nv[0]}/cuda_nvrtc", f"{nv[0]}/curand", f"{nv[0]}/cudnn"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def free_port(start: int = 29511, skip: set = frozenset()) -> int:
    """The first port from start that nothing listens on and no running job has taken (for the torchrun master)."""
    for port in range(start, start + 200):
        if port in skip:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    sys.exit(f"no free port in {start}-{start + 199}: give --port")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="dataset folder (make_dataset.py) or run folder (make_demos.sh)")
    ap.add_argument("--demos", nargs="+", default=None, help="demo numbers and ranges, e.g. 0-49 (default: all)")
    ap.add_argument("--max", type=int, default=None, help="dataset: take at most this many numbers")
    ap.add_argument("--skip_check", action="store_true", help="run folder: do not require check_references.csv")
    ap.add_argument("--framework", required=True, help="cosmos-framework folder (with .venv)")
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME"), help="Hugging Face cache with the text encoder (default: $HF_HOME)")
    ap.add_argument("--tools", default=None, help="folder with bin/uv and the uv caches (see above)")
    ap.add_argument("--checkpoint", required=True, help="Cosmos3 checkpoint folder")
    ap.add_argument("--gpus", default="0", help="GPU indices, comma separated (default 0)")
    ap.add_argument("--cp", type=int, default=1, help="context parallel size (2 for Super fp8 on 2 GPUs)")
    ap.add_argument("--port", type=int, default=None, help="torchrun master port (default: the first free from 29511)")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--dry_run", action="store_true", help="write the specs and print the command, do not run")
    args = ap.parse_args()
    if not args.hf_home:
        sys.exit("set --hf_home (or HF_HOME): the Hugging Face cache folder of the framework")
    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        sys.exit(f"no such folder: {run_dir}")
    wanted = status.parse_demos(args.demos)
    if args.max is not None and args.max < 1:
        sys.exit("--max must be 1 or more")
    dataset = status.is_dataset(run_dir)
    with contextlib.ExitStack() as stack:
        if dataset:
            if args.skip_check:
                sys.exit("--skip_check is not used in a dataset folder: status.py decides which images are ready")
            out, pairs, sources = dataset_pairs(run_dir, wanted, args.max, args, stack)
        else:
            pairs = checked_references(run_dir, wanted, args.skip_check)
            if args.max is not None:
                pairs = pairs[:args.max]
            if not pairs:
                sys.exit("no reference images to run")
            out, sources = new_run_folder(run_dir, args.checkpoint), {}
        code = run(args, run_dir, out, pairs, sources, dataset)
    if dataset and not args.dry_run:  # after .running.lock is free, so the numbers no longer count as running
        status.update(run_dir)
    sys.exit(code)


def run(args, run_dir: str, out: str, pairs: list, sources: dict, dataset: bool) -> int:
    """Write the specs, run the framework, move the videos to their group folders. -> exit code"""
    os.makedirs(f"{out}/specs", exist_ok=True)
    os.makedirs(f"{out}/refs", exist_ok=True)
    config = json.load(open(f"{out}/run_config.json")) if os.path.isfile(f"{out}/run_config.json") else {}
    port = config.get("port") or args.port or free_port()
    config.update({"checkpoint": os.path.abspath(args.checkpoint), "demos": [i for i, _ in pairs],
                   "references": [n for _, n in pairs], "prompt": args.prompt, "seed": args.seed,
                   "steps": args.steps, "gpus": args.gpus, "cp": args.cp, "port": port,
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    status.write_json(f"{out}/run_config.json", config)
    specs, meta = [], {}
    for idx, name in pairs:
        control = f"{layout.demo_dir(run_dir, idx)}/{layout.demo_name(idx)}_geoedge.mp4"
        ref = f"{layout.ref_dir(run_dir + '/refs', idx)}/{name}.png"
        for path in (control, ref):
            if not os.path.isfile(path):
                sys.exit(f"missing: {path}")
        meta[name] = {"source": sources.get(name, ""), "reference": name, "reference_sha1": status.file_sha1(ref),
                      "seed": args.seed, "cosmos_run": os.path.basename(out)}
        ref_mp4 = f"{out}/refs/{name}.mp4"
        ref_video(ref, ref_mp4, frames=81, fps=16)
        path = f"{out}/specs/{name}.json"
        with open(path, "w") as f:
            json.dump(spec(name, control, ref_mp4, args.prompt, args.seed, args.steps), f, indent=1)
        specs.append(path)
    config["policy_data"] = meta  # lets a later job take over the videos if this one ends without moving them
    status.write_json(f"{out}/run_config.json", config)

    n_gpu = len(args.gpus.split(","))
    env = framework_env(os.path.abspath(args.framework), args.tools)
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["HF_HOME"] = os.path.abspath(args.hf_home)
    cmd = [f"{os.path.abspath(args.framework)}/.venv/bin/torchrun", f"--nproc-per-node={n_gpu}", f"--master-port={port}",
           f"{os.path.dirname(os.path.abspath(__file__))}/cosmos_launch.py",
           "--parallelism-preset=throughput", f"--dp-shard-size={n_gpu}", "--dp-replicate-size=1", f"--cp-size={args.cp}",
           "--cfgp-size=1", "-i", *specs, "-o", out, "--checkpoint-path", os.path.abspath(args.checkpoint),
           "--no-guardrails", "--benchmark", "--experiment-overrides", "model.config.tokenizer.encode_chunk_frames.480=4"]
    print(f"[run_cosmos] {len(pairs)} videos: demos {status.ranges(i for i, _ in pairs)} -> {out}", flush=True)
    print("[run_cosmos]", " ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    keep = [fd for fd in [status.lock_fd(out, status.RUNNING_LOCK)] if fd is not None]
    rc, stopped = run_framework(cmd, os.path.abspath(args.framework), env, f"{out}/run.log", keep)
    if dataset:
        with status.lock(run_dir):  # the videos appear under the list lock
            missing = tidy(out, pairs, meta)
    else:
        missing = tidy(out, pairs, meta)
    print(f"[run_cosmos] exit {rc}, {len(pairs) - len(missing)}/{len(pairs)} videos in {out} (log: {out}/run.log)")
    if missing:
        print(f"[run_cosmos] {'stopped; ' if stopped else ''}missing: {', '.join(missing)}. Run the same command "
              "again to make them.")
    return 0 if rc == 0 and not missing else 1


def run_framework(cmd: list, cwd: str, env: dict, log_path: str, keep_fds: list) -> tuple[int, bool]:
    """Run torchrun in its own process group. The lock file descriptors in keep_fds stay open in it, so the claim of
    the job lasts as long as the framework runs. Ctrl-C, SIGTERM or SIGHUP stop the whole group (SIGKILL after
    120 s). -> (exit code, stopped)"""
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                pass_fds=keep_fds)
        try:
            with status.stop_signals():
                return proc.wait(), False
        except KeyboardInterrupt:
            print("[run_cosmos] stopping the framework, the finished videos are kept", flush=True)
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            return proc.returncode, True


if __name__ == "__main__":
    main()
