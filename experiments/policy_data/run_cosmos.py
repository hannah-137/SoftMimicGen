"""Make real-looking videos from the demo videos with Cosmos3 (video2video with an edge control video).

  python experiments/policy_data/run_cosmos.py <run_dir> --framework <cosmos-framework dir> \
      --checkpoint <Cosmos3 checkpoint dir> --gpus 2,3 [--cp 2] [--port 29511] [--demos 0 1]

Input: <run_dir>/videos/demo_NNN_geoedge.mp4 (control video) and the reference images in <run_dir>/refs/
(demo_NNN_<tag>.png or demo_NNN.png, checked by check_references.py). One video per reference image.
Run check_references.py first. This script refuses to start when refs/check_references.csv is missing or lists
a failed image (--skip_check overrides). Without --demos it takes every image that passed. Cosmos keeps the
reference image as frame 0 and follows the edge video.
Output: <run_dir>/cosmos/<checkpoint folder name>_<YYYYMMDD>_<HHMM>/ with, per reference, <name>.mp4 (the video)
and <name>.json (the settings the framework used), plus run_config.json, run.log, debug.log and benchmark.json.
The folder name is fixed; every run gets a new one. Temporary files (reference videos, specs, framework
folders) are removed after a successful run; after a failure everything stays for a look.

Settings are the ones of the earlier tests: resolution 480 tier (patched to 1024x512), 81 frames, 16 fps,
35 steps, guidance 3, control guidance 3, shift 5, seed 0. The framework's default negative prompt is used.
The prompt is short on purpose: the reference image gives the colors and materials. --prompt replaces it.
The Cosmos3 checkpoint must fit on the given GPUs: Super fp8 needs 2 GPUs (48 GB) with --cp 2, Nano fp8 needs 1.
The framework downloads small parts (text encoder, tokenizer) with "uv run hf download". Give --tools <dir> when
uv and its caches are not in the default places: <dir>/bin/uv, <dir>/uv_cache, <dir>/uv_python, <dir>/uv_tools.
Run with any python; the framework's own .venv python is used for torchrun.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_references  # noqa: E402

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
    """(demo index, reference name) pairs to run: the images that passed check_references.py."""
    path = f"{run_dir}/refs/check_references.csv"
    if skip_check:
        if wanted is None:
            sys.exit("--skip_check needs --demos")
        pairs = []
        for i in wanted:
            names = sorted(os.path.basename(f)[: -len(".png")] for f in glob.glob(f"{run_dir}/refs/demo_{i:03d}*.png"))
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


def tidy(out: str, names: list) -> list:
    """After the run: <out>/<name>/vision.mp4 -> <out>/<name>.mp4, sample_args.json -> <name>.json, temporary files
    removed. Returns the names whose video is missing (their folders are kept)."""
    missing = []
    for name in names:
        src = f"{out}/{name}"
        if not os.path.isfile(f"{src}/vision.mp4"):
            missing.append(name)
            continue
        os.replace(f"{src}/vision.mp4", f"{out}/{name}.mp4")
        if os.path.isfile(f"{src}/sample_args.json"):
            os.replace(f"{src}/sample_args.json", f"{out}/{name}.json")
        shutil.rmtree(src)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="demo run folder (output of make_demos.sh)")
    ap.add_argument("--demos", type=int, nargs="+", default=None, help="demo indices (default: all that passed the check)")
    ap.add_argument("--skip_check", action="store_true", help="do not require check_references.csv")
    ap.add_argument("--framework", required=True, help="cosmos-framework folder (with .venv)")
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME"), help="Hugging Face cache with the text encoder (default: $HF_HOME)")
    ap.add_argument("--tools", default=None, help="folder with bin/uv and the uv caches (see above)")
    ap.add_argument("--checkpoint", required=True, help="Cosmos3 checkpoint folder")
    ap.add_argument("--gpus", default="0", help="GPU indices, comma separated (default 0)")
    ap.add_argument("--cp", type=int, default=1, help="context parallel size (2 for Super fp8 on 2 GPUs)")
    ap.add_argument("--port", type=int, default=29511, help="torchrun master port; use another port for a second job")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--dry_run", action="store_true", help="write the specs and print the command, do not run")
    args = ap.parse_args()
    if not args.hf_home:
        sys.exit("set --hf_home (or HF_HOME): the Hugging Face cache folder of the framework")
    run_dir = os.path.abspath(args.run_dir)
    pairs = checked_references(run_dir, args.demos, args.skip_check)
    if not pairs:
        sys.exit("no reference images to run")
    name = f"{os.path.basename(os.path.normpath(args.checkpoint))}_{time.strftime('%Y%m%d_%H%M')}"
    out = f"{run_dir}/cosmos/{name}"
    os.makedirs(f"{out}/specs", exist_ok=True)
    os.makedirs(f"{out}/refs", exist_ok=True)
    with open(f"{out}/run_config.json", "w") as f:
        json.dump({"checkpoint": os.path.abspath(args.checkpoint), "references": [n for _, n in pairs], "prompt": args.prompt,
                   "seed": args.seed, "steps": args.steps, "gpus": args.gpus, "cp": args.cp, "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  f, indent=1)
    specs = []
    for idx, name in pairs:
        control = f"{run_dir}/videos/demo_{idx:03d}_geoedge.mp4"
        ref = f"{run_dir}/refs/{name}.png"
        for path in (control, ref):
            if not os.path.isfile(path):
                sys.exit(f"missing: {path}")
        ref_mp4 = f"{out}/refs/{name}.mp4"
        ref_video(ref, ref_mp4, frames=81, fps=16)
        path = f"{out}/specs/{name}.json"
        with open(path, "w") as f:
            json.dump(spec(name, control, ref_mp4, args.prompt, args.seed, args.steps), f, indent=1)
        specs.append(path)

    n_gpu = len(args.gpus.split(","))
    env = framework_env(os.path.abspath(args.framework), args.tools)
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["HF_HOME"] = os.path.abspath(args.hf_home)
    cmd = [f"{os.path.abspath(args.framework)}/.venv/bin/torchrun", f"--nproc-per-node={n_gpu}", f"--master-port={args.port}",
           f"{os.path.dirname(os.path.abspath(__file__))}/cosmos_launch.py",
           "--parallelism-preset=throughput", f"--dp-shard-size={n_gpu}", "--dp-replicate-size=1", f"--cp-size={args.cp}",
           "--cfgp-size=1", "-i", *specs, "-o", out, "--checkpoint-path", os.path.abspath(args.checkpoint),
           "--no-guardrails", "--benchmark", "--experiment-overrides", "model.config.tokenizer.encode_chunk_frames.480=4"]
    print("[run_cosmos]", " ".join(cmd), flush=True)
    if args.dry_run:
        return
    with open(f"{out}/run.log", "w") as log:
        rc = subprocess.run(cmd, cwd=os.path.abspath(args.framework), env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    missing = tidy(out, [name for _, name in pairs])
    print(f"[run_cosmos] exit {rc}, {len(pairs) - len(missing)}/{len(pairs)} videos in {out} (log: {out}/run.log)")
    if missing:
        print(f"[run_cosmos] missing: {', '.join(missing)}")
    sys.exit(0 if rc == 0 and not missing else 1)


if __name__ == "__main__":
    main()
