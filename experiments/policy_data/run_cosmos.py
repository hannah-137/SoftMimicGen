"""Make real-looking videos from the demo videos with Cosmos3 (video2video with control videos from the simulator).

  python experiments/policy_data/run_cosmos.py <run_dir> --framework <cosmos-framework dir> \
      --checkpoint <Cosmos3 checkpoint dir> --gpus 2,3 [--cp 2] [--demos 0-49] [--max 25] [--spec v2] [--port 29511]

Input: <run_dir>/demos/000-049/demo_NNN/demo_NNN_geoedge.mp4 (control video) and the reference images in
<run_dir>/refs/000-049/ (demo_NNN_<tag>.png or demo_NNN.png, checked by checks/check_references.py; 50 demos per
folder, see layout.py). One video per reference image. Cosmos keeps the reference image as frame 0 and follows the
edge video. A spec version can add controls (key cosmos.controls of specs/<version>.json). All controls have the
same weight.
  depth (from v7): a depth video from the raw depth in the demo hdf5 of <run_dir> (1 / depth as gray, near is white).
  color (from v8): a color guide. Each video is then made in two passes, one video after the other. Pass 1 uses the
      edge video only. The guide is the pass-1 video with the towel painted in one color: the color that the towel
      has in the reference image (the towel pixels come from the raw instance ids in the demo hdf5). Pass 2 gets the
      guide as the Cosmos blur control, next to the other controls of the version, and makes the final video. The
      framework starts twice per video (about 1 minute each time). The video goes into its group folder when its
      pass 2 is done, so the review can start while the job runs.

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
policy_data: source demo, reference image name and sha1, seed, spec version), and run_config.json, run.log,
debug.log and benchmark.json at the top. A video goes into its group folder only when it has all 81 frames.
Temporary files (reference videos, control videos, pass-1 videos, specs, framework folders) are removed after a
successful run; after a failure everything stays for a look.
Stop a job with Ctrl-C (or kill <pid>): the script stops the framework, keeps the finished videos and ends. The
same command then makes the rest. When a job ended without that (for example kill -9), the next run_cosmos.py on
the dataset takes over its finished videos, if the reference image and the demo are still the same.

Settings are the ones of the earlier tests: resolution 480 tier (patched to 1024x512), 81 frames, 16 fps,
35 steps, guidance 3, control guidance 3, shift 5, seed 0. The framework's default negative prompt is used.
The prompt comes from the spec version of each reference image (column spec of refs/references.csv; the file
specs/<version>.json, key cosmos, see SPEC.md): a short prompt (the reference image gives the colors and materials)
and a sentence that is filled from the axes of the image and put after the first sentence of the prompt (v2: "The
towel is pewter bamboo fiber, plain, one solid color."). The sentence helps Cosmos keep the towel pattern while the
towel moves (the reference image fixes only frame 0). So a video always follows the version of its image.
--spec v3 takes only the images of that version; the others are skipped. A json file also works, for tests in a run
folder: --spec <file.json>. An image that make_references.py did not make (no row in refs/references.csv) gets the
version of --spec, else SPEC in status.py; its spec needs an empty cosmos sentence.
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
import variations  # noqa: E402

N_FRAMES = 81  # frames of every video (spec num_frames)
FIRST = "_pass1"  # after the name of a video: its pass-1 video (control "color")
GUIDE_INSET = 6  # color guide: the towel color is taken this many px inside the towel outline of frame 0
GUIDE_MIN_PX = 150  # color guide: a view with fewer such pixels at frame 0 takes the color of the other view


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


def load_spec(version: str) -> variations.Spec:
    try:
        return variations.load(version)
    except ValueError as e:
        sys.exit(str(e))


def image_version(row: dict | None, chosen: variations.Spec | None) -> str:
    """The spec version of a reference image: its row of refs/references.csv, else --spec, else SPEC."""
    if row is not None:
        return status.row_spec(row)
    return chosen.version if chosen else status.SPEC


def video_prompt(name: str, row: dict | None, chosen: variations.Spec | None) -> tuple:
    """(spec, prompt) for the video of one reference image: the Cosmos prompt of the version of the image, with the
    sentence filled from the axes of the image (its row of refs/references.csv)."""
    version = image_version(row, chosen)
    try:
        spec = chosen if chosen and chosen.version == version else variations.load(version)
    except ValueError as e:
        sys.exit(f"{name}: {e}. For an image of a test spec, give --spec <its json file>")
    if row is None:
        if spec.cosmos.get("sentence"):
            sys.exit(f"{name}: no row in refs/references.csv (an image not made by make_references.py), so the "
                     f"sentence of spec {spec.version} cannot be filled: use a spec with an empty cosmos sentence")
        return spec, spec.cosmos["prompt"]
    try:
        return spec, spec.cosmos_prompt(row)
    except (KeyError, IndexError, ValueError) as e:
        sys.exit(f"{name}: refs/references.csv has no value for {e} of spec {spec.version}")


def video_prompts(run_dir: str, pairs: list, sources: dict, chosen: variations.Spec | None) -> dict:
    """name -> (spec, prompt) for every pair (see video_prompt). Called before the run folder is made, so a wrong
    spec stops the script before it leaves an empty folder."""
    rows = status.latest_refs(status.read_csv(f"{run_dir}/refs/references.csv"))
    return {name: video_prompt(name, rows.get((name, sources.get(name, ""))), chosen) for _, name in pairs}


def other_version(pairs: list, rows: dict, sources: dict, chosen: variations.Spec | None) -> list:
    """The (demo, name) pairs whose image is not of the version of --spec (none without --spec)."""
    if chosen is None:
        return []
    return [(i, n) for i, n in pairs
            if image_version(rows.get((n, sources.get(n, ""))), chosen) != chosen.version]


def spec(name: str, control: str, ref: str, prompt: str, seed: int, steps: int, depth: str | None = None,
         color: str | None = None) -> dict:
    out = {
        "model_mode": "video2video", "resolution": "480", "aspect_ratio": "16,9", "num_frames": 81, "fps": 16,
        "shift": 5.0, "num_steps": steps, "seed": seed, "num_video_frames_per_chunk": 81, "num_conditional_frames": 1,
        "share_vision_temporal_positions": True, "negative_metadata_mode": "none", "negative_prompt_keep_metadata": False,
        "guidance": 3.0, "control_guidance": 3.0, "name": name,
        "edge": {"control_path": control, "preset_edge_threshold": "medium"},
        "vision_path": ref, "num_first_chunk_conditional_frames": 1, "prompt": prompt,
    }
    if depth:
        out["depth"] = {"control_path": depth}
    if color:  # the color guide is the blur control of Cosmos
        out["blur"] = {"control_path": color}
    return out


def demo_hdf5(run_dir: str) -> str:
    """The demo file of a run or dataset folder: the one hdf5 without _failed."""
    path = next((f"{run_dir}/{f}" for f in sorted(os.listdir(run_dir)) if f.endswith(".hdf5") and "_failed" not in f),
                None)
    if path is None:
        sys.exit(f"no hdf5 in {run_dir}")
    return path


def depth_control(h5path: str, demo: int, mp4: str) -> None:
    """Depth control video of one demo, from its raw depth: 1 / depth as gray, so near is white; nothing rendered is
    black. Each view is scaled between the 1st and the 99th percentile of its video. Room | wrist, the frames of the
    edge video, lossless."""
    import numpy as np

    import make_videos

    with status.open_hdf5(h5path) as h5:
        obs = h5["data"][f"demo_{demo}"]["obs"]
        views = []
        for cam in make_videos.camera_keys(h5, None, None):
            key = f"{make_videos.prefix(cam)}_depth_raw"
            if key not in obs:
                sys.exit(f"{h5path}: no {key} (demos made with --no_raw): the depth control needs the raw depth")
            d = obs[key][:][make_videos.sample_idx(obs[key].shape[0])][..., 0]
            ok = np.isfinite(d) & (d > 0)
            disparity = np.where(ok, 1.0 / np.maximum(d, 1e-6), 0.0)
            lo, hi = np.percentile(disparity[ok], [1, 99])
            gray = np.clip((disparity - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            gray[~ok] = 0.0
            views.append(np.repeat(np.round(gray * 255.0).astype(np.uint8)[..., None], 3, axis=-1))
    frames = [make_videos.tile(a, b) for a, b in zip(*views)]
    make_videos.write(mp4, frames, make_videos.FPS, make_videos.LOSSLESS_RGB, check=True)


def towel_tables(h5path: str, h5, demo: int) -> dict:
    """Camera -> {instance id: prim path} of one demo, for the color guide. Stops when the demo has no raw instance
    ids or no table for them (demos made with --no_raw)."""
    import make_videos

    table_path = h5path[: -len(".hdf5")] + "_instance_ids.json"  # a run folder; a dataset has the table in each demo
    run_tables = json.load(open(table_path)) if os.path.isfile(table_path) else {}
    group = h5["data"][f"demo_{demo}"]
    tables = check_references.demo_tables(group, run_tables)
    for cam in make_videos.camera_keys(h5, None, None):
        key = f"{make_videos.prefix(cam)}_instance_raw"
        if key not in group["obs"] or not tables.get(cam):
            sys.exit(f"{h5path}: no {key} or no instance id table for it (demos made with --no_raw): the color "
                     "guide needs them to find the towel")
    return tables


def cosmos_blur(img):
    """The blur that Cosmos applies when it makes a blur control itself (preset medium): half size, bilateral filter
    (d 30, sigma color 150, sigma space 100 at 720 px), full size, then 1/10 size and back. Cosmos uses a control
    file as it is, so the color guide gets this blur here."""
    h, w = img.shape[:2]
    x = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    scale = max(x.shape[:2]) / 720.0
    d = max(1, int(round(30 * scale)))
    d = d + 1 if d % 2 == 0 else d
    x = cv2.bilateralFilter(x, d, max(1.0, 150 * scale), max(1.0, 100 * scale))
    x = cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)
    small = cv2.resize(x, (int(w / 10), int(h / 10)), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def color_guide(h5path: str, demo: int, ref_png: str, first: str, mp4: str) -> None:
    """Color guide of one demo: its pass-1 video (first) with the towel painted in one color, then cosmos_blur.
    The towel pixels are the instance ids of the object in the demo. The color is the median color (Lab) of the towel
    in the reference image at frame 0, GUIDE_INSET px inside its outline, per view. A view with fewer than
    GUIDE_MIN_PX such pixels takes the color of the other view. Everything else stays as it is in the pass-1 video.
    Room | wrist, lossless. Raises ValueError when the guide cannot be made."""
    import numpy as np

    import make_videos

    img, _, _, reason = check_references.load_and_resize(ref_png)
    if img is None:
        raise ValueError(f"{ref_png}: {reason}")
    cap, frames = cv2.VideoCapture(first), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    with status.open_hdf5(h5path) as h5:
        tables = towel_tables(h5path, h5, demo)
        obs = h5["data"][f"demo_{demo}"]["obs"]
        masks = []  # per view: (frames, H, W) bool, True on the towel
        for cam in make_videos.camera_keys(h5, None, None):
            ids = obs[f"{make_videos.prefix(cam)}_instance_raw"]
            ids = ids[:][make_videos.sample_idx(ids.shape[0])][..., 0]
            masks.append(check_references.group_ids(ids, tables[cam]) == 2)
    n, h, half = masks[0].shape
    size = frames[0].shape[1::-1] if frames else (0, 0)
    if len(frames) != n or size != (2 * half, h) or img.shape[1::-1] != size:
        raise ValueError(f"{first}: {len(frames)} frames of {size[0]}x{size[1]}, the demo has {n} frames of "
                         f"{2 * half}x{h}")
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    kernel = np.ones((2 * GUIDE_INSET + 1, 2 * GUIDE_INSET + 1), np.uint8)
    colors = []  # per view: the towel color (Lab), or None
    for k, mask in enumerate(masks):
        inner = cv2.erode(mask[0].astype(np.uint8), kernel) > 0
        view = lab[:, k * half:(k + 1) * half]
        colors.append(np.median(view[inner], axis=0) if inner.sum() >= GUIDE_MIN_PX else None)
    seen = [c for c in colors if c is not None]
    if not seen:
        raise ValueError("the towel is not visible at frame 0, so it has no color for the guide")
    rgb = [np.clip(np.round(cv2.cvtColor((seen[0] if c is None else c).reshape(1, 1, 3).astype(np.float32),
                                         cv2.COLOR_Lab2RGB) * 255.0), 0, 255).astype(np.uint8)[0, 0] for c in colors]
    out = []
    for t, frame in enumerate(frames):
        for k, mask in enumerate(masks):
            frame[:, k * half:(k + 1) * half][mask[t]] = rgb[k]
        out.append(cosmos_blur(frame))
    make_videos.write(mp4, out, make_videos.FPS, make_videos.LOSSLESS_RGB, check=True)


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
        chosen, sources, taken, other = [], {}, [], []
        for r in rows:
            if wanted is not None and r["demo"] not in wanted:
                continue
            if r["state"] == "needs_video" and not r["note"]:
                if args.spec and r["spec"] and r["spec"] != args.spec.version:
                    other.append(r["demo"])
                    continue
                chosen.append((r["demo"], r["reference"]))
                sources[r["reference"]] = r["source"]
            elif r["state"] == "needs_video":
                taken.append(r)
            elif wanted is not None:
                print(f"{layout.demo_name(r['demo'])}: {r['state']}{', ' + r['note'] if r['note'] else ''}, skipped",
                      flush=True)
        if other:
            print(f"demos {status.ranges(other)}: the reference image is not spec {args.spec.version}, skipped",
                  flush=True)
        if max_n is not None:
            chosen = chosen[:max_n]
        if not chosen:
            if taken:
                runs = sorted({t["note"][len("cosmos running ("):-1] for t in taken})
                sys.exit(f"all {len(taken)} ready demos are taken by running jobs: {', '.join(runs)}")
            sys.exit("no demo is ready for a video (state needs_video): see status.py")
        prompts = video_prompts(ds, chosen, sources, args.spec)
        busy = {c.get("port") for c in status.running_cosmos(ds).values()}  # a job may not listen on it yet
        out = new_run_folder(ds, args.checkpoint)
        stack.enter_context(status.lock(out, status.RUNNING_LOCK))
        status.write_json(f"{out}/run_config.json", {"demos": [i for i, _ in chosen],
                                                    "references": [n for _, n in chosen], "gpus": args.gpus,
                                                    "port": args.port or free_port(skip=busy)})
    return out, chosen, sources, prompts


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
    """After the run: move every complete video (place_video) that is not in its group folder yet, remove the
    temporary files when nothing is missing. Returns the names whose video is missing (their folders stay)."""
    missing = [name for idx, name in pairs if not complete(f"{out}/{layout.chunk(idx)}/{name}.mp4")
               and not place_video(out, idx, name, meta.get(name, {}))]
    if not missing:
        for d in ["refs", "specs", "controls"] + [name + FIRST for _, name in pairs]:
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
    ap.add_argument("--spec", default=None,
                    help="make videos only for the images of this spec version, e.g. v3 (default: all; each video "
                         "follows the version of its image); a json file for tests in a run folder")
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
    args.spec = load_spec(args.spec) if args.spec else None
    if args.spec and dataset and os.path.dirname(args.spec.path) != variations.SPECS_DIR:
        sys.exit(f"--spec {args.spec.path}: a dataset uses only the versions in specs/ "
                 f"({', '.join(variations.versions())}); a json file is for tests in a run folder")
    with contextlib.ExitStack() as stack:
        if dataset:
            if args.skip_check:
                sys.exit("--skip_check is not used in a dataset folder: status.py decides which images are ready")
            out, pairs, sources, prompts = dataset_pairs(run_dir, wanted, args.max, args, stack)
        else:
            pairs = checked_references(run_dir, wanted, args.skip_check)
            rows = status.latest_refs(status.read_csv(f"{run_dir}/refs/references.csv"))
            other = other_version(pairs, rows, {}, args.spec)
            if other:
                print(f"{', '.join(n for _, n in other)}: the reference image is not spec {args.spec.version}, "
                      "skipped", flush=True)
                pairs = [p for p in pairs if p not in other]
            if args.max is not None:
                pairs = pairs[:args.max]
            if not pairs:
                sys.exit("no reference images to run")
            sources, prompts = {}, video_prompts(run_dir, pairs, {}, args.spec)
            out = new_run_folder(run_dir, args.checkpoint)
        code = run(args, run_dir, out, pairs, sources, prompts, dataset)
    if dataset and not args.dry_run:  # after .running.lock is free, so the numbers no longer count as running
        status.update(run_dir)
    sys.exit(code)


def run(args, run_dir: str, out: str, pairs: list, sources: dict, prompts: dict, dataset: bool) -> int:
    """Write the specs, run the framework, move the videos to their group folders. prompts: name -> (spec, prompt)
    (video_prompts). -> exit code"""
    os.makedirs(f"{out}/specs", exist_ok=True)
    os.makedirs(f"{out}/refs", exist_ok=True)
    os.makedirs(f"{out}/controls", exist_ok=True)
    config = json.load(open(f"{out}/run_config.json")) if os.path.isfile(f"{out}/run_config.json") else {}
    port = config.get("port") or args.port or free_port()
    config.update({"checkpoint": os.path.abspath(args.checkpoint), "demos": [i for i, _ in pairs],
                   "references": [n for _, n in pairs], "seed": args.seed, "steps": args.steps, "gpus": args.gpus,
                   "cp": args.cp, "port": port, "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    status.write_json(f"{out}/run_config.json", config)
    specs, meta = [], {}  # specs: the videos that need one framework run
    two_pass = []  # (demo, name, pass-1 spec, final spec, reference image, color guide): the videos with a color guide
    for idx, name in pairs:
        control = f"{layout.demo_dir(run_dir, idx)}/{layout.demo_name(idx)}_geoedge.mp4"
        ref = f"{layout.ref_dir(run_dir + '/refs', idx)}/{name}.png"
        for path in (control, ref):
            if not os.path.isfile(path):
                sys.exit(f"missing: {path}")
        vspec, prompt = prompts[name]
        meta[name] = {"source": sources.get(name, ""), "reference": name, "reference_sha1": status.file_sha1(ref),
                      "seed": args.seed, "cosmos_run": os.path.basename(out), "prompt": prompt,
                      "spec": vspec.version, "controls": vspec.controls}
        ref_mp4 = f"{out}/refs/{name}.mp4"
        ref_video(ref, ref_mp4, frames=81, fps=16)
        depth = None
        if "depth" in vspec.controls:  # made here from the raw depth of the demo (a temporary file, like ref_mp4)
            depth = f"{out}/controls/{name}_depth.mp4"
            depth_control(demo_hdf5(run_dir), idx, depth)
        path = f"{out}/specs/{name}.json"
        if "color" in vspec.controls:  # two passes: the guide is made from the pass-1 video, after it is done
            with status.open_hdf5(demo_hdf5(run_dir)) as h5:
                towel_tables(demo_hdf5(run_dir), h5, idx)  # stops now when the demo has no instance ids
            first, guide = f"{out}/specs/{name}{FIRST}.json", f"{out}/controls/{name}_color.mp4"
            with open(first, "w") as f:  # pass 1: the edge control only
                json.dump(spec(name + FIRST, control, ref_mp4, prompt, args.seed, args.steps), f, indent=1)
            two_pass.append((idx, name, first, path, ref, guide))
        else:
            guide = None
            specs.append(path)
        with open(path, "w") as f:
            json.dump(spec(name, control, ref_mp4, prompt, args.seed, args.steps, depth, guide), f, indent=1)
    config["spec"] = ",".join(sorted({v.version for v, _ in prompts.values()}))  # each video json has its own
    config["policy_data"] = meta  # lets a later job take over the videos if this one ends without moving them
    status.write_json(f"{out}/run_config.json", config)

    n_gpu = len(args.gpus.split(","))
    env = framework_env(os.path.abspath(args.framework), args.tools)
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["HF_HOME"] = os.path.abspath(args.hf_home)

    def command(spec_files: list) -> list:
        return [f"{os.path.abspath(args.framework)}/.venv/bin/torchrun", f"--nproc-per-node={n_gpu}",
                f"--master-port={port}", f"{os.path.dirname(os.path.abspath(__file__))}/cosmos_launch.py",
                "--parallelism-preset=throughput", f"--dp-shard-size={n_gpu}", "--dp-replicate-size=1",
                f"--cp-size={args.cp}", "--cfgp-size=1", "-i", *spec_files, "-o", out, "--checkpoint-path",
                os.path.abspath(args.checkpoint), "--no-guardrails", "--benchmark", "--experiment-overrides",
                "model.config.tokenizer.encode_chunk_frames.480=4"]

    print(f"[run_cosmos] {len(pairs)} videos: demos {status.ranges(i for i, _ in pairs)} -> {out}", flush=True)
    if specs:
        print("[run_cosmos]", " ".join(command(specs)), flush=True)
    if two_pass:  # the two commands of the first of these videos; the others differ only in the spec file
        print(f"[run_cosmos] {len(two_pass)} videos with a color guide: two framework runs for each video, "
              "for example", flush=True)
        print("[run_cosmos] pass 1:", " ".join(command([two_pass[0][2]])), flush=True)
        print("[run_cosmos] pass 2:", " ".join(command([two_pass[0][3]])), flush=True)
    if args.dry_run:
        return 0
    keep = [fd for fd in [status.lock_fd(out, status.RUNNING_LOCK)] if fd is not None]
    rc, stopped = 0, False

    def framework(spec_files: list) -> None:
        nonlocal rc, stopped
        code, stopped = run_framework(command(spec_files), os.path.abspath(args.framework), env, f"{out}/run.log", keep)
        rc = rc or code

    try:
        with status.stop_signals():  # also between two framework runs
            if specs:
                framework(specs)
            failed = 0  # videos with a color guide that failed one after the other
            for idx, name, first, final, ref, guide in two_pass:
                if stopped:
                    break
                if failed == 2:
                    print("[run_cosmos] two videos in a row failed: the job stops here", flush=True)
                    break
                failed += 1
                framework([first])
                video = f"{out}/{name}{FIRST}/vision.mp4"
                if stopped or not complete(video):
                    continue  # no pass-1 video: the name is reported as missing below
                try:
                    color_guide(demo_hdf5(run_dir), idx, ref, video, guide)
                except ValueError as e:
                    print(f"[run_cosmos] {name}: no color guide: {e}", flush=True)
                    continue
                framework([final])
                with status.lock(run_dir) if dataset else contextlib.nullcontext():  # as in tidy
                    if place_video(out, idx, name, meta[name]):
                        print(f"[run_cosmos] {name}: done", flush=True)
                        failed = 0
    except KeyboardInterrupt:
        print("[run_cosmos] stopping, the finished videos are kept", flush=True)
        stopped = True
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
    120 s). The log is appended: a job with a color guide runs the framework several times. -> (exit code, stopped)"""
    with open(log_path, "a") as log:
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
