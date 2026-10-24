"""RoboRender inputs for demos of a policy_data dataset: per view a depth video and a robot RGB mask video, the
prompt, and the reference image as the optional first frame.

    conda activate softmimicgen
    python experiments/roborender/make_inputs.py <dataset> --demos 3,4,13 --out <folder>

Writes, for each number N, the sample folder RoboRender's infer.py reads (416x240, every simulator step):
    <out>/demo_NNN/original_instruction.txt            the Cosmos prompt of the current video, without the split-screen sentences
    <out>/demo_NNN/<view>/depth/original_depth_vda.mp4 inverse depth as gray, near is bright, 1st to 99th percentile
                                                       of the view's video, nothing rendered is black
    <out>/demo_NNN/<view>/mask/rgb_robot.mp4           the simulator RGB where the instance id is a robot link, else black
    <out>/demo_NNN/<view>/pre_frame.png                the half of the reference image of this view (for --use_prev_frame)
    <out>/demo_NNN/<view>/rgb.mp4                      the simulator RGB, same framing (for the comparison page)
views: room (agentview) and wrist (robot0_eye_in_hand). The square 512x512 view is letterboxed into 416x240 (black
bars left and right, nothing cut) unless --crop is given, which center-crops the height as RoboRender's loader would.
The raw depth comes from the hdf5 of the demo's source run (the dataset hdf5 has no raw depth); RGB and instance ids
too, so one file is read per demo.
"""
import argparse
import json
import os
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "policy_data"))
import layout  # noqa: E402
import make_videos  # noqa: E402
import run_cosmos  # noqa: E402
import status  # noqa: E402

VIEWS = [("room", "agentview"), ("wrist", "robot0_eye_in_hand")]
WIDTH, HEIGHT, FPS = 416, 240, 15


def fit(img: np.ndarray, crop: bool) -> np.ndarray:
    """(H, W, 3) uint8 square view -> (HEIGHT, WIDTH, 3). Letterbox (default) or center crop of the height."""
    h, w = img.shape[:2]
    if crop:
        ch = int(w / WIDTH * HEIGHT)
        top = (h - ch) // 2
        return cv2.resize(img[top:top + ch], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    scale = HEIGHT / h
    nw = int(round(w * scale))
    small = cv2.resize(img, (nw, HEIGHT), interpolation=cv2.INTER_AREA)
    out = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    left = (WIDTH - nw) // 2
    out[:, left:left + nw] = small
    return out


def inverse_depth_gray(depth: np.ndarray) -> np.ndarray:
    """(T, H, W, 1) float32 m -> (T, H, W, 3) uint8: 1 / depth scaled between the 1st and 99th percentile of the
    video (near is bright); inf or 0 depth (nothing rendered) is black."""
    d = depth[..., 0].astype(np.float32)
    ok = np.isfinite(d) & (d > 0)
    inv = np.zeros_like(d)
    inv[ok] = 1.0 / d[ok]
    lo, hi = np.percentile(inv[ok], [1, 99])
    g = np.clip((inv - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    g[~ok] = 0.0
    g = np.round(g * 255.0).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=-1)


def scene_prompt(cosmos_prompt: str) -> str:
    """The Cosmos prompt without the sentences about our split-screen layout (RoboRender gets the views apart)."""
    keep = [s.strip() for s in cosmos_prompt.split(". ")
            if not any(w in s for w in ("Split screen", "left camera", "right view", "left view", "left half", "right half"))]
    text = ". ".join(k.rstrip(".") for k in keep if k)
    return text + "."


def robot_ids(run_dir: str, camera: str) -> set:
    """Instance ids of the robot links for a camera, from <run>_instance_ids.json of the run."""
    path = next(f"{run_dir}/{f}" for f in os.listdir(run_dir) if f.endswith("_instance_ids.json"))
    table = json.load(open(path))[f"{camera}_image"]
    return {int(k) for k, prim in table.items() if "/Robot/" in prim}


def write_video(path: str, frames: np.ndarray, crop: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    make_videos.write(path, [fit(f, crop) for f in frames], FPS, make_videos.LOSSLESS_RGB, check=False)


def make_sample(ds: str, n: int, out: str, crop: bool) -> dict:
    sources = {int(r["demo"]): r for r in status.read_csv(f"{ds}/sources.csv")}
    run_dir = f"{os.path.dirname(ds)}/{sources[n]['source_run']}"
    src_demo = int(sources[n]["source_demo"])
    videos = {int(r["demo"]): r for r in status.read_csv(f"{ds}/videos.csv") if r["where"] == "current"}
    dataset = {int(r["demo"]): r for r in status.read_csv(f"{ds}/dataset.csv")}
    settings = json.load(open(f"{ds}/{videos[n]['file'][:-4]}.json"))
    sample = f"{out}/{layout.demo_name(n)}"
    os.makedirs(sample, exist_ok=True)
    with open(f"{sample}/original_instruction.txt", "w") as f:
        f.write(scene_prompt(settings["prompt"]) + "\n")
    ref = cv2.imread(f"{layout.ref_dir(f'{ds}/refs', n)}/{dataset[n]['reference']}.png")
    info = {"number": n, "source": f"{sources[n]['source_run']}:{src_demo}", "steps": None, "views": {}}
    with h5py.File(run_cosmos.demo_hdf5(run_dir), "r") as h5:
        obs = h5["data"][f"demo_{src_demo}"]["obs"]
        for i, (view, cam) in enumerate(VIEWS):
            rgb = obs[f"{cam}_image"][:]
            depth = obs[f"{cam}_depth_raw"][:]
            ids = obs[f"{cam}_instance_raw"][:][..., 0]
            robot = np.isin(ids, list(robot_ids(run_dir, cam)))
            mask_rgb = np.where(robot[..., None], rgb, 0).astype(np.uint8)
            write_video(f"{sample}/{view}/depth/original_depth_vda.mp4", inverse_depth_gray(depth), crop)
            write_video(f"{sample}/{view}/mask/rgb_robot.mp4", mask_rgb, crop)
            write_video(f"{sample}/{view}/rgb.mp4", rgb, crop)
            half = ref[:, i * ref.shape[1] // 2:(i + 1) * ref.shape[1] // 2]
            cv2.imwrite(f"{sample}/{view}/pre_frame.png", cv2.cvtColor(fit(cv2.cvtColor(half, cv2.COLOR_BGR2RGB), crop), cv2.COLOR_RGB2BGR))
            info["steps"] = int(rgb.shape[0])
            info["views"][view] = {"robot_pixels_frame0": int(robot[0].sum()), "depth_finite_share": float(np.isfinite(depth).mean())}
    json.dump(info, open(f"{sample}/inputs.json", "w"), indent=1)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--demos", required=True)
    ap.add_argument("--out", required=True, help="sample root folder")
    ap.add_argument("--crop", action="store_true", help="center-crop the height to 16:9 instead of letterboxing")
    args = ap.parse_args()
    ds = os.path.normpath(args.dataset)
    for n in status.parse_demos([args.demos]):
        info = make_sample(ds, n, args.out, args.crop)
        print(json.dumps(info))


if __name__ == "__main__":
    main()
