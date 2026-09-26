"""Videos and a summary from a demo hdf5. No GPU needed.

  python experiments/policy_data/make_videos.py <demos.hdf5> [--out <dir>] [--all_frames] [--far 2.0]

Output in <dir>/videos/ (default <dir> = folder of the hdf5), one file per demo:
  source/demo_NNN.mp4     room | wrist RGB, side by side (1024x512), lossy
  geoedge/demo_NNN.mp4    room | wrist geometry edges, lossless, 1 px white line on the border. Control video.
  depth/demo_NNN.mp4      raw depth as gray: 0 m black, far plane white (lossless)
  normals/demo_NNN.mp4    raw normals as color: (xyz + 1) * 127.5 (lossless)
  instance/demo_NNN.mp4   raw instance id as color, one fixed color per id (lossless)
  ref_sim/demo_NNN.png    frame 0 of the source video, the base for a reference image
Also <dir>/sheet_first.png, sheet_last.png (room frame 0 and last of every demo) and summary.csv.

Frames: 81 frames spread over the demo at 16 fps (same as the earlier video tests). --all_frames keeps every
step at the control rate. Needs h5py, numpy, opencv-python, imageio, imageio-ffmpeg.
"""

import argparse
import csv
import json
import os
import sys

import cv2
import h5py
import imageio
import numpy as np

N_FRAMES, FPS = 81, 16
LOSSLESS_RGB = {"codec": "libx264rgb", "pixelformat": "rgb24", "output_params": ["-qp", "0"]}
LOSSLESS_EDGE = {"codec": "libx264", "pixelformat": "yuv444p", "output_params": ["-qp", "0"]}


def sample_idx(n: int) -> np.ndarray:
    """81 frame indices spread over n steps."""
    return np.linspace(0, n - 1, N_FRAMES).astype(int)


def control_fps(h5: h5py.File, fallback: float = 20.0) -> float:
    try:
        sim = json.loads(h5["data"].attrs["env_args"])["sim_args"]
        return 1.0 / (sim["dt"] * sim["decimation"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return fallback


def prefix(camera: str) -> str:
    """agentview_image -> agentview (same rule as generate_demos.py)"""
    return camera[: -len("_image")] if camera.endswith("_image") else camera


def camera_keys(h5: h5py.File, room: str | None, wrist: str | None) -> tuple[str, str]:
    """Observation keys of the two RGB images. The wrist key contains 'eye_in_hand' or 'wrist'."""
    if room and wrist:
        return room, wrist
    obs = h5["data"][next(iter(h5["data"]))]["obs"]
    rgb = [k for k in obs if isinstance(obs[k], h5py.Dataset) and obs[k].ndim == 4 and obs[k].shape[-1] == 3
           and obs[k].dtype == np.uint8 and k.endswith("_image")]
    wrists = [k for k in rgb if "eye_in_hand" in k or "wrist" in k]
    if len(rgb) != 2 or len(wrists) != 1:
        sys.exit(f"found RGB observations {rgb}; pass --room and --wrist")
    return next(k for k in rgb if k != wrists[0]), wrists[0]


def tile(left: np.ndarray, right: np.ndarray, line: bool = False) -> np.ndarray:
    out = np.concatenate([left, right], axis=1)
    if line:
        out[:, left.shape[1] - 1] = 255
    return out


def write(path: str, frames: list, fps: float, params: dict | None, check: bool) -> None:
    """Write an mp4. With check=True, read it back and compare every frame."""
    if params is None:
        imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=9)
        return
    imageio.mimsave(path, frames, fps=fps, **params)
    if check:
        back = imageio.mimread(path, memtest=False)
        assert len(back) == len(frames) and all(np.array_equal(a, b) for a, b in zip(back, frames)), f"lossless check failed: {path}"


def depth_to_gray(d: np.ndarray, far: float) -> np.ndarray:
    """(T, H, W, 1) float32 m -> (T, H, W, 3) uint8, inf -> far."""
    d = np.where(np.isfinite(d), d, far)
    g = np.round(np.clip(d / far, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.repeat(g, 3, axis=-1)


def normals_to_rgb(n: np.ndarray) -> np.ndarray:
    return np.round((np.clip(n, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)


def instance_to_rgb(ids: np.ndarray) -> np.ndarray:
    """(T, H, W, 1) int32 -> one fixed color per id. Id 0 stays black."""
    x = ids[..., 0].astype(np.int64)
    r = ((x * 2654435761) >> 8) & 255
    g = ((x * 2246822519) >> 8) & 255
    b = ((x * 3266489917) >> 8) & 255
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    rgb[x == 0] = 0
    return rgb


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (8 + 11 * len(text), 22), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def sheet(frames: list, names: list, cols: int = 10, size: int = 160) -> np.ndarray:
    tiles = [label(cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA), n) for f, n in zip(frames, names)]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)]
    return np.concatenate(rows, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hdf5")
    ap.add_argument("--out", default=None, help="output folder (default: folder of the hdf5)")
    ap.add_argument("--room", default=None, help="observation key of the room image")
    ap.add_argument("--wrist", default=None, help="observation key of the wrist image")
    ap.add_argument("--all_frames", action="store_true", help="every step at the control rate instead of 81 frames at 16 fps")
    ap.add_argument("--far", type=float, default=2.0, help="depth video: this distance in m becomes white")
    args = ap.parse_args()
    out = args.out or os.path.dirname(os.path.abspath(args.hdf5))
    kinds = ("source", "geoedge", "depth", "normals", "instance", "ref_sim")
    for k in kinds:
        os.makedirs(f"{out}/videos/{k}", exist_ok=True)

    first, last, names, rows = [], [], [], []
    with h5py.File(args.hdf5, "r") as h5:
        demos = sorted(h5["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        if not demos:
            sys.exit(f"{args.hdf5}: no demos")
        room_key, wrist_key = camera_keys(h5, args.room, args.wrist)
        rp, wp = prefix(room_key), prefix(wrist_key)
        for key in demos:
            obs = h5["data"][key]["obs"]
            idx = int(key.split("_")[-1])
            name = f"demo_{idx:03d}"
            steps = obs[room_key].shape[0]
            sel = np.arange(steps) if args.all_frames else sample_idx(steps)
            fps = control_fps(h5) if args.all_frames else FPS

            room, wrist = obs[room_key][:][sel], obs[wrist_key][:][sel]
            write(f"{out}/videos/source/{name}.mp4", [tile(a, b) for a, b in zip(room, wrist)], fps, None, check=False)
            cv2.imwrite(f"{out}/videos/ref_sim/{name}.png", cv2.cvtColor(tile(room[0], wrist[0]), cv2.COLOR_RGB2BGR))

            e_room, e_wrist = obs[f"{rp}_geoedge"][:][sel][..., 0], obs[f"{wp}_geoedge"][:][sel][..., 0]
            edge_frames = [cv2.cvtColor(tile(a, b, line=True), cv2.COLOR_GRAY2RGB) for a, b in zip(e_room, e_wrist)]
            write(f"{out}/videos/geoedge/{name}.mp4", edge_frames, fps, LOSSLESS_EDGE, check=True)

            if f"{rp}_depth_raw" in obs:
                conv = {
                    "depth": lambda k: depth_to_gray(obs[f"{k}_depth_raw"][:][sel], args.far),
                    "normals": lambda k: normals_to_rgb(obs[f"{k}_normals_raw"][:][sel]),
                    "instance": lambda k: instance_to_rgb(obs[f"{k}_instance_raw"][:][sel]),
                }
                for kind, fn in conv.items():
                    frames = [tile(a, b) for a, b in zip(fn(rp), fn(wp))]
                    write(f"{out}/videos/{kind}/{name}.mp4", frames, fps, LOSSLESS_RGB, check=True)

            rows.append([idx, steps, round(steps / control_fps(h5), 2), bool(h5["data"][key].attrs.get("success", True))])
            first.append(room[0].copy())
            last.append(room[-1].copy())
            names.append(str(idx))
            print(f"{name}: {steps} steps, {len(sel)} frames", flush=True)

    cv2.imwrite(f"{out}/sheet_first.png", cv2.cvtColor(sheet(first, names), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{out}/sheet_last.png", cv2.cvtColor(sheet(last, names), cv2.COLOR_RGB2BGR))
    with open(f"{out}/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["demo", "steps", "seconds", "success"])
        w.writerows(rows)
    print(f"{len(rows)} demos, mean {np.mean([r[1] for r in rows]):.1f} steps -> {out}/videos")


if __name__ == "__main__":
    main()
