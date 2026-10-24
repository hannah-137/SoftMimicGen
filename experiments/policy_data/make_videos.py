"""Videos and a summary from a demo hdf5. No GPU needed.

  python experiments/policy_data/make_videos.py <demos.hdf5> [--out <dir>] [--all_frames] [--far 2.0]
      [--cameras ROOM WRIST [WRIST ...]] [--cell WxH]

Output in <dir> (default: folder of the hdf5). Per demo, in demos/000-049/demo_NNN/ (50 demos per folder, see
layout.py):
  demo_NNN_source.mp4     room | wrist RGB, side by side (1024x512), lossy
  demo_NNN_geoedge.mp4    room | wrist geometry edges, lossless, 1 px white line on the border. Control video.
  demo_NNN_depth.mp4      raw depth as gray: 0 m black, far plane white (lossless; depth, normals and instance only
                          when the hdf5 has that raw kind for every camera)
  demo_NNN_normals.mp4    raw normals as color: (xyz + 1) * 127.5 (lossless)
  demo_NNN_instance.mp4   raw instance id as color, one fixed color per id (lossless)
and in ref_sim/000-049/ (one folder per 50 demos, easy to download):
  demo_NNN_ref_sim.png    frame 0 of the source video, the base for a reference image
and in <dir>: sheet_first.png, sheet_last.png (room frame 0 and last of every demo) and summary.csv.

Frames: 81 frames spread over the demo at 16 fps (same as the earlier video tests). --all_frames keeps every
step at the control rate. Needs h5py, numpy, opencv-python, imageio, imageio-ffmpeg.

Views: without --cameras the two RGB observations (room and wrist) are found by name. --cameras gives the scene
names of the cameras in tile order (room first). The RGB of a camera is <camera>_rgb when the hdf5 has it
(generate_demos.py --rgb_full), else the observation with the camera name. Two views go side by side; three or four
go into a 2x2 grid, row by row, and an empty cell stays black (YAM: top | left wrist / right wrist | black).
--cell WxH resizes every view to W x H first (for example 320x240 gives a 640x480 grid of three views). An edge line
stays a line when it is made smaller (a cell pixel is an edge when any pixel under it is one).
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

import layout

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


def view_keys(h5: h5py.File, cameras: list | None, room: str | None, wrist: str | None) -> list[tuple[str, str]]:
    """(RGB observation key, term prefix) of every view, in tile order."""
    if not cameras:
        r, w = camera_keys(h5, room, wrist)
        return [(r, prefix(r)), (w, prefix(w))]
    obs = h5["data"][next(iter(h5["data"]))]["obs"]
    out = []
    for cam in cameras:
        p = prefix(cam)
        key = f"{p}_rgb" if f"{p}_rgb" in obs else cam
        if key not in obs:
            sys.exit(f"no RGB observation for camera {cam!r} (looked for {p}_rgb and {cam})")
        out.append((key, p))
    return out


def tile(left: np.ndarray, right: np.ndarray, line: bool = False) -> np.ndarray:
    out = np.concatenate([left, right], axis=1)
    if line:
        out[:, left.shape[1] - 1] = 255
    return out


def grid(views: list, line: bool = False) -> np.ndarray:
    """One frame from the views of one step: two side by side (same as tile), three or four in a 2x2 grid, row by
    row, with an empty cell black. With line=True a 1 px white line marks the borders between the cells."""
    if len(views) == 2:
        return tile(views[0], views[1], line)
    if not 3 <= len(views) <= 4:
        sys.exit(f"{len(views)} views: use 2, 3 or 4 cameras")
    cells = list(views) + [np.zeros_like(views[0])] * (4 - len(views))
    out = np.concatenate([np.concatenate(cells[:2], axis=1), np.concatenate(cells[2:], axis=1)], axis=0)
    if line:
        h, w = views[0].shape[:2]
        out[:, w - 1] = 255
        out[h - 1, :] = 255
    return out


def fit(frames: np.ndarray, cell: tuple | None, kind: str) -> np.ndarray:
    """Resize (T, H, W, C) frames to the cell size (W, H). kind 'edge' (0/255 lines: a pixel is an edge when any
    pixel under it is one), 'area' (RGB) or 'nearest' (colored ids, depth, normals)."""
    if cell is None or (frames.shape[2], frames.shape[1]) == cell:
        return frames
    interp = cv2.INTER_NEAREST if kind == "nearest" else cv2.INTER_AREA
    out = np.stack([cv2.resize(f, cell, interpolation=interp) for f in frames])
    if out.ndim == 3:  # cv2 drops a channel axis of size 1
        out = out[..., None]
    if kind == "edge":
        out = np.where(out > 0, 255, 0).astype(np.uint8)
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
    ap.add_argument("--cameras", nargs="+", default=None, help="scene names of the cameras in tile order, room first (see Views)")
    ap.add_argument("--cell", default=None, help="WxH: resize every view to this size before tiling (default: keep)")
    ap.add_argument("--all_frames", action="store_true", help="every step at the control rate instead of 81 frames at 16 fps")
    ap.add_argument("--far", type=float, default=2.0, help="depth video: this distance in m becomes white")
    args = ap.parse_args()
    cell = tuple(int(v) for v in args.cell.lower().split("x")) if args.cell else None
    out = os.path.abspath(args.out or os.path.dirname(os.path.abspath(args.hdf5)))
    os.makedirs(out, exist_ok=True)

    first, last, names, rows = [], [], [], []
    with h5py.File(args.hdf5, "r") as h5:
        demos = sorted(h5["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        if not demos:
            sys.exit(f"{args.hdf5}: no demos")
        views = view_keys(h5, args.cameras, args.room, args.wrist)
        prefixes = [p for _, p in views]
        print("views:", ", ".join(k for k, _ in views), f"(cell {args.cell or 'as recorded'})", flush=True)
        for key in demos:
            obs = h5["data"][key]["obs"]
            idx = int(key.split("_")[-1])
            name = layout.demo_name(idx)
            d = layout.demo_dir(out, idx)
            os.makedirs(d, exist_ok=True)
            steps = obs[views[0][0]].shape[0]
            sel = np.arange(steps) if args.all_frames else sample_idx(steps)
            fps = control_fps(h5) if args.all_frames else FPS

            rgb = [fit(obs[k][:][sel], cell, "area") for k, _ in views]
            write(f"{d}/{name}_source.mp4", [grid(list(f)) for f in zip(*rgb)], fps, None, check=False)
            rs = layout.ref_sim_dir(out, idx)
            os.makedirs(rs, exist_ok=True)
            cv2.imwrite(f"{rs}/{name}_ref_sim.png", cv2.cvtColor(grid([v[0] for v in rgb]), cv2.COLOR_RGB2BGR))

            edges = [fit(obs[f"{p}_geoedge"][:][sel], cell, "edge")[..., 0] for p in prefixes]
            edge_frames = [cv2.cvtColor(grid(list(f), line=True), cv2.COLOR_GRAY2RGB) for f in zip(*edges)]
            write(f"{d}/{name}_geoedge.mp4", edge_frames, fps, LOSSLESS_EDGE, check=True)
            del edges, edge_frames

            conv = {
                "depth": lambda k: depth_to_gray(obs[f"{k}_depth_raw"][:][sel], args.far),
                "normals": lambda k: normals_to_rgb(obs[f"{k}_normals_raw"][:][sel]),
                "instance": lambda k: instance_to_rgb(obs[f"{k}_instance_raw"][:][sel]),
            }
            for kind, fn in conv.items():
                if not all(f"{p}_{kind}_raw" in obs for p in prefixes):
                    continue  # not recorded (generate_demos.py --skip_raw or --no_raw)
                per_view = [fit(fn(p), cell, "nearest") for p in prefixes]
                frames = [grid(list(f)) for f in zip(*per_view)]
                write(f"{d}/{name}_{kind}.mp4", frames, fps, LOSSLESS_RGB, check=True)
                del per_view, frames

            rows.append([idx, steps, round(steps / control_fps(h5), 2), bool(h5["data"][key].attrs.get("success", True))])
            first.append(rgb[0][0].copy())
            last.append(rgb[0][-1].copy())
            names.append(str(idx))
            print(f"{name}: {steps} steps, {len(sel)} frames", flush=True)

    cv2.imwrite(f"{out}/sheet_first.png", cv2.cvtColor(sheet(first, names), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{out}/sheet_last.png", cv2.cvtColor(sheet(last, names), cv2.COLOR_RGB2BGR))
    with open(f"{out}/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["demo", "steps", "seconds", "success"])
        w.writerows(rows)
    print(f"{len(rows)} demos, mean {np.mean([r[1] for r in rows]):.1f} steps -> {out}/demos")


if __name__ == "__main__":
    main()
