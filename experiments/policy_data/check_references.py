"""Check reference images before video generation. No GPU needed. Run it as often as you like.

  python experiments/policy_data/check_references.py <run_dir> [--refs <folder>] [--no_layout]

Input: <run_dir>/refs/demo_NNN.png, one image per demo, and the demo hdf5 in <run_dir>.
Every demo in the hdf5 needs an image. A missing image is a failure.

Check 1, size: the image must have the 2:1 aspect of the tiled view (room | wrist). Then it is resized to 1024 x 512
and saved as <run_dir>/refs_checked/demo_NNN.png. A different aspect is a failure (make the image again).
Check 2, layout: the robot and the towel must be where the simulator has them in frame 0. Score = share of the
simulator outline (from the raw instance ids, grouped into robot / object / environment by prim path, so joints
between robot links do not count) that lies within 5 px of an edge in the image. Only the moving region (robot +
towel) counts. Room and wrist views are scored separately. Pass: both >= 0.70.
Calibration (Franka towel, 2026-09-26): simulator frame 0.91 / 0.98, the same frame resized 0.85 / 0.99, earlier
ChatGPT references 0.86-1.00, mirrored image 0.55 / 0.41, towel removed 0.38 in the wrist view.

Output: <run_dir>/check_references.csv (all demos), <run_dir>/failed_references.txt (demos to make again, with the
reason), <run_dir>/check_references/demo_NNN.png (image with the simulator outline drawn, only for failures).
Exit code 1 when any demo fails.
"""

import argparse
import csv
import json
import os
import sys

import cv2
import h5py
import numpy as np

TILE_W, TILE_H = 1024, 512
TOL = 5  # px between a simulator outline pixel and an image edge
MOTION_THR = 25  # brightness change vs frame 0 that counts as motion
MIN_OUTLINE_PX = 80  # fewer outline pixels than this in the region: no score
LAYOUT_MIN = 0.70


def prefix(camera: str) -> str:
    return camera[: -len("_image")] if camera.endswith("_image") else camera


def camera_keys(obs) -> tuple[str, str]:
    rgb = [k for k in obs if isinstance(obs[k], h5py.Dataset) and obs[k].ndim == 4 and obs[k].shape[-1] == 3
           and obs[k].dtype == np.uint8 and k.endswith("_image")]
    wrists = [k for k in rgb if "eye_in_hand" in k or "wrist" in k]
    if len(rgb) != 2 or len(wrists) != 1:
        sys.exit(f"cannot find the two camera images in the hdf5: {rgb}")
    return next(k for k in rgb if k != wrists[0]), wrists[0]


def outline(ids: np.ndarray) -> np.ndarray:
    """(H, W) ids -> (H, W) bool, True where the right or lower neighbor has another id."""
    b = np.zeros(ids.shape, bool)
    b[:, :-1] |= ids[:, :-1] != ids[:, 1:]
    b[:-1, :] |= ids[:-1, :] != ids[1:, :]
    return b


def group_ids(ids: np.ndarray, table: dict | None) -> np.ndarray:
    """Instance ids -> 1 robot, 2 object, 3 environment (by prim path). Without a table the ids stay as they are."""
    if not table:
        return ids
    out = np.full(ids.shape, 3, dtype=np.int32)
    for k, prim in table.items():
        out[ids == int(k)] = 1 if "/Robot/" in prim else 2 if "/Object/" in prim else 3
    out[ids == 0] = 0  # nothing hit
    return out


def motion_region(rgb: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) -> (H, W) bool: pixels that change during the demo, grown by 15 px."""
    diff = np.abs(rgb.astype(np.int16) - rgb[:1].astype(np.int16)).max(axis=-1).max(axis=0) > MOTION_THR
    diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(diff, np.ones((15, 15), np.uint8)) > 0


def image_edges(img_rgb: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    return cv2.Canny(g, 40, 120) > 0


def layout_score(sim_outline: np.ndarray, img_rgb: np.ndarray, region: np.ndarray) -> float:
    target = sim_outline & region
    n = int(target.sum())
    if n < MIN_OUTLINE_PX:
        return float("nan")
    near = cv2.dilate(image_edges(img_rgb).astype(np.uint8), np.ones((2 * TOL + 1, 2 * TOL + 1), np.uint8)) > 0
    return float((target & near).sum()) / n


def sim_frame0(obs, cam: str, table: dict | None):
    """(outline of frame 0, motion region) for one camera."""
    ids = obs[f"{prefix(cam)}_instance_raw"][0][..., 0]
    return outline(group_ids(ids, table)), motion_region(obs[cam][:])


def load_and_resize(path: str):
    """-> (image 1024x512 RGB or None, width, height, reason)"""
    img = cv2.imread(path)
    if img is None:
        return None, 0, 0, "cannot read"
    h, w = img.shape[:2]
    if abs(w / h - TILE_W / TILE_H) > 0.01:
        return None, w, h, f"aspect {w}x{h} is {w / h:.2f}, need 2.00"
    if (w, h) != (TILE_W, TILE_H):
        img = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), w, h, ""


def draw_fail(img_rgb: np.ndarray, outlines: list, path: str) -> None:
    """Save the image with the simulator outlines drawn in red."""
    out = img_rgb.copy()
    for x0, ol in outlines:
        ys, xs = np.nonzero(ol)
        out[ys, xs + x0] = (255, 0, 0)
    cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--refs", default=None, help="folder with demo_NNN.png (default <run_dir>/refs)")
    ap.add_argument("--hdf5", default=None, help="demo file (default: the one hdf5 in <run_dir> without _failed)")
    ap.add_argument("--no_layout", action="store_true", help="only the size check")
    args = ap.parse_args()
    run = os.path.abspath(args.run_dir)
    refs = args.refs or f"{run}/refs"
    h5path = args.hdf5 or next((f"{run}/{f}" for f in sorted(os.listdir(run)) if f.endswith(".hdf5") and "_failed" not in f), None)
    if not h5path or not os.path.isfile(h5path):
        sys.exit(f"no hdf5 in {run}")
    os.makedirs(f"{run}/refs_checked", exist_ok=True)
    os.makedirs(f"{run}/check_references", exist_ok=True)
    table_path = h5path[: -len(".hdf5")] + "_instance_ids.json"
    tables = json.load(open(table_path)) if os.path.isfile(table_path) else {}
    if not tables and not args.no_layout:
        print(f"note: {table_path} not found, the outline keeps every instance boundary (stricter score)")

    rows, failed = [], []
    with h5py.File(h5path, "r") as h5:
        demos = sorted(h5["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        room_key, wrist_key = camera_keys(h5["data"][demos[0]]["obs"])
        for key in demos:
            idx = int(key.split("_")[-1])
            name = f"demo_{idx:03d}"
            src = f"{refs}/{name}.png"
            row = {"demo": idx, "file": src, "width": 0, "height": 0, "size_ok": False, "room_score": "", "wrist_score": "", "passed": False, "reason": ""}
            if not os.path.isfile(src):
                row["reason"] = "missing"
            else:
                img, w, h, reason = load_and_resize(src)
                row.update(width=w, height=h, size_ok=img is not None, reason=reason)
                if img is not None:
                    cv2.imwrite(f"{run}/refs_checked/{name}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if args.no_layout:
                        row["passed"] = True
                    else:
                        obs = h5["data"][key]["obs"]
                        (ol_r, reg_r), (ol_w, reg_w) = sim_frame0(obs, room_key, tables.get(room_key)), sim_frame0(obs, wrist_key, tables.get(wrist_key))
                        s_r = layout_score(ol_r, img[:, :TILE_H], reg_r)
                        s_w = layout_score(ol_w, img[:, TILE_H:], reg_w)
                        row.update(room_score=round(s_r, 3), wrist_score=round(s_w, 3))
                        bad = [f"{n} layout {s:.2f} < {LAYOUT_MIN}" for n, s in (("room", s_r), ("wrist", s_w)) if not s >= LAYOUT_MIN]
                        row["passed"] = not bad
                        row["reason"] = "; ".join(bad)
                        if bad:
                            draw_fail(img, [(0, ol_r), (TILE_H, ol_w)], f"{run}/check_references/{name}.png")
            if not row["passed"]:
                failed.append(f"{name}: {row['reason']}")
                if os.path.isfile(f"{run}/refs_checked/{name}.png") and not row["size_ok"]:
                    os.remove(f"{run}/refs_checked/{name}.png")
            rows.append(row)
            print(f"{name}: {'ok' if row['passed'] else 'FAIL ' + row['reason']}", flush=True)

    with open(f"{run}/check_references.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(f"{run}/failed_references.txt", "w") as f:
        f.write("\n".join(failed) + ("\n" if failed else ""))
    print(f"{len(rows) - len(failed)}/{len(rows)} passed. Failed list: {run}/failed_references.txt")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
