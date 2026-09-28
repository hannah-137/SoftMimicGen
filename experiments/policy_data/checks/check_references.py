"""Check reference images before video generation. No GPU needed. Run it as often as you like.

  python experiments/policy_data/checks/check_references.py <run_dir> [--demos 0-49] [--refs <folder>] [--no_layout]

Input: the demo hdf5 in <run_dir> and the reference images in <run_dir>/refs/000-049/, <run_dir>/refs/050-099/, ...
(the folder of the demo, 50 demos per folder, see layout.py), named demo_NNN_<tag>.png (several images per demo,
one video per image later) or demo_NNN.png (one image). Every demo in the hdf5 needs at least one image. A demo
without an image is a failure ("missing"). An image in the wrong folder is a failure ("wrong folder").
--demos (numbers and ranges, e.g. 0-49) checks only these demos, for work in batches. The rows of the other demos
stay in check_references.csv as they were.

Check 1, size: the image must have the 2:1 aspect of the tiled view (room | wrist). A different aspect is a failure
(make the image again). The layout check and run_cosmos.py resize the image to 1024 x 512 themselves.
Check 2, layout: the robot and the towel must be where the simulator has them in frame 0. Score = share of the
simulator outline (from the raw instance ids, grouped into robot / object / environment by prim path, so joints
between robot links do not count) that lies within 5 px of an edge in the image. Only the moving region (robot +
towel) counts. Room and wrist views are scored separately. Pass: both >= 0.70.
The instance id table comes from the attribute instance_ids of data/demo_N (a dataset made by make_dataset.py: runs
can number the ids differently) or else from <hdf5 name>_instance_ids.json next to the hdf5 (a run folder).
Calibration (Franka towel, 2026-09-26): simulator frame 0.91 / 0.98, the same frame resized 0.85 / 0.99, earlier
ChatGPT references 0.86-1.00, mirrored image 0.55 / 0.41, towel removed 0.38 in the wrist view.

Output, in the refs folder: check_references.csv (one row per image, with the sha1 of the checked file, so a later
image with the same name counts as not checked). Only when something fails: failed_references.txt (images to make
again, with the reason) and check_<name>.png next to the image (the image with the simulator outline drawn). Rows
of other demos are kept only while they still describe the file on disk. Exit code 1 when anything checked in this
run fails. Both files are written under the dataset lock (see status.py); in a dataset folder, status.py runs at
the end.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layout  # noqa: E402
import status  # noqa: E402

TILE_W, TILE_H = 1024, 512
TOL = 5  # px between a simulator outline pixel and an image edge
MOTION_THR = 25  # brightness change vs frame 0 that counts as motion
MIN_OUTLINE_PX = 80  # fewer outline pixels than this in the region: no score
LAYOUT_MIN = 0.70
COLUMNS = ["demo", "name", "file", "width", "height", "size_ok", "room_score", "wrist_score", "passed", "reason",
           "sha1"]


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


def demo_tables(demo, run_tables: dict) -> dict:
    """camera -> {id: prim path} for one demo: its own attribute (dataset) or the table of the run."""
    if "instance_ids" in demo.attrs:
        return json.loads(demo.attrs["instance_ids"])
    return run_tables


def sim_frame0(obs, cam: str, table: dict | None):
    """(outline of frame 0, motion region) for one camera."""
    ids = obs[f"{prefix(cam)}_instance_raw"][0][..., 0]
    return outline(group_ids(ids, table)), motion_region(obs[cam][:])


def load_and_resize(path: str):
    """-> (image 1024x512 RGB or None, width, height, reason). run_cosmos.py uses it too."""
    with open(path, "rb") as f:
        return decode_and_resize(f.read())


def decode_and_resize(data: bytes):
    """Image file bytes -> (image 1024x512 RGB or None, width, height, reason)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
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


def still_valid(row: dict, run: str, refs: str) -> bool:
    """True when a kept row still describes the files on disk (see the docstring)."""
    if row["reason"] == "missing":  # still no image in the folder of the demo
        folder = layout.ref_dir(refs, int(row["demo"]))
        return not any(re.fullmatch(rf"{row['name']}(_[^/]+)?\.png", os.path.basename(f))
                       for f in glob.glob(f"{folder}/*.png"))
    if row["reason"].startswith("wrong folder"):
        return os.path.isfile(os.path.join(run, row["file"]))
    return status.check_is_current(refs, row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--demos", nargs="*", default=None, help="demo numbers and ranges, e.g. 0-49 (default: all)")
    ap.add_argument("--refs", default=None, help="folder with the 000-049/... image folders (default <run_dir>/refs)")
    ap.add_argument("--hdf5", default=None, help="demo file (default: the one hdf5 in <run_dir> without _failed)")
    ap.add_argument("--no_layout", action="store_true", help="only the size check")
    args = ap.parse_args()
    run = os.path.abspath(args.run_dir)
    refs = os.path.abspath(args.refs or f"{run}/refs")
    if not os.path.isdir(run):
        sys.exit(f"no such folder: {run}")
    if not os.path.isdir(refs):
        sys.exit(f"no reference folder: {refs}")
    h5path = args.hdf5 or next((f"{run}/{f}" for f in sorted(os.listdir(run)) if f.endswith(".hdf5") and "_failed" not in f), None)
    if not h5path or not os.path.isfile(h5path):
        sys.exit(f"no hdf5 in {run}")
    table_path = h5path[: -len(".hdf5")] + "_instance_ids.json"
    run_tables = json.load(open(table_path)) if os.path.isfile(table_path) else {}
    wanted = status.parse_demos(args.demos)

    rows, failed = [], []
    pending = status.pending_replacements(run) if status.is_dataset(run) else set()
    with status.open_hdf5(h5path) as h5:
        # Only data/demo_N: a stopped make_dataset.py --replace can leave a temporary key in the file.
        demos = sorted((k for k in h5["data"].keys() if re.fullmatch(r"demo_\d+", k)), key=lambda k: int(k[5:]))
        if wanted is not None:
            unknown = sorted(set(wanted) - {int(k.split("_")[-1]) for k in demos})
            if unknown:
                sys.exit(f"--demos: not in the hdf5: {status.ranges(unknown)}")
            demos = [k for k in demos if int(k.split("_")[-1]) in set(wanted)]
        for k in [k for k in demos if int(k[5:]) in pending]:
            print(f"{k}: make_dataset.py --replace is not finished for this number, not checked", flush=True)
        demos = [k for k in demos if int(k[5:]) not in pending]
        if not demos:
            sys.exit("no demos to check")
        room_key, wrist_key = camera_keys(h5["data"][demos[0]]["obs"])
        for key in demos:
            idx = int(key.split("_")[-1])
            demo = layout.demo_name(idx)
            folder = layout.ref_dir(refs, idx)
            files = sorted(f for f in glob.glob(f"{refs}/**/{demo}*.png", recursive=True)
                           if re.fullmatch(rf"{demo}(_[^/]+)?\.png", os.path.basename(f)))
            for src in [f for f in files if os.path.dirname(f) != folder]:
                name = os.path.basename(src)[: -len(".png")]
                reason = f"wrong folder, move it to {os.path.relpath(folder, run)}"
                rows.append({"demo": idx, "name": name, "file": os.path.relpath(src, run), "width": 0, "height": 0,
                             "size_ok": False, "room_score": "", "wrist_score": "", "passed": False,
                             "reason": reason})
                failed.append(f"{name}: {reason}")
                print(f"{name}: FAIL {reason} (it is in {os.path.relpath(os.path.dirname(src), run)})", flush=True)
            files = [f for f in files if os.path.dirname(f) == folder]
            if not files:
                rows.append({"demo": idx, "name": demo, "file": "", "width": 0, "height": 0, "size_ok": False,
                             "room_score": "", "wrist_score": "", "passed": False, "reason": "missing"})
                failed.append(f"{demo}: missing")
                print(f"{demo}: FAIL missing", flush=True)
                continue
            sim = None
            for src in files:
                name = os.path.basename(src)[: -len(".png")]
                with open(src, "rb") as f:
                    data = f.read()  # one read: the sha1 and the check see the same bytes
                row = {"demo": idx, "name": name, "file": os.path.relpath(src, run), "width": 0, "height": 0,
                       "size_ok": False, "room_score": "", "wrist_score": "", "passed": False, "reason": "",
                       "sha1": hashlib.sha1(data).hexdigest()}
                img, w, h, reason = decode_and_resize(data)
                row.update(width=w, height=h, size_ok=img is not None, reason=reason)
                if img is not None:
                    if args.no_layout:
                        row["passed"] = True
                    else:
                        if sim is None:
                            obs = h5["data"][key]["obs"]
                            tables = demo_tables(h5["data"][key], run_tables)
                            if not tables:
                                print(f"note: {key} has no instance id table, the outline keeps every instance "
                                      "boundary (stricter score)", flush=True)
                            sim = sim_frame0(obs, room_key, tables.get(room_key)), sim_frame0(obs, wrist_key, tables.get(wrist_key))
                        (ol_r, reg_r), (ol_w, reg_w) = sim
                        s_r = layout_score(ol_r, img[:, :TILE_H], reg_r)
                        s_w = layout_score(ol_w, img[:, TILE_H:], reg_w)
                        row.update(room_score=round(s_r, 3), wrist_score=round(s_w, 3))
                        bad = [f"{n} layout {s:.2f} < {LAYOUT_MIN}" for n, s in (("room", s_r), ("wrist", s_w)) if not s >= LAYOUT_MIN]
                        row["passed"] = not bad
                        row["reason"] = "; ".join(bad)
                        if bad:
                            draw_fail(img, [(0, ol_r), (TILE_H, ol_w)], f"{folder}/check_{name}.png")
                        elif os.path.isfile(f"{folder}/check_{name}.png"):  # drawing of an earlier failed image
                            os.remove(f"{folder}/check_{name}.png")
                if not row["passed"]:
                    failed.append(f"{name}: {row['reason']}")
                rows.append(row)
                print(f"{name}: {'ok' if row['passed'] else 'FAIL ' + row['reason']}", flush=True)

    # Keep the rows of the demos not checked in this run while they still describe the files on disk, and write both
    # files at once under the lock of the folder (the dataset or run folder, or the --refs folder outside it).
    checked = {int(k.split("_")[-1]) for k in demos}
    csv_path, failed_path = f"{refs}/check_references.csv", f"{refs}/failed_references.txt"
    lock_dir = run if status.is_dataset(run) or refs.startswith(run + os.sep) else refs
    with status.lock(lock_dir):
        kept = [r for r in status.read_csv(csv_path)
                if int(r["demo"]) not in checked and still_valid(r, run, refs)]
        merged = sorted(kept + rows, key=lambda r: (int(r["demo"]), r["name"]))
        status.write_csv(csv_path, merged, COLUMNS)
        all_failed = [f"{r['name']}: {r['reason']}" for r in merged if str(r["passed"]) != "True"]
        if all_failed:
            tmp = f"{failed_path}.tmp"
            with open(tmp, "w") as f:
                f.write("\n".join(all_failed) + "\n")
            os.replace(tmp, failed_path)
        elif os.path.isfile(failed_path):
            os.remove(failed_path)
    scope = f"demos {status.ranges(checked)}" if wanted is not None else "all demos"
    print(f"{len(rows) - len(failed)}/{len(rows)} passed ({scope})"
          + (f". Failed list (all checks so far): {failed_path}" if all_failed else ""))
    if status.is_dataset(run):
        status.update(run)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
