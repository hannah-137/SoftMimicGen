"""Check reference images before video generation. No GPU needed. Run it as often as you like.

  python experiments/policy_data/checks/check_references.py <run_dir> [--demos 0-49] [--refs <folder>] [--no_layout]
                                                            [--workers 4]

Input: the demo hdf5 in <run_dir> and the reference images in <run_dir>/refs/000-049/, <run_dir>/refs/050-099/, ...
(the folder of the demo, 50 demos per folder, see layout.py), named demo_NNN_<tag>.png (several images per demo,
one video per image later) or demo_NNN.png (one image). Every demo in the hdf5 needs at least one image. A demo
without an image is a failure ("missing"). An image in the wrong folder is a failure ("wrong folder").
--demos (numbers and ranges, e.g. 0-49) checks only these demos, for work in batches. The rows of the other demos
stay in check_references.csv as they were.

Check 1, size: the image must have the 2:1 aspect of the tiled view (room | wrist). A different aspect is a failure
(make the image again). The layout check and run_cosmos.py resize the image to 1024 x 512 themselves.
Check 2, layout (reference_layout.py): the table, the towel and the robot must have the shape, size and position
that the simulator has in frame 0. The look may differ. The simulator masks come from the raw instance ids: robot
and object by prim path, table = the instance that the object lies on. The check finds the true boundary of every
object in the image and fails the image when
  - the towel (both views), the table of the wrist view or the robot (wrist view) is more than 8 px off, or
  - the table of the room view is more than 20 px off (the image model often draws it a little larger), or
  - a table edge is missing, or
  - another surface covers more than 5 % of the table top or of the towel, or
  - the table top looks different in the room view and in the wrist view, or
  - a part of the wrist table looks like the floor of the room view (the image shows floor where the simulator shows
    the table).
A boundary that cannot be measured is not a failure. The check takes about 4 s per image on one CPU core;
--workers images are checked at the same time.
The instance id table comes from the attribute instance_ids of data/demo_N (a dataset made by make_dataset.py: runs
can number the ids differently) or else from <hdf5 name>_instance_ids.json next to the hdf5 (a run folder). The
simulator normals come from the raw normals in the hdf5, or from demo_NNN_normals.mp4 in the folder of the demo (a
dataset made with --skip_raw normals).

Output, in the refs folder: check_references.csv (one row per image, with the sha1 of the checked file, so a later
image with the same name counts as not checked). Only when something fails: failed_references.txt (images to make
again, with the reason) and check_<name>.png next to the image (the image with the simulator outlines in red, the
outlines found in the image in green, and a floor-like part of the wrist table in blue). The columns room_score and
wrist_score are empty: they belong to the earlier layout check and stay for the old rows. Rows of other demos are
kept only while they still describe the file on disk. Exit code 1 when anything checked in this run fails. Both
files are written under the dataset lock (see status.py); in a dataset folder, status.py runs at the end.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from multiprocessing import Pool

import cv2
import h5py
import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layout  # noqa: E402
import make_videos  # noqa: E402
import reference_layout  # noqa: E402
import status  # noqa: E402

TILE_W, TILE_H = 1024, 512
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


def group_ids(ids: np.ndarray, table: dict | None) -> np.ndarray:
    """Instance ids -> 1 robot, 2 object, 3 environment (by prim path). Without a table the ids stay as they are."""
    if not table:
        return ids
    out = np.full(ids.shape, 3, dtype=np.int32)
    for k, prim in table.items():
        out[ids == int(k)] = 1 if "/Robot/" in prim else 2 if "/Object/" in prim else 3
    out[ids == 0] = 0  # nothing hit
    return out


def demo_tables(demo, run_tables: dict) -> dict:
    """camera -> {id: prim path} for one demo: its own attribute (dataset) or the table of the run."""
    if "instance_ids" in demo.attrs:
        return json.loads(demo.attrs["instance_ids"])
    return run_tables


def normals_frame0(obs, cams: tuple, video: str):
    """Frame 0 of the simulator normals of both cameras as uint8, (xyz + 1) * 127.5: from the raw normals in the
    hdf5, or else from the normals video of the demo (room | wrist). None when there is neither."""
    keys = [f"{prefix(c)}_normals_raw" for c in cams]
    if all(k in obs for k in keys):
        return [make_videos.normals_to_rgb(obs[k][0]) for k in keys]
    if not os.path.isfile(video):
        return None
    with imageio.get_reader(video) as reader:
        frame = reader.get_data(0)
    w = frame.shape[1] // 2
    return [np.ascontiguousarray(frame[:, :w]), np.ascontiguousarray(frame[:, w:])]


def sim_views(obs, cams: tuple, tables: dict, normals: list):
    """Frame 0 of both cameras as the input of reference_layout.check(). The table is the environment instance that
    the object lies on: the instance with the most pixels right outside the object in the room view. None when the
    room view shows no object on such an instance."""
    views, prim = {}, None
    for vname, cam, nrm in zip(reference_layout.VIEWS, cams, normals):
        ids = obs[f"{prefix(cam)}_instance_raw"][0][..., 0]
        groups = group_ids(ids, tables[cam])
        if prim is None:  # the room view comes first
            ring = (cv2.dilate((groups == 2).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (groups == 3)
            if not ring.any():
                return None
            vals, counts = np.unique(ids[ring], return_counts=True)
            prim = tables[cam].get(str(int(vals[counts.argmax()])))
            if prim is None:
                return None
        masks = {"table": np.isin(ids, [int(k) for k, v in tables[cam].items() if v == prim]),
                 "towel": groups == 2, "robot": groups == 1}
        views[vname] = reference_layout.view(cv2.cvtColor(obs[cam][0], cv2.COLOR_RGB2BGR), masks, ids == 0, nrm,
                                             obs[f"{prefix(cam)}_geoedge"][0][..., 0] > 127)
    return views


def demo_views(demo, cams: tuple, run_tables: dict, normals_video: str):
    """Simulator views of one demo for the layout check, or a text when frame 0 cannot be used. Exits when the
    files do not hold what the check needs."""
    obs, tables = demo["obs"], demo_tables(demo, run_tables)
    missing = [f"{prefix(c)}_instance_raw" for c in cams if f"{prefix(c)}_instance_raw" not in obs]
    if missing:
        sys.exit(f"the layout check needs the raw instance ids, the hdf5 has no obs/{missing[0]} (or use --no_layout)")
    if any(not tables.get(c) for c in cams):
        sys.exit("the layout check needs the instance id table (see the docstring), it is missing (or use --no_layout)")
    normals = normals_frame0(obs, cams, normals_video)
    if normals is None:
        sys.exit(f"the layout check needs the simulator normals: the hdf5 has no raw normals and {normals_video} "
                 "is missing (or use --no_layout)")
    return sim_views(obs, cams, tables, normals) or "simulator frame 0: no object on a table in the room view"


def layout_check(job: tuple) -> list:
    """One image in a worker process: (image RGB, simulator views, path of the drawing) -> the reasons why the image
    fails (an empty list = pass). The drawing is written for a failed image and removed for a passed one."""
    img, views, drawing = job
    try:
        ref = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        reasons, found = reference_layout.check(ref, views)
    except Exception as e:  # noqa: BLE001  an image that breaks the check is not checked: it fails with the error
        return [f"check error: {type(e).__name__}: {e}"]
    if reasons:
        cv2.imwrite(drawing, reference_layout.draw(ref, views, found))
    elif os.path.isfile(drawing):  # drawing of an earlier failed image
        os.remove(drawing)
    return reasons


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
    ap.add_argument("--workers", type=int, default=4, help="images checked at the same time (layout check)")
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

    rows, failed, batch = [], [], []

    def flush():
        """Run the layout check of the waiting images and finish their rows, in the order of the demos."""
        jobs = [job for _, job in batch if job]
        results = iter(pool.map(layout_check, jobs, chunksize=1) if jobs else [])
        for row, job in batch:
            if job:
                reasons = next(results)
                row.update(passed=not reasons, reason="; ".join(reasons))
            if not row["passed"]:
                failed.append(f"{row['name']}: {row['reason']}")
            rows.append(row)
            where = f" (it is in {os.path.dirname(row['file'])})" if row["reason"].startswith("wrong folder") else ""
            print(f"{row['name']}: {'ok' if row['passed'] else 'FAIL ' + row['reason'] + where}", flush=True)
        batch.clear()

    pending = status.pending_replacements(run) if status.is_dataset(run) else set()
    # The worker processes start before the hdf5 is open.
    with Pool(max(1, args.workers)) as pool, status.open_hdf5(h5path) as h5:
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
        cams = camera_keys(h5["data"][demos[0]]["obs"])
        for key in demos:
            idx = int(key.split("_")[-1])
            demo = layout.demo_name(idx)
            folder = layout.ref_dir(refs, idx)
            files = sorted(f for f in glob.glob(f"{refs}/**/{demo}*.png", recursive=True)
                           if re.fullmatch(rf"{demo}(_[^/]+)?\.png", os.path.basename(f)))
            for src in [f for f in files if os.path.dirname(f) != folder]:
                name = os.path.basename(src)[: -len(".png")]
                reason = f"wrong folder, move it to {os.path.relpath(folder, run)}"
                batch.append(({"demo": idx, "name": name, "file": os.path.relpath(src, run), "width": 0, "height": 0,
                               "size_ok": False, "room_score": "", "wrist_score": "", "passed": False,
                               "reason": reason}, None))
            files = [f for f in files if os.path.dirname(f) == folder]
            if not files:
                batch.append(({"demo": idx, "name": demo, "file": "", "width": 0, "height": 0, "size_ok": False,
                               "room_score": "", "wrist_score": "", "passed": False, "reason": "missing"}, None))
                continue
            views = None
            for src in files:
                name = os.path.basename(src)[: -len(".png")]
                with open(src, "rb") as f:
                    data = f.read()  # one read: the sha1 and the check see the same bytes
                row = {"demo": idx, "name": name, "file": os.path.relpath(src, run), "width": 0, "height": 0,
                       "size_ok": False, "room_score": "", "wrist_score": "", "passed": False, "reason": "",
                       "sha1": hashlib.sha1(data).hexdigest()}
                img, w, h, reason = decode_and_resize(data)
                row.update(width=w, height=h, size_ok=img is not None, reason=reason)
                job = None
                if img is not None and args.no_layout:
                    row["passed"] = True
                elif img is not None:
                    if views is None:
                        views = demo_views(h5["data"][key], cams, run_tables,
                                           f"{layout.demo_dir(run, idx)}/{demo}_normals.mp4")
                    if isinstance(views, str):  # the simulator frame cannot be used
                        row["reason"] = views
                    else:
                        job = (img, views, f"{folder}/check_{name}.png")
                batch.append((row, job))
            if len(batch) >= 4 * max(1, args.workers):
                flush()
        flush()

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
