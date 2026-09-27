"""Find demos where the towel touches the image border in any frame of the room camera. No GPU needed.

The room camera can get a random pose per demo (make_demos.sh --camera_noise_pos / --camera_noise_rot). The
generator only accepts a pose where the towel is inside the image at the start. This script checks every frame
after the fact, from the raw instance ids in the hdf5.

How it works, for each demo and frame:
1. Towel pixels = pixels whose instance id maps to a prim path with "/Object/" (from <hdf5>_instance_ids.json).
2. border_px = the smallest distance from a towel pixel to the image border, in pixels.
3. The demo is marked "cut" when border_px < --margin in any frame (default margin 1: the towel touches the border).

Usage:
    python checks/check_towel_in_view.py <hdf5> [--out check_towel_in_view.csv] [--margin 1] [--stride 1]

Output: one csv row per demo (demo, frames, min_border_px, worst_frame, cut) and the list of the marked demos on
the console. --stride N checks every N-th frame (faster). Needs h5py and numpy.
"""

import argparse
import csv
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layout  # noqa: E402

COLUMNS = ["demo", "frames", "min_border_px", "worst_frame", "cut"]


def towel_ids(table_path: str, camera_key: str) -> list:
    table = json.load(open(table_path))
    ids = [int(k) for k, prim in table.get(camera_key, {}).items() if "/Object/" in prim]
    if not ids:
        sys.exit(f"no towel instance id (prim path with /Object/) for {camera_key} in {table_path}")
    return ids


def border_distance(mask: np.ndarray) -> float:
    """Smallest distance from a True pixel to the image border. inf when the mask is empty."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return float("inf")
    h, w = mask.shape
    return float(min(ys.min(), xs.min(), h - 1 - ys.max(), w - 1 - xs.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("hdf5", help="dataset written by make_demos.sh")
    ap.add_argument("--out", default="check_towel_in_view.csv", help="csv path (default: current folder)")
    ap.add_argument("--margin", type=float, default=1.0, help="a towel pixel closer than this to the border = cut")
    ap.add_argument("--stride", type=int, default=1, help="check every N-th frame")
    args = ap.parse_args()

    table_path = args.hdf5[: -len(".hdf5")] + "_instance_ids.json"
    rows = []
    with h5py.File(args.hdf5, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        obs0 = f["data"][keys[0]]["obs"]
        room = [k for k in obs0 if k.endswith("_instance_raw") and "eye_in_hand" not in k and "wrist" not in k]
        if len(room) != 1:
            sys.exit(f"cannot find the room camera instance channel: {room}")
        ids = np.array(towel_ids(table_path, room[0][: -len("_instance_raw")] + "_image"))
        for key in keys:
            data = f["data"][key]["obs"][room[0]]
            best, worst = float("inf"), -1
            for t in range(0, data.shape[0], args.stride):
                d = border_distance(np.isin(data[t][..., 0], ids))
                if d < best:
                    best, worst = d, t
            row = {"demo": layout.demo_name(int(key.split("_")[-1])), "frames": int(data.shape[0]),
                   "min_border_px": best if best != float("inf") else "", "worst_frame": worst, "cut": int(best < args.margin)}
            rows.append(row)
            print(f"{row['demo']}: min border {row['min_border_px']} px at frame {worst}{'  CUT' if row['cut'] else ''}", flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    cut = [r["demo"] for r in rows if r["cut"]]
    print(f"cut: {len(cut)} of {len(rows)} demos" + (": " + ", ".join(cut) if cut else ""))
    print(f"csv: {args.out}")


if __name__ == "__main__":
    main()
