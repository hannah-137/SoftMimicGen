"""Find demos where the towel is still in the gripper at the last frame. No GPU needed.

The success rule of the Franka towel task checks three things: the towel is folded (fold ratio
below 0.55), the gripper is fully open, and the gripper is at least 12 cm above the table. It does
not check where the towel is. The demo ends on the step where the gripper becomes fully open.
When the robot opens the gripper while it lifts, the towel is still hanging on the fingers at the
last frame. It had no time to fall. This script finds such demos from the hdf5 file.

How it works, for each demo:
1. Read the towel node positions `states/deformable/object/nodal_position` (T, N, 3) and the
   gripper position `obs/robot0_eef_pos` (T, 3).
2. Table height = median z of all towel nodes at frame 0.
3. At the last frame, measure
   - max_height_cm: height of the highest towel node above the table
   - lifted_ratio: share of the nodes more than --high_cm above the table
   - min_dist_cm: distance from the gripper to the nearest towel node
4. A demo is marked "stuck" when min_dist_cm < --near_cm. A towel lying on the table is far from
   the lifted gripper (more than 8 cm in our data). A towel in the fingers is about 3 cm from the
   gripper frame.

Usage:
    python checks/check_towel_stuck.py <hdf5> [--out check_towel_stuck.csv] [--high_cm 3] [--near_cm 5]

Output: one csv row per demo (demo, frames, max_height_cm, lifted_ratio, min_dist_cm, stuck) and the
list of the marked demos on the console. Needs h5py and numpy.
"""

import argparse
import csv

import h5py
import numpy as np

TOWEL_KEY = "states/deformable/object/nodal_position"
GRIPPER_KEY = "obs/robot0_eef_pos"
COLUMNS = ["demo", "frames", "max_height_cm", "lifted_ratio", "min_dist_cm", "stuck"]


def measure_demo(demo, high_cm):
    """Measure the towel at the last frame of one demo. Returns a dict of values."""
    nodes = demo[TOWEL_KEY]
    table_z = float(np.median(nodes[0][:, 2]))
    last = nodes[-1]
    height = last[:, 2] - table_z
    dist = np.linalg.norm(last - demo[GRIPPER_KEY][-1], axis=1)
    return {
        "frames": int(nodes.shape[0]),
        "max_height_cm": round(100.0 * float(height.max()), 2),
        "lifted_ratio": round(float(np.mean(height > high_cm / 100.0)), 4),
        "min_dist_cm": round(100.0 * float(dist.min()), 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("hdf5", help="dataset written by make_demos.sh")
    ap.add_argument("--out", default="check_towel_stuck.csv", help="csv path (default: current folder)")
    ap.add_argument("--high_cm", type=float, default=3.0, help="a node above this height counts as lifted")
    ap.add_argument("--near_cm", type=float, default=5.0, help="towel closer than this to the gripper = stuck")
    args = ap.parse_args()

    rows = []
    with h5py.File(args.hdf5, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        for key in keys:
            row = {"demo": f"demo_{int(key.split('_')[-1]):03d}"}
            row.update(measure_demo(f["data"][key], args.high_cm))
            row["stuck"] = int(row["min_dist_cm"] < args.near_cm)
            rows.append(row)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print("  ".join(f"{c:>13}" for c in COLUMNS))
    for row in rows:
        print("  ".join(f"{row[c]:>13}" for c in COLUMNS))
    stuck = [r["demo"] for r in rows if r["stuck"]]
    print(f"stuck: {len(stuck)} of {len(rows)} demos" + (": " + ", ".join(stuck) if stuck else ""))
    print(f"csv: {args.out}")


if __name__ == "__main__":
    main()
