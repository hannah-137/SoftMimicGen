"""Instance video of every simulator step for the demos of a dataset: <demo>_instance_all.mp4 in each demo folder.

    python experiments/policy_data/instance_all.py <dataset> --demos <range>

The video has the frames of the all-steps videos (one frame per step, at the control rate of the simulator), room |
wrist, one fixed color per instance id, lossless (the same look as the 81-frame demo_NNN_instance.mp4). It is made
from <camera>_instance_raw of the dataset hdf5; the 81-frame videos are not touched. A demo whose file exists with
the right frame count is skipped, so the script can run again.
"""
import argparse
import os
import sys
import time

import layout
import make_videos
import run_cosmos
import status


def instance_video(ds: str, idx: int) -> str:
    """Write <demo>_instance_all.mp4 for one demo when it is missing or has another frame count. -> "made" or "kept"."""
    d, name = layout.demo_dir(ds, idx), layout.demo_name(idx)
    out = f"{d}/{name}_instance{run_cosmos.ALL}.mp4"
    with status.open_hdf5(run_cosmos.demo_hdf5(ds)) as h5:
        fps = make_videos.control_fps(h5)
        room_key, wrist_key = make_videos.camera_keys(h5, None, None)
        obs = h5["data"][f"demo_{idx}"]["obs"]
        steps = obs[room_key].shape[0]
        if run_cosmos.video_frames(out) == steps:
            return "kept"
        keys = [f"{make_videos.prefix(k)}_instance_raw" for k in (room_key, wrist_key)]
        for key in keys:
            if key not in obs:
                sys.exit(f"{ds}: no {key} in demo_{idx}")
        room, wrist = (make_videos.instance_to_rgb(obs[k][:]) for k in keys)
    frames = [make_videos.tile(a, b) for a, b in zip(room, wrist)]
    tmp = f"{d}/.{name}_instance{run_cosmos.ALL}.tmp{os.getpid()}.mp4"
    make_videos.write(tmp, frames, fps, make_videos.LOSSLESS_RGB, check=True)
    os.replace(tmp, out)
    if run_cosmos.video_frames(out) != steps:
        sys.exit(f"{out}: {run_cosmos.video_frames(out)} frames, expected {steps}")
    return "made"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--demos", required=True, help="numbers, for example 0-349 or 3,5,10-12")
    args = ap.parse_args()
    ds = os.path.normpath(args.dataset)
    if not status.is_dataset(ds):
        sys.exit(f"{ds} is not a dataset (no sources.csv)")
    t0, made, kept = time.time(), 0, 0
    for idx in status.parse_demos([args.demos]):
        t = time.time()
        what = instance_video(ds, idx)
        made += what == "made"
        kept += what == "kept"
        print(f"{layout.demo_name(idx)} {what} {time.time() - t:.0f} s", flush=True)
    print(f"made {made} | kept {kept} | {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
