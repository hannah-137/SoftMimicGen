"""Cosmos-style edges video: obs/agentview_shadedcanny -> <prefix>_shadedcanny.mp4.

The channel is produced inside the simulator by ShadedCannyImage (the CosmosWriter "edges" graph: Canny over the
colorized shaded instance-id segmentation). Nothing is post-processed here: frames are sampled, replicated to
3 channels, encoded losslessly (libx264 qp 0, 4:4:4), and every frame is decoded back and checked against the hdf5."""
import os

import imageio
import numpy as np

from common import FPS, load_obs, parse_args, prefix_of, sample_idx


def run(hdf5: str, out_dir: str, demo: str = "demo_0"):
    sc = load_obs(hdf5, demo, "agentview_shadedcanny")  # (T, H, W, 1) uint8, values {0, 255}
    if sc is None:
        print("shaded canny: agentview_shadedcanny not in hdf5, skipped")
        return None
    frames = np.repeat(sc[sample_idx(len(sc))], 3, axis=-1)  # (N, H, W, 3), values unchanged
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix_of(hdf5)}_shadedcanny.mp4")
    imageio.mimsave(path, list(frames), fps=FPS, codec="libx264", pixelformat="yuv444p", output_params=["-qp", "0"])

    decoded = [fr for fr in imageio.get_reader(path)]
    bad = [i for i, (d, r) in enumerate(zip(decoded, frames)) if not np.array_equal(d, r)]
    ok = len(decoded) == len(frames) and not bad
    print(
        f"shaded canny: {path} ({len(frames)} frames, {FPS}fps), all frames == hdf5: {ok}"
        + ("" if ok else f"  MISMATCH: decoded {len(decoded)} frames, differing frames {bad[:10]}")
    )
    return path


if __name__ == "__main__":
    args = parse_args(__doc__)
    run(args.hdf5, args.out_dir, args.demo)
