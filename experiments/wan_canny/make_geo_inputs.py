"""Stage 3i: geometry-edge input videos (inspection only, not Wan controls).
obs/<edges>_depth, _normals, _instance -> <prefix>_depth.mp4, _normals.mp4, _instance.mp4.

  python experiments/wan_canny/make_geo_inputs.py <run_dir> [--hdf5 H5] [--prefix P] [--task T] [--demo demo_0]

The channels are the 8-bit views of GeoEdgeImage's three inputs recorded by GeoInputImage (only present when the
hdf5 was generated with --inspect): depth (0..far plane -> 0..255, grey), normals (xyz -> RGB), colourised instance
ids. Nothing is post-processed here: frames are sampled, encoded losslessly (libx264rgb qp 0, exact RGB) and every
frame is decoded back and checked against the hdf5."""
import imageio
import numpy as np

from common import FPS, load_obs, out_path, parse_args, sample_idx

KINDS = ("depth", "normals", "instance")


def run(rd):
    paths = {}
    for kind in KINDS:
        key = f"{rd.edges}_{kind}"
        arr = load_obs(rd.hdf5, rd.demo, key)  # (T, H, W, 1) for depth, (T, H, W, 3) otherwise, uint8
        if arr is None:
            print(f"{kind}: {key} not in hdf5, skipped")
            continue
        frames = arr[sample_idx(len(arr))]
        if frames.shape[-1] == 1:
            frames = np.repeat(frames, 3, axis=-1)
        path = out_path(rd, f"{kind}.mp4", "sources")
        imageio.mimsave(path, list(frames), fps=FPS, codec="libx264rgb", pixelformat="rgb24", output_params=["-qp", "0"])
        decoded = [fr for fr in imageio.get_reader(path)]
        diff = [int(np.abs(d.astype(int) - r.astype(int)).max()) for d, r in zip(decoded, frames)]
        ok = len(decoded) == len(frames) and max(diff) == 0
        print(
            f"{kind}: {path} ({len(frames)} frames, {FPS}fps), all frames == hdf5: {ok}"
            + ("" if ok else f"  (decoded {len(decoded)} frames, max abs diff {max(diff)})")
        )
        paths[kind] = path
    return paths


if __name__ == "__main__":
    run(parse_args(__doc__))
