"""Stage 3c: geometry edges video. obs/<edges>_geoedge -> <prefix>_geoedge.mp4.

  python experiments/wan_canny/scripts/make_geo_edge.py <run_dir> [--hdf5 H5] [--prefix P] [--task T] [--demo demo_0]

The channel is produced inside the simulator by GeoEdgeImage (depth jump | normal angle | instance-id change between
right/lower neighbours, 1 px lines, no Canny). Nothing is post-processed here: frames are sampled, replicated to
3 channels, encoded losslessly (libx264 qp 0, 4:4:4), and every frame is decoded back and checked against the hdf5."""
import imageio
import numpy as np

from common import FPS, load_obs, out_path, parse_args, sample_idx


def run(rd):
    key = f"{rd.edges}_geoedge"
    ge = load_obs(rd.hdf5, rd.demo, key)  # (T, H, W, 1) uint8, values {0, 255}
    if ge is None:
        print(f"geo edge: {key} not in hdf5, skipped")
        return None
    frames = np.repeat(ge[sample_idx(len(ge))], 3, axis=-1)  # (N, H, W, 3), values unchanged
    path = out_path(rd, "geoedge.mp4", "edges")
    imageio.mimsave(path, list(frames), fps=FPS, codec="libx264", pixelformat="yuv444p", output_params=["-qp", "0"])

    decoded = [fr for fr in imageio.get_reader(path)]
    bad = [i for i, (d, r) in enumerate(zip(decoded, frames)) if not np.array_equal(d, r)]
    ok = len(decoded) == len(frames) and not bad
    print(
        f"geo edge: {path} ({len(frames)} frames, {FPS}fps), all frames == hdf5: {ok}"
        + ("" if ok else f"  MISMATCH: decoded {len(decoded)} frames, differing frames {bad[:10]}")
    )
    return path


if __name__ == "__main__":
    run(parse_args(__doc__))
