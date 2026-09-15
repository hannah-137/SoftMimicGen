"""Stage 3g: union edges video. obs/<edges>_shadedcanny OR obs/<edges>_geoedge -> <prefix>_union.mp4.

  python experiments/wan_canny/scripts/make_union.py <run_dir> [--hdf5 H5] [--prefix P] [--task T] [--demo demo_0]

Pixel-wise OR of the two simulator edge channels (Cosmos-style shaded Canny and the geometry edges), so the video
carries both the shading/fold lines of the first and the depth/normal/id boundaries of the second. No other
processing: same 81-frame sampling, 16 fps, 1 px lines, lossless encoding, and a full decode check against the hdf5."""
import imageio
import numpy as np

from common import FPS, load_obs, out_path, parse_args, sample_idx


def run(rd):
    sc = load_obs(rd.hdf5, rd.demo, f"{rd.edges}_shadedcanny")
    ge = load_obs(rd.hdf5, rd.demo, f"{rd.edges}_geoedge")
    if sc is None or ge is None:
        print(f"union: {rd.edges}_shadedcanny or {rd.edges}_geoedge not in hdf5, skipped")
        return None
    idx = sample_idx(len(sc))
    union = ((sc[idx] > 0) | (ge[idx] > 0)).astype(np.uint8) * 255  # (N, H, W, 1)
    frames = np.repeat(union, 3, axis=-1)
    path = out_path(rd, "union.mp4", "edges")
    imageio.mimsave(path, list(frames), fps=FPS, codec="libx264", pixelformat="yuv444p", output_params=["-qp", "0"])

    decoded = [fr for fr in imageio.get_reader(path)]
    bad = [i for i, (d, r) in enumerate(zip(decoded, frames)) if not np.array_equal(d, r)]
    ok = len(decoded) == len(frames) and not bad
    px = (sc[idx] > 0).sum() / len(idx), (ge[idx] > 0).sum() / len(idx), (union > 0).sum() / len(idx)
    print(
        f"union: {path} ({len(frames)} frames, {FPS}fps; edge px/frame shaded {px[0]:.0f} + geo {px[1]:.0f} -> "
        f"union {px[2]:.0f}), all frames == hdf5: {ok}" + ("" if ok else f"  MISMATCH: differing frames {bad[:10]}")
    )
    return path


if __name__ == "__main__":
    run(parse_args(__doc__))
