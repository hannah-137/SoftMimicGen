"""Stage 3h: shaded segmentation video (inspection only, not a Wan control). obs/<edges>_shaded -> <prefix>_shaded.mp4.

  python experiments/wan_canny/make_shaded.py <run_dir> [--hdf5 H5] [--prefix P] [--task T] [--demo demo_0]

The channel is the colourised shaded instance-id segmentation recorded by ShadedSegImage, i.e. the image the shaded
Canny is computed from. Nothing is post-processed here: frames are sampled, encoded losslessly (libx264rgb qp 0, exact RGB)
and every frame is decoded back and checked against the hdf5."""
import imageio
import numpy as np

from common import FPS, load_obs, out_path, parse_args, sample_idx


def run(rd):
    key = f"{rd.edges}_shaded"
    sh = load_obs(rd.hdf5, rd.demo, key)  # (T, H, W, 3) uint8 RGB
    if sh is None:
        print(f"shaded: {key} not in hdf5, skipped")
        return None
    frames = sh[sample_idx(len(sh))]
    path = out_path(rd, "shaded.mp4", "sources")
    # colour frames: libx264rgb keeps RGB exactly (yuv444p qp 0 is exact only for 0/255 line images, off by 1-2 here)
    imageio.mimsave(path, list(frames), fps=FPS, codec="libx264rgb", pixelformat="rgb24", output_params=["-qp", "0"])

    decoded = [fr for fr in imageio.get_reader(path)]
    diff = [int(np.abs(d.astype(int) - r.astype(int)).max()) for d, r in zip(decoded, frames)]
    ok = len(decoded) == len(frames) and max(diff) == 0
    print(
        f"shaded: {path} ({len(frames)} frames, {FPS}fps), all frames == hdf5: {ok}"
        + ("" if ok else f"  (decoded {len(decoded)} frames, max abs diff {max(diff)})")
    )
    return path


if __name__ == "__main__":
    run(parse_args(__doc__))
