"""Stage 2: source video and reference image. obs/<camera> -> <prefix>_source.mp4 + <prefix>_ref_sim.png (frame 0).

  python experiments/wan_canny/make_source.py <run_dir> [--hdf5 H5] [--prefix P] [--task T] [--demo demo_0]
"""
import cv2
import imageio

from common import FPS, load_obs, out_path, parse_args, sample_idx


def run(rd) -> tuple[str, str]:
    imgs = load_obs(rd.hdf5, rd.demo, rd.camera)
    imgs = imgs[sample_idx(len(imgs))]
    src_path = out_path(rd, "source.mp4", "sources")
    ref_path = out_path(rd, "ref_sim.png", "images")
    imageio.mimsave(src_path, list(imgs), fps=FPS)
    cv2.imwrite(ref_path, cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR))  # first frame as candidate reference image
    h, w = imgs.shape[1:3]
    print(f"source: {src_path} ({len(imgs)} frames, {w}x{h}, {FPS}fps), reference: {ref_path}")
    return src_path, ref_path


if __name__ == "__main__":
    run(parse_args(__doc__))
