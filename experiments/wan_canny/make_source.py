"""Source video and reference image: obs/agentview_image -> <prefix>_source.mp4 + <prefix>_ref_sim.png (frame 0)."""
import os

import cv2
import imageio

from common import FPS, load_obs, parse_args, prefix_of, sample_idx


def run(hdf5: str, out_dir: str, demo: str = "demo_0") -> tuple[str, str]:
    imgs = load_obs(hdf5, demo, "agentview_image")
    imgs = imgs[sample_idx(len(imgs))]
    os.makedirs(out_dir, exist_ok=True)
    prefix = prefix_of(hdf5)
    src_path = os.path.join(out_dir, f"{prefix}_source.mp4")
    ref_path = os.path.join(out_dir, f"{prefix}_ref_sim.png")
    imageio.mimsave(src_path, list(imgs), fps=FPS)
    cv2.imwrite(ref_path, cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR))  # first frame as candidate reference image
    h, w = imgs.shape[1:3]
    print(f"source: {src_path} ({len(imgs)} frames, {w}x{h}, {FPS}fps), reference: {ref_path}")
    return src_path, ref_path


if __name__ == "__main__":
    args = parse_args(__doc__)
    run(args.hdf5, args.out_dir, args.demo)
