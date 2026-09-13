"""RGB Canny control video for Wan: obs/agentview_image -> <prefix>_canny.mp4.

CRAFT-style post-processing on the rendered RGB frames: median blur, per-channel Canny, dilation, closing,
small-blob removal. Independent of the shaded-segmentation channel (see make_shaded_canny.py)."""
import os

import cv2
import imageio
import numpy as np

from common import FPS, load_obs, parse_args, prefix_of, sample_idx

CANNY_LO, CANNY_HI = 10, 70
BLUR_K = 7     # median blur: kills fabric texture, keeps outlines (odd number)
DILATE_K = 2   # edge thickening kernel
CLOSE_K = 3    # gap bridging kernel
MIN_AREA = 50  # drop isolated edge blobs smaller than this (pixels)

_dilate_kernel = np.ones((DILATE_K, DILATE_K), np.uint8)
_close_kernel = np.ones((CLOSE_K, CLOSE_K), np.uint8)


def canny_frame(im: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB uint8 -> (H, W, 3) edge image (0/255 replicated to 3 channels)."""
    im_b = cv2.medianBlur(im, BLUR_K)  # remove fine texture before edge detection
    edges = np.zeros(im.shape[:2], np.uint8)
    for c in range(3):  # per-channel Canny, then combine: catches colour edges (blue towel on dark table)
        edges = cv2.bitwise_or(edges, cv2.Canny(im_b[:, :, c], CANNY_LO, CANNY_HI))
    edges = cv2.dilate(edges, _dilate_kernel)                        # thicken
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _close_kernel)  # bridge gaps
    n, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    for i in range(1, n):                                            # drop small blobs
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            edges[labels == i] = 0
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def run(hdf5: str, out_dir: str, demo: str = "demo_0") -> str:
    imgs = load_obs(hdf5, demo, "agentview_image")
    imgs = imgs[sample_idx(len(imgs))]
    frames = [canny_frame(im) for im in imgs]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix_of(hdf5)}_canny.mp4")
    imageio.mimsave(path, frames, fps=FPS)
    print(
        f"rgb canny: {path} ({len(frames)} frames, {FPS}fps; canny {CANNY_LO}/{CANNY_HI}, blur {BLUR_K}, "
        f"dilate {DILATE_K}, close {CLOSE_K}, min_area {MIN_AREA})"
    )
    return path


if __name__ == "__main__":
    args = parse_args(__doc__)
    run(args.hdf5, args.out_dir, args.demo)
