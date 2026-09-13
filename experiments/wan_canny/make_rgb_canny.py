"""Stage 3a: RGB Canny control video. obs/<camera> -> <prefix>_canny.mp4.

  python experiments/wan_canny/make_rgb_canny.py <run_dir> [--hdf5 H5] [--prefix P] [--task T] [--demo demo_0]

Pipeline on the rendered RGB frames: bilateral filter -> colour Canny (3-channel Sobel, per-pixel max-magnitude
channel) -> close / small-blob removal, no dilation (1 px lines like the shaded-canny channel, Wan-Fun's own
preprocessor and CRAFT). Encoded losslessly.
Filter and thresholds are the top-1 configuration of the 2026-09-12 benchmark against the shaded-canny (Cosmos
edges) ground truth on franka_towel_cosmos_v4 (512 configs, 27 frames, boundary F1 at 2 px tolerance); the
post-processing is the best variant without dilation: F1 0.838 (precision 0.831, recall 0.846). With dilate 2 the
same pipeline scores 0.843; the previous median-7 / per-channel Canny 10/70 / dilate 2 pipeline scored 0.823.
Independent of the shaded-segmentation channel (see make_shaded_canny.py)."""
import cv2
import imageio
import numpy as np

from common import FPS, load_obs, out_path, parse_args, sample_idx

BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE = 9, 50, 7  # edge-preserving pre-filter (kills weave)
CANNY_LO, CANNY_HI = 30, 100   # hysteresis thresholds on the L1 gradient (|dx| + |dy|)
CANNY_L2 = False
DILATE_K = 1     # edge thickening kernel; 1 = no dilation (1 px lines)
CLOSE_K = 3      # gap bridging kernel
MIN_AREA = 50    # drop isolated edge blobs smaller than this (pixels)

_dilate_kernel = np.ones((DILATE_K, DILATE_K), np.uint8)
_close_kernel = np.ones((CLOSE_K, CLOSE_K), np.uint8)


def canny_frame(im: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB uint8 -> (H, W, 3) edge image (0/255 replicated to 3 channels)."""
    im_b = cv2.bilateralFilter(im, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
    # colour Canny: OpenCV's (dx, dy) overload with 3-channel CV_16S gradients uses, per pixel, the channel with the
    # largest magnitude, so colour edges (blue towel on black table) are found without OR-ing three binary maps
    dx = cv2.Sobel(im_b, cv2.CV_16S, 1, 0, ksize=3)
    dy = cv2.Sobel(im_b, cv2.CV_16S, 0, 1, ksize=3)
    edges = cv2.Canny(dx, dy, CANNY_LO, CANNY_HI, L2gradient=CANNY_L2)
    if DILATE_K > 1:
        edges = cv2.dilate(edges, _dilate_kernel)                    # thicken
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _close_kernel)  # bridge gaps
    n, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    for i in range(1, n):                                            # drop small blobs
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            edges[labels == i] = 0
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def run(rd) -> str:
    imgs = load_obs(rd.hdf5, rd.demo, rd.camera)
    imgs = imgs[sample_idx(len(imgs))]
    frames = [canny_frame(im) for im in imgs]
    path = out_path(rd, "canny.mp4")
    # lossless (libx264 qp 0, 4:4:4): binary edge frames compress smaller than lossy and keep exact 0/255 values
    imageio.mimsave(path, frames, fps=FPS, codec="libx264", pixelformat="yuv444p", output_params=["-qp", "0"])
    print(
        f"rgb canny: {path} ({len(frames)} frames, {FPS}fps; bilateral {BILATERAL_D}/{BILATERAL_SIGMA_COLOR}/"
        f"{BILATERAL_SIGMA_SPACE}, colour canny {CANNY_LO}/{CANNY_HI} L2={CANNY_L2}, dilate {DILATE_K}, "
        f"close {CLOSE_K}, min_area {MIN_AREA})"
    )
    return path


if __name__ == "__main__":
    run(parse_args(__doc__))
