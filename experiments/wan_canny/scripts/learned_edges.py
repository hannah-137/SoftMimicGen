"""Learned edge/line detectors (HED, PiDiNet, TEED, LineArt) on rendered RGB frames.

Runs in the `rgb_edge` conda env (docker/setup_rgb_edge.sh), NOT in softmimicgen. All four detectors return a soft
edge map (uint8 0..255, bright = edge); `binarize()` turns it into 1 px lines so the outputs are comparable with the
Canny videos. Two binarization paths (EDGE_BENCHMARK.md, 2026-09-13):
  - base:        threshold -> morphological close -> skeletonize -> drop small blobs
  - region+nms:  a lower threshold inside a region of interest (the towel, where the detectors' fold response is weak
                 but still 10-20x above the non-fold response) and NMS ridge thinning instead of skeletonize.
"""
import time

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize

DETECTORS = ("hed", "pidinet", "teed", "lineart")

_cache = {}


def load_detector(name: str, device: str):
    """Lazy-load one controlnet_aux detector (weights come from HF_HOME, see docker/setup_rgb_edge.sh)."""
    if name in _cache:
        return _cache[name]
    if name == "hed":
        from controlnet_aux import HEDdetector

        det = HEDdetector.from_pretrained("lllyasviel/Annotators")
    elif name == "pidinet":
        from controlnet_aux import PidiNetDetector

        det = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
    elif name == "teed":
        from controlnet_aux import TEEDdetector

        det = TEEDdetector.from_pretrained("fal-ai/teed", filename="5_model.pth")
    elif name == "lineart":
        from controlnet_aux import LineartDetector

        det = LineartDetector.from_pretrained("lllyasviel/Annotators")
    else:
        raise ValueError(f"unknown detector {name!r}, expected one of {DETECTORS}")
    _cache[name] = det.to(device)
    return _cache[name]


def soft_map(name: str, frame: np.ndarray, device: str, **opts) -> np.ndarray:
    """(H, W, 3) RGB uint8 -> (H, W) uint8 soft edge map at the frame's own resolution.

    opts (detector specific, all optional): hed: safe; pidinet: safe, apply_filter; teed: safe_steps;
    lineart: coarse. detect_resolution defaults to the frame size.
    """
    det = load_detector(name, device)
    h, w = frame.shape[:2]
    res = opts.pop("detect_resolution", max(h, w))
    if name == "teed":
        out = det(frame, detect_resolution=res, output_type="np", **opts)
    else:
        out = det(frame, detect_resolution=res, image_resolution=max(h, w), output_type="np", **opts)
    out = np.asarray(out)
    if out.ndim == 3:
        out = out[..., 0]
    if out.shape != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def roi_mask(rgb: np.ndarray, kind: str = "hsv_blue", dilate_k: int = 7) -> np.ndarray | None:
    """Region where the lower threshold applies. 'hsv_blue' = the blue towel (largest blue blob, dilated so the
    outline is included); 'none' disables the region-adaptive threshold. Other scenes need their own rule here."""
    if kind == "none":
        return None
    if kind != "hsv_blue":
        raise ValueError(f"unknown roi kind {kind!r}")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, (95, 80, 40), (130, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (labels == big).astype(np.uint8) * 255
    return cv2.dilate(m, np.ones((dilate_k, dilate_k), np.uint8)) > 0


def nms_ridge(soft01: np.ndarray, sigma: float) -> np.ndarray:
    """Directional local-maximum (ridge) pixels of a soft map, as in controlnet_aux.util.nms: after Gaussian
    smoothing keep pixels that are the maximum of their 3-neighbourhood along at least one of 4 orientations."""
    x = cv2.GaussianBlur(soft01.astype(np.float32), (0, 0), sigma) if sigma > 0 else soft01.astype(np.float32)
    kernels = [
        np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], np.uint8),
        np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], np.uint8),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.uint8),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], np.uint8),
    ]
    y = np.zeros_like(x)
    for k in kernels:
        np.putmask(y, cv2.dilate(x, k) == x, x)
    return y > 0.0


def binarize(
    soft: np.ndarray,
    threshold: float,
    roi: np.ndarray | None = None,
    threshold_in: float | None = None,
    nms_sigma: float | None = None,
    close_k: int = 3,
    min_area: int = 50,
) -> np.ndarray:
    """uint8 soft map -> bool 1 px line map. Thresholds are on the 0..1 scale.

    threshold applies everywhere; inside `roi` (bool mask) `threshold_in` applies instead when both are given.
    With nms_sigma the lines are the NMS ridge pixels that pass the threshold (use close_k=1 there); otherwise the
    thresholded map is closed and skeletonized. Finally connected components smaller than min_area are dropped.
    """
    s = soft.astype(np.float32) / 255.0
    b = s > threshold
    if roi is not None and threshold_in is not None:
        b = np.where(roi, s > threshold_in, b)
    if nms_sigma is not None:
        b = nms_ridge(s, nms_sigma) & b
    u = b.astype(np.uint8) * 255
    if close_k > 1:
        u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    thin = skeletonize(u > 0)
    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(thin.astype(np.uint8), connectivity=8)
        small = np.zeros(n, bool)
        small[1:] = stats[1:, cv2.CC_STAT_AREA] < min_area
        thin[small[labels]] = False
    return thin


def pick_device() -> str:
    """Use the CUDA device with the most free memory (the GPUs are shared with other users), else CPU."""
    if not torch.cuda.is_available():
        return "cpu"
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best, best_free = i, free
    return f"cuda:{best}"


def timed_soft_maps(name: str, frames: np.ndarray, device: str, **opts):
    """Soft maps for a stack of frames plus mean seconds per frame."""
    t0 = time.time()
    maps = np.stack([soft_map(name, f, device, **dict(opts)) for f in frames])
    return maps, (time.time() - t0) / len(frames)
