"""RGB Canny parameter sweep for the 2026-09-15 meeting report: blur strength x Canny thresholds, and camera size.

  python experiments/wan_canny/scripts/sweep_rgb_canny.py <source_run_dir> <sweep_dir>

Reads the rendered frames of <source_run_dir> (a finished run, e.g. runs/v3/franka_towel_v3) and writes one RGB Canny
control video per setting into <sweep_dir>/edges/<prefix>_<name>.mp4, plus the source run's reference images and a link
to its hdf5 (for the layout check), so make_wan.py can run on <sweep_dir> with --controls <names>:
  blur:   b0 = none, b1 = bilateral 9/50/7, b2 = bilateral 15/100/15
  thresh: t1 = 10/50, t2 = 30/100, t3 = 80/180
  size:   s128, s256 = frames downscaled to that camera size, Canny at that size, lines upscaled back to 512 (nearest),
          with b1 t2; the 512 case is b1_t2.
Everything else is the pipeline's RGB Canny: colour Canny on 3-channel Sobel, close 3, drop blobs under 50 px, 1 px lines.
The results are judged on the Wan videos, not on the edge maps.
"""
import os
import shutil
import sys

import cv2
import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FPS, load_obs, resolve_run, sample_idx  # noqa: E402

BLUR = {"b0": None, "b1": (9, 50, 7), "b2": (15, 100, 15)}
THRESH = {"t1": (10, 50), "t2": (30, 100), "t3": (80, 180)}
SIZES = {"s128": 128, "s256": 256}


def canny(im, blur, lo, hi, close_k=3, min_area=50):
    if blur:
        im = cv2.bilateralFilter(im, *blur)
    dx = cv2.Sobel(im, cv2.CV_16S, 1, 0, ksize=3)
    dy = cv2.Sobel(im, cv2.CV_16S, 0, 1, ksize=3)
    e = cv2.Canny(dx, dy, lo, hi)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(e, connectivity=8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area * (e.shape[0] / 512) ** 2:
            e[lab == i] = 0
    return e


def main(src, dst):
    rd = resolve_run(src)
    prefix = os.path.basename(os.path.normpath(dst))
    for sub in ("edges", "images", "sources"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
    link = os.path.join(dst, f"{prefix}.hdf5")
    if not os.path.exists(link):
        os.symlink(os.path.abspath(rd.hdf5), link)
    shutil.copy(os.path.join(src, "images", f"{rd.prefix}_ref_sim.png"), os.path.join(dst, "images", f"{prefix}_ref_sim.png"))
    shutil.copy(os.path.join(src, "images", f"{rd.prefix}_ref_ai_01.png"), os.path.join(dst, "images", f"{prefix}_ref_ai.png"))
    shutil.copy(os.path.join(src, "sources", f"{rd.prefix}_source.mp4"), os.path.join(dst, "sources", f"{prefix}_source.mp4"))
    imgs = load_obs(rd.hdf5, rd.demo, rd.camera)
    imgs = imgs[sample_idx(len(imgs))]
    h, w = imgs.shape[1:3]
    jobs = {f"{b}_{t}": (BLUR[b], *THRESH[t], None) for b in BLUR for t in THRESH}
    jobs.update({s: (BLUR["b1"], *THRESH["t2"], size) for s, size in SIZES.items()})
    for name, (blur, lo, hi, size) in jobs.items():
        frames = []
        for im in imgs:
            if size:
                small = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
                e = cv2.resize(canny(small, blur, lo, hi), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                e = canny(im, blur, lo, hi)
            frames.append(cv2.cvtColor(e, cv2.COLOR_GRAY2RGB))
        path = os.path.join(dst, "edges", f"{prefix}_{name}.mp4")
        imageio.mimsave(path, frames, fps=FPS, codec="libx264", pixelformat="yuv444p", output_params=["-qp", "0"])
        px = float(np.mean([(f[..., 0] > 0).sum() for f in frames]))
        print(f"sweep: {name}: blur={blur} canny={lo}/{hi} size={size or w} -> {os.path.basename(path)} ({px:.0f} edge px/frame)")
    print("controls:", ",".join(jobs))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
