"""Stage 3d: learned edge/line videos. obs/<camera> -> <prefix>_hed.mp4, _pidinet.mp4, _teed.mp4, _lineart.mp4.

Runs in the rgb_edge env (docker/setup_rgb_edge.sh), NOT in softmimicgen:
  source /opt/miniconda3/etc/profile.d/conda.sh && conda activate /workspace/tools/envs/rgb_edge
  HF_HOME=/workspace/tools/rgb_edge_models python experiments/wan_canny/make_learned_edges.py <run_dir> \
      [--hdf5 H5] [--prefix P] [--task T] [--demo demo_0] [--detectors hed,pidinet,teed,lineart] [--device cuda:N]
      [--roi hsv_blue|none]

Each detector's soft map (0..255) is binarized to 1 px lines with learned_edges.binarize so the videos are comparable
with _canny.mp4 / _shadedcanny.mp4. The per-detector DEFAULTS are the best settings of the 2026-09-13 benchmark
against the shaded-canny ground truth (EDGE_BENCHMARK.md): HED / PiDiNet / LineArt use the region-adaptive
threshold (lower inside the ROI) with NMS ridge thinning; TEED uses the plain threshold path. The ROI comes from
tasks.TASKS[task]["roi"] (hsv_blue = the blue towel) unless --roi is given.
Same 81-frame sampling, 16 fps and lossless encoding as the other make_*.py scripts.
"""
import argparse

import imageio
import numpy as np

import learned_edges as LE
from common import FPS, add_run_args, load_obs, out_path, resolve_run, sample_idx

# binarize kwargs: threshold (everywhere), threshold_in (inside the ROI), nms_sigma, close_k, min_area
DEFAULTS = {
    "hed": dict(opts=dict(safe=False), detect_resolution=512,
                binarize=dict(threshold=0.4, threshold_in=0.03, nms_sigma=1.0, close_k=1, min_area=50)),
    "pidinet": dict(opts=dict(safe=False), detect_resolution=1024,
                    binarize=dict(threshold=0.4, threshold_in=0.03, nms_sigma=2.0, close_k=1, min_area=50)),
    "teed": dict(opts=dict(safe_steps=0), detect_resolution=1024,
                 binarize=dict(threshold=0.6, threshold_in=None, nms_sigma=None, close_k=3, min_area=50)),
    "lineart": dict(opts=dict(coarse=False), detect_resolution=512,
                    binarize=dict(threshold=0.2, threshold_in=0.03, nms_sigma=1.0, close_k=1, min_area=50)),
}


def run(rd, detectors=LE.DETECTORS, device: str | None = None, roi: str | None = None):
    roi = roi or rd.roi
    imgs = load_obs(rd.hdf5, rd.demo, rd.camera)
    imgs = imgs[sample_idx(len(imgs))]
    device = device or LE.pick_device()
    rois = [LE.roi_mask(im, roi) for im in imgs]
    paths = {}
    for name in detectors:
        cfg = DEFAULTS[name]
        maps, spf = LE.timed_soft_maps(name, imgs, device, detect_resolution=cfg["detect_resolution"], **cfg["opts"])
        lines = np.stack([LE.binarize(m, roi=r, **cfg["binarize"]) for m, r in zip(maps, rois)]).astype(np.uint8) * 255
        frames = np.repeat(lines[..., None], 3, axis=-1)
        path = out_path(rd, f"{name}.mp4")
        imageio.mimsave(path, list(frames), fps=FPS, codec="libx264", pixelformat="yuv444p", output_params=["-qp", "0"])
        decoded = [fr for fr in imageio.get_reader(path)]
        ok = len(decoded) == len(frames) and all(np.array_equal(d, r) for d, r in zip(decoded, frames))
        print(
            f"{name}: {path} ({len(frames)} frames, {FPS}fps; {cfg['opts']} detect_resolution={cfg['detect_resolution']} "
            f"binarize={cfg['binarize']} roi={roi}, {spf * 1000:.0f} ms/frame on {device}), lossless check: {ok}"
        )
        paths[name] = path
    return paths


if __name__ == "__main__":
    p = add_run_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--detectors", default=",".join(LE.DETECTORS), help="comma-separated subset of " + ",".join(LE.DETECTORS))
    p.add_argument("--device", default=None, help="cuda:N or cpu; default = CUDA device with the most free memory")
    p.add_argument("--roi", default=None, choices=["hsv_blue", "none"],
                   help="region for the lower threshold (default: tasks.TASKS[task]['roi']); none = single threshold")
    a = p.parse_args()
    rd = resolve_run(a.run_dir, a.hdf5, a.prefix, a.task, a.demo)
    run(rd, [d.strip() for d in a.detectors.split(",") if d.strip()], a.device, a.roi)
