"""Layout checks for generated reference images and Wan videos against the simulator (2026-09-14).

A generated image/video must keep the simulator's layout: the robot and the manipulated object where the control video
has them. Background, table and lighting may change freely, so only the moving region is compared:

  motion mask  = pixels that change anywhere in the episode's rgb frames (robot + object), dilated
  sim edges    = object boundaries from obs/<edges>_instance (simulator instance-id segmentation): a pixel whose right or
                 lower neighbour belongs to another object. This is ground truth from the simulator and is not any of
                 the control channels, so no control type is favoured (2026-09-15; before, shadedcanny was used, which
                 favoured the shadedcanny videos). The channel is recorded only with --inspect (gen.sh/make_hdf5.sh).
  score        = recall: share of sim edges inside the region that lie within TOL px of a Canny edge of the generated frame

Reference images use frame 0; kind "empty" (object removed) checks the static region instead (camera, table outline,
robot base must not move). Videos are scored on every frame; the video score is the lowest frame score.

  python experiments/wan_canny/scripts/quality.py <run_dir> <image_or_video> [...] [--kind var|empty] [--json]
"""
import argparse
import json
import os

import cv2
import imageio
import numpy as np

from common import load_obs, resolve_run, sample_idx

TOL = 3            # px tolerance between a sim edge and a generated edge
MOTION_THR = 25    # max-channel abs difference to frame 0 that counts as motion
MIN_EDGE_PX = 80   # fewer sim edge pixels than this in the region: no score (NaN)
# Thresholds for instance boundaries (2026-09-15, franka_towel v2/v3 re-scored): layout-matching references 0.93-0.97,
# real-photo references with another camera view (cause of the v2 double towel) 0.35-0.52; empty reference 0.65 vs 0.00.
# Video scores overlap (v3 good 0.77-0.96, v2 double towel 0.73-0.93): recall does not penalise extra lines, so the video
# check only catches a robot or object in the wrong place; the reference check is what stops a mismatched layout.
REF_MIN = 0.80        # reference image (faithful or varied) passes when score >= REF_MIN
REF_EMPTY_MIN = 0.40  # empty reference (object removed, static region scored) passes when score >= REF_EMPTY_MIN
VIDEO_MIN = 0.60      # video passes when its lowest frame score >= VIDEO_MIN

_cache = {}
_lock = __import__("threading").Lock()


def sim_data(run_dir: str):
    """(rgb frames, instance boundary frames, motion mask) for demo_0 of a run dir, 81 sampled frames; cached, thread-safe."""
    run_dir = os.path.normpath(run_dir)
    with _lock:
        if run_dir not in _cache:
            _cache[run_dir] = _load_sim(run_dir)
        return _cache[run_dir]


def _load_sim(run_dir: str):
    rd = resolve_run(run_dir)
    rgb = load_obs(rd.hdf5, rd.demo, rd.camera)
    idx = sample_idx(len(rgb))
    rgb = rgb[idx]
    ins = load_obs(rd.hdf5, rd.demo, f"{rd.edges}_instance")
    if ins is None:
        raise KeyError(f"{rd.edges}_instance not in {rd.hdf5}: make the hdf5 with --inspect to record it")
    sc = instance_boundary(ins[idx])
    diff = np.abs(rgb.astype(np.int16) - rgb[:1].astype(np.int16)).max(axis=-1).max(axis=0) > MOTION_THR
    diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    motion = cv2.dilate(diff, np.ones((15, 15), np.uint8)) > 0
    return rgb, sc, motion


def instance_boundary(ins: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) colourised instance ids -> (T, H, W) bool, True where the right or lower neighbour is another object."""
    key = ins[..., 0].astype(np.int32) << 16 | ins[..., 1].astype(np.int32) << 8 | ins[..., 2].astype(np.int32)
    b = np.zeros(key.shape, bool)
    b[:, :-1] |= key[:, :-1] != key[:, 1:]
    b[:, :, :-1] |= key[:, :, :-1] != key[:, :, 1:]
    return b


def edges_of(img_rgb: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    return cv2.Canny(g, 40, 120) > 0


def recall(sim_edge: np.ndarray, gen_rgb: np.ndarray, region: np.ndarray) -> float:
    target = sim_edge & region
    n = int(target.sum())
    if n < MIN_EDGE_PX:
        return float("nan")
    near = cv2.dilate(edges_of(gen_rgb).astype(np.uint8), np.ones((2 * TOL + 1, 2 * TOL + 1), np.uint8)) > 0
    return float((target & near).sum()) / n


def check_ref(run_dir: str, image_path: str, kind: str = "var") -> dict:
    """kind 'var' (faithful or varied reference): robot + object region; 'empty': static region."""
    _, sc, motion = sim_data(run_dir)
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    if img.shape[:2] != sc.shape[1:]:
        img = cv2.resize(img, (sc.shape[2], sc.shape[1]), interpolation=cv2.INTER_AREA)
    region = ~motion if kind == "empty" else motion
    s = recall(sc[0], img, region)
    thr = REF_EMPTY_MIN if kind == "empty" else REF_MIN
    return dict(kind=kind, score=round(s, 3), threshold=thr, passed=bool(s >= thr))


def extra_edges(sim_edge: np.ndarray, gen_rgb: np.ndarray, region: np.ndarray) -> tuple[float, float]:
    """(precision, ghost ratio): share of generated edges in the region that lie on a sim edge, and generated edge pixels in
    a 12 px band around the sim outline but off it, per sim edge pixel (a doubled outline raises it). Texture also raises
    both (waffle towels, wood grain), so they are logged, not used as a gate (2026-09-14)."""
    e = edges_of(gen_rgb)
    near = cv2.dilate(sim_edge.astype(np.uint8), np.ones((2 * TOL + 1, 2 * TOL + 1), np.uint8)) > 0
    band = (cv2.dilate(sim_edge.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0) & region & ~near
    g = e & region
    return float((g & near).sum()) / max(int(g.sum()), 1), float((e & band).sum()) / max(int((sim_edge & region).sum()), 1)


def check_video(run_dir: str, video_path: str) -> dict:
    _, sc, motion = sim_data(run_dir)
    frames = [f for f in imageio.get_reader(video_path)]
    n = min(len(frames), len(sc))
    scores = np.array([recall(sc[t], frames[t], motion) for t in range(n)])
    ex = np.array([extra_edges(sc[t], frames[t], motion) for t in range(n)])
    valid = scores[~np.isnan(scores)]
    low = float(valid.min()) if len(valid) else float("nan")
    worst = int(np.nanargmin(scores)) if len(valid) else -1
    return dict(score=round(low, 3), worst_frame=worst, mean=round(float(valid.mean()), 3) if len(valid) else None,
                threshold=VIDEO_MIN, passed=bool(low >= VIDEO_MIN), frames=n,
                precision_mean=round(float(ex[:, 0].mean()), 3), ghost_mean=round(float(ex[:, 1].mean()), 3),
                ghost_max=round(float(ex[:, 1].max()), 3))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("files", nargs="+")
    p.add_argument("--kind", default="var", choices=["var", "empty"])
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    for f in a.files:
        r = check_video(a.run_dir, f) if f.lower().endswith(".mp4") else check_ref(a.run_dir, f, a.kind)
        print(json.dumps({"file": os.path.basename(f), **r}) if a.json else
              f"{'PASS' if r['passed'] else 'FAIL'}  {r['score']:.3f}  {os.path.basename(f)}"
              + (f"  (worst frame {r['worst_frame']}, mean {r['mean']})" if "worst_frame" in r else ""))
