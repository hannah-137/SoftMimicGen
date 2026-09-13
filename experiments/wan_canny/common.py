"""Shared helpers for the wan_canny stage scripts: run-dir resolution, hdf5 loading, frame sampling, CLI.

Every stage script takes one pipeline run folder  experiments/wan_canny/runs/<task>_<tag>/  (see tasks.py):
it reads <prefix>.hdf5 there and writes its outputs next to it, prefix = folder name. --hdf5 / --prefix / --task
override the defaults (e.g. to process an hdf5 that lives elsewhere)."""
import argparse
import os
from types import SimpleNamespace

import h5py
import numpy as np

from tasks import TASKS, task_of

N_FRAMES = 81  # Wan 2.2 clip length
FPS = 16


def add_run_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("run_dir", help="pipeline run folder experiments/wan_canny/runs/<task>_<tag>/ (holds <prefix>.hdf5, receives the outputs)")
    parser.add_argument("--hdf5", default=None, help="generated dataset (default <run_dir>/<prefix>.hdf5)")
    parser.add_argument("--prefix", default=None, help="output file prefix (default: run_dir folder name)")
    parser.add_argument("--task", default=None, help="task key in tasks.TASKS (default: derived from the prefix)")
    parser.add_argument("--demo", default="demo_0", help="episode key under data/")
    return parser


def resolve_run(run_dir: str, hdf5: str | None = None, prefix: str | None = None, task: str | None = None,
                demo: str = "demo_0") -> SimpleNamespace:
    """Everything a stage needs: run_dir, prefix, hdf5, task, camera (rgb obs key), edges (edge-channel key prefix),
    roi, demo."""
    run_dir = os.path.normpath(run_dir)
    prefix = prefix or os.path.basename(run_dir)
    task = task or task_of(prefix)
    hdf5 = hdf5 or os.path.join(run_dir, f"{prefix}.hdf5")
    if not os.path.isfile(hdf5):
        raise FileNotFoundError(f"hdf5 not found: {hdf5} (pass --hdf5, or run docker/make_hdf5.sh first)")
    os.makedirs(run_dir, exist_ok=True)
    cfg = TASKS[task]
    return SimpleNamespace(run_dir=run_dir, prefix=prefix, hdf5=hdf5, task=task, camera=cfg["camera"], edges=cfg["edges"],
                           roi=cfg["roi"], demo=demo)


def parse_args(description: str) -> SimpleNamespace:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = add_run_args(parser).parse_args()
    return resolve_run(a.run_dir, a.hdf5, a.prefix, a.task, a.demo)


def out_path(rd: SimpleNamespace, suffix: str) -> str:
    """<run_dir>/<prefix>_<suffix>"""
    return os.path.join(rd.run_dir, f"{rd.prefix}_{suffix}")


def load_obs(hdf5: str, demo: str, key: str):
    """Return obs/<key> of one demo as a numpy array, or None if the key is absent."""
    with h5py.File(hdf5, "r") as f:
        obs = f[f"data/{demo}/obs"]
        return obs[key][:] if key in obs else None


def sample_idx(n: int, n_frames: int = N_FRAMES) -> np.ndarray:
    """Uniform frame indices n -> n_frames. Every script uses this so the videos stay frame-aligned."""
    return np.linspace(0, n - 1, n_frames).astype(int)
