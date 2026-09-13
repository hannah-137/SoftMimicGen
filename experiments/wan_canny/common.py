"""Shared helpers for the wan_canny video scripts (hdf5 loading, frame sampling, CLI)."""
import argparse
import os

import h5py
import numpy as np

N_FRAMES = 81  # Wan 2.2 clip length
FPS = 16
DEFAULT_OUT_DIR = "experiments/wan_canny/inputs"


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("hdf5", help="generated dataset, e.g. datasets/generated_dataset/<name>.hdf5")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--demo", default="demo_0", help="episode key under data/")
    return parser.parse_args()


def prefix_of(hdf5: str) -> str:
    """Output file prefix: the hdf5 basename without extension."""
    return os.path.splitext(os.path.basename(hdf5))[0]


def load_obs(hdf5: str, demo: str, key: str):
    """Return obs/<key> of one demo as a numpy array, or None if the key is absent."""
    with h5py.File(hdf5, "r") as f:
        obs = f[f"data/{demo}/obs"]
        return obs[key][:] if key in obs else None


def sample_idx(n: int, n_frames: int = N_FRAMES) -> np.ndarray:
    """Uniform frame indices n -> n_frames. Every script uses this so the videos stay frame-aligned."""
    return np.linspace(0, n - 1, n_frames).astype(int)
