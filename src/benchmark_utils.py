from __future__ import annotations

from pathlib import Path
import json
import random
import numpy as np
import torch


def set_all_seeds(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

def load_cube(path):
    d = np.load(path, allow_pickle=True)
    cube = np.asarray(d["cube"], dtype=np.float32)  # (m, tenor, date)
    m_grid = np.asarray(d["m_grid"], dtype=np.float32)
    tenor_days = np.asarray(d["tenor_days"], dtype=np.float32)
    asof_dates = np.asarray(d["asof_dates"]) if "asof_dates" in d.files else np.arange(cube.shape[-1])
    return cube, m_grid, tenor_days, asof_dates

def save_cube(path, cube, m_grid, tenor_days, asof_dates):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, cube=cube, m_grid=m_grid, tenor_days=tenor_days, asof_dates=asof_dates)

def chronological_split_cube(input_path, train_path, test_path, train_fraction=0.8):
    cube, m_grid, tenor_days, dates = load_cube(input_path)
    n = cube.shape[-1]
    cut = int(np.floor(float(train_fraction) * n))
    cut = min(max(cut, 10), n - 2)
    save_cube(train_path, cube[..., :cut], m_grid, tenor_days, dates[:cut])
    save_cube(test_path, cube[..., cut:], m_grid, tenor_days, dates[cut:])
    return cut

def cube_to_fm_surfaces(cube):
    """(m, tenor, date) -> (date, 1, tenor, m), matching IVS_AE."""
    return np.transpose(cube, (2, 1, 0))[:, None, :, :].astype(np.float32)

def normalizer_fit(surfaces_pct, q_lo=0.04, q_hi=0.96):
    flat = np.asarray(surfaces_pct, np.float32).reshape(-1)
    return float(np.quantile(flat, q_lo)), float(np.quantile(flat, q_hi))

def normalize_pct(x, lo, hi):
    return (2.0 * (np.asarray(x, np.float32) - lo) / max(hi - lo, 1e-8) - 1.0).astype(np.float32)

def denormalize_pct(x, lo, hi):
    return (((np.asarray(x, np.float32) + 1.0) / 2.0) * (hi - lo) + lo).astype(np.float32)

def orient_paths_to_tenor_money(paths, n_tenor=None, n_money=None):
    """Return path array as (P,L,T,M). Existing SB/Cont outputs are usually (P,L,M,T)."""
    x = np.asarray(paths, np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected 4D paths, got {x.shape}")
    if n_tenor is not None and n_money is not None:
        if x.shape[-2:] == (n_tenor, n_money):
            return x
        if x.shape[-2:] == (n_money, n_tenor):
            return np.transpose(x, (0, 1, 3, 2))
    # fallback: tenor is normally the smaller axis for the 16x32 SPX grid
    if x.shape[-2] > x.shape[-1]:
        return np.transpose(x, (0, 1, 3, 2))
    return x

def to_fraction(x):
    x = np.asarray(x, np.float32)
    return x / 100.0 if np.nanmedian(x) > 2.0 else x

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
