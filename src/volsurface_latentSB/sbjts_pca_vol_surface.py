"""
Schrödinger Bridge with Jumps for implied-volatility surface paths — PCA latent version.

Input NPZ format:
    cube        : (n_moneyness, n_tenor, n_dates), IV in percent or fraction
    m_grid      : (n_moneyness,)
    tenor_days  : (n_tenor,)
    asof_dates  : optional, length n_dates

Saved output NPZ:
    paths       : (n_output_paths, path_length, n_moneyness, n_tenor)
    cube        : (n_moneyness, n_tenor, n_output_paths * path_length)
    m_grid, tenor_days, asof_dates, selected_candidate_path_indices, meta

Model summary:
    1. Convert IV surfaces to log-IV in fraction units.
    2. Fit PCA factors exactly in the same spirit as simulations.py: PCA basis is
       fitted on daily log-IV changes; levels are projected onto that basis.
    3. Simulate latent paths with a data-driven jump-diffusion bridge estimator.
       The estimator is a practical discrete-time counterpart of the SBJTS paper:
       local kernel weights estimate the conditional drift and jump propensity;
       large historical increments become jump candidates.
    4. Decode latent paths to surfaces and optionally select low-arbitrage paths
       by the same path-level weighted Monte Carlo idea used in simulations.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json
import math
import numpy as np
import pandas as pd
from scipy.special import ndtr

@dataclass(frozen=True)
class SBJTSPCAConfig:
    input_path: Path = Path("data/treated/vol_surface_cube.npz")
    output_path: Path = Path("data/treated/sbjts_pca_vol_surface.npz")
    #candidate_path_diagnostics_path: Path = Path("data/treated/sbjts_pca_candidate_path_diagnostics.csv")
    #selected_path_diagnostics_path: Path = Path("data/treated/sbjts_pca_selected_path_diagnostics.csv")
    #selected_surface_diagnostics_path: Path = Path("data/treated/sbjts_pca_selected_surface_diagnostics.csv")

    n_output_paths: int = 10
    path_length: int = 64
    n_candidate_paths: Optional[int] = None
    candidate_multiplier: int = 500
    candidate_batch_size: int = 512

    # Latent representation
    n_factors: int = 4
    residual_covariance: str = "diag"  # local covariance regularisation; diag or full

    # SBJTS-style local estimator
    memory_order: int = 2          # k in the paper: number of previous latent states used as context
    bandwidth: float = 1.0         # kernel h after robust context scaling
    min_effective_neighbors: float = 15.0
    ridge: float = 1.0e-8

    # Jump/diffusion decomposition
    jump_quantile: float = 0.90    # top 10% Mahalanobis increments are jump candidates
    lambda0: Optional[float] = None  # if None, inferred from historical jump frequency
    max_jump_prob: float = 0.85
    jump_scale: float = 1.0
    diffusion_scale: float = 1.0
    drift_scale: float = 1.0

    # Starting state
    seed: int = 42
    start_mode: str = "last"  # last, random_history
    start_index: Optional[int] = None

    # Path selection using static-arbitrage penalty
    beta: float = 1.0e4
    require_zero_path: bool = False
    surface_arb_tol: float = 1.0e-10
    path_arb_tol: Optional[float] = None

    # IV surface bounds
    r: float = 0.0
    vol_floor: float = 1.0e-6
    vol_cap: float = 5.0


@dataclass
class SBJTSResult:
    paths: np.ndarray
    cube: np.ndarray
    m_grid: np.ndarray
    tenor_days: np.ndarray
    asof_dates: np.ndarray
    selected_candidate_path_indices: np.ndarray
    candidate_path_diagnostics: pd.DataFrame
    selected_path_diagnostics: pd.DataFrame
    selected_surface_diagnostics: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)


# Input/output and surface utilities
def load_cube_npz(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    cube = np.asarray(data["cube"], dtype=float)
    m_grid = np.asarray(data["m_grid"], dtype=float)
    tenor_days = np.asarray(data["tenor_days"], dtype=float)
    asof_dates = np.asarray(data["asof_dates"]) if "asof_dates" in data.files else np.arange(cube.shape[2]).astype(str)

    meta: Dict[str, Any] = {}
    if "meta" in data.files:
        try:
            raw = data["meta"]
            meta = raw.item() if getattr(raw, "shape", None) == () else {"raw_meta": raw.tolist()}
        except Exception:
            meta = {"raw_meta": str(data["meta"])}

    if cube.ndim != 3:
        raise ValueError(f"Expected cube shape (n_m,n_tenor,n_dates), got {cube.shape}")
    if cube.shape[0] != len(m_grid):
        raise ValueError("cube first dimension must match len(m_grid)")
    if cube.shape[1] != len(tenor_days):
        raise ValueError("cube second dimension must match len(tenor_days)")
    return cube, m_grid, tenor_days, asof_dates, meta


def save_result_npz(path: str | Path, result: SBJTSResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        paths=result.paths,
        cube=result.cube,
        m_grid=result.m_grid,
        tenor_days=result.tenor_days,
        asof_dates=np.asarray(result.asof_dates),
        selected_candidate_path_indices=result.selected_candidate_path_indices,
        meta=np.array(result.meta, dtype=object),
    )


def cube_to_batch_m_t(cube: np.ndarray) -> np.ndarray:
    """(n_m,n_tenor,n_dates) -> (n_dates,n_m,n_tenor)."""
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got {cube.shape}")
    return np.moveaxis(cube, 2, 0)


def batch_m_t_to_cube(batch: np.ndarray) -> np.ndarray:
    """(n_dates,n_m,n_tenor) -> (n_m,n_tenor,n_dates)."""
    batch = np.asarray(batch, dtype=float)
    if batch.ndim != 3:
        raise ValueError(f"Expected 3D batch, got {batch.shape}")
    return np.moveaxis(batch, 0, 2)


def _to_fraction_vol(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=float)
    if np.nanmedian(out) > 2.0:
        out = out / 100.0
    return out


def _from_fraction_vol(x: np.ndarray, output_in_percent: bool) -> np.ndarray:
    return np.asarray(x, dtype=float) * (100.0 if output_in_percent else 1.0)


def _fill_nan_surface_batch(batch: np.ndarray) -> np.ndarray:
    out = np.asarray(batch, dtype=float).copy()
    for k in range(out.shape[0]):
        df = pd.DataFrame(out[k])
        out[k] = (
            df.interpolate(axis=0, limit_direction="both")
            .interpolate(axis=1, limit_direction="both")
            .ffill()
            .bfill()
            .values
        )
    return out



# Static arbitrage diagnostics copied in spirit from simulations.py
def relative_call_from_iv_batch(
    sigma_frac: np.ndarray,
    m_grid: np.ndarray,
    tau_grid: np.ndarray,
    r: float = 0.0,
) -> np.ndarray:
    """Relative Black-Scholes call prices from IV surfaces.

    sigma_frac: (B,n_m,n_tau) in fraction units.
    Returns C with shape (B,n_m,n_tau).
    """
    sigma = np.clip(np.asarray(sigma_frac, dtype=float), 1.0e-12, None)
    if sigma.ndim == 2:
        sigma = sigma[None, :, :]
    if sigma.ndim != 3:
        raise ValueError(f"sigma_frac must be 2D or 3D, got {sigma.shape}")

    m_grid = np.asarray(m_grid, dtype=float)
    tau_grid = np.maximum(np.asarray(tau_grid, dtype=float), 1.0e-12)
    if sigma.shape[1:] != (len(m_grid), len(tau_grid)):
        raise ValueError(
            f"Surface shape {sigma.shape[1:]} incompatible with grids "
            f"({len(m_grid)}, {len(tau_grid)})"
        )

    M, T = np.meshgrid(m_grid, tau_grid, indexing="ij")
    M = M[None, :, :]
    T = T[None, :, :]
    d1 = (-np.log(M) + T * (float(r) + 0.5 * sigma**2)) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return ndtr(d1) - M * np.exp(-float(r) * T) * ndtr(d2)


def arbitrage_penalties_from_calls_batch(
    C: np.ndarray,
    m_grid: np.ndarray,
    tau_grid: np.ndarray,
    eps: float = 1.0e-10,
) -> pd.DataFrame:
    C = np.asarray(C, dtype=float)
    if C.ndim == 2:
        C = C[None, :, :]
    if C.ndim != 3:
        raise ValueError(f"C must be 2D or 3D, got {C.shape}")

    m_grid = np.asarray(m_grid, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    dm = np.diff(m_grid)
    dt = np.diff(tau_grid)
    if np.any(dm <= 0):
        raise ValueError("m_grid must be strictly increasing")
    if np.any(dt <= 0):
        raise ValueError("tau_grid must be strictly increasing")

    calendar = np.maximum(0.0, (C[:, :, :-1] - C[:, :, 1:]) / dt[None, None, :])
    mono = np.maximum(0.0, (C[:, 1:, :] - C[:, :-1, :]) / dm[None, :, None])
    left = (C[:, 1:-1, :] - C[:, :-2, :]) / dm[None, :-1, None]
    right = (C[:, 2:, :] - C[:, 1:-1, :]) / dm[None, 1:, None]
    butterfly = np.maximum(0.0, left - right)

    p_calendar = calendar.sum(axis=(1, 2))
    p_monotonicity = mono.sum(axis=(1, 2))
    p_butterfly = butterfly.sum(axis=(1, 2))
    phi = p_calendar + p_monotonicity + p_butterfly
    return pd.DataFrame(
        {
            "p_calendar": p_calendar,
            "p_monotonicity": p_monotonicity,
            "p_butterfly": p_butterfly,
            "phi": phi,
            "n_calendar_viol": (calendar > float(eps)).sum(axis=(1, 2)).astype(int),
            "n_monotonicity_viol": (mono > float(eps)).sum(axis=(1, 2)).astype(int),
            "n_butterfly_viol": (butterfly > float(eps)).sum(axis=(1, 2)).astype(int),
        }
    )


def arbitrage_penalties_from_iv_batch(
    sigma_frac: np.ndarray,
    m_grid: np.ndarray,
    tau_grid: np.ndarray,
    r: float = 0.0,
    eps: float = 1.0e-10,
) -> pd.DataFrame:
    C = relative_call_from_iv_batch(sigma_frac, m_grid=m_grid, tau_grid=tau_grid, r=r)
    return arbitrage_penalties_from_calls_batch(C, m_grid=m_grid, tau_grid=tau_grid, eps=eps)


def diagnose_paths(
    paths_frac: np.ndarray,
    m_grid: np.ndarray,
    tau_grid: np.ndarray,
    r: float = 0.0,
    surface_eps: float = 1.0e-10,
    chunk_size: int = 8192,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (path_diagnostics, surface_diagnostics).

    paths_frac shape: (n_paths,path_length,n_m,n_tau), IV in fraction units.
    """
    paths_frac = np.asarray(paths_frac, dtype=float)
    if paths_frac.ndim != 4:
        raise ValueError(f"Expected paths shape (n_paths,path_length,n_m,n_tau), got {paths_frac.shape}")
    n_paths, path_length, n_m, n_tau = paths_frac.shape
    flat = paths_frac.reshape(n_paths * path_length, n_m, n_tau)

    pieces = []
    for start in range(0, len(flat), int(chunk_size)):
        stop = min(start + int(chunk_size), len(flat))
        pieces.append(
            arbitrage_penalties_from_iv_batch(
                flat[start:stop], m_grid=m_grid, tau_grid=tau_grid, r=r, eps=surface_eps
            )
        )
    surf_diag = pd.concat(pieces, ignore_index=True)
    surf_diag.insert(0, "surface_global_idx", np.arange(len(surf_diag), dtype=int))
    surf_diag.insert(1, "candidate_path_idx", np.repeat(np.arange(n_paths, dtype=int), path_length))
    surf_diag.insert(2, "step_idx", np.tile(np.arange(path_length, dtype=int), n_paths))

    grouped = surf_diag.groupby("candidate_path_idx", sort=True)
    path_diag = grouped.agg(
        path_phi=("phi", "sum"),
        mean_surface_phi=("phi", "mean"),
        median_surface_phi=("phi", "median"),
        max_surface_phi=("phi", "max"),
        p_calendar_sum=("p_calendar", "sum"),
        p_monotonicity_sum=("p_monotonicity", "sum"),
        p_butterfly_sum=("p_butterfly", "sum"),
        n_calendar_viol=("n_calendar_viol", "sum"),
        n_monotonicity_viol=("n_monotonicity_viol", "sum"),
        n_butterfly_viol=("n_butterfly_viol", "sum"),
    ).reset_index()
    zero_surface = grouped["phi"].apply(lambda x: float(np.mean(np.asarray(x) <= surface_eps)))
    path_diag["zero_surface_pct"] = path_diag["candidate_path_idx"].map(zero_surface) * 100.0
    path_diag["path_length"] = int(path_length)
    return path_diag, surf_diag


def stable_exp_weights(phi: np.ndarray, beta: float) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 1 or len(phi) == 0:
        raise ValueError("phi must be a non-empty one-dimensional array")
    logw = -float(beta) * phi
    logw -= np.nanmax(logw)
    weights = np.exp(logw)
    total = float(np.nansum(weights))
    if not np.isfinite(total) or total <= 0.0:
        out = np.zeros_like(phi, dtype=float)
        out[int(np.nanargmin(phi))] = 1.0
        return out
    return weights / total


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(1.0 / np.sum(weights**2))


def relative_entropy_to_uniform(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    positive = weights > 0
    n = len(weights)
    return float(np.sum(weights[positive] * np.log(weights[positive] * n)))



# PCA latent codec
@dataclass
class PCALogVolCodec:
    n_factors: int = 4
    vol_floor: float = 1.0e-6
    vol_cap: float = 5.0

    mean_: Optional[np.ndarray] = field(default=None, init=False)
    components_: Optional[np.ndarray] = field(default=None, init=False)
    shape_: Optional[Tuple[int, int]] = field(default=None, init=False)
    explained_variance_ratio_: Optional[np.ndarray] = field(default=None, init=False)
    output_in_percent_: bool = field(default=False, init=False)

    def fit_transform(self, cube: np.ndarray) -> np.ndarray:
        """Fit PCA basis on daily log-IV changes and return level scores."""
        batch = cube_to_batch_m_t(cube)
        self.output_in_percent_ = bool(np.nanmedian(batch) > 2.0)
        batch_frac = _to_fraction_vol(batch)
        batch_frac = _fill_nan_surface_batch(batch_frac)
        batch_frac = np.clip(batch_frac, float(self.vol_floor), float(self.vol_cap))

        n_dates, n_m, n_tau = batch_frac.shape
        if n_dates < 3:
            raise ValueError("Need at least 3 historical surfaces to fit PCA factors.")
        self.shape_ = (int(n_m), int(n_tau))

        Y = np.log(batch_frac).reshape(n_dates, n_m * n_tau)
        self.mean_ = Y.mean(axis=0)
        dY = np.diff(Y, axis=0)
        dY_centered = dY - dY.mean(axis=0, keepdims=True)
        _, singular_values, Vt = np.linalg.svd(dY_centered, full_matrices=False)
        k = min(int(self.n_factors), Vt.shape[0])
        self.components_ = Vt[:k]
        total_var = float(np.sum(singular_values**2))
        self.explained_variance_ratio_ = (singular_values[:k] ** 2) / max(total_var, 1.0e-14)
        return (Y - self.mean_[None, :]) @ self.components_.T

    def decode(self, scores: np.ndarray) -> np.ndarray:
        """Decode latent scores to IV surfaces in fraction units."""
        if self.mean_ is None or self.components_ is None or self.shape_ is None:
            raise RuntimeError("Codec is not fitted.")
        scores = np.asarray(scores, dtype=float)
        original_shape = scores.shape[:-1]
        flat_scores = scores.reshape(-1, scores.shape[-1])
        Y = self.mean_[None, :] + flat_scores @ self.components_
        Y = np.clip(Y, np.log(float(self.vol_floor)), np.log(float(self.vol_cap)))
        surfaces = np.exp(Y).reshape(original_shape + self.shape_)
        return np.clip(surfaces, float(self.vol_floor), float(self.vol_cap))

    def meta(self) -> Dict[str, Any]:
        return {
            "latent_codec": "PCA on daily log-IV changes, levels projected on change eigensurfaces",
            "n_factors": int(self.components_.shape[0]) if self.components_ is not None else int(self.n_factors),
            "surface_shape": self.shape_,
            "output_in_percent": bool(self.output_in_percent_),
            "explained_variance_ratio": None
            if self.explained_variance_ratio_ is None
            else self.explained_variance_ratio_.tolist(),
            "cumulative_explained_variance": None
            if self.explained_variance_ratio_ is None
            else np.cumsum(self.explained_variance_ratio_).tolist(),
        }


# SBJTS latent jump-diffusion estimator
def _weighted_mean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.sum(x * w[:, None], axis=0)


def _weighted_cov(x: np.ndarray, w: np.ndarray, ridge: float = 1.0e-8, diag: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(x) <= 1:
        return np.eye(x.shape[1]) * ridge
    mu = _weighted_mean(x, w)
    xc = x - mu[None, :]
    denom = max(1.0 - float(np.sum(w**2)), 1.0e-8)
    cov = (xc * w[:, None]).T @ xc / denom
    cov = 0.5 * (cov + cov.T)
    if diag:
        cov = np.diag(np.maximum(np.diag(cov), ridge))
    else:
        eigval, eigvec = np.linalg.eigh(cov)
        eigval = np.maximum(eigval, ridge)
        cov = (eigvec * eigval[None, :]) @ eigvec.T
    return cov


@dataclass
class SBJTSLatentModel:
    """Practical discrete-time SBJTS estimator in latent space.

    The paper's exact optimal dynamics involves ratios of conditional h-functions.
    Here those functions are approximated by local kernel conditioning on the last k
    latent states.  The drift is the local mean of non-jump increments; the jump
    intensity is a local weighted jump frequency; jump sizes are sampled from the
    empirical jump increments under the same local weights.
    """

    memory_order: int = 2
    bandwidth: float = 1.0
    jump_quantile: float = 0.90
    lambda0: Optional[float] = None
    max_jump_prob: float = 0.85
    diffusion_scale: float = 1.0
    jump_scale: float = 1.0
    drift_scale: float = 1.0
    min_effective_neighbors: float = 15.0
    ridge: float = 1.0e-8
    residual_covariance: str = "diag"

    latent_: Optional[np.ndarray] = field(default=None, init=False)
    contexts_: Optional[np.ndarray] = field(default=None, init=False)
    increments_: Optional[np.ndarray] = field(default=None, init=False)
    jump_mask_: Optional[np.ndarray] = field(default=None, init=False)
    context_scale_: Optional[np.ndarray] = field(default=None, init=False)
    increment_scale_: Optional[np.ndarray] = field(default=None, init=False)
    global_cont_cov_: Optional[np.ndarray] = field(default=None, init=False)
    global_cont_mean_: Optional[np.ndarray] = field(default=None, init=False)
    empirical_jump_rate_: float = field(default=0.0, init=False)

    def fit(self, latent: np.ndarray) -> "SBJTSLatentModel":
        latent = np.asarray(latent, dtype=float)
        if latent.ndim != 2:
            raise ValueError(f"latent must have shape (n_dates,d), got {latent.shape}")
        n_dates, d = latent.shape
        k = int(self.memory_order)
        if k < 1:
            raise ValueError("memory_order must be >= 1")
        if n_dates <= k + 1:
            raise ValueError("Not enough dates for requested memory_order")
        if not (0.0 < float(self.jump_quantile) < 1.0):
            raise ValueError("jump_quantile must be in (0,1)")
        if self.residual_covariance not in {"diag", "full"}:
            raise ValueError("residual_covariance must be 'diag' or 'full'")

        contexts = []
        increments = []
        for i in range(k - 1, n_dates - 1):
            contexts.append(latent[i - k + 1 : i + 1].reshape(-1))
            increments.append(latent[i + 1] - latent[i])
        contexts = np.asarray(contexts, dtype=float)
        increments = np.asarray(increments, dtype=float)

        inc_scale = np.nanstd(increments, axis=0)
        inc_scale = np.where(inc_scale < 1.0e-10, 1.0, inc_scale)
        maha_diag = np.sum((increments / inc_scale[None, :]) ** 2, axis=1)
        threshold = float(np.nanquantile(maha_diag, float(self.jump_quantile)))
        jump_mask = maha_diag >= threshold
        # guarantee both classes exist
        if np.all(jump_mask):
            jump_mask[np.argmin(maha_diag)] = False
        if not np.any(jump_mask):
            jump_mask[np.argmax(maha_diag)] = True

        ctx_scale = np.nanstd(contexts, axis=0)
        ctx_scale = np.where(ctx_scale < 1.0e-10, 1.0, ctx_scale)

        cont_increments = increments[~jump_mask]
        self.global_cont_mean_ = cont_increments.mean(axis=0)
        self.global_cont_cov_ = np.cov(cont_increments, rowvar=False) if len(cont_increments) > 1 else np.eye(d) * self.ridge
        self.global_cont_cov_ = np.atleast_2d(self.global_cont_cov_)
        if self.residual_covariance == "diag":
            self.global_cont_cov_ = np.diag(np.maximum(np.diag(self.global_cont_cov_), self.ridge))
        else:
            eigval, eigvec = np.linalg.eigh(0.5 * (self.global_cont_cov_ + self.global_cont_cov_.T))
            eigval = np.maximum(eigval, self.ridge)
            self.global_cont_cov_ = (eigvec * eigval[None, :]) @ eigvec.T

        self.latent_ = latent
        self.contexts_ = contexts
        self.increments_ = increments
        self.jump_mask_ = jump_mask
        self.context_scale_ = ctx_scale
        self.increment_scale_ = inc_scale
        self.empirical_jump_rate_ = float(np.mean(jump_mask))
        return self

    @property
    def latent_dim(self) -> int:
        if self.latent_ is None:
            raise RuntimeError("Model is not fitted.")
        return int(self.latent_.shape[1])

    def _kernel_weights(self, context: np.ndarray) -> np.ndarray:
        if self.contexts_ is None or self.context_scale_ is None:
            raise RuntimeError("Model is not fitted.")
        context = np.asarray(context, dtype=float).reshape(-1)
        diff = (self.contexts_ - context[None, :]) / self.context_scale_[None, :]
        d2 = np.sum(diff**2, axis=1)
        h = max(float(self.bandwidth), 1.0e-8)
        logw = -0.5 * d2 / (h * h)
        logw -= np.max(logw)
        w = np.exp(logw)
        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0.0:
            w = np.ones(len(d2), dtype=float) / len(d2)
        else:
            w /= total

        ess = 1.0 / max(float(np.sum(w**2)), 1.0e-12)
        if ess < float(self.min_effective_neighbors):
            # Blend with a small uniform component to avoid path collapse in sparse regions.
            alpha = min(0.75, (float(self.min_effective_neighbors) - ess) / float(self.min_effective_neighbors))
            w = (1.0 - alpha) * w + alpha / len(w)
            w /= np.sum(w)
        return w

    def local_parameters(self, history: np.ndarray) -> Dict[str, Any]:
        if self.increments_ is None or self.jump_mask_ is None or self.global_cont_cov_ is None:
            raise RuntimeError("Model is not fitted.")
        k = int(self.memory_order)
        hist = np.asarray(history, dtype=float)
        if hist.shape[0] < k:
            raise ValueError(f"Need at least memory_order={k} states in history")
        context = hist[-k:].reshape(-1)
        w = self._kernel_weights(context)

        jump_w_mass = float(np.sum(w[self.jump_mask_]))
        lambda_base = self.empirical_jump_rate_ if self.lambda0 is None else float(self.lambda0)
        # Ratio-like local intensity: baseline lambda0 multiplied by local/global jump propensity.
        local_ratio = jump_w_mass / max(self.empirical_jump_rate_, 1.0e-8)
        p_jump = float(np.clip(lambda_base * local_ratio, 0.0, float(self.max_jump_prob)))

        cont_idx = ~self.jump_mask_
        w_cont = w[cont_idx]
        if float(np.sum(w_cont)) <= 1.0e-12:
            w_cont = np.ones(np.sum(cont_idx), dtype=float) / np.sum(cont_idx)
        else:
            w_cont = w_cont / np.sum(w_cont)
        cont_increments = self.increments_[cont_idx]
        drift = _weighted_mean(cont_increments, w_cont) * float(self.drift_scale)
        cov = _weighted_cov(
            cont_increments,
            w_cont,
            ridge=float(self.ridge),
            diag=(self.residual_covariance == "diag"),
        )
        cov *= float(self.diffusion_scale) ** 2

        jump_idx = self.jump_mask_
        w_jump = w[jump_idx]
        if np.sum(jump_idx) == 0:
            jump_increments = np.zeros((1, self.latent_dim), dtype=float)
            w_jump = np.ones(1, dtype=float)
        else:
            jump_increments = self.increments_[jump_idx]
            if float(np.sum(w_jump)) <= 1.0e-12:
                w_jump = np.ones(len(jump_increments), dtype=float) / len(jump_increments)
            else:
                w_jump = w_jump / np.sum(w_jump)
        return {
            "drift": drift,
            "cov": cov,
            "p_jump": p_jump,
            "jump_increments": jump_increments,
            "jump_weights": w_jump,
            "kernel_ess": float(1.0 / np.sum(w**2)),
            "local_jump_mass": jump_w_mass,
        }

    def initial_history(self, rng: np.random.Generator, n_paths: int, mode: str = "last", start_index: Optional[int] = None) -> np.ndarray:
        if self.latent_ is None:
            raise RuntimeError("Model is not fitted.")
        latent = self.latent_
        k = int(self.memory_order)
        n_dates = latent.shape[0]
        if start_index is not None:
            idx = int(start_index)
            if idx < 0:
                idx = n_dates + idx
            idx = int(np.clip(idx, k - 1, n_dates - 1))
            hist = latent[idx - k + 1 : idx + 1]
            return np.repeat(hist[None, :, :], int(n_paths), axis=0)
        mode = str(mode).lower()
        if mode == "last":
            hist = latent[-k:]
            return np.repeat(hist[None, :, :], int(n_paths), axis=0)
        if mode == "random_history":
            end_idxs = rng.integers(k - 1, n_dates, size=int(n_paths))
            return np.stack([latent[j - k + 1 : j + 1] for j in end_idxs], axis=0)
        raise ValueError("start_mode must be one of: last, random_history")

    def simulate_latent_paths(
        self,
        n_paths: int,
        path_length: int,
        seed: int,
        start_mode: str = "last",
        start_index: Optional[int] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        rng = np.random.default_rng(int(seed))
        n_paths = int(n_paths)
        path_length = int(path_length)
        if n_paths <= 0 or path_length <= 0:
            raise ValueError("n_paths and path_length must be positive")
        d = self.latent_dim
        histories = self.initial_history(rng, n_paths=n_paths, mode=start_mode, start_index=start_index)
        paths = np.empty((n_paths, path_length, d), dtype=float)
        rows = []
        for p in range(n_paths):
            hist = histories[p].copy()
            for t in range(path_length):
                params = self.local_parameters(hist)
                drift = params["drift"]
                cov = params["cov"]
                try:
                    diffusion = rng.multivariate_normal(np.zeros(d), cov)
                except np.linalg.LinAlgError:
                    diffusion = rng.multivariate_normal(np.zeros(d), np.diag(np.maximum(np.diag(cov), self.ridge)))

                did_jump = rng.random() < float(params["p_jump"])
                if did_jump:
                    j = rng.choice(len(params["jump_increments"]), p=params["jump_weights"])
                    jump = np.asarray(params["jump_increments"][j], dtype=float) * float(self.jump_scale)
                else:
                    jump = np.zeros(d, dtype=float)
                prev_x = hist[-1].copy()
                new_x = prev_x + drift + diffusion + jump
                paths[p, t] = new_x
                hist = np.vstack([hist[1:], new_x[None, :]])
                rows.append(
                    {
                        "candidate_path_idx": p,
                        "step_idx": t,
                        "p_jump": float(params["p_jump"]),
                        "did_jump": int(did_jump),
                        "kernel_ess": float(params["kernel_ess"]),
                        "local_jump_mass": float(params["local_jump_mass"]),
                        "latent_step_norm": float(np.linalg.norm(new_x - prev_x)),
                    }
                )
        return paths, pd.DataFrame(rows)

    def meta(self) -> Dict[str, Any]:
        return {
            "latent_dynamics": "SBJTS-style local kernel jump-diffusion estimator",
            "memory_order": int(self.memory_order),
            "bandwidth": float(self.bandwidth),
            "jump_quantile": float(self.jump_quantile),
            "lambda0": None if self.lambda0 is None else float(self.lambda0),
            "lambda0_effective": float(self.empirical_jump_rate_ if self.lambda0 is None else self.lambda0),
            "empirical_jump_rate": float(self.empirical_jump_rate_),
            "max_jump_prob": float(self.max_jump_prob),
            "diffusion_scale": float(self.diffusion_scale),
            "jump_scale": float(self.jump_scale),
            "drift_scale": float(self.drift_scale),
            "residual_covariance": str(self.residual_covariance),
        }


# Candidate streaming and WMC selection

def _total_candidate_paths(cfg: SBJTSPCAConfig) -> int:
    return int(cfg.n_candidate_paths) if cfg.n_candidate_paths is not None else max(
        int(cfg.n_output_paths) * int(cfg.candidate_multiplier), 1000
    )


def _candidate_batch_ranges(n_total: int, batch_size: int):
    batch_size = max(1, int(batch_size))
    for start in range(0, int(n_total), batch_size):
        stop = min(start + batch_size, int(n_total))
        yield start, stop


def _seed_for_candidate_batch(base_seed: int, batch_start: int) -> int:
    return int((int(base_seed) + 1000003 * int(batch_start) + 9176) % (2**32 - 1))


def _config_for_candidate_batch(cfg: SBJTSPCAConfig, batch_start: int, batch_n: int) -> SBJTSPCAConfig:
    return replace(cfg, n_candidate_paths=int(batch_n), seed=_seed_for_candidate_batch(int(cfg.seed), int(batch_start)))


def select_path_indices_wmc(candidate_path_diag: pd.DataFrame, cfg: SBJTSPCAConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(int(cfg.seed) + 271828)
    phi = candidate_path_diag["path_phi"].to_numpy(dtype=float)
    path_tol = float(cfg.path_arb_tol) if cfg.path_arb_tol is not None else float(cfg.surface_arb_tol) * int(cfg.path_length)
    if cfg.require_zero_path:
        eligible = np.where(phi <= path_tol)[0]
        if len(eligible) == 0:
            best = float(np.nanmin(phi))
            raise RuntimeError(
                "No zero-penalty SBJTS path found. "
                f"Best path_phi={best:.12g}, path_arb_tol={path_tol:.12g}. "
                "Increase candidates, relax tolerance, or use finite beta."
            )
        weights = np.zeros_like(phi, dtype=float)
        weights[eligible] = 1.0 / len(eligible)
        selection_mode = "beta_infinity_uniform_over_zero_penalty_paths"
    else:
        weights = stable_exp_weights(phi, beta=float(cfg.beta))
        selection_mode = "finite_beta_path_weighted_monte_carlo"
    selected_idx = rng.choice(len(phi), size=int(cfg.n_output_paths), replace=True, p=weights).astype(int)
    meta = {
        "selection_mode": selection_mode,
        "beta": float(cfg.beta),
        "require_zero_path": bool(cfg.require_zero_path),
        "path_arb_tol": path_tol,
        "n_candidate_paths": int(len(phi)),
        "candidate_batch_size": int(cfg.candidate_batch_size),
        "n_zero_penalty_paths": int((phi <= path_tol).sum()),
        "candidate_path_phi_min": float(np.nanmin(phi)),
        "candidate_path_phi_median": float(np.nanmedian(phi)),
        "candidate_path_phi_mean": float(np.nanmean(phi)),
        "candidate_path_phi_max": float(np.nanmax(phi)),
        "wmc_effective_sample_size": effective_sample_size(weights),
        "wmc_effective_sample_size_pct": 100.0 * effective_sample_size(weights) / len(weights),
        "wmc_relative_entropy_to_uniform": relative_entropy_to_uniform(weights),
    }
    return selected_idx, weights, meta


def _simulate_candidate_batch(
    latent_model: SBJTSLatentModel,
    codec: PCALogVolCodec,
    cfg: SBJTSPCAConfig,
) -> Tuple[np.ndarray, pd.DataFrame]:
    latent_paths, latent_diag = latent_model.simulate_latent_paths(
        n_paths=int(cfg.n_candidate_paths),
        path_length=int(cfg.path_length),
        seed=int(cfg.seed),
        start_mode=str(cfg.start_mode),
        start_index=cfg.start_index,
    )
    paths_frac = codec.decode(latent_paths)
    paths_frac = np.clip(paths_frac, float(cfg.vol_floor), float(cfg.vol_cap))
    return paths_frac, latent_diag


def diagnose_candidate_paths_streaming(
    latent_model: SBJTSLatentModel,
    codec: PCALogVolCodec,
    cfg: SBJTSPCAConfig,
    m_grid: np.ndarray,
    tau_grid: np.ndarray,
) -> pd.DataFrame:
    n_total = _total_candidate_paths(cfg)
    pieces = []
    for start, stop in _candidate_batch_ranges(n_total, int(cfg.candidate_batch_size)):
        batch_cfg = _config_for_candidate_batch(cfg, start, stop - start)
        batch_paths_frac, latent_diag = _simulate_candidate_batch(latent_model, codec, batch_cfg)
        batch_path_diag, _ = diagnose_paths(
            batch_paths_frac,
            m_grid=m_grid,
            tau_grid=tau_grid,
            r=float(cfg.r),
            surface_eps=float(cfg.surface_arb_tol),
        )
        batch_path_diag = batch_path_diag.copy()
        latent_summary = latent_diag.groupby("candidate_path_idx").agg(
            n_latent_jumps=("did_jump", "sum"),
            mean_p_jump=("p_jump", "mean"),
            mean_kernel_ess=("kernel_ess", "mean"),
            mean_latent_step_norm=("latent_step_norm", "mean"),
            max_latent_step_norm=("latent_step_norm", "max"),
        ).reset_index()
        batch_path_diag = batch_path_diag.merge(latent_summary, on="candidate_path_idx", how="left")
        batch_path_diag["candidate_path_idx"] = batch_path_diag["candidate_path_idx"].astype(int) + int(start)
        batch_path_diag["candidate_batch_start"] = int(start)
        batch_path_diag["candidate_batch_stop"] = int(stop)
        pieces.append(batch_path_diag)
        del batch_paths_frac
    if not pieces:
        raise RuntimeError("No candidate paths were generated.")
    out = pd.concat(pieces, ignore_index=True).sort_values("candidate_path_idx").reset_index(drop=True)
    expected = np.arange(n_total, dtype=int)
    got = out["candidate_path_idx"].to_numpy(dtype=int)
    if len(got) != n_total or not np.array_equal(got, expected):
        raise RuntimeError("Internal error: candidate diagnostics are not contiguous/sorted.")
    return out


def regenerate_selected_paths(
    latent_model: SBJTSLatentModel,
    codec: PCALogVolCodec,
    cfg: SBJTSPCAConfig,
    selected_candidate_idx: np.ndarray,
) -> np.ndarray:
    selected_candidate_idx = np.asarray(selected_candidate_idx, dtype=int)
    n_total = _total_candidate_paths(cfg)
    positions_by_candidate: Dict[int, list[int]] = {}
    for out_pos, candidate_id in enumerate(selected_candidate_idx.tolist()):
        if candidate_id < 0 or candidate_id >= n_total:
            raise ValueError(f"selected candidate id out of range: {candidate_id}")
        positions_by_candidate.setdefault(int(candidate_id), []).append(int(out_pos))
    selected_paths_frac: Optional[np.ndarray] = None
    needed = set(positions_by_candidate)
    for start, stop in _candidate_batch_ranges(n_total, int(cfg.candidate_batch_size)):
        ids_in_batch = [cid for cid in needed if start <= cid < stop]
        if not ids_in_batch:
            continue
        batch_cfg = _config_for_candidate_batch(cfg, start, stop - start)
        batch_paths_frac, _ = _simulate_candidate_batch(latent_model, codec, batch_cfg)
        if selected_paths_frac is None:
            selected_paths_frac = np.empty((len(selected_candidate_idx),) + batch_paths_frac.shape[1:], dtype=batch_paths_frac.dtype)
        for cid in ids_in_batch:
            local = int(cid - start)
            for out_pos in positions_by_candidate[cid]:
                selected_paths_frac[out_pos] = batch_paths_frac[local]
        del batch_paths_frac
    if selected_paths_frac is None:
        raise RuntimeError("No selected paths could be regenerated.")
    return selected_paths_frac


# -----------------------------------------------------------------------------
# Top-level training/simulation function
# -----------------------------------------------------------------------------


def simulate_sbjts_pca_paths(config: SBJTSPCAConfig) -> SBJTSResult:
    cube, m_grid, tenor_days, asof_dates_input, input_meta = load_cube_npz(config.input_path)
    tau_grid = np.asarray(tenor_days, dtype=float) / 365.0
    output_in_percent = bool(np.nanmedian(cube) > 2.0)

    codec = PCALogVolCodec(n_factors=int(config.n_factors), vol_floor=float(config.vol_floor), vol_cap=float(config.vol_cap))
    latent = codec.fit_transform(cube)
    latent_model = SBJTSLatentModel(
        memory_order=int(config.memory_order),
        bandwidth=float(config.bandwidth),
        jump_quantile=float(config.jump_quantile),
        lambda0=config.lambda0,
        max_jump_prob=float(config.max_jump_prob),
        diffusion_scale=float(config.diffusion_scale),
        jump_scale=float(config.jump_scale),
        drift_scale=float(config.drift_scale),
        min_effective_neighbors=float(config.min_effective_neighbors),
        ridge=float(config.ridge),
        residual_covariance=str(config.residual_covariance),
    ).fit(latent)

    candidate_path_diag = diagnose_candidate_paths_streaming(
        latent_model, codec, config, m_grid=m_grid, tau_grid=tau_grid
    )
    selected_candidate_idx, weights, wmc_meta = select_path_indices_wmc(candidate_path_diag, config)
    selected_paths_frac = regenerate_selected_paths(latent_model, codec, config, selected_candidate_idx)

    selected_path_diag, selected_surface_diag = diagnose_paths(
        selected_paths_frac,
        m_grid=m_grid,
        tau_grid=tau_grid,
        r=float(config.r),
        surface_eps=float(config.surface_arb_tol),
    )
    selected_path_diag.insert(0, "output_path_idx", np.arange(len(selected_path_diag), dtype=int))
    selected_path_diag.insert(1, "selected_candidate_path_idx", selected_candidate_idx)
    selected_path_diag["candidate_weight"] = weights[selected_candidate_idx]

    selected_surface_diag = selected_surface_diag.rename(columns={"candidate_path_idx": "output_path_idx"})
    selected_surface_diag.insert(2, "selected_candidate_path_idx", np.repeat(selected_candidate_idx, int(config.path_length)))

    candidate_path_diag = candidate_path_diag.copy()
    candidate_path_diag["wmc_weight"] = weights
    candidate_path_diag["rank_by_path_phi"] = candidate_path_diag["path_phi"].rank(method="first")

    selected_paths = _from_fraction_vol(selected_paths_frac, output_in_percent=output_in_percent)
    flat_selected = selected_paths.reshape(
        int(config.n_output_paths) * int(config.path_length), selected_paths.shape[2], selected_paths.shape[3]
    )
    out_cube = batch_m_t_to_cube(flat_selected)
    out_dates = np.array([f"sbjts_pca_path_{p:04d}_t_{t:04d}" for p in range(int(config.n_output_paths)) for t in range(int(config.path_length))])

    meta: Dict[str, Any] = {
        "model": "SBJTS PCA latent jump-diffusion bridge for IV surfaces",
        "input_path": str(config.input_path),
        "input_meta": input_meta,
        "n_historical_dates": int(cube.shape[2]),
        "path_length": int(config.path_length),
        "n_output_paths": int(config.n_output_paths),
        "output_in_percent": bool(output_in_percent),
        **codec.meta(),
        **latent_model.meta(),
        **wmc_meta,
    }

    return SBJTSResult(
        paths=selected_paths,
        cube=out_cube,
        m_grid=m_grid,
        tenor_days=tenor_days,
        asof_dates=out_dates,
        selected_candidate_path_indices=selected_candidate_idx,
        candidate_path_diagnostics=candidate_path_diag,
        selected_path_diagnostics=selected_path_diag,
        selected_surface_diagnostics=selected_surface_diag,
        meta=meta,
    )



def main() -> None:
    import argparse

    p = argparse.ArgumentParser("sbjts_pca_vol_surface: Schrödinger bridge with jumps in PCA latent space")
    p.add_argument("--input", default="data/treated/vol_surface_cube.npz")
    p.add_argument("--out", default="data/treated/sbjts_pca_vol_surface.npz")
    p.add_argument("--candidate-path-diagnostics", default="data/treated/sbjts_pca_candidate_path_diagnostics.csv")
    p.add_argument("--selected-path-diagnostics", default="data/treated/sbjts_pca_selected_path_diagnostics.csv")
    p.add_argument("--selected-surface-diagnostics", default="data/treated/sbjts_pca_selected_surface_diagnostics.csv")

    p.add_argument("--n-output-paths", type=int, default=10)
    p.add_argument("--path-length", type=int, default=64)
    p.add_argument("--n-candidate-paths", type=int, default=None)
    p.add_argument("--candidate-multiplier", type=int, default=500)
    p.add_argument("--candidate-batch-size", type=int, default=512)

    p.add_argument("--n-factors", type=int, default=4)
    p.add_argument("--residual-covariance", choices=["diag", "full"], default="diag")
    p.add_argument("--memory-order", type=int, default=2)
    p.add_argument("--bandwidth", type=float, default=1.0)
    p.add_argument("--min-effective-neighbors", type=float, default=15.0)
    p.add_argument("--jump-quantile", type=float, default=0.90)
    p.add_argument("--lambda0", type=float, default=None)
    p.add_argument("--max-jump-prob", type=float, default=0.85)
    p.add_argument("--jump-scale", type=float, default=1.0)
    p.add_argument("--diffusion-scale", type=float, default=1.0)
    p.add_argument("--drift-scale", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-mode", choices=["last", "random_history"], default="last")
    p.add_argument("--start-index", type=int, default=None)

    p.add_argument("--beta", type=float, default=1.0e4)
    p.add_argument("--require-zero-path", action="store_true")
    p.add_argument("--surface-arb-tol", type=float, default=1.0e-10)
    p.add_argument("--path-arb-tol", type=float, default=None)
    p.add_argument("--r", type=float, default=0.0)
    p.add_argument("--vol-floor", type=float, default=1.0e-6)
    p.add_argument("--vol-cap", type=float, default=5.0)

    args = p.parse_args()
    cfg = SBJTSPCAConfig(
        input_path=Path(args.input),
        output_path=Path(args.out),
        #candidate_path_diagnostics_path=Path(args.candidate_path_diagnostics),
        #selected_path_diagnostics_path=Path(args.selected_path_diagnostics),
        #selected_surface_diagnostics_path=Path(args.selected_surface_diagnostics),
        n_output_paths=args.n_output_paths,
        path_length=args.path_length,
        n_candidate_paths=args.n_candidate_paths,
        candidate_multiplier=args.candidate_multiplier,
        candidate_batch_size=args.candidate_batch_size,
        n_factors=args.n_factors,
        residual_covariance=args.residual_covariance,
        memory_order=args.memory_order,
        bandwidth=args.bandwidth,
        min_effective_neighbors=args.min_effective_neighbors,
        jump_quantile=args.jump_quantile,
        lambda0=args.lambda0,
        max_jump_prob=args.max_jump_prob,
        jump_scale=args.jump_scale,
        diffusion_scale=args.diffusion_scale,
        drift_scale=args.drift_scale,
        seed=args.seed,
        start_mode=args.start_mode,
        start_index=args.start_index,
        beta=args.beta,
        require_zero_path=args.require_zero_path,
        surface_arb_tol=args.surface_arb_tol,
        path_arb_tol=args.path_arb_tol,
        r=args.r,
        vol_floor=args.vol_floor,
        vol_cap=args.vol_cap,
    )
    result = simulate_sbjts_pca_paths(cfg)
    save_result_npz(cfg.output_path, result)
    #cfg.candidate_path_diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    #result.candidate_path_diagnostics.to_csv(cfg.candidate_path_diagnostics_path, index=False)
    #result.selected_path_diagnostics.to_csv(cfg.selected_path_diagnostics_path, index=False)
    #result.selected_surface_diagnostics.to_csv(cfg.selected_surface_diagnostics_path, index=False)

    print(f"Saved SBJTS PCA output paths/cube to {cfg.output_path}")
    #print(f"Saved candidate path diagnostics to {cfg.candidate_path_diagnostics_path}")
    #print(f"Saved selected path diagnostics to {cfg.selected_path_diagnostics_path}")
    #print(f"Saved selected surface diagnostics to {cfg.selected_surface_diagnostics_path}")
    print(f"paths shape: {result.paths.shape}")
    print(f"cube shape: {result.cube.shape}")
    print(json.dumps(result.meta, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
