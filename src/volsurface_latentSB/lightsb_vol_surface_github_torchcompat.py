"""
Light Schrödinger Bridge for implied-volatility surface paths.

This version uses a diagonal Torch LightSB core adapted from "LIGHT SCHRODINGER BRIDGE" A.Korotin, N.Gushchin, and E. Burnaev's GitHub implementation.

Pipeline:
1. Encode IV surfaces with the existing PCA log-IV codec.
2. Fit a diagonal Gaussian-mixture LightSB conditional transition in standardized
   latent space, using the paper objective E[log C_theta(X0)] - E[log v_theta(X1)].
3. Generate candidate latent paths from
       v_theta(y) = sum_k alpha_k N(y | r_k, epsilon S_k)
       p_theta(y|x) = sum_k w_k(x) N(y | r_k + S_k x, epsilon S_k)
   with diagonal S_k > 0.
4. Decode paths back to IV surfaces and reuse the existing static-arbitrage WMC
   selector from the SBJTS PCA implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json
import math
import time

import numpy as np
import pandas as pd

from volsurface_latentSB.sbjts_pca_vol_surface import (
    SBJTSResult,
    PCALogVolCodec,
    load_cube_npz,
    save_result_npz,
    batch_m_t_to_cube,
    _from_fraction_vol,
    diagnose_paths,
    stable_exp_weights,
    effective_sample_size,
    relative_entropy_to_uniform,
)


@dataclass(frozen=True)
class LightSBVolConfig:
    input_path: Path = Path("data/treated/vol_surface_cube.npz")
    output_path: Path = Path("data/treated/lightsb_official_vol_surface.npz")

    n_output_paths: int = 10
    path_length: int = 64
    n_candidate_paths: Optional[int] = None
    candidate_multiplier: int = 500
    candidate_batch_size: int = 512

    # Surface codec. PCA is deterministic and less prone than a decoder network
    # to creating off-manifold smiles/term structures.
    n_factors: int = 8

    # LightSB transition model.
    n_components: int = 64
    epsilon: float = 0.12
    epochs: int = 1200
    batch_size: int = 256
    lr: float = 2.0e-3
    weight_decay: float = 1.0e-4
    grad_clip: float = 10.0
    paired_nll_weight: float = 0.0
    cov_reg_weight: float = 0.0
    s_min: float = 0.015
    s_max: float = 1.50

    device: str = "cpu"

    # Sampling controls. These are the main quality knobs.
    temperature: float = 1.0        # official conditional sampling uses no sharpening
    innovation_scale: float = 1.0   # official conditional sampling uses full sqrt(epsilon*S) noise
    manifold_shrink: float = 0.0    # disabled for faithful LightSB objective/sampling
    manifold_k: int = 12
    manifold_bandwidth: float = 1.0

    # Starting state.
    seed: int = 42
    start_mode: str = "last"  # last, random_history
    start_index: Optional[int] = None

    # WMC path selection using static-arbitrage penalties.
    beta: float = 1.0e4
    require_zero_path: bool = False
    surface_arb_tol: float = 1.0e-10
    path_arb_tol: Optional[float] = None

    # IV surface bounds.
    r: float = 0.0
    vol_floor: float = 1.0e-6
    vol_cap: float = 5.0


def _as_float64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _logsumexp(a: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + 1.0e-300)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def _softmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.exp(a - _logsumexp(a, axis=axis, keepdims=True))


def _init_kmeans_plus_plus(y: np.ndarray, k: int, rng: np.random.Generator, n_iter: int = 8) -> np.ndarray:
    """Tiny dependency-free k-means++ initializer for component locations."""
    y = _as_float64(y)
    n, d = y.shape
    k = int(max(1, min(k, max(n, 1))))
    centers = np.empty((k, d), dtype=np.float64)
    first = int(rng.integers(0, n))
    centers[0] = y[first]
    d2 = np.sum((y - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        probs = d2 / max(float(d2.sum()), 1.0e-300)
        idx = int(rng.choice(n, p=probs))
        centers[j] = y[idx]
        d2 = np.minimum(d2, np.sum((y - centers[j]) ** 2, axis=1))

    labels = np.zeros(n, dtype=int)
    for _ in range(int(n_iter)):
        dist2 = np.sum((y[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dist2, axis=1)
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = y[mask].mean(axis=0)
            else:
                centers[j] = y[int(rng.integers(0, n))]
    return centers

try:
    import torch
    from torch import nn
except Exception as _torch_exc:  # pragma: no cover
    torch = None
    nn = object
    _TORCH_IMPORT_ERROR = _torch_exc
else:
    _TORCH_IMPORT_ERROR = None


def _torch_device(name: str):
    if torch is None:
        raise RuntimeError(f"Torch could not be imported: {_TORCH_IMPORT_ERROR}")
    if str(name) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(name))


def _torch_from_np_safe(x: np.ndarray, device, dtype=None):
    """Convert NumPy/list to Torch without using Torch's NumPy C bridge."""
    if torch is None:
        raise RuntimeError(f"Torch could not be imported: {_TORCH_IMPORT_ERROR}")
    if dtype is None:
        dtype = torch.float32
    arr = np.asarray(x, dtype=np.float32)
    return torch.tensor(arr.tolist(), dtype=dtype, device=device)


def _np_from_torch_safe(x) -> np.ndarray:
    """Convert Torch tensor to NumPy without calling tensor.numpy()."""
    return np.asarray(x.detach().cpu().tolist(), dtype=np.float64)


class GitHubDiagonalLightSB(nn.Module):
    """Diagonal LightSB class following the official paper's formulae and GitHub implementation.
    """

    def __init__(
        self,
        dim: int = 2,
        n_potentials: int = 5,
        epsilon: float = 1.0,
        sampling_batch_size: int = 4096,
        S_diagonal_init: float = 0.1,
    ):
        super().__init__()
        self.is_diagonal = True
        self.dim = int(dim)
        self.n_potentials = int(n_potentials)
        self.register_buffer("epsilon", torch.tensor(float(epsilon), dtype=torch.float32))
        self.sampling_batch_size = int(sampling_batch_size)
        self.log_alpha_raw = nn.Parameter(
            self.epsilon * torch.log(torch.ones(self.n_potentials, dtype=torch.float32) / self.n_potentials)
        )
        self.r = nn.Parameter(0.05 * torch.randn(self.n_potentials, self.dim, dtype=torch.float32))
        self.S_log_diagonal_matrix = nn.Parameter(
            torch.log(float(S_diagonal_init) * torch.ones(self.n_potentials, self.dim, dtype=torch.float32))
        )

    @torch.no_grad()
    def init_r_by_samples(self, samples):
        if samples.shape[0] != self.r.shape[0]:
            raise ValueError(f"Expected {self.r.shape[0]} centers, got {samples.shape[0]}")
        self.r.data = samples.to(self.r.device).clone()

    def get_S(self):
        return torch.exp(self.S_log_diagonal_matrix)

    def get_r(self):
        return self.r

    def get_log_alpha(self):
        # (1/epsilon) * log_alpha_raw.
        return (1.0 / self.epsilon) * self.log_alpha_raw

    def _component_logits(self, x):
        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        log_alpha = self.get_log_alpha()
        x_S_x = (x[:, None, :] * S[None, :, :] * x[:, None, :]).sum(dim=-1)
        x_r = (x[:, None, :] * r[None, :, :]).sum(dim=-1)
        return (x_S_x + 2.0 * x_r) / (2.0 * epsilon) + log_alpha[None, :]

    def get_log_C(self, x):
        return torch.logsumexp(self._component_logits(x), dim=-1)

    def get_log_potential(self, y):
        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        log_alpha = self.get_log_alpha()
        diff = y[:, None, :] - r[None, :, :]
        logdet = torch.sum(torch.log(epsilon * S), dim=-1)[None, :]
        maha = torch.sum(diff.pow(2) / (epsilon * S[None, :, :]), dim=-1)
        log_norm = -0.5 * (self.dim * math.log(2.0 * math.pi) + logdet + maha)
        return torch.logsumexp(log_alpha[None, :] + log_norm, dim=-1)

    def get_log_conditional(self, x, y):
        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        A = self._component_logits(x)
        mean = r[None, :, :] + S[None, :, :] * x[:, None, :]
        diff = y[:, None, :] - mean
        logdet = torch.sum(torch.log(epsilon * S), dim=-1)[None, :]
        maha = torch.sum(diff.pow(2) / (epsilon * S[None, :, :]), dim=-1)
        log_norm = -0.5 * (self.dim * math.log(2.0 * math.pi) + logdet + maha)
        return torch.logsumexp(A + log_norm, dim=-1) - torch.logsumexp(A, dim=-1)

    @torch.no_grad()
    def forward(self, x, temperature: float = 1.0, innovation_scale: float = 1.0):
        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        logits = self._component_logits(x) / max(float(temperature), 1.0e-4)
        probs = torch.softmax(logits, dim=-1)
        comp = torch.multinomial(probs, num_samples=1).squeeze(-1)
        S_c = S[comp]
        r_c = r[comp]
        mean = r_c + S_c * x
        return mean + float(innovation_scale) * torch.sqrt(epsilon * S_c) * torch.randn_like(mean)


@dataclass
class LightSBLatentModel:
    n_components: int = 64
    epsilon: float = 0.12
    epochs: int = 1200
    batch_size: int = 256
    lr: float = 2.0e-3
    weight_decay: float = 1.0e-4
    grad_clip: float = 10.0
    paired_nll_weight: float = 0.0
    cov_reg_weight: float = 0.0
    s_min: float = 0.015
    s_max: float = 1.50
    device: str = "auto"
    seed: int = 42

    model_: Optional[GitHubDiagonalLightSB] = field(default=None, init=False)
    latent_: Optional[np.ndarray] = field(default=None, init=False)
    latent_mean_: Optional[np.ndarray] = field(default=None, init=False)
    latent_scale_: Optional[np.ndarray] = field(default=None, init=False)
    reference_std_: Optional[np.ndarray] = field(default=None, init=False)
    train_history_: Dict[str, Any] = field(default_factory=dict, init=False)

    def fit(self, latent: np.ndarray) -> "LightSBLatentModel":
        if torch is None:
            raise RuntimeError(f"Torch could not be imported: {_TORCH_IMPORT_ERROR}")
        latent = np.asarray(latent, dtype=np.float64)
        if latent.ndim != 2:
            raise ValueError(f"latent must have shape (n_dates,d), got {latent.shape}")
        if latent.shape[0] < 4:
            raise ValueError("Need at least 4 latent observations for one-step LightSB training.")

        torch.manual_seed(int(self.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.seed))
        device = _torch_device(self.device)

        self.latent_ = latent.copy()
        self.latent_mean_ = latent.mean(axis=0).astype(np.float64)
        scale = latent.std(axis=0).astype(np.float64)
        scale = np.where(scale < 1.0e-6, 1.0, scale).astype(np.float64)
        self.latent_scale_ = scale

        z = (latent - self.latent_mean_[None, :]) / self.latent_scale_[None, :]
        x0 = z[:-1].astype(np.float64)
        x1 = z[1:].astype(np.float64)
        self.reference_std_ = z.astype(np.float64)

        rng = np.random.default_rng(int(self.seed))
        k = min(int(self.n_components), max(1, len(x1)))
        centers = _init_kmeans_plus_plus(x1, k, rng=rng, n_iter=10)
        if centers.shape[0] < k:
            extra = centers[rng.choice(centers.shape[0], k - centers.shape[0], replace=True)]
            centers = np.vstack([centers, extra])

        slopes = []
        for j in range(x0.shape[1]):
            denom = float(np.dot(x0[:, j], x0[:, j])) + 1.0e-8
            slopes.append(float(np.dot(x0[:, j], x1[:, j]) / denom))
        slope0 = float(np.nanmedian(np.abs(slopes))) if slopes else 0.35
        s0 = float(np.clip(slope0, float(self.s_min) * 1.5, float(self.s_max) * 0.65))

        model = GitHubDiagonalLightSB(
            dim=x0.shape[1],
            n_potentials=k,
            epsilon=float(self.epsilon),
            sampling_batch_size=max(1, int(self.batch_size)),
            S_diagonal_init=s0,
        ).to(device)
        model.init_r_by_samples(_torch_from_np_safe(centers[:k], device=device))

        x0_t = _torch_from_np_safe(x0, device=device)
        x1_t = _torch_from_np_safe(x1, device=device)
        opt = torch.optim.Adam(model.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))

        best_loss = float("inf")
        best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        hist = []
        t0 = time.time()
        n = int(x0.shape[0])
        batch_size = min(max(1, int(self.batch_size)), n)

        for epoch in range(int(self.epochs)):
            order = rng.permutation(n).tolist()
            sums = {"loss": 0.0, "lightsb": 0.0, "nll": 0.0, "cov_reg": 0.0}
            seen = 0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                idx_t = torch.tensor(idx, dtype=torch.long, device=device)
                xb = x0_t.index_select(0, idx_t)
                yb = x1_t.index_select(0, idx_t)

                opt.zero_grad(set_to_none=True)
                lightsb = model.get_log_C(xb).mean() - model.get_log_potential(yb).mean()
                nll = -model.get_log_conditional(xb, yb).mean()
                cov_reg = model.S_log_diagonal_matrix.pow(2).mean()
                loss = lightsb + float(self.paired_nll_weight) * nll + float(self.cov_reg_weight) * cov_reg
                loss.backward()
                if float(self.grad_clip) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(self.grad_clip))
                opt.step()
                with torch.no_grad():
                    model.S_log_diagonal_matrix.clamp_(min=math.log(float(self.s_min)), max=math.log(float(self.s_max)))

                b = len(idx)
                sums["loss"] += float(loss.detach().cpu().item()) * b
                sums["lightsb"] += float(lightsb.detach().cpu().item()) * b
                sums["nll"] += float(nll.detach().cpu().item()) * b
                sums["cov_reg"] += float(cov_reg.detach().cpu().item()) * b
                seen += b

            row = {key: value / max(seen, 1) for key, value in sums.items()}
            row["epoch"] = int(epoch + 1)
            hist.append(row)
            if np.isfinite(row["loss"]) and row["loss"] < best_loss:
                best_loss = float(row["loss"])
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

        model.load_state_dict(best_state)
        self.model_ = model.eval()
        self.train_history_ = {
            "best_loss": float(best_loss),
            "last_loss": float(hist[-1]["loss"]),
            "last_lightsb": float(hist[-1]["lightsb"]),
            "last_paired_nll": float(hist[-1]["nll"]),
            "last_cov_reg": float(hist[-1]["cov_reg"]),
            "epochs": int(self.epochs),
            "fit_elapsed_sec": float(time.time() - t0),
            "optimizer": "torch_adam_official_light_sb_style_no_numpy_bridge",
        }
        return self

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if self.latent_mean_ is None or self.latent_scale_ is None:
            raise RuntimeError("LightSBLatentModel is not fitted.")
        return (np.asarray(x, dtype=np.float64) - self.latent_mean_[None, :]) / self.latent_scale_[None, :]

    def _unstandardize(self, x: np.ndarray) -> np.ndarray:
        if self.latent_mean_ is None or self.latent_scale_ is None:
            raise RuntimeError("LightSBLatentModel is not fitted.")
        return np.asarray(x, dtype=np.float64) * self.latent_scale_[None, None, :] + self.latent_mean_[None, None, :]

    def _start_latent(self, cfg: LightSBVolConfig, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        if self.latent_ is None:
            raise RuntimeError("LightSBLatentModel is not fitted.")
        z = self.latent_
        if cfg.start_index is not None:
            idx = int(np.clip(int(cfg.start_index), 0, len(z) - 1))
            start = np.repeat(z[idx : idx + 1], int(n_paths), axis=0)
        elif str(cfg.start_mode) == "random_history":
            idx = rng.integers(0, len(z), size=int(n_paths))
            start = z[idx]
        else:
            start = np.repeat(z[-1:], int(n_paths), axis=0)
        return self._standardize(start)

    @staticmethod
    def _knn_barycenter(x: np.ndarray, ref: np.ndarray, k: int, bandwidth: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        k = min(max(int(k), 1), len(ref))
        d2 = np.sum((x[:, None, :] - ref[None, :, :]) ** 2, axis=2)
        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        local_d2 = np.take_along_axis(d2, idx, axis=1)
        bw2 = max(float(bandwidth) ** 2, 1.0e-8)
        w = np.exp(-0.5 * local_d2 / bw2)
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1.0e-12)
        return np.sum(ref[idx] * w[:, :, None], axis=1).astype(np.float64)

    def simulate_latent_paths(self, cfg: LightSBVolConfig, n_paths: int, seed: int) -> np.ndarray:
        if self.model_ is None or self.reference_std_ is None:
            raise RuntimeError("LightSBLatentModel is not fitted.")
        rng = np.random.default_rng(int(seed))
        device = next(self.model_.parameters()).device
        current_np = self._start_latent(cfg, n_paths=int(n_paths), rng=rng).astype(np.float64)
        current = _torch_from_np_safe(current_np, device=device)
        out_std = np.zeros((int(n_paths), int(cfg.path_length), current_np.shape[1]), dtype=np.float64)

        for t in range(int(cfg.path_length)):
            with torch.no_grad():
                current = self.model_(
                    current,
                    temperature=float(cfg.temperature),
                    innovation_scale=float(cfg.innovation_scale),
                )
            current_np = _np_from_torch_safe(current)
            if float(cfg.manifold_shrink) > 0.0:
                bary = self._knn_barycenter(
                    current_np,
                    self.reference_std_,
                    k=int(cfg.manifold_k),
                    bandwidth=float(cfg.manifold_bandwidth),
                )
                rho = float(np.clip(float(cfg.manifold_shrink), 0.0, 1.0))
                current_np = ((1.0 - rho) * current_np + rho * bary).astype(np.float64)
                current = _torch_from_np_safe(current_np, device=device)
            out_std[:, t, :] = current_np

        return self._unstandardize(out_std).astype(np.float32)

    def meta(self) -> Dict[str, Any]:
        return {
            "latent_dynamics": "GitHub-style LightSB diagonal Gaussian-mixture transition; objective mean(log C_theta(X0))-mean(log v_theta(X1))",
            "implementation": "torch_compat_github_diagonal_no_numpy_bridge_no_geotorch",
            "training_objective": "mean_x0 log C_theta(x0) - mean_x1 log v_theta(x1)",
            "strict_lightsb_defaults": True,
            "n_components": int(self.model_.n_potentials if self.model_ is not None else self.n_components),
            "epsilon": float(self.epsilon),
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "paired_nll_weight": float(self.paired_nll_weight),
            "cov_reg_weight": float(self.cov_reg_weight),
            "s_min": float(self.s_min),
            "s_max": float(self.s_max),
            **self.train_history_,
        }

def _total_candidate_paths(cfg: LightSBVolConfig) -> int:
    if cfg.n_candidate_paths is not None:
        return int(cfg.n_candidate_paths)
    return max(int(cfg.n_output_paths) * int(cfg.candidate_multiplier), 1000)


def _candidate_batch_ranges(n_total: int, batch_size: int):
    batch_size = max(1, int(batch_size))
    for start in range(0, int(n_total), batch_size):
        yield start, min(start + batch_size, int(n_total))


def _seed_for_candidate_batch(base_seed: int, batch_start: int) -> int:
    return int((int(base_seed) + 1000003 * int(batch_start) + 9176) % (2**32 - 1))


def _diagnose_candidate_paths_streaming(
    latent_model: LightSBLatentModel,
    codec: PCALogVolCodec,
    cfg: LightSBVolConfig,
    m_grid: np.ndarray,
    tau_grid: np.ndarray,
) -> pd.DataFrame:
    n_total = _total_candidate_paths(cfg)
    pieces = []
    for start, stop in _candidate_batch_ranges(n_total, int(cfg.candidate_batch_size)):
        seed = _seed_for_candidate_batch(int(cfg.seed), int(start))
        latent_paths = latent_model.simulate_latent_paths(cfg, n_paths=stop - start, seed=seed)
        paths_frac = codec.decode(latent_paths)
        path_diag, _ = diagnose_paths(
            paths_frac,
            m_grid=m_grid,
            tau_grid=tau_grid,
            r=float(cfg.r),
            surface_eps=float(cfg.surface_arb_tol),
        )
        path_diag["candidate_path_idx"] = path_diag["candidate_path_idx"].astype(int) + int(start)
        pieces.append(path_diag)
    return pd.concat(pieces, ignore_index=True)


def _select_path_indices(candidate_path_diag: pd.DataFrame, cfg: LightSBVolConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(int(cfg.seed) + 271828)
    phi = candidate_path_diag["path_phi"].to_numpy(dtype=float)
    path_tol = float(cfg.path_arb_tol) if cfg.path_arb_tol is not None else float(cfg.surface_arb_tol) * int(cfg.path_length)
    if bool(cfg.require_zero_path):
        eligible = np.where(phi <= path_tol)[0]
        if len(eligible) == 0:
            best = float(np.nanmin(phi))
            raise RuntimeError(
                "No zero-penalty LightSB path found. "
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
        "wmc_effective_sample_size_pct": 100.0 * effective_sample_size(weights) / max(len(weights), 1),
        "wmc_relative_entropy_to_uniform": relative_entropy_to_uniform(weights),
    }
    return selected_idx, weights, meta


def _regenerate_selected_paths(
    latent_model: LightSBLatentModel,
    codec: PCALogVolCodec,
    cfg: LightSBVolConfig,
    selected_candidate_idx: np.ndarray,
) -> np.ndarray:
    n_total = _total_candidate_paths(cfg)
    selected_candidate_idx = np.asarray(selected_candidate_idx, dtype=int)
    need = {int(idx): [] for idx in np.unique(selected_candidate_idx)}
    for pos, idx in enumerate(selected_candidate_idx):
        need[int(idx)].append(int(pos))

    output = None
    for start, stop in _candidate_batch_ranges(n_total, int(cfg.candidate_batch_size)):
        wanted = [idx for idx in need.keys() if start <= idx < stop]
        if not wanted:
            continue
        seed = _seed_for_candidate_batch(int(cfg.seed), int(start))
        latent_paths = latent_model.simulate_latent_paths(cfg, n_paths=stop - start, seed=seed)
        paths_frac = codec.decode(latent_paths)
        if output is None:
            output = np.zeros((len(selected_candidate_idx),) + paths_frac.shape[1:], dtype=paths_frac.dtype)
        for idx in wanted:
            local = int(idx) - int(start)
            for pos in need[idx]:
                output[pos] = paths_frac[local]
    if output is None:
        raise RuntimeError("Failed to regenerate selected LightSB paths.")
    return output


def simulate_lightsb_paths(config: LightSBVolConfig) -> SBJTSResult:
    cube, m_grid, tenor_days, asof_dates_input, input_meta = load_cube_npz(config.input_path)
    tau_grid = np.asarray(tenor_days, dtype=float) / 365.0
    output_in_percent = bool(np.nanmedian(cube) > 2.0)

    codec = PCALogVolCodec(
        n_factors=int(config.n_factors),
        vol_floor=float(config.vol_floor),
        vol_cap=float(config.vol_cap),
    )
    latent = codec.fit_transform(cube)

    latent_model = LightSBLatentModel(
        n_components=int(config.n_components),
        epsilon=float(config.epsilon),
        epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
        grad_clip=float(config.grad_clip),
        paired_nll_weight=float(config.paired_nll_weight),
        cov_reg_weight=float(config.cov_reg_weight),
        s_min=float(config.s_min),
        s_max=float(config.s_max),
        device=str(config.device),
        seed=int(config.seed),
    ).fit(latent)

    candidate_path_diag = _diagnose_candidate_paths_streaming(
        latent_model, codec, config, m_grid=m_grid, tau_grid=tau_grid
    )
    selected_candidate_idx, weights, wmc_meta = _select_path_indices(candidate_path_diag, config)
    selected_paths_frac = _regenerate_selected_paths(latent_model, codec, config, selected_candidate_idx)

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
    out_dates = np.array(
        [
            f"lightsb_path_{p:04d}_t_{t:04d}"
            for p in range(int(config.n_output_paths))
            for t in range(int(config.path_length))
        ]
    )

    meta: Dict[str, Any] = {
        "model": "LightSB PCA latent Gaussian-mixture transition for IV surfaces",
        "input_path": str(config.input_path),
        "input_meta": input_meta,
        "n_historical_dates": int(cube.shape[2]),
        "path_length": int(config.path_length),
        "n_output_paths": int(config.n_output_paths),
        "output_in_percent": bool(output_in_percent),
        "sampling_temperature": float(config.temperature),
        "innovation_scale": float(config.innovation_scale),
        "manifold_shrink": float(config.manifold_shrink),
        "manifold_k": int(config.manifold_k),
        "manifold_bandwidth": float(config.manifold_bandwidth),
        **codec.meta(),
        **latent_model.meta(),
        **wmc_meta,
    }

    return SBJTSResult(
        paths=selected_paths,
        cube=out_cube,
        m_grid=np.asarray(m_grid),
        tenor_days=np.asarray(tenor_days),
        asof_dates=out_dates,
        selected_candidate_path_indices=selected_candidate_idx,
        candidate_path_diagnostics=candidate_path_diag,
        selected_path_diagnostics=selected_path_diag,
        selected_surface_diagnostics=selected_surface_diag,
        meta=meta,
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser("lightsb_vol_surface_github_torchcompat: GitHub-style Torch LightSB in PCA latent space for IV surfaces")
    p.add_argument("--input", default="data/treated/vol_surface_cube.npz")
    p.add_argument("--out", default="data/treated/lightsb_vol_surface.npz")

    p.add_argument("--n-output-paths", type=int, default=10)
    p.add_argument("--path-length", type=int, default=64)
    p.add_argument("--n-candidate-paths", type=int, default=None)
    p.add_argument("--candidate-multiplier", type=int, default=500)
    p.add_argument("--candidate-batch-size", type=int, default=512)

    p.add_argument("--n-factors", type=int, default=8)
    p.add_argument("--n-components", type=int, default=64)
    p.add_argument("--epsilon", type=float, default=0.12)
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--paired-nll-weight", type=float, default=0.0, help="Extra time-pair NLL; keep 0 for strict LightSB objective.")
    p.add_argument("--cov-reg-weight", type=float, default=0.0)
    p.add_argument("--s-min", type=float, default=0.015)
    p.add_argument("--s-max", type=float, default=1.50)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu", help="Ignored in pure NumPy version; kept for compatibility.")

    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--innovation-scale", type=float, default=1.0)
    p.add_argument("--manifold-shrink", type=float, default=0.0)
    p.add_argument("--manifold-k", type=int, default=12)
    p.add_argument("--manifold-bandwidth", type=float, default=1.0)

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
    cfg = LightSBVolConfig(
        input_path=Path(args.input),
        output_path=Path(args.out),
        n_output_paths=args.n_output_paths,
        path_length=args.path_length,
        n_candidate_paths=args.n_candidate_paths,
        candidate_multiplier=args.candidate_multiplier,
        candidate_batch_size=args.candidate_batch_size,
        n_factors=args.n_factors,
        n_components=args.n_components,
        epsilon=args.epsilon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        paired_nll_weight=args.paired_nll_weight,
        cov_reg_weight=args.cov_reg_weight,
        s_min=args.s_min,
        s_max=args.s_max,
        device=args.device,
        temperature=args.temperature,
        innovation_scale=args.innovation_scale,
        manifold_shrink=args.manifold_shrink,
        manifold_k=args.manifold_k,
        manifold_bandwidth=args.manifold_bandwidth,
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
    result = simulate_lightsb_paths(cfg)
    save_result_npz(cfg.output_path, result)
    print(f"Saved LightSB output paths/cube to {cfg.output_path}")
    print(f"paths shape: {result.paths.shape}")
    print(f"cube shape: {result.cube.shape}")
    print(json.dumps(result.meta, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
