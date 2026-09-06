"""
Schrödinger Bridge with Jumps for implied-volatility surface paths — beta-VAE latent version.
This is the second SBJTS model. It uses the same jump-diffusion bridge estimator as sbjts_pca_vol_surface.py, but replaces PCA factors with an arbitrage-aware
beta-VAE latent representation. SBJTS is simulated on posterior-mean latent time series, then decoded back to IV surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json
import math
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split



def _torch_tensor_from_np(x: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    arr = np.asarray(x, dtype=np.float32)
    t = torch.tensor(arr.tolist(), dtype=torch.float32)
    return t.to(device) if device is not None else t


def _np_from_torch_tensor(x: torch.Tensor, dtype=np.float32) -> np.ndarray:
    return np.asarray(x.detach().cpu().tolist(), dtype=dtype)


def _torch_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _torch_relative_call_from_iv(
    sigma_frac: torch.Tensor,
    m_grid: torch.Tensor,
    tau_grid: torch.Tensor,
    r: float = 0.0,
) -> torch.Tensor:
    sigma = torch.clamp(sigma_frac, min=1.0e-8, max=5.0)
    M, T = torch.meshgrid(m_grid, tau_grid, indexing="ij")
    M = M.to(sigma.device)[None, :, :]
    T = torch.clamp(T.to(sigma.device), min=1.0e-8)[None, :, :]
    d1 = (-torch.log(M) + T * (float(r) + 0.5 * sigma.pow(2))) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)
    return _torch_normal_cdf(d1) - M * torch.exp(-float(r) * T) * _torch_normal_cdf(d2)


def _torch_three_noarb_penalty_from_decoded_log_norm(
    x_hat_norm: torch.Tensor,
    mean_log: torch.Tensor,
    std_log: torch.Tensor,
    shape: Tuple[int, int],
    m_grid: torch.Tensor,
    tau_grid: torch.Tensor,
    r: float = 0.0,
    vol_floor: float = 1.0e-6,
    vol_cap: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All three static-arbitrage losses from decoded normalized log-IV.

    The model decoder outputs normalized log-IV. We map it back to IV in
    fraction units, compute relative call prices, then penalize:

    1. calendar:     C(m,tau_j) <= C(m,tau_{j+1})
    2. monotonicity: C(m_{i+1},tau) <= C(m_i,tau)
    3. butterfly:    increasing finite-difference slopes in moneyness

    Returns mean-squared positive violations (calendar, monotonicity, butterfly).
    """
    n_m, n_tau = int(shape[0]), int(shape[1])
    log_sigma = x_hat_norm * std_log[None, :] + mean_log[None, :]
    log_sigma = torch.clamp(log_sigma, min=math.log(float(vol_floor)), max=math.log(float(vol_cap)))
    sigma = torch.exp(log_sigma).reshape(-1, n_m, n_tau)

    C = _torch_relative_call_from_iv(sigma, m_grid=m_grid, tau_grid=tau_grid, r=r)

    dm = torch.diff(m_grid).to(C.device)
    dt = torch.diff(tau_grid).to(C.device)

    calendar = F.relu((C[:, :, :-1] - C[:, :, 1:]) / dt[None, None, :])
    monotonicity = F.relu((C[:, 1:, :] - C[:, :-1, :]) / dm[None, :, None])

    left = (C[:, 1:-1, :] - C[:, :-2, :]) / dm[None, :-1, None]
    right = (C[:, 2:, :] - C[:, 1:-1, :]) / dm[None, 1:, None]
    butterfly = F.relu(left - right)

    return calendar.pow(2).mean(), monotonicity.pow(2).mean(), butterfly.pow(2).mean()


from volsurface_latentSB.sbjts_pca_vol_surface import (
    SBJTSResult,
    SBJTSLatentModel,
    load_cube_npz,
    save_result_npz,
    cube_to_batch_m_t,
    batch_m_t_to_cube,
    _to_fraction_vol,
    _from_fraction_vol,
    _fill_nan_surface_batch,
    diagnose_paths,
    stable_exp_weights,
    effective_sample_size,
    relative_entropy_to_uniform,
)


@dataclass(frozen=True)
class SBJTSAEConfig:
    input_path: Path = Path("data/treated/vol_surface_cube.npz")
    output_path: Path = Path("data/treated/sbjts_autoencoder_vol_surface.npz")
    #candidate_path_diagnostics_path: Path = Path("data/treated/sbjts_ae_candidate_path_diagnostics.csv")
    #selected_path_diagnostics_path: Path = Path("data/treated/sbjts_ae_selected_path_diagnostics.csv")
    #selected_surface_diagnostics_path: Path = Path("data/treated/sbjts_ae_selected_surface_diagnostics.csv")
    autoencoder_checkpoint_path: Path = Path("data/treated/sbjts_surface_autoencoder.pt")

    n_output_paths: int = 10
    path_length: int = 64
    n_candidate_paths: Optional[int] = None
    candidate_multiplier: int = 500
    candidate_batch_size: int = 512

    # Beta-VAE latent representation
    latent_dim: int = 8
    hidden_dim_1: int = 256
    hidden_dim_2: int = 96
    ae_epochs: int = 300
    ae_batch_size: int = 64
    ae_lr: float = 1.0e-3
    ae_weight_decay: float = 1.0e-5
    # VAE regularisation. SBJTS uses posterior means as latent time series,
    # while the KL term makes generated latents safer for the decoder.
    ae_beta_kl: float = 1.0e-3
    ae_kl_warmup_epochs: int = 100

    # Differentiable no-arbitrage penalties inside the autoencoder loss.
    # These are computed from decoded IV -> relative Black-Scholes calls and
    # include all three static-arbitrage conditions:
    #   calendar      : C(m,tau) non-decreasing in tau
    #   monotonicity  : C(m,tau) non-increasing in moneyness
    #   butterfly     : C(m,tau) convex in moneyness
    ae_lambda_calendar: float = 10.0
    ae_lambda_monotonicity: float = 10.0
    ae_lambda_butterfly: float = 10.0

    ae_val_fraction: float = 0.15
    ae_patience: int = 40
    load_autoencoder_if_exists: bool = False
    save_autoencoder: bool = True
    device: str = "auto"  # auto, cpu, cuda

    # SBJTS-style local estimator
    memory_order: int = 2
    bandwidth: float = 1.0
    min_effective_neighbors: float = 15.0
    ridge: float = 1.0e-8
    residual_covariance: str = "diag"

    # Jump/diffusion decomposition
    jump_quantile: float = 0.90
    lambda0: Optional[float] = None
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

# Beta VAE
class MLPSurfaceBetaVAE(nn.Module):
    """Dense beta-VAE for flattened normalized log-IV surfaces.

    This replaces the previous deterministic autoencoder.  SBJTS uses the
    posterior mean mu(x_t) as the historical latent time series, while the KL
    term regularizes the latent cloud so generated SBJTS states are less likely
    to fall into decoder extrapolation regions.
    """

    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim_1: int = 256, hidden_dim_2: int = 96):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder_body = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim_1)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim_1)),
            nn.Linear(int(hidden_dim_1), int(hidden_dim_2)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim_2)),
        )
        self.mu = nn.Linear(int(hidden_dim_2), self.latent_dim)
        self.logvar = nn.Linear(int(hidden_dim_2), self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, int(hidden_dim_2)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim_2)),
            nn.Linear(int(hidden_dim_2), int(hidden_dim_1)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim_1)),
            nn.Linear(int(hidden_dim_1), self.input_dim),
        )

    def encode_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_body(x)
        return self.mu(h), self.logvar(h).clamp(min=-12.0, max=8.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def encode(self, x: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        mu, logvar = self.encode_stats(x)
        return mu if bool(use_mean) else self.reparameterize(mu, logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_stats(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

MLPSurfaceAutoencoder=MLPSurfaceBetaVAE #for older pipelines


@dataclass
class AutoencoderLogVolCodec:
    """SBJTS surface codec using an arbitrage-aware beta-VAE in log-IV space.

    The public name is kept for compatibility with the existing SBJTS-AE file,
    but the implementation is now variational:
      * chronological train/validation split,
      * KL warmup,
      * exact relative-call no-arbitrage penalties,
      * reconstruction diagnostics,
      * latent OOD diagnostics for generated paths,
      * decoder clipping diagnostics.
    """

    latent_dim: int = 8
    hidden_dim_1: int = 256
    hidden_dim_2: int = 96
    vol_floor: float = 1.0e-6
    vol_cap: float = 5.0
    seed: int = 42
    device: str = "auto"
    beta_kl: float = 1.0e-3
    kl_warmup_epochs: int = 100

    shape_: Optional[Tuple[int, int]] = field(default=None, init=False)
    output_in_percent_: bool = field(default=False, init=False)
    mean_: Optional[np.ndarray] = field(default=None, init=False)
    std_: Optional[np.ndarray] = field(default=None, init=False)
    model_: Optional[MLPSurfaceBetaVAE] = field(default=None, init=False)
    train_loss_: float = field(default=float("nan"), init=False)
    val_loss_: float = field(default=float("nan"), init=False)
    train_recon_loss_: float = field(default=float("nan"), init=False)
    val_recon_loss_: float = field(default=float("nan"), init=False)
    train_kl_loss_: float = field(default=float("nan"), init=False)
    val_kl_loss_: float = field(default=float("nan"), init=False)
    train_calendar_loss_: float = field(default=float("nan"), init=False)
    val_calendar_loss_: float = field(default=float("nan"), init=False)
    train_monotonicity_loss_: float = field(default=float("nan"), init=False)
    val_monotonicity_loss_: float = field(default=float("nan"), init=False)
    train_butterfly_loss_: float = field(default=float("nan"), init=False)
    val_butterfly_loss_: float = field(default=float("nan"), init=False)
    n_epochs_fit_: int = field(default=0, init=False)
    diagnostics_: Dict[str, Any] = field(default_factory=dict, init=False)
    historical_latent_: Optional[np.ndarray] = field(default=None, init=False)
    latent_mean_: Optional[np.ndarray] = field(default=None, init=False)
    latent_cov_inv_: Optional[np.ndarray] = field(default=None, init=False)
    latent_train_reference_: Optional[np.ndarray] = field(default=None, init=False)

    def _device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _prepare_log_surface_matrix(self, cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch = cube_to_batch_m_t(cube)
        self.output_in_percent_ = bool(np.nanmedian(batch) > 2.0)
        batch_frac = _to_fraction_vol(batch)
        batch_frac = _fill_nan_surface_batch(batch_frac)
        batch_frac = np.clip(batch_frac, float(self.vol_floor), float(self.vol_cap))
        n_dates, n_m, n_tau = batch_frac.shape
        self.shape_ = (int(n_m), int(n_tau))
        X_log = np.log(batch_frac).reshape(n_dates, n_m * n_tau).astype(np.float32)
        return X_log, batch_frac.astype(np.float32)

    @staticmethod
    def _kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

    def _loss_components(
        self,
        xb: torch.Tensor,
        pred: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta_now: float,
        criterion: nn.Module,
        use_noarb: bool,
        mean_t: Optional[torch.Tensor],
        std_t: Optional[torch.Tensor],
        m_grid_t: Optional[torch.Tensor],
        tau_grid_t: Optional[torch.Tensor],
        r: float,
        lambda_calendar: float,
        lambda_monotonicity: float,
        lambda_butterfly: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = criterion(pred, xb)
        kl_loss = self._kl_loss(mu, logvar)
        if use_noarb:
            l_cal, l_mono, l_but = _torch_three_noarb_penalty_from_decoded_log_norm(
                pred,
                mean_log=mean_t,
                std_log=std_t,
                shape=self.shape_,
                m_grid=m_grid_t,
                tau_grid=tau_grid_t,
                r=float(r),
                vol_floor=float(self.vol_floor),
                vol_cap=float(self.vol_cap),
            )
        else:
            zero = torch.zeros((), dtype=xb.dtype, device=xb.device)
            l_cal = l_mono = l_but = zero
        total = (
            recon_loss
            + float(beta_now) * kl_loss
            + float(lambda_calendar) * l_cal
            + float(lambda_monotonicity) * l_mono
            + float(lambda_butterfly) * l_but
        )
        return total, recon_loss, kl_loss, l_cal, l_mono, l_but

    def _evaluate_loader(
        self,
        loader: DataLoader,
        beta_now: float,
        criterion: nn.Module,
        use_noarb: bool,
        mean_t: Optional[torch.Tensor],
        std_t: Optional[torch.Tensor],
        m_grid_t: Optional[torch.Tensor],
        tau_grid_t: Optional[torch.Tensor],
        r: float,
        lambda_calendar: float,
        lambda_monotonicity: float,
        lambda_butterfly: float,
    ) -> Dict[str, float]:
        if self.model_ is None:
            raise RuntimeError("VAE model is not initialised.")
        self.model_.eval()
        sums = {k: 0.0 for k in ["loss", "recon", "kl", "calendar", "monotonicity", "butterfly"]}
        total_n = 0
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self._device())
                pred, mu, logvar = self.model_(xb)
                comps = self._loss_components(
                    xb, pred, mu, logvar, beta_now, criterion, use_noarb,
                    mean_t, std_t, m_grid_t, tau_grid_t, r,
                    lambda_calendar, lambda_monotonicity, lambda_butterfly,
                )
                for key, value in zip(sums, comps):
                    sums[key] += float(value.item()) * len(xb)
                total_n += len(xb)
        return {k: v / max(total_n, 1) for k, v in sums.items()}

    def fit_transform(
        self,
        cube: np.ndarray,
        epochs: int = 300,
        batch_size: int = 64,
        lr: float = 1.0e-3,
        weight_decay: float = 1.0e-5,
        val_fraction: float = 0.15,
        patience: int = 40,
        m_grid: Optional[np.ndarray] = None,
        tau_grid: Optional[np.ndarray] = None,
        r: float = 0.0,
        lambda_calendar: float = 10.0,
        lambda_monotonicity: float = 10.0,
        lambda_butterfly: float = 10.0,
        checkpoint_path: Optional[Path] = None,
        load_if_exists: bool = False,
        save_checkpoint: bool = True,
        surface_arb_tol: float = 1.0e-10,
    ) -> np.ndarray:
        torch.manual_seed(int(self.seed))
        np.random.seed(int(self.seed))
        X_log, batch_frac = self._prepare_log_surface_matrix(cube)
        if X_log.shape[0] < 4:
            raise ValueError("Need at least 4 surfaces to train a VAE latent model.")
        self.mean_ = X_log.mean(axis=0).astype(np.float32)
        self.std_ = X_log.std(axis=0)
        self.std_ = np.where(self.std_ < 1.0e-8, 1.0, self.std_).astype(np.float32)
        X_norm = ((X_log - self.mean_[None, :]) / self.std_[None, :]).astype(np.float32)
        input_dim = X_norm.shape[1]
        device = self._device()
        self.model_ = MLPSurfaceBetaVAE(
            input_dim=input_dim,
            latent_dim=int(self.latent_dim),
            hidden_dim_1=int(self.hidden_dim_1),
            hidden_dim_2=int(self.hidden_dim_2),
        ).to(device)

        checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        if load_if_exists and checkpoint_path is not None and checkpoint_path.exists():
            payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
            codec_type = str(payload.get("codec_type", ""))
            if codec_type not in {"SBJTS_BETA_VAE", "MLPSurfaceBetaVAE"}:
                raise RuntimeError(
                    f"Checkpoint {checkpoint_path} is not a compatible SBJTS beta-VAE checkpoint "
                    f"(codec_type={codec_type!r}). Train a new checkpoint or remove --load-autoencoder-if-exists."
                )
            self.model_.load_state_dict(payload["model_state_dict"])
            self.mean_ = np.asarray(payload["mean"], dtype=np.float32)
            self.std_ = np.asarray(payload["std"], dtype=np.float32)
            self.shape_ = tuple(payload["shape"])
            self.output_in_percent_ = bool(payload.get("output_in_percent", self.output_in_percent_))
            self.train_loss_ = float(payload.get("train_loss", float("nan")))
            self.val_loss_ = float(payload.get("val_loss", float("nan")))
            self.train_recon_loss_ = float(payload.get("train_recon_loss", float("nan")))
            self.val_recon_loss_ = float(payload.get("val_recon_loss", float("nan")))
            self.train_kl_loss_ = float(payload.get("train_kl_loss", float("nan")))
            self.val_kl_loss_ = float(payload.get("val_kl_loss", float("nan")))
            self.train_calendar_loss_ = float(payload.get("train_calendar_loss", float("nan")))
            self.val_calendar_loss_ = float(payload.get("val_calendar_loss", float("nan")))
            self.train_monotonicity_loss_ = float(payload.get("train_monotonicity_loss", float("nan")))
            self.val_monotonicity_loss_ = float(payload.get("val_monotonicity_loss", float("nan")))
            self.train_butterfly_loss_ = float(payload.get("train_butterfly_loss", float("nan")))
            self.val_butterfly_loss_ = float(payload.get("val_butterfly_loss", float("nan")))
            self.n_epochs_fit_ = int(payload.get("n_epochs_fit", 0))
            self.diagnostics_ = dict(payload.get("diagnostics", {}))
            latent = self.encode_from_normalized(X_norm, use_mean=True)
            self._fit_latent_reference(latent)
            return latent

        tensor = _torch_tensor_from_np(X_norm)
        dataset = TensorDataset(tensor)
        n_val = int(round(float(val_fraction) * len(dataset)))
        n_val = min(max(n_val, 1), len(dataset) - 1)
        n_train = len(dataset) - n_val
        # Chronological split: validation is the most recent block, not a random subset.
        train_tensor = tensor[:n_train]
        val_tensor = tensor[n_train:]
        train_ds = TensorDataset(train_tensor)
        val_ds = TensorDataset(val_tensor)
        train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, drop_last=False)

        opt = torch.optim.AdamW(self.model_.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_val = float("inf")
        bad_epochs = 0
        criterion = nn.MSELoss()

        use_noarb = (
            m_grid is not None
            and tau_grid is not None
            and (float(lambda_calendar) > 0.0 or float(lambda_monotonicity) > 0.0 or float(lambda_butterfly) > 0.0)
        )
        if use_noarb:
            if self.shape_ is None:
                raise RuntimeError("Surface shape is not fitted before no-arbitrage loss setup.")
            mean_t = _torch_tensor_from_np(self.mean_.astype(np.float32), device)
            std_t = _torch_tensor_from_np(self.std_.astype(np.float32), device)
            m_grid_t = _torch_tensor_from_np(np.asarray(m_grid, dtype=np.float32), device)
            tau_grid_t = _torch_tensor_from_np(np.asarray(tau_grid, dtype=np.float32), device)
            if torch.any(torch.diff(m_grid_t) <= 0):
                raise ValueError("m_grid must be strictly increasing for no-arbitrage penalties.")
            if torch.any(torch.diff(tau_grid_t) <= 0):
                raise ValueError("tau_grid must be strictly increasing for no-arbitrage penalties.")
        else:
            mean_t = std_t = m_grid_t = tau_grid_t = None

        last_train_metrics: Dict[str, float] = {}
        best_val_metrics: Dict[str, float] = {}
        for epoch in range(int(epochs)):
            beta_now = float(self.beta_kl) * min(1.0, float(epoch + 1) / max(float(self.kl_warmup_epochs), 1.0))
            self.model_.train()
            for (xb,) in train_loader:
                xb = xb.to(device)
                pred, mu, logvar = self.model_(xb)
                loss, _, _, _, _, _ = self._loss_components(
                    xb, pred, mu, logvar, beta_now, criterion, use_noarb,
                    mean_t, std_t, m_grid_t, tau_grid_t, r,
                    lambda_calendar, lambda_monotonicity, lambda_butterfly,
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            last_train_metrics = self._evaluate_loader(
                train_loader, beta_now, criterion, use_noarb, mean_t, std_t, m_grid_t, tau_grid_t, r,
                lambda_calendar, lambda_monotonicity, lambda_butterfly,
            )
            val_metrics = self._evaluate_loader(
                val_loader, beta_now, criterion, use_noarb, mean_t, std_t, m_grid_t, tau_grid_t, r,
                lambda_calendar, lambda_monotonicity, lambda_butterfly,
            )
            val_loss = float(val_metrics["loss"])
            if val_loss < best_val - 1.0e-8:
                best_val = val_loss
                best_val_metrics = val_metrics
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                bad_epochs += 1
            self.n_epochs_fit_ = epoch + 1
            if bad_epochs >= int(patience):
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        # Re-evaluate with the final beta after restoring the best state.
        final_beta = float(self.beta_kl)
        train_metrics = self._evaluate_loader(
            train_loader, final_beta, criterion, use_noarb, mean_t, std_t, m_grid_t, tau_grid_t, r,
            lambda_calendar, lambda_monotonicity, lambda_butterfly,
        )
        val_metrics = self._evaluate_loader(
            val_loader, final_beta, criterion, use_noarb, mean_t, std_t, m_grid_t, tau_grid_t, r,
            lambda_calendar, lambda_monotonicity, lambda_butterfly,
        ) if best_val_metrics else {}
        self.train_loss_ = float(train_metrics.get("loss", float("nan")))
        self.val_loss_ = float(val_metrics.get("loss", best_val))
        self.train_recon_loss_ = float(train_metrics.get("recon", float("nan")))
        self.val_recon_loss_ = float(val_metrics.get("recon", float("nan")))
        self.train_kl_loss_ = float(train_metrics.get("kl", float("nan")))
        self.val_kl_loss_ = float(val_metrics.get("kl", float("nan")))
        self.train_calendar_loss_ = float(train_metrics.get("calendar", float("nan")))
        self.val_calendar_loss_ = float(val_metrics.get("calendar", float("nan")))
        self.train_monotonicity_loss_ = float(train_metrics.get("monotonicity", float("nan")))
        self.val_monotonicity_loss_ = float(val_metrics.get("monotonicity", float("nan")))
        self.train_butterfly_loss_ = float(train_metrics.get("butterfly", float("nan")))
        self.val_butterfly_loss_ = float(val_metrics.get("butterfly", float("nan")))

        latent = self.encode_from_normalized(X_norm, use_mean=True)
        self._fit_latent_reference(latent)
        self._compute_reconstruction_diagnostics(
            X_norm=X_norm,
            batch_frac=batch_frac,
            latent=latent,
            m_grid=m_grid,
            tau_grid=tau_grid,
            r=float(r),
            surface_arb_tol=float(surface_arb_tol),
            n_train=n_train,
            n_val=n_val,
        )

        if save_checkpoint and checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "codec_type": "SBJTS_BETA_VAE",
                    "model_state_dict": self.model_.state_dict(),
                    "mean": self.mean_,
                    "std": self.std_,
                    "shape": self.shape_,
                    "output_in_percent": self.output_in_percent_,
                    "train_loss": self.train_loss_,
                    "val_loss": self.val_loss_,
                    "train_recon_loss": self.train_recon_loss_,
                    "val_recon_loss": self.val_recon_loss_,
                    "train_kl_loss": self.train_kl_loss_,
                    "val_kl_loss": self.val_kl_loss_,
                    "train_calendar_loss": self.train_calendar_loss_,
                    "val_calendar_loss": self.val_calendar_loss_,
                    "train_monotonicity_loss": self.train_monotonicity_loss_,
                    "val_monotonicity_loss": self.val_monotonicity_loss_,
                    "train_butterfly_loss": self.train_butterfly_loss_,
                    "val_butterfly_loss": self.val_butterfly_loss_,
                    "n_epochs_fit": self.n_epochs_fit_,
                    "ae_lambda_calendar": float(lambda_calendar),
                    "ae_lambda_monotonicity": float(lambda_monotonicity),
                    "ae_lambda_butterfly": float(lambda_butterfly),
                    "beta_kl": float(self.beta_kl),
                    "kl_warmup_epochs": int(self.kl_warmup_epochs),
                    "latent_dim": int(self.latent_dim),
                    "hidden_dim_1": int(self.hidden_dim_1),
                    "hidden_dim_2": int(self.hidden_dim_2),
                    "diagnostics": self.diagnostics_,
                },
                checkpoint_path,
            )
        return latent

    def _fit_latent_reference(self, latent: np.ndarray) -> None:
        z = np.asarray(latent, dtype=float)
        self.historical_latent_ = z.copy()
        self.latent_train_reference_ = z.copy()
        self.latent_mean_ = z.mean(axis=0)
        cov = np.cov(z, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        cov = np.asarray(cov, dtype=float)
        cov = cov + np.eye(cov.shape[0]) * 1.0e-6
        self.latent_cov_inv_ = np.linalg.pinv(cov)

    def encode_from_normalized(self, X_norm: np.ndarray, use_mean: bool = True) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("VAE codec is not fitted.")
        device = self._device()
        self.model_.eval()
        outs = []
        with torch.no_grad():
            for start in range(0, len(X_norm), 2048):
                xb = _torch_tensor_from_np(X_norm[start:start + 2048].astype(np.float32), device)
                outs.append(_np_from_torch_tensor(self.model_.encode(xb, use_mean=bool(use_mean))))
        return np.concatenate(outs, axis=0).astype(float)

    def decode_raw_log(self, latent: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.mean_ is None or self.std_ is None or self.shape_ is None:
            raise RuntimeError("VAE codec is not fitted.")
        z = np.asarray(latent, dtype=np.float32)
        flat_z = z.reshape(-1, z.shape[-1])
        device = self._device()
        self.model_.eval()
        outs = []
        with torch.no_grad():
            for start in range(0, len(flat_z), 2048):
                zb = _torch_tensor_from_np(flat_z[start:start + 2048], device)
                outs.append(_np_from_torch_tensor(self.model_.decode(zb)))
        X_norm = np.concatenate(outs, axis=0)
        return X_norm * self.std_[None, :] + self.mean_[None, :]

    def decode_with_clip_info(self, latent: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        if self.shape_ is None:
            raise RuntimeError("VAE codec is not fitted.")
        z = np.asarray(latent, dtype=np.float32)
        original_shape = z.shape[:-1]
        X_log_raw = self.decode_raw_log(z)
        log_floor = math.log(float(self.vol_floor))
        log_cap = math.log(float(self.vol_cap))
        floor_mask = X_log_raw < log_floor
        cap_mask = X_log_raw > log_cap
        X_log = np.clip(X_log_raw, log_floor, log_cap)
        surfaces = np.exp(X_log).reshape(original_shape + self.shape_)
        info = {
            "decoder_clip_rate_floor": float(np.mean(floor_mask)),
            "decoder_clip_rate_cap": float(np.mean(cap_mask)),
            "decoder_clip_rate_total": float(np.mean(floor_mask | cap_mask)),
            "decoder_raw_log_min": float(np.nanmin(X_log_raw)),
            "decoder_raw_log_max": float(np.nanmax(X_log_raw)),
        }
        return np.clip(surfaces, float(self.vol_floor), float(self.vol_cap)), info

    def decode(self, latent: np.ndarray) -> np.ndarray:
        surfaces, _ = self.decode_with_clip_info(latent)
        return surfaces

    def latent_ood_scores(self, latent: np.ndarray, chunk_size: int = 8192) -> Dict[str, np.ndarray]:
        if self.latent_mean_ is None or self.latent_cov_inv_ is None or self.latent_train_reference_ is None:
            raise RuntimeError("Latent reference statistics are not fitted.")
        z = np.asarray(latent, dtype=float)
        original_shape = z.shape[:-1]
        flat = z.reshape(-1, z.shape[-1])
        centered = flat - self.latent_mean_[None, :]
        mahal = np.sqrt(np.maximum(np.sum((centered @ self.latent_cov_inv_) * centered, axis=1), 0.0))
        ref = np.asarray(self.latent_train_reference_, dtype=float)
        nn = np.empty(len(flat), dtype=float)
        for start in range(0, len(flat), int(chunk_size)):
            stop = min(start + int(chunk_size), len(flat))
            diff = flat[start:stop, None, :] - ref[None, :, :]
            nn[start:stop] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
        return {
            "latent_mahalanobis": mahal.reshape(original_shape),
            "latent_nearest_neighbor": nn.reshape(original_shape),
        }

    @staticmethod
    def _summary_stats(prefix: str, arr: np.ndarray) -> Dict[str, float]:
        a = np.asarray(arr, dtype=float)
        return {
            f"{prefix}_mean": float(np.nanmean(a)),
            f"{prefix}_median": float(np.nanmedian(a)),
            f"{prefix}_q95": float(np.nanquantile(a, 0.95)),
            f"{prefix}_max": float(np.nanmax(a)),
        }

    def generated_latent_path_diagnostics(self, latent_paths: np.ndarray) -> pd.DataFrame:
        scores = self.latent_ood_scores(latent_paths)
        mahal = scores["latent_mahalanobis"]
        nn = scores["latent_nearest_neighbor"]
        X_log_raw = self.decode_raw_log(latent_paths)
        n_paths, path_length = latent_paths.shape[0], latent_paths.shape[1]
        X_log_raw = X_log_raw.reshape(n_paths, path_length, -1)
        log_floor = math.log(float(self.vol_floor))
        log_cap = math.log(float(self.vol_cap))
        floor_mask = X_log_raw < log_floor
        cap_mask = X_log_raw > log_cap
        rows = []
        for i in range(n_paths):
            row: Dict[str, Any] = {"candidate_path_idx": int(i)}
            row.update(self._summary_stats("latent_mahalanobis", mahal[i]))
            row.update(self._summary_stats("latent_nearest_neighbor", nn[i]))
            row["decoder_clip_rate_floor"] = float(np.mean(floor_mask[i]))
            row["decoder_clip_rate_cap"] = float(np.mean(cap_mask[i]))
            row["decoder_clip_rate_total"] = float(np.mean(floor_mask[i] | cap_mask[i]))
            row["decoder_raw_log_min"] = float(np.nanmin(X_log_raw[i]))
            row["decoder_raw_log_max"] = float(np.nanmax(X_log_raw[i]))
            rows.append(row)
        return pd.DataFrame(rows)

    def _compute_reconstruction_diagnostics(
        self,
        X_norm: np.ndarray,
        batch_frac: np.ndarray,
        latent: np.ndarray,
        m_grid: Optional[np.ndarray],
        tau_grid: Optional[np.ndarray],
        r: float,
        surface_arb_tol: float,
        n_train: int,
        n_val: int,
    ) -> None:
        recon_frac, clip_info = self.decode_with_clip_info(latent)
        err = recon_frac - batch_frac
        train_slice = slice(0, int(n_train))
        val_slice = slice(int(n_train), int(n_train) + int(n_val))

        def rmse(a: np.ndarray) -> float:
            return float(np.sqrt(np.nanmean(np.asarray(a, dtype=float) ** 2)))

        def rel_rmse(e: np.ndarray, ref: np.ndarray) -> float:
            denom = float(np.sqrt(np.nanmean(np.asarray(ref, dtype=float) ** 2)))
            return rmse(e) / max(denom, 1.0e-12)

        diag: Dict[str, Any] = {
            "chronological_train_size": int(n_train),
            "chronological_val_size": int(n_val),
            "reconstruction_rmse_all": rmse(err),
            "reconstruction_rmse_train": rmse(err[train_slice]),
            "reconstruction_rmse_val": rmse(err[val_slice]),
            "reconstruction_relative_rmse_all": rel_rmse(err, batch_frac),
            "reconstruction_relative_rmse_train": rel_rmse(err[train_slice], batch_frac[train_slice]),
            "reconstruction_relative_rmse_val": rel_rmse(err[val_slice], batch_frac[val_slice]),
            "reconstruction_max_abs_error_all": float(np.nanmax(np.abs(err))),
            "reconstruction_max_abs_error_train": float(np.nanmax(np.abs(err[train_slice]))),
            "reconstruction_max_abs_error_val": float(np.nanmax(np.abs(err[val_slice]))),
            **clip_info,
        }
        scores = self.latent_ood_scores(latent)
        diag.update(self._summary_stats("historical_latent_mahalanobis", scores["latent_mahalanobis"]))
        diag.update(self._summary_stats("historical_latent_nearest_neighbor", scores["latent_nearest_neighbor"]))

        if m_grid is not None and tau_grid is not None:
            hist_path_diag, _ = diagnose_paths(
                batch_frac[:, None, :, :],
                m_grid=np.asarray(m_grid, dtype=float),
                tau_grid=np.asarray(tau_grid, dtype=float),
                r=float(r),
                surface_eps=float(surface_arb_tol),
            )
            recon_path_diag, _ = diagnose_paths(
                recon_frac[:, None, :, :],
                m_grid=np.asarray(m_grid, dtype=float),
                tau_grid=np.asarray(tau_grid, dtype=float),
                r=float(r),
                surface_eps=float(surface_arb_tol),
            )
            diag.update({
                "historical_surface_phi_mean": float(hist_path_diag["path_phi"].mean()),
                "historical_surface_phi_median": float(hist_path_diag["path_phi"].median()),
                "historical_zero_surface_pct": float(hist_path_diag["zero_surface_pct"].mean()),
                "reconstructed_surface_phi_mean": float(recon_path_diag["path_phi"].mean()),
                "reconstructed_surface_phi_median": float(recon_path_diag["path_phi"].median()),
                "reconstructed_zero_surface_pct": float(recon_path_diag["zero_surface_pct"].mean()),
            })
        self.diagnostics_ = diag

    def meta(self) -> Dict[str, Any]:
        return {
            "latent_codec": "arbitrage-aware beta-VAE on normalized log-IV surfaces",
            "latent_dim": int(self.latent_dim),
            "hidden_dim_1": int(self.hidden_dim_1),
            "hidden_dim_2": int(self.hidden_dim_2),
            "beta_kl": float(self.beta_kl),
            "kl_warmup_epochs": int(self.kl_warmup_epochs),
            "surface_shape": self.shape_,
            "output_in_percent": bool(self.output_in_percent_),
            "ae_train_loss": float(self.train_loss_),
            "ae_val_loss": float(self.val_loss_),
            "ae_train_recon_loss": float(self.train_recon_loss_),
            "ae_val_recon_loss": float(self.val_recon_loss_),
            "ae_train_kl_loss": float(self.train_kl_loss_),
            "ae_val_kl_loss": float(self.val_kl_loss_),
            "ae_train_calendar_loss": float(self.train_calendar_loss_),
            "ae_val_calendar_loss": float(self.val_calendar_loss_),
            "ae_train_monotonicity_loss": float(self.train_monotonicity_loss_),
            "ae_val_monotonicity_loss": float(self.val_monotonicity_loss_),
            "ae_train_butterfly_loss": float(self.train_butterfly_loss_),
            "ae_val_butterfly_loss": float(self.val_butterfly_loss_),
            "ae_epochs_fit": int(self.n_epochs_fit_),
            "ae_noarb_loss": "relative_call_price_calendar_monotonicity_butterfly",
            **self.diagnostics_,
        }



# -----------------------------------------------------------------------------
# Candidate streaming and WMC selection for AE config
# -----------------------------------------------------------------------------


def _total_candidate_paths(cfg: SBJTSAEConfig) -> int:
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


def _config_for_candidate_batch(cfg: SBJTSAEConfig, batch_start: int, batch_n: int) -> SBJTSAEConfig:
    return replace(cfg, n_candidate_paths=int(batch_n), seed=_seed_for_candidate_batch(int(cfg.seed), int(batch_start)))


def select_path_indices_wmc(candidate_path_diag: pd.DataFrame, cfg: SBJTSAEConfig):
    rng = np.random.default_rng(int(cfg.seed) + 271828)
    phi = candidate_path_diag["path_phi"].to_numpy(dtype=float)
    path_tol = float(cfg.path_arb_tol) if cfg.path_arb_tol is not None else float(cfg.surface_arb_tol) * int(cfg.path_length)
    if cfg.require_zero_path:
        eligible = np.where(phi <= path_tol)[0]
        if len(eligible) == 0:
            best = float(np.nanmin(phi))
            raise RuntimeError(
                "No zero-penalty SBJTS-AE path found. "
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


def _simulate_candidate_batch(latent_model: SBJTSLatentModel, codec: AutoencoderLogVolCodec, cfg: SBJTSAEConfig):
    latent_paths, latent_diag = latent_model.simulate_latent_paths(
        n_paths=int(cfg.n_candidate_paths),
        path_length=int(cfg.path_length),
        seed=int(cfg.seed),
        start_mode=str(cfg.start_mode),
        start_index=cfg.start_index,
    )
    paths_frac, clip_info = codec.decode_with_clip_info(latent_paths)
    paths_frac = np.clip(paths_frac, float(cfg.vol_floor), float(cfg.vol_cap))
    latent_path_diag = codec.generated_latent_path_diagnostics(latent_paths)
    return paths_frac, latent_diag, latent_path_diag


def diagnose_candidate_paths_streaming(
    latent_model: SBJTSLatentModel,
    codec: AutoencoderLogVolCodec,
    cfg: SBJTSAEConfig,
    m_grid: np.ndarray,
    tau_grid: np.ndarray,
) -> pd.DataFrame:
    n_total = _total_candidate_paths(cfg)
    pieces = []
    for start, stop in _candidate_batch_ranges(n_total, int(cfg.candidate_batch_size)):
        batch_cfg = _config_for_candidate_batch(cfg, start, stop - start)
        batch_paths_frac, latent_diag, latent_path_diag = _simulate_candidate_batch(latent_model, codec, batch_cfg)
        batch_path_diag, _ = diagnose_paths(
            batch_paths_frac,
            m_grid=m_grid,
            tau_grid=tau_grid,
            r=float(cfg.r),
            surface_eps=float(cfg.surface_arb_tol),
        )
        latent_summary = latent_diag.groupby("candidate_path_idx").agg(
            n_latent_jumps=("did_jump", "sum"),
            mean_p_jump=("p_jump", "mean"),
            mean_kernel_ess=("kernel_ess", "mean"),
            mean_latent_step_norm=("latent_step_norm", "mean"),
            max_latent_step_norm=("latent_step_norm", "max"),
        ).reset_index()
        batch_path_diag = batch_path_diag.merge(latent_summary, on="candidate_path_idx", how="left")
        batch_path_diag = batch_path_diag.merge(latent_path_diag, on="candidate_path_idx", how="left")
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
    codec: AutoencoderLogVolCodec,
    cfg: SBJTSAEConfig,
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
        batch_paths_frac, _, _ = _simulate_candidate_batch(latent_model, codec, batch_cfg)
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


def simulate_sbjts_autoencoder_paths(config: SBJTSAEConfig) -> SBJTSResult:
    t_start = time.perf_counter()
    cube, m_grid, tenor_days, asof_dates_input, input_meta = load_cube_npz(config.input_path)
    tau_grid = np.asarray(tenor_days, dtype=float) / 365.0
    output_in_percent = bool(np.nanmedian(cube) > 2.0)

    codec = AutoencoderLogVolCodec(
        latent_dim=int(config.latent_dim),
        hidden_dim_1=int(config.hidden_dim_1),
        hidden_dim_2=int(config.hidden_dim_2),
        vol_floor=float(config.vol_floor),
        vol_cap=float(config.vol_cap),
        seed=int(config.seed),
        device=str(config.device),
        beta_kl=float(config.ae_beta_kl),
        kl_warmup_epochs=int(config.ae_kl_warmup_epochs),
    )
    latent = codec.fit_transform(
        cube,
        epochs=int(config.ae_epochs),
        batch_size=int(config.ae_batch_size),
        lr=float(config.ae_lr),
        weight_decay=float(config.ae_weight_decay),
        val_fraction=float(config.ae_val_fraction),
        patience=int(config.ae_patience),
        m_grid=m_grid,
        tau_grid=tau_grid,
        r=float(config.r),
        lambda_calendar=float(config.ae_lambda_calendar),
        lambda_monotonicity=float(config.ae_lambda_monotonicity),
        lambda_butterfly=float(config.ae_lambda_butterfly),
        checkpoint_path=Path(config.autoencoder_checkpoint_path),
        load_if_exists=bool(config.load_autoencoder_if_exists),
        save_checkpoint=bool(config.save_autoencoder),
        surface_arb_tol=float(config.surface_arb_tol),
    )

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
    out_dates = np.array([f"sbjts_ae_path_{p:04d}_t_{t:04d}" for p in range(int(config.n_output_paths)) for t in range(int(config.path_length))])

    meta: Dict[str, Any] = {
        "model": "SBJTS beta-VAE latent jump-diffusion bridge for IV surfaces",
        "input_path": str(config.input_path),
        "input_meta": input_meta,
        "n_historical_dates": int(cube.shape[2]),
        "path_length": int(config.path_length),
        "n_output_paths": int(config.n_output_paths),
        "output_in_percent": bool(output_in_percent),
        "elapsed_sec": float(time.perf_counter() - t_start),
        "ae_lambda_calendar": float(config.ae_lambda_calendar),
        "ae_lambda_monotonicity": float(config.ae_lambda_monotonicity),
        "ae_lambda_butterfly": float(config.ae_lambda_butterfly),
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


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    import argparse

    p = argparse.ArgumentParser("sbjts_autoencoder_vol_surface: Schrödinger bridge with jumps in beta-VAE latent space")
    p.add_argument("--input", default="data/treated/vol_surface_cube.npz")
    p.add_argument("--out", default="data/treated/sbjts_autoencoder_vol_surface.npz")
    p.add_argument("--candidate-path-diagnostics", default="data/treated/sbjts_ae_candidate_path_diagnostics.csv")
    p.add_argument("--selected-path-diagnostics", default="data/treated/sbjts_ae_selected_path_diagnostics.csv")
    p.add_argument("--selected-surface-diagnostics", default="data/treated/sbjts_ae_selected_surface_diagnostics.csv")
    p.add_argument("--autoencoder-checkpoint", default="data/treated/sbjts_surface_autoencoder.pt")

    p.add_argument("--n-output-paths", type=int, default=10)
    p.add_argument("--path-length", type=int, default=64)
    p.add_argument("--n-candidate-paths", type=int, default=None)
    p.add_argument("--candidate-multiplier", type=int, default=500)
    p.add_argument("--candidate-batch-size", type=int, default=512)

    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--hidden-dim-1", type=int, default=256)
    p.add_argument("--hidden-dim-2", type=int, default=96)
    p.add_argument("--ae-epochs", type=int, default=300)
    p.add_argument("--ae-batch-size", type=int, default=64)
    p.add_argument("--ae-lr", type=float, default=1.0e-3)
    p.add_argument("--ae-weight-decay", type=float, default=1.0e-5)
    p.add_argument("--ae-beta-kl", type=float, default=1.0e-3)
    p.add_argument("--ae-kl-warmup-epochs", type=int, default=100)
    p.add_argument("--ae-lambda-calendar", type=float, default=10.0)
    p.add_argument("--ae-lambda-monotonicity", type=float, default=10.0)
    p.add_argument("--ae-lambda-butterfly", type=float, default=10.0)
    p.add_argument("--ae-val-fraction", type=float, default=0.15)
    p.add_argument("--ae-patience", type=int, default=40)
    p.add_argument("--load-autoencoder-if-exists", action="store_true")
    p.add_argument("--no-save-autoencoder", action="store_true")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

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
    cfg = SBJTSAEConfig(
        input_path=Path(args.input),
        output_path=Path(args.out),
        #candidate_path_diagnostics_path=Path(args.candidate_path_diagnostics),
        #selected_path_diagnostics_path=Path(args.selected_path_diagnostics),
        #selected_surface_diagnostics_path=Path(args.selected_surface_diagnostics),
        autoencoder_checkpoint_path=Path(args.autoencoder_checkpoint),
        n_output_paths=args.n_output_paths,
        path_length=args.path_length,
        n_candidate_paths=args.n_candidate_paths,
        candidate_multiplier=args.candidate_multiplier,
        candidate_batch_size=args.candidate_batch_size,
        latent_dim=args.latent_dim,
        hidden_dim_1=args.hidden_dim_1,
        hidden_dim_2=args.hidden_dim_2,
        ae_epochs=args.ae_epochs,
        ae_batch_size=args.ae_batch_size,
        ae_lr=args.ae_lr,
        ae_weight_decay=args.ae_weight_decay,
        ae_beta_kl=args.ae_beta_kl,
        ae_kl_warmup_epochs=args.ae_kl_warmup_epochs,
        ae_lambda_calendar=args.ae_lambda_calendar,
        ae_lambda_monotonicity=args.ae_lambda_monotonicity,
        ae_lambda_butterfly=args.ae_lambda_butterfly,
        ae_val_fraction=args.ae_val_fraction,
        ae_patience=args.ae_patience,
        load_autoencoder_if_exists=args.load_autoencoder_if_exists,
        save_autoencoder=not args.no_save_autoencoder,
        device=args.device,
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
    result = simulate_sbjts_autoencoder_paths(cfg)
    save_result_npz(cfg.output_path, result)
    #cfg.candidate_path_diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    #result.candidate_path_diagnostics.to_csv(cfg.candidate_path_diagnostics_path, index=False)
    #result.selected_path_diagnostics.to_csv(cfg.selected_path_diagnostics_path, index=False)
    #result.selected_surface_diagnostics.to_csv(cfg.selected_surface_diagnostics_path, index=False)

    print(f"Saved SBJTS beta-VAE output paths/cube to {cfg.output_path}")
    #print(f"Saved candidate path diagnostics to {cfg.candidate_path_diagnostics_path}")
    #print(f"Saved selected path diagnostics to {cfg.selected_path_diagnostics_path}")
    #print(f"Saved selected surface diagnostics to {cfg.selected_surface_diagnostics_path}")
    print(f"paths shape: {result.paths.shape}")
    print(f"cube shape: {result.cube.shape}")
    print(json.dumps(result.meta, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
