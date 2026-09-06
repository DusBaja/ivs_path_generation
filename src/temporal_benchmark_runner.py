from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import time
import numpy as np
import torch

from benchmark_utils import (
    set_all_seeds, load_cube, cube_to_fm_surfaces, normalizer_fit,
    normalize_pct, denormalize_pct, orient_paths_to_tenor_money,
    write_json,
)
from volsurface_latentFM.ivs_autoencoder import IVS_AE
from volsurface_latentFM.temporal_fm_trigo import TemporalFMConfig, TemporalLatentFlowMatchTrigo
from volsurface_latentSB.sbjts_pca_vol_surface import SBJTSPCAConfig, simulate_sbjts_pca_paths
from volsurface_latentSB.sbjts_autoencoder_vol_surface import SBJTSAEConfig, simulate_sbjts_autoencoder_paths
from volsurface_latentSB.lightsb_vol_surface_github_torchcompat import LightSBVolConfig, simulate_lightsb_paths
from other_models.cont_simulations import SimulationConfig, simulate_arbitrage_free_paths


@dataclass
class BenchmarkConfig:
    train_cube_path: str = "data/treated/benchmark_train.npz"
    test_cube_path: str = "data/treated/benchmark_test.npz"
    output_dir: str = "data/treated/temporal_benchmark_500"
    n_paths: int = 500
    path_length: int = 100
    n_candidate_paths: int = 2000
    candidate_batch_size: int = 128
    seed: int = 42
    beta_wmc: float = 1.0e4
    # Temporal FM
    fm_latent_dim: int = 8
    fm_context_lags: int = 3
    fm_ae_epochs: int = 1200
    fm_epochs: int = 1200
    fm_batch_size: int = 64
    fm_ode_steps: int = 64
    # SBJTS VAE
    sbjts_vae_latent_dim: int = 8
    sbjts_vae_epochs: int = 1200
    # LightSB
    lightsb_factors: int = 8
    lightsb_components: int = 32
    lightsb_epochs: int = 1200
    # VolGAN
    volgan_epochs: int = 1200
    volgan_grad_epochs: int = 100
    volgan_noise_dim: int = 32
    volgan_hidden_dim: int = 256


def _save_paths(out_dir: Path, name: str, paths: np.ndarray, m_grid, tenor_days):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npz"
    np.savez_compressed(path, paths=np.asarray(paths, np.float32), m_grid=m_grid, tenor_days=tenor_days)
    print(f"saved {name}: {paths.shape} -> {path}")
    return path


def train_temporal_fm(cfg: BenchmarkConfig, device):
    cube, m_grid, tenor_days, _ = load_cube(cfg.train_cube_path)
    surfaces = cube_to_fm_surfaces(cube)  # percent/fraction unchanged
    # Existing FM code expects percent IV. Convert if input is fractional.
    if np.nanmedian(surfaces) < 2.0:
        surfaces = surfaces * 100.0
    lo, hi = normalizer_fit(surfaces)
    surfaces_n = normalize_pct(surfaces, lo, hi)
    tau_grid = tenor_days / 365.0

    ae = IVS_AE(z_dim=cfg.fm_latent_dim)
    ae.fit(
        surfaces_n, m_grid=m_grid, tau_grid=tau_grid,
        n_epochs=cfg.fm_ae_epochs, batch_size=cfg.fm_batch_size,
        lr=1e-3, warmup_epochs=max(50, cfg.fm_ae_epochs // 5),
        patience=max(100, cfg.fm_ae_epochs // 3), print_every=100,
        lo=lo, hi=hi, device=device,
    )
    z = ae.encode(surfaces_n, device=device, use_mean=True)

    fm_cfg = TemporalFMConfig(
        context_lags=cfg.fm_context_lags,
        epochs=cfg.fm_epochs,
        batch_size=cfg.fm_batch_size,
        ode_steps=cfg.fm_ode_steps,
    )
    fm = TemporalLatentFlowMatchTrigo(cfg.fm_latent_dim, fm_cfg)
    fm.fit(z, device=device, print_every=100)

    out_dir = Path(cfg.output_dir)
    ae.save(out_dir / "temporal_fm_ae.pt", surf_lo=lo, surf_hi=hi)
    fm.save(out_dir / "temporal_fm_trigo.pt")

    z_paths = fm.sample_paths(
        initial_history=z[-cfg.fm_context_lags:],
        n_paths=cfg.n_paths,
        path_length=cfg.path_length,
        device=device,
    )
    flat = z_paths.reshape(-1, cfg.fm_latent_dim)
    decoded_n = ae.decode(flat, device=device).detach().cpu().numpy().reshape(
        cfg.n_paths, cfg.path_length, surfaces.shape[-2], surfaces.shape[-1]
    )
    paths_pct = denormalize_pct(decoded_n, lo, hi)
    return paths_pct

