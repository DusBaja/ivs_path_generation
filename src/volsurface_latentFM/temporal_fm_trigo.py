from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class SinusoidalEmbed(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dimension must be even")
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=t.dtype)
            * (math.log(10000.0) / max(half - 1, 1))
        )
        angles = t[:, None] * freqs[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ConditionalVelocityMLP(nn.Module):
    """Velocity field v_theta(x_s, s | z_t, ..., z_{t-L+1})."""

    def __init__(
        self,
        z_dim: int,
        context_lags: int = 3,
        hidden: Iterable[int] = (128, 128, 128),
        t_emb_dim: int = 32,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.context_lags = int(context_lags)
        self.context_dim = self.z_dim * self.context_lags
        self.t_emb_dim = int(t_emb_dim)
        self.t_embed = SinusoidalEmbed(self.t_emb_dim)

        dims = [self.z_dim + self.context_dim + self.t_emb_dim, *map(int, hidden), self.z_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.SiLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x_s: torch.Tensor, s: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_s, context, self.t_embed(s)], dim=-1))


def make_temporal_pairs(z_codes: np.ndarray, context_lags: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Build chronological (context, next latent increment) observations."""
    z = np.asarray(z_codes, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"z_codes must be 2D, got {z.shape}")
    L = int(context_lags)
    if len(z) <= L:
        raise ValueError(f"Need more than {L} latent observations, got {len(z)}")

    contexts, deltas = [], []
    for t in range(L - 1, len(z) - 1):
        contexts.append(z[t - L + 1 : t + 1].reshape(-1))
        deltas.append(z[t + 1] - z[t])
    return np.asarray(contexts, np.float32), np.asarray(deltas, np.float32)


@dataclass
class TemporalFMConfig:
    context_lags: int = 3
    hidden: tuple[int, ...] = (128, 128, 128)
    t_emb_dim: int = 32
    ode_steps: int = 64
    epochs: int = 1200
    batch_size: int = 64
    lr: float = 3.0e-4
    weight_decay: float = 0.0
    grad_clip: float = 5.0


class TemporalLatentFlowMatchTrigo(nn.Module):
    """
    Conditional latent Flow Matching on one-step increments.

    Historical time and FM interpolation time are distinct:
      context c_t = (z_{t-L+1}, ..., z_t)
      target  r_t = z_{t+1} - z_t
      x_s = cos(pi s/2) eps + sin(pi s/2) r_t

    The model learns v_theta(x_s, s | c_t). Path simulation repeatedly samples
    a conditional increment and updates z_{t+1}=z_t+r_t.
    """

    def __init__(self, z_dim: int, config: TemporalFMConfig | None = None):
        super().__init__()
        self.z_dim = int(z_dim)
        self.config = config or TemporalFMConfig()
        self.net = ConditionalVelocityMLP(
            z_dim=self.z_dim,
            context_lags=self.config.context_lags,
            hidden=self.config.hidden,
            t_emb_dim=self.config.t_emb_dim,
        )

    @staticmethod
    def _trigo_interpolate(noise: torch.Tensor, target: torch.Tensor, s: torch.Tensor):
        s_ = s[:, None]
        angle = 0.5 * torch.pi * s_
        alpha = torch.cos(angle)
        sigma = torch.sin(angle)
        x_s = alpha * noise + sigma * target
        v_target = (
            -0.5 * torch.pi * torch.sin(angle) * noise
            + 0.5 * torch.pi * torch.cos(angle) * target
        )
        return x_s, v_target

    def fit(self, z_codes: np.ndarray, device=None, print_every: int = 100):
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)
        contexts, deltas = make_temporal_pairs(z_codes, self.config.context_lags)
        dataset = TensorDataset(torch.from_numpy(contexts), torch.from_numpy(deltas))
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False)
        opt = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        history = []

        for epoch in range(1, self.config.epochs + 1):
            self.train()
            total, n_batches = 0.0, 0
            for context, target_delta in loader:
                context = context.to(dev)
                target_delta = target_delta.to(dev)
                B = target_delta.shape[0]
                s = torch.rand(B, device=dev)
                noise = torch.randn_like(target_delta)
                x_s, v_target = self._trigo_interpolate(noise, target_delta, s)
                v_pred = self.net(x_s, s, context)
                loss = F.mse_loss(v_pred, v_target)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)
                opt.step()
                total += float(loss.item())
                n_batches += 1

            history.append(total / max(n_batches, 1))
            if epoch == 1 or epoch % int(print_every) == 0 or epoch == self.config.epochs:
                print(f"[Temporal FM-trigo] epoch {epoch:4d}/{self.config.epochs} loss={history[-1]:.6f}")
        return history

    @torch.no_grad()
    def sample_increment(self, context: torch.Tensor) -> torch.Tensor:
        """Vectorized conditional increment generation for a batch of path contexts."""
        self.eval()
        dev = next(self.parameters()).device
        context = context.to(dev)
        n = context.shape[0]
        x = torch.randn(n, self.z_dim, device=dev)
        dt = 1.0 / int(self.config.ode_steps)
        for i in range(int(self.config.ode_steps)):
            s = torch.full((n,), i / self.config.ode_steps, device=dev)
            x = x + dt * self.net(x, s, context)
        return x

    @torch.no_grad()
    def sample_paths(
        self,
        initial_history: np.ndarray,
        n_paths: int = 500,
        path_length: int = 100,
        device=None,
    ) -> np.ndarray:
        """Return latent paths with shape (n_paths, path_length, z_dim)."""
        dev = device or next(self.parameters()).device
        L = int(self.config.context_lags)
        init = np.asarray(initial_history, dtype=np.float32)
        if init.shape != (L, self.z_dim):
            raise ValueError(f"initial_history must have shape {(L, self.z_dim)}, got {init.shape}")

        history = torch.tensor(init, dtype=torch.float32, device=dev)[None, :, :].repeat(int(n_paths), 1, 1)
        out = torch.empty(int(n_paths), int(path_length), self.z_dim, device=dev)

        for t in range(int(path_length)):
            context = history.reshape(int(n_paths), -1)
            delta = self.sample_increment(context)
            z_next = history[:, -1, :] + delta
            out[:, t, :] = z_next
            history = torch.cat([history[:, 1:, :], z_next[:, None, :]], dim=1)
        return out.cpu().numpy()

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "z_dim": self.z_dim,
                "config": self.config.__dict__,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device=None):
        ck = torch.load(path, map_location=device or "cpu", weights_only=False)
        model = cls(ck["z_dim"], TemporalFMConfig(**ck["config"]))
        model.load_state_dict(ck["state_dict"])
        if device is not None:
            model.to(device)
        return model
