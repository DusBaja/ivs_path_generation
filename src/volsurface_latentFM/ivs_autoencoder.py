"""
Stage 1: Arbitrage-Constrained VAE
====================================
Encoder/Decoder trained with:
  - Reconstruction loss (MSE)
  - KL regularisation (beta-warmup) to latent space ≈ N(0,I)
  - Calendar-spread penalty     to decoded surface non-decreasing in tau
  - Butterfly penalty           to decoded smile convex in m

After training, the encoder is frozen. Latent codes {z_i} are extracted
and used as the training set for any latent-space generative model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple



# Architecture
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net      = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                                      nn.Linear(out_dim, out_dim))
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    def forward(self, x):
        return F.relu(self.net(x) + self.shortcut(x))


class Encoder(nn.Module):
    def __init__(self, x_dim=512, z_dim=8, hidden=(256, 128, 64)):
        super().__init__()
        layers, prev = [], x_dim
        for h in hidden:
            layers.append(ResBlock(prev, h)); prev = h
        self.net    = nn.Sequential(*layers)
        self.mu     = nn.Linear(prev, z_dim)
        self.logvar = nn.Linear(prev, z_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, x_dim=512, z_dim=8, hidden=(64, 128, 256)):
        super().__init__()
        layers, prev = [], z_dim
        for h in hidden:
            layers.append(ResBlock(prev, h)); prev = h
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(prev, x_dim)

    def forward(self, z):
        return self.out(self.net(z))

def fit_normalizer(x_train: torch.Tensor):
    
    flat = x_train.flatten()
    lo = torch.quantile(flat, 0.04)
    hi = torch.quantile(flat, 0.96)# BEST AT 0.95 
    return lo.detach(), hi.detach()

def normalize(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
    den = (hi - lo).clamp_min(1e-8)
    x01 = (x - lo) / den
    return (x01 * 2.0 - 1.0).contiguous()

def denormalize(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
    x01 = (x + 1.0) / 2.0 #REMOVED Clamp for better queue!!!
    #x01 = (x.clamp(-1, 1) + 1.0) / 2.0
    return x01 * (hi - lo) + lo

# Arbitrage penalties  (computed on decoded surface in normalised space)

"""def _arbitrage_penalty(x_hat, n_tenor=16, n_money=32, lo=None, hi=None):
   """ """
    Soft arbitrage penalties on a batch of decoded surfaces.

    x_hat : (B, 512)  decoded surface in normalised space
    lo/hi : scalar tensors — surface normaliser. If provided, penalties are
            computed in % space for better calibration. Otherwise computed
            in normalised space (still a valid proxy).

    Returns
    -------
    l_cal : calendar-spread penalty  (scalar)
    l_but : butterfly penalty        (scalar)
    """"""
    # Reshape to (B, tenor, moneyness)
    s = x_hat.reshape(-1, n_tenor, n_money)

    # Optionally denormalise to get σ in % for interpretable penalties
    if lo is not None and hi is not None:
        s = (s+1)/2*(hi-lo)+lo           # → σ in %
        #s = s * (hi - lo) + lo 
    # ── Calendar spread ───────────────────────────────────────────────────
    # Total implied variance  w = σ² × τ  should be non-decreasing in τ.
    # Proxy: σ itself non-decreasing in τ at each moneyness
    # (exact condition would require τ values; this is a sufficient condition
    #  under the assumption σ is smooth and τ values are increasing)
    diff_tau = s[:, 1:, :] - s[:, :-1, :]          # (B, 15, 32)
    #l_cal    = F.relu(-diff_tau).pow(2).mean()
    neg_cal  = torch.clamp(-diff_tau, min=0, max=10)
    l_cal    = (torch.exp(neg_cal) - 1).mean()
    # ── Butterfly ─────────────────────────────────────────────────────────
    # Smile must be convex in m: ∂²σ/∂m² ≥ 0
    # Finite-difference second derivative along moneyness axis
    d2m   = s[:, :, 2:] - 2*s[:, :, 1:-1] + s[:, :, :-2]   # (B, 16, 30)
    #l_but = F.relu(-d2m).pow(2).mean()
    neg_but  = torch.clamp(-d2m, min=0, max=10)
    l_but    = (torch.exp(neg_but) - 1).mean()
    
    return l_cal, l_but
"""

def _to_tensor(v, dtype, device):
    if isinstance(v, torch.Tensor):
        return v.to(dtype=dtype, device=device)
    return torch.tensor(v, dtype=dtype, device=device)

def _arbitrage_penalty(x_hat, m_grid, tau_grid,
                        n_tenor=16, n_money=32, lo=None, hi=None):
    """
    Three no-arbitrage penalties.
 
    Calendar and call spread use direct σ differences — proven sufficient.
    Butterfly uses Gatheral g(k) ≥ 0 — exact necessary and sufficient condition
    for each maturity slice.
 
    Parameters
    ----------
    x_hat    : (B, 512)  decoded surface, normalised [-1,1]
    m_grid   : (M,)      actual moneyness K/S  (e.g. linspace(0.7, 1.3, 32))
    tau_grid : (T,)      maturities in years   (e.g. [1/12, ..., 1.0])
    lo / hi  : float     surface normaliser scalars
 
    Returns
    -------
    l_cal  : calendar penalty   (scalar)
    l_call : call-spread penalty (scalar)
    l_g    : Gatheral butterfly penalty (scalar)
    """
    m_grid   = _to_tensor(m_grid,   x_hat.dtype, x_hat.device)
    tau_grid = _to_tensor(tau_grid, x_hat.dtype, x_hat.device)
 
    s = x_hat.reshape(-1, n_tenor, n_money)
    if lo is not None and hi is not None:
        lo_ = _to_tensor(lo, s.dtype, s.device)
        hi_ = _to_tensor(hi, s.dtype, s.device)
        s   = (s + 1) / 2 * (hi_ - lo_) + lo_
 
    # ── 1. Calendar: ∂_τ σ ≥ 0 ──────────
    diff_tau = s[:, 1:, :] - s[:, :-1, :]          # (B, T-1, M)
    l_cal    = F.relu(-diff_tau).mean()
 
    """# ── 2. Call spread: ∂_m σ ≤ 0  (sufficient for ∂_K C ≤ 0) ──────────
    diff_m   = s[:, :, 1:] - s[:, :, :-1]          # (B, T, M-1)
    l_call   = F.relu(diff_m).mean()"""
 
    # ── 3. Butterfly: Gatheral g(k) >= 0  ───────────────
    # Total variance w = (sigma/100)^2 x tau
    vol = s / 100                                   
    tau = tau_grid[None, :, None]                   # (1, T, 1)
    w   = vol**2 * tau                              # (B, T, M)
 
    # Log-moneyness k = log(K/S)
    k  = torch.log(m_grid)                          # (M,) 

    dk = float((k[-1] - k[0]).item() / (len(k) - 1))
 
    w_l = w[:, :, :-2]                             # (B, T, M-2)
    w_c = w[:, :, 1:-1]
    w_r = w[:, :, 2:]
 
    # First and second derivatives via differences
    w_p  = (w_r - w_l) / (2 * dk)                  # ∂_k w
    w_pp = (w_r - 2 * w_c + w_l) / (dk ** 2)       # ∂²_k w
 
    k_c  = k[1:-1][None, None, :]                   # (1, 1, M-2)
 
    w_cs = w_c.clamp(min=1e-4)
 
    # Gatheral 
    g = ((1 - k_c * w_p / (2 * w_cs)) ** 2
         - (w_p ** 2 / 4) * (1 / w_cs + 0.25)
         + w_pp / 2)                                # (B, T, M-2)
 
    l_g = F.relu(-g).mean()
 
    return l_cal, l_g

# VAE
class IVS_AE(nn.Module):
    """
    Arbitrage-constrained β-VAE.

    Parameters
    ----------
    z_dim      : latent dimension (8 recommended for approx 70 training surfaces)
    beta       : target KL weight (after warmup)
    lambda_cal : calendar-spread penalty weight
    lambda_but : butterfly penalty weight
    """
    def __init__(self, x_dim=512, z_dim=8,
                 hidden_enc=(256, 128, 64), hidden_dec=(64, 128, 256),
                 beta=1e-3, lambda_cal=1e-2, lambda_but=1e-3):
        super().__init__()
        self.z_dim      = z_dim
        self.beta       = beta
        self.lambda_cal = lambda_cal
        self.lambda_but = lambda_but

        self.encoder = Encoder(x_dim, z_dim, hidden_enc)
        self.decoder = Decoder(x_dim, z_dim, hidden_dec)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"IVS_AE  |  z_dim={z_dim}  β={beta}  "
              f"λ_cal={lambda_cal}  λ_but={lambda_but}  "
              f"params={n_params:,}")

    @staticmethod
    def reparametrize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.reparametrize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss(self, x, x_hat, m_grid, tau_grid, mu, logvar, beta_now, lo=None, hi=None):
        recon    = F.mse_loss(x_hat, x)
        kl       = -0.5 * torch.mean(1 + logvar - torch.pow(mu, 2) - logvar.exp())
        l_cal, l_but = _arbitrage_penalty(x_hat,m_grid, tau_grid, lo=lo, hi=hi)
        total    = (recon
                    + beta_now       * kl
                    + self.lambda_cal * l_cal
                    + self.lambda_but * l_but)
        return total, recon, kl, l_cal, l_but

    # Training

    def fit(self, surf_train_n, m_grid, tau_grid, surf_val_n=None,
            n_epochs=3000, batch_size=16, lr=1e-3,
            warmup_epochs=500, patience=400, print_every=300,
            lo=None, hi=None,
            device=None):
        """
        surf_train_n : (n, 1, 16, 32) or (n, 512)  normalised surfaces in %
        lo / hi      : surface normaliser scalars (passed to arbitrage penalty
                       for calibrated thresholds); optional but recommended
        """
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)

        lo_t = torch.tensor(lo, dtype=torch.float32, device=dev) if lo is not None else None
        hi_t = torch.tensor(hi, dtype=torch.float32, device=dev) if hi is not None else None

        def prep(s):
            a = np.asarray(s, dtype=np.float32)
            if a.ndim == 4: a = a[:, 0]
            return torch.tensor(a.reshape(len(a), -1)).to(dev)

        x_tr  = prep(surf_train_n)
        loader = DataLoader(TensorDataset(x_tr), batch_size=batch_size,
                            shuffle=True, drop_last=False)
        has_val = surf_val_n is not None
        if has_val:
            x_val = prep(surf_val_n)

        opt  = torch.optim.Adam(self.parameters(), lr=lr)
        hist = {k: [] for k in ["loss","recon","kl","cal","but","val_loss"]}
        best_val, best_state, no_improve = float("inf"), None, 0

        for epoch in range(1, n_epochs + 1):
            beta_now = self.beta * min(1.0, epoch / max(warmup_epochs, 1))
            self.train()
            ep = {k: 0. for k in ["loss","recon","kl","cal","but"]}
            for (xb,) in loader:
                opt.zero_grad()
                x_hat, mu, lv = self(xb)
                tot, recon, kl, cal, but_ = self.loss(xb, x_hat, m_grid, tau_grid, mu, lv, beta_now, lo_t, hi_t)
                tot.backward(); opt.step()
                ep["loss"]+=tot.item(); ep["recon"]+=recon.item()
                ep["kl"]+=kl.item(); ep["cal"]+=cal.item(); ep["but"]+=but_.item()
            n = len(loader)
            for k in ep: hist[k].append(ep[k]/n)
            hist["val_loss"].append(None)

            if has_val:
                self.eval()
                with torch.no_grad():
                    xh, mu, lv = self(x_val)
                    vl,*_ = self.loss(x_val, xh, m_grid, tau_grid, mu, lv, beta_now, lo_t, hi_t)
                hist["val_loss"][-1] = vl.item()
                if vl.item() < best_val - 1e-7:
                    best_val   = vl.item()
                    best_state = {k: v.cpu().clone() for k,v in self.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    print(f"\n  Early stop  epoch={epoch}  best_val={best_val:.5f}")
                    break

            if epoch % print_every == 0 or epoch == 1:
                msg = (f"  Ep {epoch:>5}  recon={hist['recon'][-1]:.5f}  "
                       f"kl={hist['kl'][-1]:.4f}  "
                       f"cal={hist['cal'][-1]:.5f}  but={hist['but'][-1]:.5f}  "
                       f"β={beta_now:.1e}")
                if has_val and hist["val_loss"][-1]:
                    msg += f"  val={hist['val_loss'][-1]:.5f}"
                print(msg)

        if best_state:
            self.load_state_dict({k: v.to(dev) for k,v in best_state.items()})
            print(f"  Restored best  val={best_val:.5f}")
        return hist

    # Encode dataset to latent codes

    @torch.no_grad()
    def encode(self, surf_n, device=None, use_mean=True):
        """
        Extract latent codes from normalised surfaces.

        use_mean=True  to use mu (deterministic, recommended for downstream training)
        use_mean=False to sample z following q(z|x)

        Returns
        -------
        z : (n, z_dim)  numpy array
        """
        dev = device or next(self.parameters()).device
        self.eval()
        a = np.asarray(surf_n, dtype=np.float32)
        if a.ndim == 4: a = a[:, 0]
        x = torch.tensor(a.reshape(len(a), -1)).to(dev)
        mu, logvar = self.encoder(x)
        z = mu if use_mean else self.reparametrize(mu, logvar)
        return z.cpu().numpy()

    @torch.no_grad()
    def decode(self, z, device=None):
        """z : (n, z_dim) numpy or tensor → (n, 512) tensor"""
        dev = device or next(self.parameters()).device
        self.eval()
        zt = torch.tensor(np.asarray(z, dtype=np.float32)).to(dev)
        return self.decoder(zt)

    # Persistence

    def save(self, path, surf_lo=None, surf_hi=None):
        torch.save(dict(
            state_dict  = self.state_dict(),
            z_dim       = self.z_dim,
            beta        = self.beta,
            lambda_cal  = self.lambda_cal,
            lambda_but  = self.lambda_but,
            surf_lo     = surf_lo,
            surf_hi     = surf_hi,
        ), path)
        print(f"  Saved AE → {path}")

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location=device or "cpu", weights_only=False)
        ae = cls(z_dim=ck["z_dim"], beta=ck["beta"],
                 lambda_cal=ck["lambda_cal"], lambda_but=ck["lambda_but"])
        ae.load_state_dict(ck["state_dict"])
        ae.surf_lo = ck.get("surf_lo")
        ae.surf_hi = ck.get("surf_hi")
        return ae
