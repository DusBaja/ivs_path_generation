from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, CubicSpline, LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import least_squares
from scipy.special import ndtr


_DATE_IN_FILENAME = re.compile(
    r"^(0[1-9]|[12]\d|3[01])(0[1-9]|1[0-2])(\d{4})\.xlsx$",
    re.IGNORECASE,
)


def parse_asof_from_filename(path: Path) -> pd.Timestamp:
    m = _DATE_IN_FILENAME.match(path.name)
    if not m:
        raise ValueError(f"File name not in ddmmyyyy.xlsx format: {path.name}")
    dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
    return pd.Timestamp(f"{yyyy}-{mm}-{dd}")


_EXPIRY_DATE_RE = re.compile(r"(\d{1,2})-([A-Za-z]{3})-(\d{2,4})")
_FWD_RE = re.compile(r"\bifwd\b\s*([0-9]+(?:[.,][0-9]+)?)", re.IGNORECASE)

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_expiry_from_row(row: List[object]) -> Optional[pd.Timestamp]:
    for cell in row:
        if pd.isna(cell):
            continue
        s = str(cell).strip()
        m = _EXPIRY_DATE_RE.search(s)
        if not m:
            continue
        dd = int(m.group(1))
        mon = _MONTHS.get(m.group(2).upper())
        yy = m.group(3)
        if mon is None:
            continue
        yyyy = int(yy)
        if yyyy < 100:
            yyyy += 2000
        try:
            return pd.Timestamp(year=yyyy, month=mon, day=dd)
        except Exception:
            continue
    return None


def to_float(x) -> Optional[float]:
    if pd.isna(x):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            return float(x)
        except Exception:
            return None
    s = str(x).strip()
    if s == "":
        return None
    s = s.replace(" ", "")
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    s = s.replace("%", "")
    try:
        return float(s)
    except Exception:
        return None


def parse_ifwd_from_row(row: List[object]) -> Optional[float]:
    for cell in row:
        if pd.isna(cell):
            continue
        s = str(cell)
        m = _FWD_RE.search(s)
        if m:
            return to_float(m.group(1))
    return None


def _norm(x) -> str:
    return str(x).strip().lower() if not pd.isna(x) else ""


def is_header_row(row: List[object]) -> bool:
    cells = [_norm(c) for c in row]
    return cells.count("strike") >= 2 and cells.count("ivm") >= 2


def find_header_row(df0: pd.DataFrame, max_scan: int = 120) -> int:
    for r in range(min(max_scan, len(df0))):
        if is_header_row(df0.iloc[r, :].tolist()):
            return r
    raise ValueError("Could not find header row with 2x(Strike, IVM).")


def locate_calls_puts_columns(header_row: List[object]) -> Tuple[dict, dict, bool]:
    cells = [_norm(c) for c in header_row]
    strike_pos = [i for i, c in enumerate(cells) if c == "strike"]
    ivm_pos = [i for i, c in enumerate(cells) if c == "ivm"]
    dm_pos = [i for i, c in enumerate(cells) if c == "dm"]

    if len(strike_pos) < 2 or len(ivm_pos) < 2:
        raise ValueError(
            f"Header found but missing 2x strike/ivm. strike_pos={strike_pos}, ivm_pos={ivm_pos}"
        )

    has_dm = len(dm_pos) >= 2
    calls = {"strike": strike_pos[0], "ivm": ivm_pos[0], "dm": (dm_pos[0] if has_dm else None)}
    puts = {"strike": strike_pos[1], "ivm": ivm_pos[1], "dm": (dm_pos[1] if has_dm else None)}
    return calls, puts, has_dm


def extract_points_from_excel(
    xlsx_path: Path,
    asof: pd.Timestamp,
    sheet_name: int | str = 0,
) -> pd.DataFrame:
    df0 = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    df0 = df0.iloc[:, :80].copy()

    header_r = find_header_row(df0)
    header = df0.iloc[header_r, :].tolist()
    calls_idx, puts_idx, has_dm = locate_calls_puts_columns(header)

    points = []
    current_expiry: Optional[pd.Timestamp] = None
    current_ifwd: Optional[float] = None

    for r in range(header_r + 1, df0.shape[0]):
        row = df0.iloc[r, :].tolist()

        expiry = parse_expiry_from_row(row)
        if expiry is not None:
            current_expiry = expiry
            fwd = parse_ifwd_from_row(row)
            if fwd is not None:
                current_ifwd = fwd
            continue

        if current_expiry is None:
            continue

        tenor = (current_expiry - asof).days
        if tenor <= 0:
            continue

        cK = to_float(row[calls_idx["strike"]] if calls_idx["strike"] < len(row) else None)
        if cK is not None:
            cIV = to_float(row[calls_idx["ivm"]] if calls_idx["ivm"] < len(row) else None)
            cDM = None
            if has_dm and calls_idx["dm"] is not None and calls_idx["dm"] < len(row):
                cDM = to_float(row[calls_idx["dm"]])
            if cIV is not None:
                points.append(
                    dict(
                        asof=asof,
                        tenor_days=float(tenor),
                        strike=float(cK),
                        iv=float(cIV),
                        dm=cDM,
                        ifwd=current_ifwd,
                        side="call",
                    )
                )

        pK = to_float(row[puts_idx["strike"]] if puts_idx["strike"] < len(row) else None)
        if pK is not None:
            pIV = to_float(row[puts_idx["ivm"]] if puts_idx["ivm"] < len(row) else None)
            pDM = None
            if has_dm and puts_idx["dm"] is not None and puts_idx["dm"] < len(row):
                pDM = to_float(row[puts_idx["dm"]])
            if pIV is not None:
                points.append(
                    dict(
                        asof=asof,
                        tenor_days=float(tenor),
                        strike=float(pK),
                        iv=float(pIV),
                        dm=pDM,
                        ifwd=current_ifwd,
                        side="put",
                    )
                )

    out = pd.DataFrame(points)
    if out.empty:
        raise ValueError(
            f"No points extracted from {xlsx_path.name}. Most likely: wrong sheet index, or header row differs."
        )
    return out



# ATM proxy

def infer_k_atm_from_dm(points: pd.DataFrame) -> float:
    candidates: list[float] = []

    calls = points[(points["side"] == "call") & points["dm"].notna()]
    if not calls.empty:
        dm = calls["dm"].to_numpy(dtype=float)
        i = int(np.nanargmin(np.abs(dm - 0.5)))
        candidates.append(float(calls.iloc[i]["strike"]))

    puts = points[(points["side"] == "put") & points["dm"].notna()]
    if not puts.empty:
        dm = puts["dm"].to_numpy(dtype=float)
        i_neg = int(np.nanargmin(np.abs(dm + 0.5)))
        candidates.append(float(puts.iloc[i_neg]["strike"]))
        i_pos = int(np.nanargmin(np.abs(dm - 0.5)))
        candidates.append(float(puts.iloc[i_pos]["strike"]))

    if candidates:
        return float(np.nanmean(candidates))

    return float(np.median(points["strike"].to_numpy(dtype=float)))


def infer_k_atm(points: pd.DataFrame) -> float:
    if "dm" in points.columns and points["dm"].notna().any():
        return infer_k_atm_from_dm(points)
    if "ifwd" in points.columns and points["ifwd"].notna().any():
        return float(np.nanmedian(points["ifwd"].to_numpy(dtype=float)))
    return float(np.median(points["strike"].to_numpy(dtype=float)))



# Smile models
def _smooth_smile_on_grid(m: np.ndarray, iv: np.ndarray, m_grid: np.ndarray, kind: str = "pchip") -> np.ndarray:
    m = np.asarray(m, dtype=float)
    iv = np.asarray(iv, dtype=float)
    mask = np.isfinite(m) & np.isfinite(iv) & (m > 0) & (iv > 0)
    m = m[mask]
    iv = iv[mask]

    if len(m) == 0:
        return np.full_like(m_grid, np.nan, dtype=float)

    order = np.argsort(m)
    m = m[order]
    iv = iv[order]

    um, idx = np.unique(m, return_inverse=True)
    if len(um) != len(m):
        iv2 = np.zeros_like(um, dtype=float)
        cnt = np.zeros_like(um, dtype=float)
        for k in range(len(m)):
            iv2[idx[k]] += iv[k]
            cnt[idx[k]] += 1
        iv = iv2 / np.maximum(cnt, 1)
        m = um

    if len(m) == 1:
        return np.full_like(m_grid, float(iv[0]), dtype=float)

    if len(m) < 3:
        return np.interp(m_grid, m, iv)

    if kind == "cubic":
        f = CubicSpline(m, iv, bc_type="natural", extrapolate=True)
        out = f(m_grid)
    else:
        f = PchipInterpolator(m, iv, extrapolate=True)
        out = f(m_grid)

    return np.asarray(out, dtype=float)


# SVI raw: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
def svi_raw_w(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    x = k - m
    return a + b * (rho * x + np.sqrt(x * x + sigma * sigma))


def _relative_call_from_total_variance(m_grid: np.ndarray, tau: float, w: np.ndarray) -> np.ndarray:
    """
    Relative BS call price on m=K/S grid, assuming zero rates.
    Used only for an optional SVI fit penalty, not for raw quote conversion.
    """
    m_grid = np.asarray(m_grid, dtype=float)
    tau = max(float(tau), 1e-12)
    w = np.maximum(np.asarray(w, dtype=float), 1e-12)
    vol_sqrt_t = np.sqrt(w)
    d1 = (-np.log(m_grid) + 0.5 * w) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return ndtr(d1) - m_grid * ndtr(d2)


def _svi_static_arb_residual(
    x: np.ndarray,
    k_grid: np.ndarray,
    tau: float,
    lambda_mono: float,
    lambda_bfly: float,
) -> np.ndarray:
    """
    Soft penalty for call monotonicity and convexity on the SVI curve.
    This keeps the model SVI, but discourages the bad strike-shape behaviour.
    """
    if (lambda_mono <= 0) and (lambda_bfly <= 0):
        return np.empty(0, dtype=float)

    a, b, rho, m0, sig = x
    m_grid = np.exp(k_grid)
    w_grid = svi_raw_w(k_grid, a, b, rho, m0, sig)
    if not np.all(np.isfinite(w_grid)):
        return np.full(1, 1e6, dtype=float)

    C = _relative_call_from_total_variance(m_grid, tau, w_grid)
    dm = np.diff(m_grid)
    res_parts = []

    if lambda_mono > 0 and len(C) >= 2:
        # Call price must decrease with m=K/S. Violation: positive slope.
        mono = np.maximum(0.0, np.diff(C) / dm)
        res_parts.append(np.sqrt(lambda_mono) * mono)

    if lambda_bfly > 0 and len(C) >= 3:
        # Call price must be convex in m. Violation: slope decreases.
        left = (C[1:-1] - C[:-2]) / dm[:-1]
        right = (C[2:] - C[1:-1]) / dm[1:]
        bfly = np.maximum(0.0, left - right)
        res_parts.append(np.sqrt(lambda_bfly) * bfly)

    if not res_parts:
        return np.empty(0, dtype=float)
    return np.concatenate(res_parts)


def fit_svi_raw(
    k: np.ndarray,
    w: np.ndarray,
    wgt: Optional[np.ndarray] = None,
    max_nfev: int = 1000,
    arb_penalty: bool = False,
    k_penalty_grid: Optional[np.ndarray] = None,
    T_years: Optional[float] = None,
    lambda_mono: float = 0.0,
    lambda_bfly: float = 0.0,
) -> Tuple[float, float, float, float, float]:
    """
    Fit raw SVI parameters to data.
    Defaults reproduce the old behaviour: single-start bounded least squares,
    no static-arbitrage penalty. The optional penalty is deliberately opt-in.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(k) & np.isfinite(w) & (w > 0)
    k = k[mask]
    w = w[mask]
    if wgt is None:
        wgt = np.ones_like(w)
    else:
        wgt = np.asarray(wgt, dtype=float)[mask]

    if len(k) < 3:
        w0 = float(np.nanmean(w)) if len(w) else 0.0
        return (w0, 1e-6, 0.0, float(np.nanmean(k)) if len(k) else 0.0, 0.1)

    a0 = float(np.nanmin(w)) * 0.9
    b0 = max(1e-4, 0.5 * (float(np.nanmax(w)) - float(np.nanmin(w))) / max(1e-4, np.ptp(k)))
    rho0 = 0.0
    m0 = float(np.nanmean(k))
    sigma0 = max(1e-3, 0.5 * float(np.nanstd(k)) if np.nanstd(k) > 0 else 0.1)

    x0 = np.array([a0, b0, rho0, m0, sigma0], dtype=float)
    lb = np.array([-1.0, 1e-8, -0.999, -2.0, 1e-6], dtype=float)
    ub = np.array([10.0, 10.0, 0.999, 2.0, 5.0], dtype=float)

    use_arb_penalty = bool(
        arb_penalty
        and k_penalty_grid is not None
        and T_years is not None
        and (lambda_mono > 0 or lambda_bfly > 0)
    )

    def resid(x: np.ndarray) -> np.ndarray:
        a, b, rho, m_fit, sig = x
        r = svi_raw_w(k, a, b, rho, m_fit, sig) - w
        base = np.sqrt(np.maximum(wgt, 1e-12)) * r
        if not use_arb_penalty:
            return base
        arb = _svi_static_arb_residual(
            x=x,
            k_grid=np.asarray(k_penalty_grid, dtype=float),
            tau=float(T_years),
            lambda_mono=float(lambda_mono),
            lambda_bfly=float(lambda_bfly),
        )
        return np.concatenate([base, arb])

    res = least_squares(
        resid,
        x0=x0,
        bounds=(lb, ub),
        method="trf",
        max_nfev=int(max_nfev),
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    a, b, rho, m_fit, sig = res.x
    return float(a), float(b), float(rho), float(m_fit), float(sig)


def svi_iv_on_m_grid(
    m_grid: np.ndarray,
    m_obs: np.ndarray,
    iv_obs_pct: np.ndarray,
    T_years: float,
    max_nfev: int = 1000,
    arb_penalty: bool = False,
    lambda_mono: float = 0.0,
    lambda_bfly: float = 0.0,
) -> np.ndarray:
    """
    Fit SVI in total variance on log-moneyness k=log(m) and evaluate on m_grid.
    IV inputs/outputs are in percent.
    """
    m_obs = np.asarray(m_obs, dtype=float)
    iv_obs_pct = np.asarray(iv_obs_pct, dtype=float)
    mask = np.isfinite(m_obs) & np.isfinite(iv_obs_pct) & (m_obs > 0) & (iv_obs_pct > 0)
    m_obs = m_obs[mask]
    iv_obs_pct = iv_obs_pct[mask]

    if len(m_obs) < 3:
        return _smooth_smile_on_grid(m_obs, iv_obs_pct, m_grid, kind="pchip")

    k = np.log(m_obs)
    iv = iv_obs_pct / 100.0
    w = np.maximum((iv * iv) * max(float(T_years), 1e-12), 1e-12)
    wgt = np.ones_like(w)

    a, b, rho, m0, sig = fit_svi_raw(
        k,
        w,
        wgt=wgt,
        max_nfev=max_nfev,
        arb_penalty=arb_penalty,
        k_penalty_grid=np.log(np.asarray(m_grid, dtype=float)),
        T_years=T_years,
        lambda_mono=lambda_mono,
        lambda_bfly=lambda_bfly,
    )

    k_grid = np.log(np.asarray(m_grid, dtype=float))
    w_grid = svi_raw_w(k_grid, a, b, rho, m0, sig)
    w_grid = np.maximum(w_grid, 1e-12)

    iv_grid = np.sqrt(w_grid / max(float(T_years), 1e-12)) * 100.0
    return iv_grid


# Tenor interpolation + calendar enforcement
def _interp_along_tenor(tenors_src: np.ndarray, values_src: np.ndarray, tenors_tgt: np.ndarray) -> np.ndarray:
    mask = np.isfinite(values_src) & np.isfinite(tenors_src)
    t = tenors_src[mask]
    v = values_src[mask]
    if len(t) == 0:
        return np.full_like(tenors_tgt, np.nan, dtype=float)
    if len(t) == 1:
        return np.full_like(tenors_tgt, float(v[0]), dtype=float)

    order = np.argsort(t)
    t = t[order]
    v = v[order]

    ut, idx = np.unique(t, return_inverse=True)
    if len(ut) != len(t):
        v2 = np.zeros_like(ut, dtype=float)
        cnt = np.zeros_like(ut, dtype=float)
        for k in range(len(t)):
            v2[idx[k]] += v[k]
            cnt[idx[k]] += 1
        t = ut
        v = v2 / np.maximum(cnt, 1)

    if len(t) == 1:
        return np.full_like(tenors_tgt, float(v[0]), dtype=float)

    f = PchipInterpolator(t, v, extrapolate=False)
    return np.asarray(f(tenors_tgt), dtype=float)

#total var is better
'''
def _interp_iv_along_tenor_total_variance(
    tenors_src: np.ndarray,
    iv_src_pct: np.ndarray,
    tenors_tgt: np.ndarray,
) -> np.ndarray:
    tenors_src = np.asarray(tenors_src, dtype=float)
    tenors_tgt = np.asarray(tenors_tgt, dtype=float)
    iv_src_pct = np.asarray(iv_src_pct, dtype=float)

    mask = (
        np.isfinite(tenors_src)
        & np.isfinite(iv_src_pct)
        & (tenors_src > 0)
        & (iv_src_pct > 0)
    )

    t = tenors_src[mask]
    iv = iv_src_pct[mask]

    if len(t) == 0:
        return np.full_like(tenors_tgt, np.nan, dtype=float)

    if len(t) == 1:
        return np.full_like(tenors_tgt, float(iv[0]), dtype=float)

    order = np.argsort(t)
    t = t[order]
    iv = iv[order]

    tau = t / 365.0
    w = (iv / 100.0) ** 2 * tau

    # Calendar no-arb on observed tenor points.
    w = np.maximum.accumulate(w)

    f = PchipInterpolator(t, w, extrapolate=False)
    w_tgt = np.asarray(f(tenors_tgt), dtype=float)

    # Fill missing edges with nearest available total variance.
    missing = ~np.isfinite(w_tgt)
    if np.any(missing):
        ok = np.isfinite(w_tgt)
        if ok.any():
            idx_ok = np.where(ok)[0]
            for j in np.where(missing)[0]:
                nearest = idx_ok[np.argmin(np.abs(idx_ok - j))]
                w_tgt[j] = w_tgt[nearest]
        else:
            w_tgt[:] = float(np.nanmedian(w))

    # Calendar no-arb again on final 16-tenor grid.
    w_tgt = np.maximum.accumulate(w_tgt)

    tau_tgt = np.maximum(tenors_tgt / 365.0, 1e-12)
    iv_tgt = np.sqrt(np.maximum(w_tgt, 1e-12) / tau_tgt) * 100.0

    return iv_tgt
'''
def _interp_iv_along_tenor_total_variance(
    tenors_src: np.ndarray,
    iv_src_pct: np.ndarray,
    tenors_tgt: np.ndarray,
) -> np.ndarray:
    tenors_src = np.asarray(tenors_src, dtype=float)
    tenors_tgt = np.asarray(tenors_tgt, dtype=float)
    iv_src_pct = np.asarray(iv_src_pct, dtype=float)

    mask = (
        np.isfinite(tenors_src)
        & np.isfinite(iv_src_pct)
        & (tenors_src > 0)
        & (iv_src_pct > 0)
    )

    t = tenors_src[mask]
    iv = iv_src_pct[mask]

    if len(t) == 0:
        return np.full_like(tenors_tgt, np.nan, dtype=float)

    order = np.argsort(t)
    t = t[order]
    iv = iv[order]

    # Remove duplicate tenors, averaging IVs.
    ut, inv = np.unique(t, return_inverse=True)
    if len(ut) != len(t):
        iv2 = np.zeros_like(ut, dtype=float)
        cnt = np.zeros_like(ut, dtype=float)
        for k in range(len(t)):
            iv2[inv[k]] += iv[k]
            cnt[inv[k]] += 1.0
        t = ut
        iv = iv2 / np.maximum(cnt, 1.0)

    tau = t / 365.0
    sigma2 = (iv / 100.0) ** 2
    w = sigma2 * tau

    # Calendar no-arb on observed points.
    w = np.maximum.accumulate(w)

    tau_tgt = np.maximum(tenors_tgt / 365.0, 1e-12)
    w_tgt = np.full_like(tenors_tgt, np.nan, dtype=float)

    if len(t) == 1:
        # Flat IV extrapolation everywhere.
        sigma2_flat = w[0] / tau[0]
        w_tgt = sigma2_flat * tau_tgt
    else:
        # Interpolate total variance only inside observed tenor range.
        inside = (tenors_tgt >= t[0]) & (tenors_tgt <= t[-1])

        f = PchipInterpolator(t, w, extrapolate=False)
        w_tgt[inside] = f(tenors_tgt[inside])

        # LEFT extrapolation: flat IV, not flat total variance.
        left = tenors_tgt < t[0]
        if np.any(left):
            sigma2_left = w[0] / tau[0]
            w_tgt[left] = sigma2_left * tau_tgt[left]

        # RIGHT extrapolation: flat IV from last observed tenor.
        # This keeps total variance increasing with maturity.
        right = tenors_tgt > t[-1]
        if np.any(right):
            sigma2_right = w[-1] / tau[-1]
            w_tgt[right] = sigma2_right * tau_tgt[right]

    # Final calendar safety.
    w_tgt = np.maximum.accumulate(np.maximum(w_tgt, 1e-12))

    iv_tgt = np.sqrt(w_tgt / tau_tgt) * 100.0
    return iv_tgt
def enforce_calendar_no_arb_iv(surf_iv_pct: np.ndarray, tenor_days: np.ndarray) -> np.ndarray:
    """
    Enforce w(T,m) non-decreasing in T by cummax in total variance.
    surf_iv_pct shape: (n_m, n_T), IV in percent.
    """
    T = np.asarray(tenor_days, dtype=float) / 365.0
    T = np.maximum(T, 1e-12)

    iv = np.asarray(surf_iv_pct, dtype=float) / 100.0
    w = (iv * iv) * T[None, :]
    w2 = np.maximum.accumulate(w, axis=1)

    iv2 = np.sqrt(w2 / T[None, :]) * 100.0
    return iv2
'''
def make_target_tenor_grid_from_observed(
    observed_tenors: np.ndarray,
    nt: int = 16,
    method: str = "quantile",
) -> np.ndarray:
    t = np.asarray(observed_tenors, dtype=float)
    t = np.sort(np.unique(t[np.isfinite(t) & (t > 0)]))

    if len(t) == 0:
        raise ValueError("No observed tenor_days found.")

    if len(t) == nt:
        return t.astype(float)

    if method == "quantile":
        grid = np.quantile(t, np.linspace(0.0, 1.0, nt))
        grid = np.round(grid)

    elif method == "linear":
        grid = np.linspace(t[0], t[-1], nt)
        grid = np.round(grid)

    elif method == "log":
        grid = np.exp(np.linspace(np.log(t[0]), np.log(t[-1]), nt))
        grid = np.round(grid)

    else:
        raise ValueError("method must be 'quantile', 'linear', or 'log'")

    grid = np.sort(np.unique(grid.astype(float)))

    if len(grid) != nt:
        grid = np.linspace(t[0], t[-1], nt)

    return grid.astype(float)
'''
def make_target_tenor_grid_from_observed(
    observed_tenors: np.ndarray,
    nt: int = 16,
    method: str = "quantile",
) -> np.ndarray:
    t = np.asarray(observed_tenors, dtype=float)
    t = t[np.isfinite(t) & (t > 0)]

    # Important: avoid very short and very long expiries.
    # The cube target maturity range should be stable.
    t_min = 7.0
    t_max = 365.0
    t = t[(t >= t_min) & (t <= t_max)]

    t = np.sort(np.unique(t))

    if len(t) == 0:
        raise ValueError(
            "No observed tenor_days found inside [7, 365]. "
            "Check expiry parsing or change t_min/t_max."
        )

    if len(t) == nt:
        return t.astype(float)

    if len(t) < nt:
        # Not enough distinct observed tenors, so create 16 points
        # between the observed min and max inside [7, 365].
        return np.linspace(float(t[0]), float(t[-1]), nt)

    if method == "quantile":
        grid = np.quantile(t, np.linspace(0.0, 1.0, nt))
    elif method == "linear":
        grid = np.linspace(float(t[0]), float(t[-1]), nt)
    elif method == "log":
        grid = np.exp(np.linspace(np.log(float(t[0])), np.log(float(t[-1])), nt))
    else:
        raise ValueError("method must be 'quantile', 'linear', or 'log'")

    grid = np.round(grid).astype(float)
    grid = np.sort(np.unique(grid))

    if len(grid) != nt:
        # If rounding created duplicates, fallback to a clean 16-point grid.
        grid = np.linspace(float(t[0]), float(t[-1]), nt)

    return np.asarray(grid, dtype=float).ravel()


@dataclass(frozen=True)
class SurfaceCube:
    cube: np.ndarray          # (n_moneyness, n_tenor, n_dates), IV percent
    m_grid: np.ndarray
    tenor_days: np.ndarray
    asof_dates: list[pd.Timestamp]


def build_surface_cube(
    data_dir: str | Path,
    m_grid: np.ndarray | None = None,
    tenor_days: np.ndarray | None = None,
    sheet_name: int | str = 0,
    method: str = "pchip",
    fill_method: str | None = "nearest",
    m_min: float = 0.90,
    m_max: float = 1.10,
    average_call_put: bool = True,
    smile_model: str = "svi",
    enforce_calendar: bool = True,
    max_files: int | None = None,
    verbose: bool = False,
    svi_max_nfev: int = 1000,
    svi_arb_penalty: bool = True,
    svi_lambda_mono: float = 100.0,
    svi_lambda_bfly: float = 100.0,
) -> SurfaceCube:
    """
    Old SVI-first cube builder, with additive safety/debug controls only.

    Defaults keep the original, better baseline:
      - m_min=0.90, m_max=1.10 # now 0.70-1.30
      - average_call_put=True
      - smile_model='svi'
      - calendar enforcement in total variance

    Optional improvement:
      set svi_arb_penalty=True with small lambdas to keep SVI while discouraging
      call-price monotonicity / convexity violations during the SVI fit.
    """
    data_dir = Path(data_dir)

    if m_grid is None:
        m_grid = np.linspace(m_min, m_max, 32)
    else:
        m_grid = np.asarray(m_grid, dtype=float)

    
        ### We do not build tenor days yet if it is none, we loook at the files to extract the observed ones 
        #tenor_days = np.array([7, 14, 21, 30, 45, 60, 90, 125,150, 180, 235, 270, 365], dtype=float)
        ##observed_tenors_all=np.concatenate([pts["tenor_days"].to_numpy(dtype=float) for pts in per_date_pts])
        ##tenor_days = make_target_tenor_grid_from_observed(observed_tenors_all,nt=16,method="quantile")
    ##else:
    if tenor_days is not None:
        tenor_days = np.asarray(tenor_days, dtype=float).ravel()

    files = sorted([p for p in data_dir.glob("*.xlsx") if _DATE_IN_FILENAME.match(p.name)])
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No ddmmyyyy.xlsx files found in {data_dir}")

    asof_dates: list[pd.Timestamp] = []
    per_date_pts: list[pd.DataFrame] = []

    for i_file, f in enumerate(files, start=1):
        if verbose:
            print(f"[{i_file}/{len(files)}] reading {f.name}")

        asof = parse_asof_from_filename(f)
        pts = extract_points_from_excel(f, asof=asof, sheet_name=sheet_name)

        k_atm = infer_k_atm(pts)
        pts = pts.copy()
        pts["moneyness"] = pts["strike"] / k_atm

        pts = pts[(pts["moneyness"] >= m_min) & (pts["moneyness"] <= m_max)]
        if pts.empty:
            raise ValueError(f"{f.name}: empty after moneyness filter [{m_min},{m_max}] (K_ref={k_atm}).")

        if average_call_put:
            pts = (
                pts.groupby(["tenor_days", "moneyness"], as_index=False)["iv"]
                .mean()
                .assign(asof=asof)
            )
        else:
            pts = pts[["asof", "tenor_days", "moneyness", "iv"]].copy()

        asof_dates.append(asof)
        per_date_pts.append(pts)

    order = np.argsort(asof_dates)
    asof_dates = [asof_dates[i] for i in order]
    per_date_pts = [per_date_pts[i] for i in order]
    #correcting the above: we extract the tenors in our files 
    if tenor_days is None:
        observed_tenors_all = np.concatenate(
            [pts["tenor_days"].to_numpy(dtype=float) for pts in per_date_pts]
        )

        tenor_days = make_target_tenor_grid_from_observed(
            observed_tenors_all,
            nt=16,
            method="quantile",
        )

        if verbose:
            print(f"target tenor grid inferred from files: {tenor_days}")
    cube = np.full((len(m_grid), len(tenor_days), len(asof_dates)), np.nan, dtype=float)
    if tenor_days is None or len(tenor_days) != 16 or not np.all(np.isfinite(tenor_days)):
        raise ValueError(f"Invalid tenor_days before cube allocation: {tenor_days}")
    for k_date, pts in enumerate(per_date_pts):
        if verbose:
            print(f"building surface {k_date + 1}/{len(per_date_pts)}: {asof_dates[k_date].date()}")

        tenors_src = np.sort(pts["tenor_days"].unique().astype(float))
        smile_src = np.full((len(m_grid), len(tenors_src)), np.nan, dtype=float)

        for j, T_days in enumerate(tenors_src):
            sub = pts[pts["tenor_days"] == T_days]
            m_obs = sub["moneyness"].to_numpy(dtype=float)
            iv_obs = sub["iv"].to_numpy(dtype=float)

            if len(m_obs) < 2:
                continue

            if smile_model.lower() == "svi":
                T_years = float(T_days) / 365.0
                smile_src[:, j] = svi_iv_on_m_grid(
                    m_grid,
                    m_obs,
                    iv_obs,
                    T_years=T_years,
                    max_nfev=svi_max_nfev,
                    arb_penalty=svi_arb_penalty,
                    lambda_mono=svi_lambda_mono,
                    lambda_bfly=svi_lambda_bfly,
                )
            else:
                kind = "cubic" if smile_model.lower() == "cubic" or method == "cubic" else "pchip"
                smile_src[:, j] = _smooth_smile_on_grid(m_obs, iv_obs, m_grid, kind=kind)

        #surf = np.full((len(m_grid), len(tenor_days)), np.nan, dtype=float)
        #for i in range(len(m_grid)):
        #    surf[i, :] = _interp_along_tenor(tenors_src, smile_src[i, :], tenor_days)

        surf = np.full((len(m_grid), len(tenor_days)), np.nan, dtype=float)
        for i in range(len(m_grid)):
            surf[i, :] = _interp_iv_along_tenor_total_variance(
                tenors_src=tenors_src,
                iv_src_pct=smile_src[i, :],
                tenors_tgt=tenor_days,
            )    
        if fill_method == "nearest":
            for i in range(len(m_grid)):
                v = surf[i, :]
                if np.any(~np.isfinite(v)):
                    ok = np.isfinite(v)
                    if ok.any():
                        idx_ok = np.where(ok)[0]
                        for jj in np.where(~ok)[0]:
                            nearest = idx_ok[np.argmin(np.abs(idx_ok - jj))]
                            v[jj] = v[nearest]
                    surf[i, :] = v

        if enforce_calendar:
            surf = enforce_calendar_no_arb_iv(surf, tenor_days)

        cube[:, :, k_date] = surf

    return SurfaceCube(cube=cube, m_grid=m_grid, tenor_days=tenor_days, asof_dates=asof_dates)



#########################################################################################
# Notebook workflow: SPX EOD .txt files // NOT SX5E AND NOT CALIBRATED BUT ALREADY READY
#########################################################################################
_SPX_TXT_COLS = [
    " [QUOTE_DATE]",
    " [UNDERLYING_LAST]",
    " [DTE]",
    " [C_IV]",
    " [STRIKE]",
    " [P_IV]",
]


def _import_tqdm():
    try:
        from tqdm.auto import tqdm
        return tqdm
    except Exception:
        def _identity(iterable, **_: object):
            return iterable
        return _identity


def _try_import_polars():
    try:
        import polars as pl  # type: ignore
        return pl
    except Exception:
        return None


def _default_spx_m_grid(m_min: float = 0.80, m_max: float = 1.20, nm: int = 32) -> np.ndarray:
    return np.linspace(float(m_min), float(m_max), int(nm), dtype=float)


def _default_spx_tenor_grid(t_min: float = 7.0, t_max: float = 365.0, nt: int = 16) -> np.ndarray:
    return np.exp(np.linspace(np.log(float(t_min)), np.log(float(t_max)), int(nt))).astype(float)


def read_spx_eod_txt(
    path: str | Path,
    *,
    dte_min: float = 7.0,
    dte_max: float = 365.0,
    m_min: float = 0.80,
    m_max: float = 1.20,
    iv_min: float = 0.01,
    iv_max: float = 5.0,
    prefer_polars: bool = True,
):
    """
    Read one SPX EOD .txt file and return filtered quote rows.

    Returned columns are QUOTE_DATE, UNDERLYING_LAST, DTE, C_IV, STRIKE, P_IV,
    moneyness, and iv.  The selected IV follows the notebook rule:
      - OTM put for moneyness <= 1 when valid, else call
      - OTM call for moneyness > 1 when valid, else put
    The raw input IVs are assumed to be decimals, e.g. 0.20 for 20%.
    """
    path = Path(path)
    pl = _try_import_polars() if prefer_polars else None

    if pl is not None:
        schema = {c: (pl.Utf8 if "DATE" in c else pl.Float32) for c in _SPX_TXT_COLS}
        try:
            df = pl.read_csv(
                path,
                columns=_SPX_TXT_COLS,
                schema_overrides=schema,
                ignore_errors=True,
                truncate_ragged_lines=True,
            ).rename({c: c.strip().strip("[]") for c in _SPX_TXT_COLS})

            df = df.with_columns(
                pl.col("QUOTE_DATE").str.strip_chars(),
                (pl.col("STRIKE") / pl.col("UNDERLYING_LAST")).alias("moneyness"),
            ).filter(
                pl.col("DTE").is_between(float(dte_min), float(dte_max))
                & pl.col("moneyness").is_between(float(m_min), float(m_max))
            )

            atm = pl.col("moneyness") <= 1.0
            c_ok = pl.col("C_IV").is_between(float(iv_min), float(iv_max))
            p_ok = pl.col("P_IV").is_between(float(iv_min), float(iv_max))
            return df.with_columns(
                pl.when(atm & p_ok).then(pl.col("P_IV"))
                .when(atm & c_ok).then(pl.col("C_IV"))
                .when((~atm) & c_ok).then(pl.col("C_IV"))
                .when((~atm) & p_ok).then(pl.col("P_IV"))
                .otherwise(None)
                .alias("iv")
            ).filter(pl.col("iv").is_not_null())
        except Exception:
            # Some files have slightly different headers/delimiters; pandas fallback
            # below is more forgiving because it can use positional columns.
            pass

    # pandas fallback: matches the notebook's positional read.
    df = pd.read_csv(
        path,
        usecols=[2, 4, 7, 13, 19, 29],
        names=["QUOTE_DATE", "UNDERLYING_LAST", "DTE", "C_IV", "STRIKE", "P_IV"],
        header=0,
        skipinitialspace=True,
        dtype={
            "QUOTE_DATE": str,
            "UNDERLYING_LAST": "float32",
            "DTE": "float32",
            "C_IV": "float32",
            "STRIKE": "float32",
            "P_IV": "float32",
        },
    )
    df["QUOTE_DATE"] = df["QUOTE_DATE"].str.strip()
    df["moneyness"] = df["STRIKE"] / df["UNDERLYING_LAST"]
    df = df[df["DTE"].between(dte_min, dte_max) & df["moneyness"].between(m_min, m_max)].copy()

    atm = df["moneyness"] <= 1.0
    c_ok = df["C_IV"].between(iv_min, iv_max)
    p_ok = df["P_IV"].between(iv_min, iv_max)
    df["iv"] = np.where(
        atm & p_ok,
        df["P_IV"],
        np.where(
            atm & c_ok,
            df["C_IV"],
            np.where((~atm) & c_ok, df["C_IV"], np.where((~atm) & p_ok, df["P_IV"], np.nan)),
        ),
    )
    return df.dropna(subset=["iv"])


def _build_spx_txt_surface_2d(
    m_np: np.ndarray,
    d_np: np.ndarray,
    v_np_pct: np.ndarray,
    m_grid: np.ndarray,
    tenor_days: np.ndarray,
) -> np.ndarray:
    """2-D linear interpolation fallback (no arbitrage corrections)."""
    nt, nm = len(tenor_days), len(m_grid)
    if len(v_np_pct) == 0:
        return np.full((nt, nm), np.nan, dtype=np.float32)
    gm, gt = np.meshgrid(np.log(m_grid), np.log(tenor_days))
    xi = np.column_stack([gt.ravel(), gm.ravel()])
    pts = np.column_stack([np.log(d_np), np.log(m_np)])
    try:
        if len(v_np_pct) >= 3:
            lin = LinearNDInterpolator(pts, v_np_pct)
            surface = lin(xi).reshape(nt, nm)
        else:
            surface = np.full((nt, nm), np.nan, dtype=float)
    except Exception:
        surface = np.full((nt, nm), np.nan, dtype=float)
    try:
        nn = NearestNDInterpolator(pts, v_np_pct)
        surface = np.where(np.isnan(surface), nn(xi).reshape(nt, nm), surface)
    except Exception:
        surface = np.where(np.isnan(surface), float(np.nanmedian(v_np_pct)), surface)
    if np.isnan(surface).any():
        surface = np.nan_to_num(surface, nan=float(np.nanmedian(v_np_pct)))
    return surface.astype(np.float32)


def _build_spx_txt_surface(
    m_np: np.ndarray,
    d_np: np.ndarray,
    v_np_pct: np.ndarray,
    m_grid: np.ndarray,
    tenor_days: np.ndarray,
    *,
    svi_arb_penalty: bool = True,
    svi_lambda_mono: float = 100.0,
    svi_lambda_bfly: float = 100.0,
    svi_max_nfev: int = 1000,
    enforce_calendar: bool = True,
) -> np.ndarray:
    """
    Build a (n_tenor, n_moneyness) surface from scattered observations using
    the same pipeline as build_surface_cube (SX5E):
      1. SVI fit per tenor slice  (arb-free smile: no butterfly / monotonicity violation)
      2. Total-variance interpolation along tenor  (calendar-safe by construction)
      3. enforce_calendar_no_arb_iv  (final calendar safety pass)
    Falls back to 2-D interpolation for degenerate date sets.
    """
    m_np = np.asarray(m_np, dtype=float)
    d_np = np.asarray(d_np, dtype=float)
    v_np_pct = np.asarray(v_np_pct, dtype=float)

    mask = (
        np.isfinite(m_np) & np.isfinite(d_np) & np.isfinite(v_np_pct)
        & (m_np > 0) & (d_np > 0) & (v_np_pct > 0)
    )
    m_np, d_np, v_np_pct = m_np[mask], d_np[mask], v_np_pct[mask]

    nt, nm = len(tenor_days), len(m_grid)
    if len(v_np_pct) == 0:
        return np.full((nt, nm), np.nan, dtype=np.float32)

    # ── Step 1: group by DTE and fit SVI per tenor slice ──────────────────────
    # SPX DTE values are integers; round to avoid float noise.
    dte_int = np.round(d_np).astype(int)
    unique_dtes = np.unique(dte_int)

    tenors_src: list[float] = []
    smile_cols: list[np.ndarray] = []

    for dte_i in unique_dtes:
        idx = dte_int == dte_i
        T_days = float(np.mean(d_np[idx]))
        T_years = T_days / 365.0
        m_obs = m_np[idx]
        iv_obs = v_np_pct[idx]

        um = np.unique(m_obs)
        if len(um) < 2:
            continue

        if len(um) >= 3:
            try:
                col = svi_iv_on_m_grid(
                    m_grid, m_obs, iv_obs, T_years=T_years,
                    max_nfev=svi_max_nfev,
                    arb_penalty=svi_arb_penalty,
                    lambda_mono=svi_lambda_mono,
                    lambda_bfly=svi_lambda_bfly,
                )
            except Exception:
                col = _smooth_smile_on_grid(m_obs, iv_obs, m_grid, kind="pchip")
        else:
            col = _smooth_smile_on_grid(m_obs, iv_obs, m_grid, kind="pchip")

        tenors_src.append(T_days)
        smile_cols.append(np.asarray(col, dtype=float))

    if len(tenors_src) == 0:
        # Degenerate: fall back to 2-D interpolation.
        return _build_spx_txt_surface_2d(m_np, d_np, v_np_pct, m_grid, tenor_days)

    tenors_src_arr = np.asarray(tenors_src, dtype=float)   # (n_slices,)
    smile_src_arr  = np.column_stack(smile_cols)            # (nm, n_slices)

    # ── Step 2: total-variance interpolation along tenor ──────────────────────
    surf_nm_nt = np.full((nm, nt), np.nan, dtype=float)
    for i in range(nm):
        surf_nm_nt[i, :] = _interp_iv_along_tenor_total_variance(
            tenors_src=tenors_src_arr,
            iv_src_pct=smile_src_arr[i, :],
            tenors_tgt=tenor_days,
        )

    # Nearest-neighbour fill for any remaining NaNs.
    for i in range(nm):
        v = surf_nm_nt[i, :]
        nan_mask = ~np.isfinite(v)
        if nan_mask.any():
            ok = np.where(~nan_mask)[0]
            if len(ok):
                for jj in np.where(nan_mask)[0]:
                    v[jj] = v[ok[np.argmin(np.abs(ok - jj))]]
            surf_nm_nt[i, :] = v

    # ── Step 3: calendar no-arb ────────────────────────────────────────────────
    if enforce_calendar:
        surf_nm_nt = enforce_calendar_no_arb_iv(surf_nm_nt, tenor_days)  # (nm, nt)

    return surf_nm_nt.T.astype(np.float32)   # → (nt, nm)


def process_spx_txt_file(
    path: str | Path,
    *,
    m_grid: np.ndarray,
    tenor_days: np.ndarray,
    date_workers: int = 4,
    dte_min: float = 7.0,
    dte_max: float = 365.0,
    m_min: float = 0.80,
    m_max: float = 1.20,
    iv_min: float = 0.01,
    iv_max: float = 5.0,
    prefer_polars: bool = True,
    svi_arb_penalty: bool = True,
    svi_lambda_mono: float = 100.0,
    svi_lambda_bfly: float = 100.0,
    svi_max_nfev: int = 1000,
    enforce_calendar: bool = True,
) -> tuple[list[str], np.ndarray, float]:
    """Read one spx_eod_YYYYMM.txt file -> (dates, cube_slice, seconds).

    cube_slice has notebook layout (n_dates, n_tenor, n_moneyness).
    Empty files return an empty cube slice instead of crashing with max_workers=0.
    """
    t0 = time.perf_counter()
    path = Path(path)
    df = read_spx_eod_txt(
        path,
        dte_min=dte_min,
        dte_max=dte_max,
        m_min=m_min,
        m_max=m_max,
        iv_min=iv_min,
        iv_max=iv_max,
        prefer_polars=prefer_polars,
    )

    pl = _try_import_polars() if prefer_polars else None
    is_polars_df = pl is not None and df.__class__.__module__.startswith("polars")

    if is_polars_df:
        dates = sorted(df["QUOTE_DATE"].unique().to_list())
        groups = {d: df.filter(pl.col("QUOTE_DATE") == d) for d in dates}

        def _args(d: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            g = groups[d]
            return (
                g["moneyness"].to_numpy(),
                g["DTE"].to_numpy(),
                g["iv"].to_numpy() * 100.0,
            )
    else:
        dates = sorted(df["QUOTE_DATE"].unique().tolist())
        groups = {d: df[df["QUOTE_DATE"] == d] for d in dates}

        def _args(d: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            g = groups[d]
            return (
                g["moneyness"].to_numpy(),
                g["DTE"].to_numpy(),
                g["iv"].to_numpy() * 100.0,
            )

    if len(dates) == 0:
        empty = np.empty((0, len(tenor_days), len(m_grid)), dtype=np.float32)
        return [], empty, time.perf_counter() - t0

    ordered: dict[str, np.ndarray] = {}
    def _build_fn(m: np.ndarray, d: np.ndarray, v: np.ndarray) -> np.ndarray:
        return _build_spx_txt_surface(
            m, d, v, m_grid, tenor_days,
            svi_arb_penalty=svi_arb_penalty,
            svi_lambda_mono=svi_lambda_mono,
            svi_lambda_bfly=svi_lambda_bfly,
            svi_max_nfev=svi_max_nfev,
            enforce_calendar=enforce_calendar,
        )

    n_workers = max(1, min(int(date_workers), len(dates)))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_build_fn, *_args(d)): d for d in dates}
        for fut in as_completed(futs):
            ordered[futs[fut]] = fut.result()

    cube_slice = np.stack([ordered[d] for d in dates], axis=0)
    return dates, cube_slice, time.perf_counter() - t0


def build_spx_txt_surface_cube(
    data_dir: str | Path,
    *,
    m_grid: np.ndarray | None = None,
    tenor_days: np.ndarray | None = None,
    nm: int = 32,
    nt: int = 16,
    m_min: float = 0.80,
    m_max: float = 1.20,
    t_min: float = 7.0,
    t_max: float = 365.0,
    iv_min: float = 0.01,
    iv_max: float = 5.0,
    date_workers: int = 4,
    max_files: int | None = None,
    skip_empty_files: bool = True,
    prefer_polars: bool = True,
    verbose: bool = False,
    svi_arb_penalty: bool = True,
    svi_lambda_mono: float = 100.0,
    svi_lambda_bfly: float = 100.0,
    svi_max_nfev: int = 1000,
    enforce_calendar: bool = True,
) -> SurfaceCube:
    """
    Build a SurfaceCube from monthly SPX EOD .txt files. Normalizes to
    SurfaceCube.cube layout (n_moneyness, n_tenor, n_dates).
    """
    data_dir = Path(data_dir)
    if m_grid is None:
        m_grid = _default_spx_m_grid(m_min=m_min, m_max=m_max, nm=nm)
    else:
        m_grid = np.asarray(m_grid, dtype=float)
        m_min, m_max = float(np.nanmin(m_grid)), float(np.nanmax(m_grid))

    if tenor_days is None:
        tenor_days = _default_spx_tenor_grid(t_min=t_min, t_max=t_max, nt=nt)
    else:
        tenor_days = np.asarray(tenor_days, dtype=float)
        t_min, t_max = float(np.nanmin(tenor_days)), float(np.nanmax(tenor_days))

    files = sorted(data_dir.glob("*.txt"))
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir!r}")

    tqdm = _import_tqdm()
    all_dates: list[str] = []
    all_cubes: list[np.ndarray] = []
    skipped: list[Path] = []
    t_start = time.perf_counter()

    iterator = tqdm(files, desc="Files", unit="file") if verbose else files
    for i, path in enumerate(iterator, start=1):
        dates, cube_slice, elapsed = process_spx_txt_file(
            path,
            m_grid=m_grid,
            tenor_days=tenor_days,
            date_workers=date_workers,
            dte_min=t_min,
            dte_max=t_max,
            m_min=m_min,
            m_max=m_max,
            iv_min=iv_min,
            iv_max=iv_max,
            prefer_polars=prefer_polars,
            svi_arb_penalty=svi_arb_penalty,
            svi_lambda_mono=svi_lambda_mono,
            svi_lambda_bfly=svi_lambda_bfly,
            svi_max_nfev=svi_max_nfev,
            enforce_calendar=enforce_calendar,
        )

        if len(dates) == 0:
            skipped.append(path)
            if not skip_empty_files:
                raise ValueError(f"{path.name}: no valid quote dates after filters")
            if verbose:
                print(f"  [{i:>2}/{len(files)}] skipped {path.name:<35s} no valid dates")
            continue

        all_dates.extend(dates)
        all_cubes.append(cube_slice)

        if verbose:
            print(
                f"  [{i:>2}/{len(files)}] done  {path.name:<35s}"
                f"  {len(dates):2d} dates"
                f"  IV [{np.nanmin(cube_slice):.1f}%–{np.nanmax(cube_slice):.1f}%]"
                f"  {elapsed:.1f}s"
            )

    if not all_cubes:
        raise ValueError(f"No surfaces built from {data_dir}. Check filters and input files.")

    dates_arr = np.array(all_dates, dtype="U10")
    cube_dtm = np.concatenate(all_cubes, axis=0)  # notebook layout: dates x tenor x moneyness

    unique_dates, first_idx = np.unique(dates_arr, return_index=True)
    order = np.argsort(unique_dates)
    unique_dates = unique_dates[order]
    first_idx = first_idx[order]
    cube_dtm = cube_dtm[first_idx]

    asof_dates = [pd.Timestamp(str(d)) for d in unique_dates]
    cube_mtd = np.transpose(cube_dtm, (2, 1, 0))  # SurfaceCube layout: moneyness x tenor x dates

    if verbose:
        total = time.perf_counter() - t_start
        print(
            f"Built SPX txt cube with shape {cube_mtd.shape} "
            f"(moneyness x tenor x dates) in {total:.1f}s"
        )
        if skipped:
            print("Skipped empty files:", ", ".join(p.name for p in skipped))

    return SurfaceCube(cube=cube_mtd, m_grid=m_grid, tenor_days=tenor_days, asof_dates=asof_dates)


# -----------------------------------------------------------------------------
# Loading and saving already-constructed cubes
# -----------------------------------------------------------------------------

def load_cube_npz(npz_path: str | Path) -> SurfaceCube:
    """
    Load an existing .npz cube using the project canonical layout only.

    Required format:
      - cube shape = (n_moneyness, n_tenor, n_dates)
      - keys = cube, m_grid, tenor_days, asof_dates

    This deliberately does not accept the old notebook layout
    (n_dates, n_tenor, n_moneyness), because the whole project should use
    one convention only.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        required = {"cube", "m_grid", "tenor_days", "asof_dates"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{npz_path} is missing required keys: {missing}")

        cube = np.asarray(data["cube"])
        m_grid = np.asarray(data["m_grid"], dtype=float)
        tenor_days = np.asarray(data["tenor_days"], dtype=float)
        raw_dates = data["asof_dates"]

    dates = [pd.Timestamp(str(d)) for d in raw_dates]
    expected_shape = (len(m_grid), len(tenor_days), len(dates))
    if cube.shape != expected_shape:
        raise ValueError(
            f"Invalid cube shape for canonical project layout: {cube.shape}. "
            f"Expected (moneyness, tenor, dates) = {expected_shape}. "
            "Regenerate or convert the file before loading."
        )

    return SurfaceCube(cube=cube, m_grid=m_grid, tenor_days=tenor_days, asof_dates=dates)



def surface_cube_summary(sc: SurfaceCube) -> str:
    dates = [pd.Timestamp(str(d)) for d in sc.asof_dates]
    start = dates[0].strftime("%Y-%m-%d") if dates else "n/a"
    end = dates[-1].strftime("%Y-%m-%d") if dates else "n/a"
    return (
        f"cube shape   : {sc.cube.shape} (moneyness x tenor x dates)\n"
        f"date range   : {start} to {end}\n"
        f"m grid       : {len(sc.m_grid)} points [{np.nanmin(sc.m_grid):.4g}, {np.nanmax(sc.m_grid):.4g}]\n"
        f"tenor grid   : {len(sc.tenor_days)} points [{np.nanmin(sc.tenor_days):.4g}, {np.nanmax(sc.tenor_days):.4g}]\n"
        f"cube min/max : {np.nanmin(sc.cube):.2f}% / {np.nanmax(sc.cube):.2f}%\n"
        f"NaN count    : {int(np.isnan(sc.cube).sum())}"
    )


def save_cube_npz(out_path: str | Path, sc: SurfaceCube) -> None:
    """
    Save a SurfaceCube using the project canonical layout only.

    Saved format:
      - cube shape = (n_moneyness, n_tenor, n_dates)
      - date key = asof_dates
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        cube=sc.cube,
        m_grid=sc.m_grid,
        tenor_days=sc.tenor_days,
        asof_dates=np.array([pd.Timestamp(str(d)).strftime("%Y-%m-%d") for d in sc.asof_dates], dtype="U10"),
    )



def sanity_plot_cube(sc: SurfaceCube, *, indices: tuple[int, int, int] | None = None) -> None:
    """Notebook-style quick visual check for first/middle/last surfaces."""
    import matplotlib.pyplot as plt

    cube_dtm = np.transpose(sc.cube, (2, 1, 0))
    if indices is None:
        indices = (0, len(cube_dtm) // 2, -1)

    fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 4))
    if len(indices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        im = ax.imshow(
            cube_dtm[idx],
            origin="lower",
            aspect="auto",
            extent=[sc.m_grid[0], sc.m_grid[-1], np.log(sc.tenor_days[0]), np.log(sc.tenor_days[-1])],
        )
        ax.set_title(pd.Timestamp(sc.asof_dates[idx]).strftime("%Y-%m-%d"))
        ax.set_xlabel("Moneyness K/S")
        ax.set_ylabel("log(DTE)")
        plt.colorbar(im, ax=ax, label="IV %")
    plt.suptitle("Vol surfaces: first / mid / last date", y=1.02)
    plt.tight_layout()
    plt.show()


def _parse_list(s: Optional[str]) -> Optional[np.ndarray]:
    if not s:
        return None
    return np.array([float(x.strip()) for x in s.split(",") if x.strip()], dtype=float)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser("volsurf-cube")
    p.add_argument(
        "--input-format",
        default="excel",
        choices=["excel", "spx-txt", "npz"],
        help="excel: ddmmyyyy.xlsx files; spx-txt: spx_eod_YYYYMM.txt files; npz: use an already-built cube",
    )
    p.add_argument("--data-dir", default="data/raw", help="Folder containing raw input files")
    p.add_argument("--existing-npz", default=None, help="Path to an already constructed .npz cube; implies --input-format npz")
    p.add_argument("--out", default="data/treated/vol_surface_cube.npz", help="Output .npz path")

    # Shared grid controls.
    p.add_argument("--m-grid", default=None, help="Comma-separated moneyness grid")
    p.add_argument("--tenor-days", default=None, help="Comma-separated tenor grid in days")
    p.add_argument("--m-min", type=float, default=None, help="Minimum moneyness. Defaults depend on input format.")
    p.add_argument("--m-max", type=float, default=None, help="Maximum moneyness. Defaults depend on input format.")
    p.add_argument("--max-files", type=int, default=None, help="Debug: only read the first N raw files")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--sanity-plot", action="store_true", help="Show first/middle/last surface plots after building/loading")

    # Excel builder options: old constructor_surface_cube path.
    p.add_argument("--sheet", default="0")
    p.add_argument("--smile-model", default="svi", choices=["pchip", "cubic", "svi"])
    p.add_argument("--fill", default="nearest", choices=["nearest", ""])
    p.add_argument("--no-avg", action="store_true")
    p.add_argument("--no-calendar", action="store_true", help="Disable calendar monotonicity enforcement")
    p.add_argument("--svi-max-nfev", type=int, default=1000)
    p.add_argument("--svi-arb-penalty", action="store_true", help="Keep SVI but add soft call-price no-arb penalty during fit")
    p.add_argument("--svi-lambda-mono", type=float, default=100.0)
    p.add_argument("--svi-lambda-bfly", type=float, default=100.0)

    # SPX txt notebook options.
    p.add_argument("--nm", type=int, default=32, help="SPX txt notebook moneyness grid size")
    p.add_argument("--nt", type=int, default=16, help="SPX txt notebook tenor grid size")
    p.add_argument("--t-min", type=float, default=7.0)
    p.add_argument("--t-max", type=float, default=365.0)
    p.add_argument("--iv-min", type=float, default=0.01)
    p.add_argument("--iv-max", type=float, default=5.0)
    p.add_argument("--date-workers", type=int, default=4)
    p.add_argument("--no-polars", action="store_true", help="Force pandas reader for SPX txt files")
    p.add_argument("--fail-on-empty-file", action="store_true", help="Raise if a monthly txt file has zero valid dates after filtering")
    p.add_argument("--no-svi", action="store_true", help="SPX txt: use raw 2-D interpolation instead of SVI per slice (disables arb corrections)")
    p.add_argument("--no-calendar-spx", action="store_true", help="SPX txt: disable calendar enforcement")
    p.add_argument("--spx-lambda-mono", type=float, default=100.0, help="SPX txt SVI monotonicity penalty weight")
    p.add_argument("--spx-lambda-bfly", type=float, default=100.0, help="SPX txt SVI butterfly penalty weight")

    args = p.parse_args()
    input_format = "npz" if args.existing_npz else args.input_format

    if input_format == "npz":
        npz_path = args.existing_npz or args.data_dir
        sc = load_cube_npz(npz_path)

    elif input_format == "spx-txt":
        m_min = 0.80 if args.m_min is None else args.m_min
        m_max = 1.20 if args.m_max is None else args.m_max
        sc = build_spx_txt_surface_cube(
            data_dir=args.data_dir,
            m_grid=_parse_list(args.m_grid),
            tenor_days=_parse_list(args.tenor_days),
            nm=args.nm,
            nt=args.nt,
            m_min=m_min,
            m_max=m_max,
            t_min=args.t_min,
            t_max=args.t_max,
            iv_min=args.iv_min,
            iv_max=args.iv_max,
            date_workers=args.date_workers,
            max_files=args.max_files,
            skip_empty_files=(not args.fail_on_empty_file),
            prefer_polars=(not args.no_polars),
            verbose=args.verbose,
            svi_arb_penalty=(not args.no_svi),
            svi_lambda_mono=args.spx_lambda_mono,
            svi_lambda_bfly=args.spx_lambda_bfly,
            enforce_calendar=(not args.no_calendar_spx),
        )

    else:  # excel
        sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
        m_min = 0.90 if args.m_min is None else args.m_min
        m_max = 1.10 if args.m_max is None else args.m_max
        sc = build_surface_cube(
            data_dir=args.data_dir,
            m_grid=_parse_list(args.m_grid),
            tenor_days=_parse_list(args.tenor_days),
            sheet_name=sheet,
            m_min=m_min,
            m_max=m_max,
            average_call_put=(not args.no_avg),
            smile_model=args.smile_model,
            fill_method=(args.fill or None),
            enforce_calendar=(not args.no_calendar),
            max_files=args.max_files,
            verbose=args.verbose,
            svi_max_nfev=args.svi_max_nfev,
            svi_arb_penalty=args.svi_arb_penalty,
            svi_lambda_mono=args.svi_lambda_mono,
            svi_lambda_bfly=args.svi_lambda_bfly,
        )

    save_cube_npz(args.out, sc)
    print(f"Saved cube to {args.out}")
    print(surface_cube_summary(sc))

    if args.sanity_plot:
        sanity_plot_cube(sc)


if __name__ == "__main__":
    main()
