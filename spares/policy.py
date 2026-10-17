"""Safety stock and reorder point from the forecast and its backtest error.

    SS  = z(service level) * sigma * sqrt(L + R)
    ROP = forecast demand over the next L + R weeks + SS

sigma is the RMSE of the chosen model's backtest errors (weekly), L the supplier
lead time in weeks and R the review period. Treating weekly errors as independent
is the textbook approximation; for lumpy slow movers it understates risk.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd


def z_value(service_level: float) -> float:
    if not 0.5 <= service_level < 1:
        raise ValueError("service level must be in [0.5, 1)")
    return NormalDist().inv_cdf(service_level)


def safety_stock(sigma: float, lead_time_weeks: int, review_weeks: int, service_level: float) -> float:
    if sigma < 0 or lead_time_weeks < 0 or review_weeks < 0:
        raise ValueError("sigma, lead time and review period must be non-negative")
    return z_value(service_level) * sigma * math.sqrt(lead_time_weeks + review_weeks)


def reorder_point(weekly_forecast: np.ndarray, lead_time_weeks: int, review_weeks: int, ss: float) -> float:
    cover = lead_time_weeks + review_weeks
    fc = np.asarray(weekly_forecast, dtype=float)
    if len(fc) < cover:
        fc = np.concatenate([fc, np.repeat(fc[-1] if len(fc) else 0.0, cover - len(fc))])
    return float(fc[:cover].sum() + ss)


def build_policy(items: pd.DataFrame, sigma: pd.Series, fc_wide: pd.DataFrame, service_level: float,
                 review_weeks: int) -> pd.DataFrame:
    """items: unique_id, sku, dc, unit_cost_usd, lead_time_weeks (one row per SKU-DC).
    sigma: RMSE per unique_id. fc_wide: unique_id x week forecast matrix (columns in order)."""
    out = items.copy()
    out["sigma"] = out["unique_id"].map(sigma).fillna(0.0)
    out["safety_stock"] = [
        safety_stock(s, int(L), review_weeks, service_level)
        for s, L in zip(out["sigma"], out["lead_time_weeks"])
    ]
    out["reorder_point"] = [
        reorder_point(fc_wide.loc[u].to_numpy(), int(L), review_weeks, ss)
        for u, L, ss in zip(out["unique_id"], out["lead_time_weeks"], out["safety_stock"])
    ]
    out["mean_weekly_forecast"] = [float(fc_wide.loc[u].mean()) for u in out["unique_id"]]
    out["safety_stock_value_usd"] = out["safety_stock"] * out["unit_cost_usd"]
    return out
