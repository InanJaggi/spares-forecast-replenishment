"""Forecasting with statsforecast: segment, backtest, pick a model per series.

Fast movers (smooth or erratic demand) get SeasonalNaive and AutoETS. Slow movers
(intermittent or lumpy) get SeasonalNaive and three intermittent-demand methods.
statsforecast's own cross_validation does the rolling-origin backtest; I only add
the segmentation, the metrics and the per-series choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ADIDA, IMAPA, AutoETS, CrostonOptimized, SeasonalNaive

BASELINE = "SeasonalNaive"
MODEL_ORDER = ["SeasonalNaive", "AutoETS", "CrostonOptimized", "ADIDA", "IMAPA"]
FREQ = "W-MON"


def fast_models(season_length: int) -> list:
    return [SeasonalNaive(season_length=season_length), AutoETS(season_length=season_length)]


def slow_models(season_length: int) -> list:
    return [SeasonalNaive(season_length=season_length), CrostonOptimized(), ADIDA(), IMAPA()]


def segment(df: pd.DataFrame) -> pd.DataFrame:
    """Syntetos-Boylan classes from ADI (average demand interval) and CV^2 of sizes."""
    rows = []
    for uid, g in df.groupby("unique_id", sort=True):
        y = g["y"].to_numpy()
        nz = y[y > 0]
        adi = len(y) / max(len(nz), 1)
        cv2 = float((nz.std() / nz.mean()) ** 2) if len(nz) > 1 else 0.0
        if adi < 1.32:
            cls = "smooth" if cv2 < 0.49 else "erratic"
        else:
            cls = "intermittent" if cv2 < 0.49 else "lumpy"
        rows.append({"unique_id": uid, "adi": adi, "cv2": cv2, "sbc_class": cls,
                     "segment": "fast" if adi < 1.32 else "slow"})
    return pd.DataFrame(rows)


def _by_segment(df: pd.DataFrame, seg: pd.DataFrame):
    fast_ids = set(seg.loc[seg["segment"] == "fast", "unique_id"])
    yield "fast", df[df["unique_id"].isin(fast_ids)]
    yield "slow", df[~df["unique_id"].isin(fast_ids)]


def backtest(df: pd.DataFrame, seg: pd.DataFrame, season_length: int, h: int, n_windows: int,
             step_size: int | None = None) -> pd.DataFrame:
    """Rolling-origin cross-validation with statsforecast, one call per segment.

    AutoETS is fitted once on the first window and rolled forward with its
    `forward` method (refit=False); the intermittent models are cheap and refit
    every window (they have no forward method).
    """
    step_size = step_size or h
    parts = []
    for name, part in _by_segment(df, seg):
        if part.empty:
            continue
        models = fast_models(season_length) if name == "fast" else slow_models(season_length)
        sf = StatsForecast(models=models, freq=FREQ, n_jobs=1)
        cv = sf.cross_validation(df=part, h=h, n_windows=n_windows, step_size=step_size,
                                 refit=(name != "fast"))
        parts.append(cv)
    cv = pd.concat(parts, ignore_index=True)
    for m in MODEL_ORDER:
        if m not in cv:
            cv[m] = np.nan
    return cv[["unique_id", "ds", "cutoff", "y", *MODEL_ORDER]]


def naive_scale(train: pd.DataFrame) -> pd.Series:
    """In-sample one-step naive MAE per series (the MASE denominator)."""
    return train.sort_values("ds").groupby("unique_id")["y"].apply(lambda s: s.diff().abs().mean())


def series_scores(cv: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    """Per series and model: MAE, RMSE, MASE, absolute-error and demand sums."""
    rows = []
    for m in [c for c in MODEL_ORDER + ["Selected"] if c in cv]:
        g = cv.dropna(subset=[m]).assign(err=lambda d: d["y"] - d[m])
        agg = g.groupby("unique_id").agg(
            mae=("err", lambda e: e.abs().mean()),
            rmse=("err", lambda e: float(np.sqrt((e ** 2).mean()))),
            abs_err=("err", lambda e: e.abs().sum()),
            demand=("y", "sum"),
        )
        agg["mase"] = agg["mae"] / scale.reindex(agg.index).replace(0, np.nan)
        agg["model"] = m
        rows.append(agg.reset_index())
    return pd.concat(rows, ignore_index=True)


def select(scores: pd.DataFrame) -> pd.Series:
    """Lowest backtest RMSE per series; ties go to the simpler model.

    RMSE, not MAE: the safety-stock math needs an unbiased mean, and MAE rewards
    forecasts near zero on intermittent series (it picked SeasonalNaive, mostly
    zeros, for a third of the slow movers when I tried it).
    """
    s = scores[scores["model"].isin(MODEL_ORDER)].copy()
    s["rank"] = s["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    s = s.sort_values(["unique_id", "rmse", "rank"])
    return s.groupby("unique_id").first()["model"]


def add_selected(frame: pd.DataFrame, selection: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    chosen = out["unique_id"].map(selection)
    vals = np.full(len(out), np.nan)
    for m in MODEL_ORDER:
        mask = (chosen == m).to_numpy()
        if m in out:
            vals[mask] = out.loc[mask, m].to_numpy()
    out["Selected"] = vals
    return out


def summary_table(scores: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    """WAPE (pooled) and mean MASE per model, overall and by segment."""
    s = scores.merge(seg[["unique_id", "segment"]], on="unique_id")
    rows = []
    for m in [x for x in MODEL_ORDER + ["Selected"] if x in set(s["model"])]:
        g = s[s["model"] == m]
        row = {"model": m, "series": len(g)}
        for name, part in [("all", g), ("fast", g[g["segment"] == "fast"]), ("slow", g[g["segment"] == "slow"])]:
            row[f"wape_{name}"] = part["abs_err"].sum() / part["demand"].sum() if len(part) else np.nan
            row[f"mase_{name}"] = part["mase"].mean() if len(part) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def forecast(df: pd.DataFrame, seg: pd.DataFrame, selection: pd.Series, season_length: int, h: int) -> pd.DataFrame:
    """Fit on all of df and forecast h weeks; returns every menu model plus Selected."""
    parts = []
    for name, part in _by_segment(df, seg):
        if part.empty:
            continue
        models = fast_models(season_length) if name == "fast" else slow_models(season_length)
        parts.append(StatsForecast(models=models, freq=FREQ, n_jobs=1).forecast(df=part, h=h))
    fc = pd.concat(parts, ignore_index=True)
    for m in MODEL_ORDER:
        if m not in fc:
            fc[m] = np.nan
    return add_selected(fc[["unique_id", "ds", *MODEL_ORDER]], selection)


def rolling_forecasts(df: pd.DataFrame, seg: pd.DataFrame, selection: pd.Series, season_length: int,
                      first_origin: pd.Timestamp, n_origins: int, h: int) -> pd.DataFrame:
    """The forecasts a planner would have had each week from first_origin on.

    Uses statsforecast cross_validation with step_size=1. To get h-week forecasts
    from origins near the end of the data, each series is padded with h-1 dummy
    weeks; models only ever see data before each cutoff, so the padding never
    enters a fit. The `y` column is dropped for that reason.
    """
    hist = df[df["ds"] < first_origin + pd.Timedelta(weeks=n_origins)]
    pad_weeks = pd.date_range(hist["ds"].max() + pd.Timedelta(weeks=1), periods=h - 1, freq=FREQ)
    ids = hist["unique_id"].unique()
    pad = pd.MultiIndex.from_product([ids, pad_weeks], names=["unique_id", "ds"]).to_frame(index=False)
    pad["y"] = 0.0
    padded = pd.concat([hist, pad], ignore_index=True)
    padded = padded.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    n_total = padded.groupby("unique_id").size().iloc[0]
    n_before = (df[df["unique_id"] == ids[0]]["ds"] < first_origin).sum()
    # cross_validation puts the first cutoff at n_total - (h + n_windows - 1)
    assert n_total - (h + n_origins - 1) == n_before, "origin alignment"
    cv = backtest(padded, seg, season_length, h=h, n_windows=n_origins, step_size=1)
    cv = add_selected(cv.drop(columns=["y"]), selection)
    cv["origin"] = cv["cutoff"] + pd.Timedelta(weeks=1)   # first forecast week
    return cv
