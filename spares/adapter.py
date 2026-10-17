"""Turn Corvane's ERP CSV exports into the long format statsforecast expects.

statsforecast wants one row per series and period: unique_id, ds, y. The ERP gives
shipped order lines with inconsistent DC codes, re-exported lines, return credits
and no rows for weeks without demand. This module fixes those and logs counts.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def normalize_dc(code: str, valid: list[str]) -> str | None:
    """'dc-cmh', 'CMH', 'DC CMH ' -> 'DC-CMH'. Returns None if unknown."""
    letters = re.sub(r"[^A-Z]", "", str(code).upper())
    if letters.startswith("DC"):
        letters = letters[2:]
    candidate = f"DC-{letters}"
    return candidate if candidate in valid else None


def load_sku_master(path: Path) -> tuple[pd.DataFrame, dict]:
    m = pd.read_csv(path)
    m["sku"] = m["sku"].str.strip().str.upper()
    log = {"sku_rows": len(m), "pallet_qty_filled": int(m["pallet_qty"].isna().sum())}
    # Blank pallet quantity: use the category median (what the DC team does today).
    m["pallet_qty"] = m["pallet_qty"].fillna(m.groupby("category")["pallet_qty"].transform("median"))
    m["pallet_qty"] = m["pallet_qty"].fillna(m["pallet_qty"].median()).astype(int)
    for col in ["lead_time_weeks", "moq"]:
        m[col] = m[col].astype(int)
    bad = m[(m["unit_cost_usd"] <= 0) | (m["pallet_qty"] <= 0) | (m["lead_time_weeks"] < 1) | (m["moq"] < 1)]
    if len(bad):
        raise ValueError(f"invalid SKU master rows: {bad['sku'].tolist()}")
    return m, log


def load_demand(path: Path, master: pd.DataFrame, dcs: list[str], start_week: str, n_weeks: int) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, dtype={"order_id": str, "dc_code": str, "sku": str})
    log = {"export_rows": len(raw)}
    df = raw.copy()
    df["sku"] = df["sku"].str.strip().str.upper()
    norm = df["dc_code"].map(lambda c: normalize_dc(c, dcs))
    log["dc_codes_normalized"] = int((norm != df["dc_code"]).sum() - norm.isna().sum())
    log["unknown_dc_rows_dropped"] = int(norm.isna().sum())
    df["dc_code"] = norm
    df = df.dropna(subset=["dc_code"])
    before = len(df)
    df = df.drop_duplicates(subset=["order_id"], keep="first")
    log["duplicate_lines_dropped"] = before - len(df)
    returns = df["qty"] < 0
    log["return_lines_dropped"] = int(returns.sum())
    df = df[~returns]
    unknown_sku = ~df["sku"].isin(master["sku"])
    log["unknown_sku_rows_dropped"] = int(unknown_sku.sum())
    df = df[~unknown_sku]
    ship = pd.to_datetime(df["ship_date"])
    df["ds"] = (ship - pd.to_timedelta(ship.dt.weekday, unit="D")).dt.normalize()
    weeks = pd.date_range(start_week, periods=n_weeks, freq="W-MON")
    outside = ~df["ds"].isin(weeks)
    log["rows_outside_history_dropped"] = int(outside.sum())
    df = df[~outside]
    df["unique_id"] = df["sku"] + "|" + df["dc_code"]
    weekly = df.groupby(["unique_id", "ds"], as_index=False)["qty"].sum().rename(columns={"qty": "y"})
    # Every SKU-DC pair in the master gets a full weekly grid; missing weeks are zero demand.
    ids = [f"{s}|{d}" for s in master["sku"] for d in dcs]
    grid = pd.MultiIndex.from_product([ids, weeks], names=["unique_id", "ds"]).to_frame(index=False)
    out = grid.merge(weekly, on=["unique_id", "ds"], how="left")
    log["zero_weeks_filled"] = int(out["y"].isna().sum())
    out["y"] = out["y"].fillna(0).astype(float)
    log["series"] = len(ids)
    log["weeks"] = n_weeks
    return out, log


def load_inventory(path: Path, dcs: list[str]) -> pd.DataFrame:
    inv = pd.read_csv(path, dtype={"on_order_due_week": str})
    inv["sku"] = inv["sku"].str.strip().str.upper()
    inv["dc_code"] = inv["dc_code"].map(lambda c: normalize_dc(c, dcs))
    inv["unique_id"] = inv["sku"] + "|" + inv["dc_code"]
    inv["on_order_due_week"] = pd.to_datetime(inv["on_order_due_week"].replace("", None))
    return inv


def split_uid(uid: str) -> tuple[str, str]:
    sku, dc = uid.split("|")
    return sku, dc
