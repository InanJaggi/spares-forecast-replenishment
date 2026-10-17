"""Seeded synthetic data for Corvane Industrial spare parts.

Writes three ERP-style CSV exports to data/:
  sku_master.csv          one row per SKU (cost, pallet size, lead time, MOQ, supplier)
  erp_demand_export.csv   shipped order lines per DC, with the usual export mess
  inventory_snapshot.csv  on-hand and open purchase orders at the end of history

Demand classes: smooth fast movers, fast movers with a 13-week cycle (customers'
quarterly PM schedules), and intermittent slow movers. All numbers are invented.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# category: description, supplier, unit cost range, pallet qty, lead time options,
#           MOQ, weekly base demand range at DC-CMH (fast movers), mean lot size (slow movers),
#           (smooth, seasonal, intermittent) SKU counts in the 40-SKU catalog
CATEGORIES = {
    "SEAL": ("Mechanical seal", "Pellum Seal Co.", (180, 650), 60, (3, 4), 10, (6, 25), 2.0, (3, 3, 2)),
    "BRG": ("Bearing", "Brightline Bearings", (35, 220), 200, (2, 3), 25, (15, 60), 6.0, (4, 2, 2)),
    "GSK": ("Gasket kit", "Pellum Seal Co.", (8, 40), 500, (2,), 100, (40, 120), 20.0, (3, 2, 1)),
    "IMP": ("Impeller", "Halvard Castings", (400, 1600), 12, (5, 6), 4, (2, 6), 1.5, (0, 0, 7)),
    "CPL": ("Coupling", "Keystone MRO Supply", (60, 300), 80, (1, 2), 10, (5, 20), 2.5, (2, 1, 3)),
    "WRG": ("Wear ring", "Halvard Castings", (90, 260), 100, (5, 6), 20, (4, 12), 4.0, (0, 0, 5)),
}
FAMILIES = ["CP-100", "CP-200", "CP-400", "GB-20", "GB-40"]
DC_SCALE = {"DC-CMH": 1.0, "DC-DFW": 0.75, "DC-ATL": 0.55}
DC_CODE_VARIANTS = {
    "DC-CMH": ["dc-cmh", "CMH", "DC-CMH ", "DC CMH"],
    "DC-DFW": ["dc-dfw", "DFW", "DC-DFW ", "DC DFW"],
    "DC-ATL": ["dc-atl", "ATL", "DC-ATL ", "DC ATL"],
}
# Round-robin order so any prefix of the catalog mixes all three classes (3:2:5).
CLASS_PATTERN = ["smooth", "intermittent", "seasonal", "intermittent", "smooth",
                 "intermittent", "seasonal", "intermittent", "smooth", "intermittent"]


@dataclass
class SyntheticData:
    sku_master: pd.DataFrame
    erp_export: pd.DataFrame
    inventory: pd.DataFrame
    truth: pd.DataFrame          # clean weekly demand: unique_id, ds, y (for tests only)


def _catalog(rng: np.random.Generator, n_skus: int) -> pd.DataFrame:
    queues: dict[str, list] = {"smooth": [], "seasonal": [], "intermittent": []}
    for cat, spec in CATEGORIES.items():
        counts = dict(zip(["smooth", "seasonal", "intermittent"], spec[8]))
        numbers = rng.choice(np.arange(10, 9999), size=sum(counts.values()), replace=False)
        j = 0
        for cls, n in counts.items():
            for _ in range(n):
                queues[cls].append((cat, int(numbers[j])))
                j += 1
    for q in queues.values():
        rng.shuffle(q)
    total = sum(len(q) for q in queues.values())
    if n_skus > total:
        raise ValueError(f"n_skus={n_skus} exceeds catalog size {total}")
    order = []
    i = 0
    while len(order) < total:
        cls = CLASS_PATTERN[i % len(CLASS_PATTERN)]
        if queues[cls]:
            order.append((cls, *queues[cls].pop(0)))
        i += 1
    rows = []
    for cls, cat, num in order[:n_skus]:
        desc, supplier, (clo, chi), pallet, lts, moq, (blo, bhi), lot, _ = CATEGORIES[cat]
        rows.append({
            "sku": f"SP-{cat}-{num:04d}",
            "description": f"{desc}, {FAMILIES[int(rng.integers(len(FAMILIES)))]}",
            "category": cat,
            "supplier": supplier,
            "unit_cost_usd": round(float(rng.uniform(clo, chi)), 2),
            "pallet_qty": int(pallet),
            "lead_time_weeks": int(rng.choice(lts)),
            "moq": int(moq),
            "demand_class": cls,           # generator label; the pipeline never reads it
            "base_rate": float(rng.uniform(blo, bhi)),
            "lot_mean": float(lot * rng.uniform(0.7, 1.3)),
            "occurrence": float(rng.uniform(0.08, 0.35)),
            "growth": float(rng.uniform(-0.10, 0.20)),
            "amplitude": float(rng.uniform(0.35, 0.60)),
            "peak": int(rng.integers(9, 13)),
        })
    return pd.DataFrame(rows)


def _demand(rng: np.random.Generator, cat: pd.DataFrame, dcs: list[str], weeks: pd.DatetimeIndex) -> pd.DataFrame:
    t = np.arange(len(weeks))
    frames = []
    for row in cat.itertuples():
        for dc in dcs:
            scale = DC_SCALE.get(dc, 0.7) * rng.uniform(0.8, 1.2)
            if row.demand_class == "intermittent":
                occurs = rng.random(len(t)) < row.occurrence * (0.6 + 0.4 * scale)
                k = 2.0
                mu = max(row.lot_mean - 1, 0.1)
                sizes = 1 + rng.negative_binomial(k, k / (k + mu), size=len(t))
                y = np.where(occurs, sizes, 0)
            else:
                lam = row.base_rate * scale * (1 + row.growth) ** (t / 52)
                if row.demand_class == "seasonal":
                    lam = lam * (1 + row.amplitude * np.cos(2 * np.pi * ((t % 13) - row.peak) / 13))
                k = 8.0
                y = rng.negative_binomial(k, k / (k + lam))
            frames.append(pd.DataFrame({"unique_id": f"{row.sku}|{dc}", "ds": weeks, "y": y.astype(int)}))
    return pd.concat(frames, ignore_index=True)


def _erp_export(rng: np.random.Generator, truth: pd.DataFrame) -> pd.DataFrame:
    """Split weekly demand into shipped order lines and add export mess."""
    pos = truth[truth["y"] > 0]
    lines = []
    order_no = 100000
    for r in pos.itertuples():
        sku, dc = r.unique_id.split("|")
        n_lines = 1 + int(rng.binomial(min(r.y - 1, 3), 0.4))
        qtys = rng.multinomial(r.y - n_lines, np.ones(n_lines) / n_lines) + 1
        for q in qtys:
            order_no += 1
            lines.append({
                "order_id": f"SO-{order_no}",
                "ship_date": r.ds + pd.Timedelta(days=int(rng.integers(0, 5))),
                "dc_code": dc,
                "sku": sku,
                "qty": int(q),
            })
    exp = pd.DataFrame(lines)
    n = len(exp)
    # 1.5% of rows carry a non-standard DC code, 0.5% a lower-case SKU.
    idx = rng.choice(n, size=int(0.015 * n), replace=False)
    exp.loc[idx, "dc_code"] = [rng.choice(DC_CODE_VARIANTS.get(d, [d])) for d in exp.loc[idx, "dc_code"]]
    idx = rng.choice(n, size=int(0.005 * n), replace=False)
    exp.loc[idx, "sku"] = exp.loc[idx, "sku"].str.lower()
    # 0.4% of order lines exported twice.
    dup = exp.iloc[rng.choice(n, size=int(0.004 * n), replace=False)]
    # 0.6% return credits (negative quantities) that are not demand.
    ret = exp.iloc[rng.choice(n, size=int(0.006 * n), replace=False)].copy()
    ret["order_id"] = [f"RMA-{i}" for i in range(5000, 5000 + len(ret))]
    ret["qty"] = -ret["qty"]
    ret["ship_date"] = ret["ship_date"] + pd.Timedelta(days=14)
    exp = pd.concat([exp, dup, ret], ignore_index=True)
    exp = exp.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    exp["ship_date"] = pd.to_datetime(exp["ship_date"]).dt.strftime("%Y-%m-%d")
    return exp


def _inventory(rng: np.random.Generator, cat: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    last = truth["ds"].max()
    recent = truth[truth["ds"] > last - pd.Timedelta(weeks=13)].groupby("unique_id")["y"].mean()
    lt = cat.set_index("sku")["lead_time_weeks"]
    moq = cat.set_index("sku")["moq"]
    rows = []
    for uid, mean in recent.items():
        sku, dc = uid.split("|")
        cover = rng.choice([0.0, rng.uniform(0.5, 3), rng.uniform(3, 8), rng.uniform(8, 20)], p=[0.1, 0.3, 0.35, 0.25])
        on_hand = int(round(mean * cover))
        if rng.random() < 0.3:
            qty = int(max(moq[sku], round(mean * 4)))
            due = last + pd.Timedelta(weeks=int(rng.integers(1, lt[sku] + 1)))
        else:
            qty, due = 0, pd.NaT
        rows.append({"sku": sku, "dc_code": dc, "on_hand": on_hand, "on_order_qty": qty,
                     "on_order_due_week": "" if pd.isna(due) else due.strftime("%Y-%m-%d")})
    return pd.DataFrame(rows)


def generate(cfg: dict) -> SyntheticData:
    d = cfg["data"]
    rng = np.random.default_rng(d["seed"])
    weeks = pd.date_range(d["start_week"], periods=d["n_weeks"], freq="W-MON")
    cat = _catalog(rng, d["n_skus"])
    truth = _demand(rng, cat, d["dcs"], weeks)
    erp = _erp_export(rng, truth)
    inv = _inventory(rng, cat, truth)
    master = cat[["sku", "description", "category", "supplier", "unit_cost_usd",
                  "pallet_qty", "lead_time_weeks", "moq"]].copy()
    master["pallet_qty"] = master["pallet_qty"].astype("Int64")
    # Two SKUs arrive with a blank pallet quantity, as in the real item master.
    if len(master) >= 10:
        master.loc[rng.choice(len(master), size=2, replace=False), "pallet_qty"] = pd.NA
    return SyntheticData(master, erp, inv, truth)


def write(data: SyntheticData, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data.sku_master.to_csv(out_dir / "sku_master.csv", index=False)
    data.erp_export.to_csv(out_dir / "erp_demand_export.csv", index=False)
    data.inventory.to_csv(out_dir / "inventory_snapshot.csv", index=False)


if __name__ == "__main__":
    from .config import ROOT, load_config

    out = ROOT / "data"
    write(generate(load_config()), out)
    print(f"wrote synthetic Corvane data to {out}")
