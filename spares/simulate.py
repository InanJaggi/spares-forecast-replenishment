"""Replay held-out weeks of synthetic demand under two replenishment policies.

Each week: open POs due this week arrive, the policy places orders (arriving after
the supplier lead time), then the week's actual demand is served from stock.
Demand that cannot be served is lost. Both policies start from the same stock and
see the same demand; only the ordering differs. Everything here is simulated on
synthetic data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .optimize import cost_rates, solve_plan

Policy = Callable[[int, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class SimResult:
    name: str
    weekly: pd.DataFrame     # unique_id, week, demand, served, lost, on_hand_end, order_qty
    orders: pd.DataFrame     # unique_id, dc, week, arrival_week, qty, pallets, value_usd
    metrics: dict
    solver_log: list


def hand_rule_levels(items: pd.DataFrame, history: pd.DataFrame, trailing_weeks: int,
                     min_weeks: float, max_weeks: float) -> pd.DataFrame:
    """Fixed min/max per SKU-DC: weeks of trailing average demand."""
    last = history["ds"].max()
    recent = history[history["ds"] > last - pd.Timedelta(weeks=trailing_weeks)]
    avg = recent.groupby("unique_id")["y"].mean()
    out = items[["unique_id"]].copy()
    out["avg_weekly"] = out["unique_id"].map(avg).fillna(0.0).to_numpy()
    out["min_level"] = np.ceil(out["avg_weekly"] * min_weeks)
    out["max_level"] = np.ceil(out["avg_weekly"] * max_weeks)
    return out


def hand_rule_policy(items: pd.DataFrame, levels: pd.DataFrame) -> Policy:
    mn = levels["min_level"].to_numpy()
    mx = levels["max_level"].to_numpy()
    moq = items["moq"].to_numpy()

    def decide(t: int, on_hand: np.ndarray, pipeline: np.ndarray) -> np.ndarray:
        position = on_hand + pipeline.sum(axis=1)
        q = np.where((position <= mn) & (mx - position > 0), mx - position, 0)
        return np.where(q > 0, np.maximum(q, moq), 0).astype(int)

    return decide


def optimized_policy(items: pd.DataFrame, forecasts: dict[int, np.ndarray], cfg: dict, log: list) -> Policy:
    """Re-solve the CP-SAT plan every week with that week's forecast; place week-0 orders."""
    def decide(t: int, on_hand: np.ndarray, pipeline: np.ndarray) -> np.ndarray:
        fc = forecasts[t]
        T = fc.shape[1]
        pipe = np.zeros((len(items), T), dtype=int)
        width = min(T - 1, pipeline.shape[1])
        pipe[:, 1:1 + width] = pipeline[:, :width]
        state = items.copy()
        state["on_hand"] = on_hand.astype(int)
        res = solve_plan(state, fc, pipe, cfg, time_limit_s=cfg["plan"]["sim_time_limit_s"])
        log.append({"week": t, "status": res.status, "objective_usd": res.objective_usd,
                    "bound_usd": res.bound_usd, "wall_time_s": res.wall_time_s})
        q = np.zeros(len(items), dtype=int)
        week0 = res.orders[res.orders["order_week"] == 0]
        idx = {u: i for i, u in enumerate(items["unique_id"])}
        for u, qty in zip(week0["unique_id"], week0["qty"]):
            q[idx[u]] = qty
        return q

    return decide


def simulate(name: str, items: pd.DataFrame, actual: np.ndarray, on_hand0: np.ndarray,
             decide: Policy, cfg: dict, segments: pd.Series, solver_log: list | None = None) -> SimResult:
    items = items.reset_index(drop=True)
    n, H = actual.shape
    lt = items["lead_time_weeks"].to_numpy().astype(int)
    arrivals = np.zeros((n, H + lt.max() + 1), dtype=int)
    on_hand = on_hand0.astype(int).copy()
    rows, order_rows = [], []
    for t in range(H):
        on_hand += arrivals[:, t]
        q = decide(t, on_hand.copy(), arrivals[:, t + 1:].copy())
        for i in np.nonzero(q)[0]:
            arrivals[i, t + lt[i]] += q[i]
            it = items.loc[i]
            order_rows.append({"unique_id": it.unique_id, "dc": it.dc, "week": t, "arrival_week": t + lt[i],
                               "qty": int(q[i]), "pallets": math.ceil(q[i] / it.pallet_qty),
                               "value_usd": float(q[i] * it.unit_cost_usd)})
        served = np.minimum(on_hand, actual[:, t])
        lost = actual[:, t] - served
        on_hand -= served
        for i in range(n):
            rows.append((items.at[i, "unique_id"], t, int(actual[i, t]), int(served[i]), int(lost[i]),
                         int(on_hand[i]), int(q[i])))
    weekly = pd.DataFrame(rows, columns=["unique_id", "week", "demand", "served", "lost", "on_hand_end", "order_qty"])
    orders = pd.DataFrame(order_rows, columns=["unique_id", "dc", "week", "arrival_week", "qty", "pallets", "value_usd"])
    metrics = score(weekly, orders, items, cfg, segments)
    return SimResult(name, weekly, orders, metrics, solver_log or [])


def score(weekly: pd.DataFrame, orders: pd.DataFrame, items: pd.DataFrame, cfg: dict, segments: pd.Series) -> dict:
    items = items.reset_index(drop=True)
    rates = cost_rates(items, cfg).set_index(items["unique_id"])
    cost = items.set_index("unique_id")["unit_cost_usd"]
    w = weekly.copy()
    w["segment"] = w["unique_id"].map(segments)
    w["inv_value"] = w["on_hand_end"] * w["unique_id"].map(cost)
    cap = cfg["capacity"]
    m: dict = {}
    for seg_name, g in [("all", w), ("fast", w[w["segment"] == "fast"]), ("slow", w[w["segment"] == "slow"])]:
        m[f"fill_rate_{seg_name}"] = g["served"].sum() / max(g["demand"].sum(), 1)
        m[f"stockout_weeks_{seg_name}"] = int((g["lost"] > 0).sum())
        m[f"avg_inventory_value_{seg_name}"] = g.groupby("week")["inv_value"].sum().mean()
    m["item_weeks"] = len(w)
    m["holding_usd"] = float((w["on_hand_end"] * w["unique_id"].map(rates["hold"])).sum())
    m["shortage_usd"] = float((w["lost"] * w["unique_id"].map(rates["short"])).sum())
    m["order_lines"] = len(orders)
    m["purchase_spend_usd"] = float(orders["value_usd"].sum())
    by_dock = orders.groupby(["dc", "arrival_week"])["pallets"].sum().reset_index()
    by_dock["trucks"] = np.ceil(by_dock["pallets"] / cap["truck_pallets"])
    by_dock["limit"] = by_dock["dc"].map(cap["max_trucks_per_week"]) * cap["truck_pallets"]
    m["trucks"] = int(by_dock["trucks"].sum())
    m["dock_weeks_over_capacity"] = int((by_dock["pallets"] > by_dock["limit"]).sum())
    spend = orders.groupby("week")["value_usd"].sum()
    m["weeks_over_budget"] = int((spend > cap["weekly_budget_usd"]).sum())
    m["pallets"] = int(orders["pallets"].sum())
    m["ordering_usd"] = (m["order_lines"] * cfg["costs"]["order_line_usd"]
                         + m["pallets"] * cfg["costs"]["freight_per_pallet_usd"])
    m["total_cost_usd"] = m["holding_usd"] + m["shortage_usd"] + m["ordering_usd"]
    return m
