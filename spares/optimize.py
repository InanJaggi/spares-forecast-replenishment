"""Replenishment plan as one CP-SAT model (OR-Tools).

For every SKU-DC the model decides how much to order in each of the next P weeks.
An order placed in week k arrives at the start of week k + lead time.

Minimize   holding cost on end-of-week stock
         + shortage penalty on demand that cannot be served (lost sales)
         + a soft penalty for every unit-week below safety stock
         + a fixed cost per order line and a freight cost per pallet
subject to inventory balance with the forecast as demand,
           MOQ (an order is 0 or at least MOQ units),
           pallets = ceil(units / pallet qty),
           inbound pallets per DC-week <= dock truck slots x pallets per truck,
           purchase spend per order week <= the budget.

CP-SAT needs integers, so demand is rounded on its running total (no demand is
lost to rounding) and costs are in thousandths of a dollar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

SCALE = 1000  # objective units per dollar


@dataclass
class PlanResult:
    orders: pd.DataFrame       # unique_id, sku, dc, order_week, arrival_week, qty, pallets, value_usd
    trucks: pd.DataFrame       # dc, week, pallets, trucks, max_trucks
    projection: pd.DataFrame   # unique_id, week, demand, arrivals, inventory, short, below_ss
    status: str
    objective_usd: float
    bound_usd: float
    wall_time_s: float
    costs: dict


def integer_demand(f: np.ndarray) -> np.ndarray:
    c = np.round(np.cumsum(np.maximum(np.asarray(f, dtype=float), 0.0)))
    return np.diff(np.concatenate([[0.0], c])).astype(int)


def cost_rates(items: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["costs"]
    r = pd.DataFrame(index=items.index)
    r["hold"] = items["unit_cost_usd"] * c["holding_rate_annual"] / 52.0
    r["short"] = items["unit_cost_usd"] * c["shortage_pct_of_cost"] + c["shortage_fixed_usd"]
    r["gap"] = r["short"] * c["ss_shortfall_share"]
    return r


def reorder_point_hint(on_hand: int, pipe: np.ndarray, D: np.ndarray, L: int, W: int, P: int, ss: int,
                       moq: int, qmax: int):
    """Starting solution for CP-SAT: the plain reorder-point rule. In each order week,
    order enough to bring stock at the arrival week back to safety stock (at least MOQ).
    It ignores budget and dock limits; CP-SAT repairs and improves it."""
    q = [0] * P
    arrive = np.zeros(W + 1, dtype=int)

    def project():
        inv, short, cur = [], [], on_hand
        for k in range(W):
            avail = cur + (int(pipe[k]) if k >= 1 else 0) + int(arrive[k])
            inv.append(max(avail - int(D[k]), 0))
            short.append(max(int(D[k]) - avail, 0))
            cur = inv[-1]
        return inv, short

    for k in range(P):
        a = k + L
        if a >= W:
            continue
        inv, _ = project()
        before = inv[a - 1] if a >= 1 else on_hand
        net = before + (int(pipe[a]) if a >= 1 else 0) + int(arrive[a]) - int(D[a])
        if net < ss:
            q[k] = min(max(moq, ss - net), qmax)
            arrive[a] += q[k]
    inv, short = project()
    return q, inv, short


def solve_plan(items: pd.DataFrame, demand: np.ndarray, pipeline: np.ndarray, cfg: dict,
               time_limit_s: float | None = None) -> PlanResult:
    """items (one row per SKU-DC, positional order matches demand/pipeline rows) needs:
    unique_id, sku, dc, unit_cost_usd, pallet_qty, lead_time_weeks, moq, safety_stock, on_hand.
    demand: forecast, shape (n_items, T). pipeline: open-PO arrivals, shape (n_items, T);
    column 0 must already be included in on_hand."""
    items = items.reset_index(drop=True)
    n, T = demand.shape
    P = cfg["plan"]["order_weeks"]
    cap = cfg["capacity"]
    truck_pal = cap["truck_pallets"]
    rates = cost_rates(items, cfg)
    line_cost = int(round(cfg["costs"]["order_line_usd"] * SCALE))
    pallet_cost = int(round(cfg["costs"]["freight_per_pallet_usd"] * SCALE))
    budget_cents = int(round(cap["weekly_budget_usd"] * 100))

    m = cp_model.CpModel()
    D = np.vstack([integer_demand(demand[i]) for i in range(n)])
    hold_c = np.round(rates["hold"].to_numpy() * SCALE).astype(int)
    short_c = np.round(rates["short"].to_numpy() * SCALE).astype(int)
    gap_c = np.round(rates["gap"].to_numpy() * SCALE).astype(int)
    lead = items["lead_time_weeks"].to_numpy().astype(int)
    q, y, pal = {}, {}, {}
    inv, short, gap = {}, {}, {}
    obj_vars, obj_coefs = [], []

    def cost(var, coef):
        obj_vars.append(var)
        obj_coefs.append(int(coef))

    for it in items.itertuples():
        i = it.Index
        L = int(it.lead_time_weeks)
        W = min(T, L + P)                       # weeks this item's decisions can affect
        ss = int(round(it.safety_stock))
        ps = int(it.pallet_qty)
        moq = int(it.moq)
        need = int(math.ceil(D[i, :W].sum() + 2 * ss))
        qmax = max(moq, need)
        hq, h_inv, h_short = reorder_point_hint(int(it.on_hand), pipeline[i], D[i], L, W, P, ss, moq, qmax)
        for k in range(P):
            if k + L >= T:
                continue
            q[i, k] = m.new_int_var(0, qmax, f"q_{i}_{k}")
            y[i, k] = m.new_bool_var(f"y_{i}_{k}")
            pal[i, k] = m.new_int_var(0, math.ceil(qmax / ps), f"pal_{i}_{k}")
            m.add(q[i, k] >= moq * y[i, k])
            m.add(q[i, k] <= qmax * y[i, k])
            m.add(pal[i, k] * ps >= q[i, k])
            cost(y[i, k], line_cost)
            cost(pal[i, k], pallet_cost)
            m.add_hint(q[i, k], hq[k])
            m.add_hint(y[i, k], int(hq[k] > 0))
            m.add_hint(pal[i, k], math.ceil(hq[k] / ps))
        ub = int(it.on_hand + pipeline[i].sum() + P * qmax)
        prev = int(it.on_hand)
        for k in range(W):
            arrivals = int(pipeline[i, k]) if k >= 1 else 0
            expr = prev + arrivals - int(D[i, k])
            if (i, k - L) in q:
                expr = expr + q[i, k - L]
            inv[i, k] = m.new_int_var(0, ub, f"inv_{i}_{k}")
            short[i, k] = m.new_int_var(0, int(D[i, k]), f"short_{i}_{k}")
            gap[i, k] = m.new_int_var(0, ss, f"gap_{i}_{k}")
            m.add(inv[i, k] == expr + short[i, k])
            m.add(inv[i, k] + gap[i, k] >= ss)
            m.add_hint(inv[i, k], h_inv[k])
            m.add_hint(short[i, k], h_short[k])
            m.add_hint(gap[i, k], max(ss - h_inv[k], 0))
            cost(inv[i, k], hold_c[i])
            cost(short[i, k], short_c[i])
            cost(gap[i, k], gap_c[i])
            prev = inv[i, k]

    # Inbound pallets per DC and arrival week, including open POs already on the way,
    # must fit the trucks the dock can receive.
    for dc in items["dc"].unique():
        idx = np.nonzero((items["dc"] == dc).to_numpy())[0]
        max_t = int(cap["max_trucks_per_week"][dc])
        pipe_pal = np.ceil(pipeline[idx] / items["pallet_qty"].to_numpy()[idx, None]).astype(int).sum(axis=0)
        for k in range(1, T):
            new = [pal[i, k - lead[i]] for i in idx if (i, k - lead[i]) in pal]
            if not new:
                continue
            pipe = int(pipe_pal[k])
            m.add(cp_model.LinearExpr.sum(new) + pipe <= max(max_t * truck_pal, pipe))

    # Corvane-wide purchase budget per order week.
    cents = np.round(items["unit_cost_usd"].to_numpy() * 100).astype(int)
    for k in range(P):
        keys = [i for i in range(n) if (i, k) in q]
        if keys:
            m.add(cp_model.LinearExpr.weighted_sum([q[i, k] for i in keys], [int(cents[i]) for i in keys])
                  <= budget_cents)

    m.minimize(cp_model.LinearExpr.weighted_sum(obj_vars, obj_coefs))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s or cfg["plan"]["solver_time_limit_s"]
    solver.parameters.num_workers = cfg["plan"]["solver_workers"]
    solver.parameters.random_seed = cfg["data"]["seed"]
    solver.parameters.relative_gap_limit = cfg["plan"].get("relative_gap_limit", 0.0)
    status = solver.solve(m)
    if status == cp_model.UNKNOWN:              # no solution yet under a busy CPU: allow more time once
        solver.parameters.max_time_in_seconds *= 3
        status = solver.solve(m)
    status_name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT returned {status_name}")

    uid = items["unique_id"].tolist()
    pq = items["pallet_qty"].to_numpy()
    hold_r, short_r, gap_r = (rates[c].to_numpy() for c in ("hold", "short", "gap"))
    order_rows = []
    for (i, k), var in q.items():
        qty = solver.value(var)
        if qty > 0:
            it = items.loc[i]
            order_rows.append({
                "unique_id": it.unique_id, "sku": it.sku, "dc": it.dc, "order_week": k,
                "arrival_week": k + int(it.lead_time_weeks), "qty": qty,
                "pallets": math.ceil(qty / it.pallet_qty), "value_usd": round(qty * it.unit_cost_usd, 2),
            })
    orders = pd.DataFrame(order_rows, columns=["unique_id", "sku", "dc", "order_week", "arrival_week",
                                               "qty", "pallets", "value_usd"])
    truck_rows = []
    for dc in items["dc"].unique():
        idx = np.nonzero((items["dc"] == dc).to_numpy())[0]
        for k in range(1, T):
            pipe = sum(math.ceil(pipeline[i, k] / pq[i]) for i in idx)
            new = orders.loc[(orders["dc"] == dc) & (orders["arrival_week"] == k), "pallets"].sum()
            if pipe + new == 0:
                continue
            truck_rows.append({"dc": dc, "week": k, "pallets_new": int(new), "pallets_open_po": int(pipe),
                               "pallets": int(pipe + new), "trucks": math.ceil((pipe + new) / truck_pal),
                               "max_trucks": int(cap["max_trucks_per_week"][dc])})
    truck_df = pd.DataFrame(truck_rows, columns=["dc", "week", "pallets_new", "pallets_open_po", "pallets",
                                                 "trucks", "max_trucks"])
    proj_rows = []
    for (i, k), var in inv.items():
        L = lead[i]
        arr = (int(pipeline[i, k]) if k >= 1 else 0) + (solver.value(q[i, k - L]) if (i, k - L) in q else 0)
        proj_rows.append({"unique_id": uid[i], "week": k, "demand": int(D[i, k]),
                          "arrivals": arr, "inventory": solver.value(var),
                          "short": solver.value(short[i, k]), "below_ss": solver.value(gap[i, k])})
    projection = pd.DataFrame(proj_rows)
    costs = {
        "holding_usd": float(sum(hold_r[i] * solver.value(v) for (i, k), v in inv.items())),
        "shortage_usd": float(sum(short_r[i] * solver.value(v) for (i, k), v in short.items())),
        "below_ss_usd": float(sum(gap_r[i] * solver.value(v) for (i, k), v in gap.items())),
        "order_lines_usd": float(cfg["costs"]["order_line_usd"] * sum(solver.value(v) for v in y.values())),
        "freight_usd": float(cfg["costs"]["freight_per_pallet_usd"] * orders["pallets"].sum()),
    }
    return PlanResult(orders, truck_df, projection, status_name, solver.objective_value / SCALE,
                      solver.best_objective_bound / SCALE, solver.wall_time, costs)
