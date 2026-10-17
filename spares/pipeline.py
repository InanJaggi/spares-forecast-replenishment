"""One command for the whole Corvane spares run.

    python -m spares.pipeline            # full run, writes data/ and out/
    python -m spares.pipeline --config path/to.toml --out out_dir

Stages: synthetic ERP data -> adapter -> holdout replay (backtest, rolling
forecasts, policy, optimized vs hand-rule simulation) -> production plan
(backtest on all history, forecast, safety stock, CP-SAT plan) -> CSVs + HTML.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import adapter, generate, policy, report
from . import forecast as F
from .config import DEFAULT_CONFIG, ROOT, load_config
from .optimize import solve_plan
from .simulate import hand_rule_levels, hand_rule_policy, optimized_policy, simulate

_T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - _T0:6.1f}s] {msg}", flush=True)


def build_items(master: pd.DataFrame, dcs: list[str]) -> pd.DataFrame:
    rows = [{"unique_id": f"{s}|{d}", "sku": s, "dc": d} for s in master["sku"] for d in dcs]
    items = pd.DataFrame(rows).merge(master, on="sku")
    return items.sort_values("unique_id").reset_index(drop=True)


def wide(fc: pd.DataFrame, items: pd.DataFrame, col: str = "Selected") -> pd.DataFrame:
    w = fc.pivot(index="unique_id", columns="ds", values=col).clip(lower=0)
    return w.loc[items["unique_id"]]


def run_backtest(df: pd.DataFrame, seg: pd.DataFrame, fcfg: dict):
    cv = F.backtest(df, seg, fcfg["season_length"], fcfg["cv_horizon"], fcfg["cv_windows"])
    scale = F.naive_scale(df[df["ds"] <= cv["cutoff"].min()])
    selection = F.select(F.series_scores(cv, scale))
    cv = F.add_selected(cv, selection)
    scores = F.series_scores(cv, scale)
    sigma = scores[scores["model"] == "Selected"].set_index("unique_id")["rmse"]
    return cv, scores, selection, sigma


def run(cfg: dict, data_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    d, fcfg, pcfg = cfg["data"], cfg["forecast"], cfg["policy"]
    m = fcfg["season_length"]

    # 1. Synthetic ERP exports, then the adapter that cleans them.
    generate.write(generate.generate(cfg), data_dir)
    master, mlog = adapter.load_sku_master(data_dir / "sku_master.csv")
    demand, dlog = adapter.load_demand(data_dir / "erp_demand_export.csv", master, d["dcs"], d["start_week"], d["n_weeks"])
    snapshot = adapter.load_inventory(data_dir / "inventory_snapshot.csv", d["dcs"])
    cleaning = {**mlog, **dlog}
    log(f"data: {cleaning['series']} series x {cleaning['weeks']} weeks; cleaning {json.dumps(cleaning)}")
    items = build_items(master, d["dcs"])
    T = cfg["plan"]["order_weeks"] + int(items["lead_time_weeks"].max())
    last_week = demand["ds"].max()

    # 2. Holdout replay: everything below uses only weeks before the holdout.
    H = fcfg["holdout_weeks"]
    holdout_start = last_week - pd.Timedelta(weeks=H - 1)
    train = demand[demand["ds"] < holdout_start]
    seg_h = F.segment(train)
    cv_h, scores_h, sel_h, sigma_h = run_backtest(train, seg_h, fcfg)
    log(f"holdout replay: backtest on {train['ds'].nunique()} weeks before {holdout_start.date()} done")
    rolling = F.rolling_forecasts(demand, seg_h, sel_h, m, holdout_start, H, T)
    log(f"holdout replay: rolling {T}-week forecasts from {H} weekly origins done")
    horizon = (rolling["ds"] - rolling["origin"]).dt.days // 7
    acc = rolling[(horizon < fcfg["cv_horizon"]) & (rolling["ds"] <= last_week)]
    acc = acc.merge(demand, on=["unique_id", "ds"])
    holdout_acc = F.summary_table(F.series_scores(acc, F.naive_scale(train)), seg_h)

    fc_by_week = {}
    for t in range(H):
        origin = holdout_start + pd.Timedelta(weeks=t)
        fc_by_week[t] = wide(rolling[rolling["origin"] == origin], items).to_numpy()
    pol_h = policy.build_policy(items, sigma_h, wide(rolling[rolling["origin"] == holdout_start], items),
                                pcfg["service_level"], pcfg["review_period_weeks"])
    actual = demand[demand["ds"] >= holdout_start].pivot(index="unique_id", columns="ds", values="y")
    actual = actual.loc[items["unique_id"]].to_numpy().astype(int)
    hr = cfg["hand_rule"]
    trailing = hand_rule_levels(items, train, hr["trailing_weeks"], 1, 1)["avg_weekly"].to_numpy()
    on_hand0 = np.round(trailing * (items["lead_time_weeks"].to_numpy() + 2)).astype(int)
    segments_h = seg_h.set_index("unique_id")["segment"]

    solver_log: list = []
    sim_opt = simulate("Optimized (forecast + CP-SAT)", pol_h, actual, on_hand0,
                       optimized_policy(pol_h, fc_by_week, cfg, solver_log), cfg, segments_h, solver_log)
    log(f"holdout replay: optimized policy simulated ({len(solver_log)} weekly CP-SAT solves, "
        f"{sum(s['wall_time_s'] for s in solver_log):.1f}s solver time)")
    levels = hand_rule_levels(items, train, hr["trailing_weeks"], hr["min_weeks"], hr["max_weeks"])
    sim_hand = simulate(f"Hand rule (min {hr['min_weeks']} / max {hr['max_weeks']} weeks)", items, actual,
                        on_hand0, hand_rule_policy(items, levels), cfg, segments_h)
    sweep = []
    for mn in hr["sweep_min_weeks"]:
        mx = mn + hr["sweep_gap_weeks"]
        lv = hand_rule_levels(items, train, hr["trailing_weeks"], mn, mx)
        r = simulate(f"min {mn} / max {mx}", items, actual, on_hand0, hand_rule_policy(items, lv), cfg, segments_h)
        sweep.append({"min_weeks": mn, "max_weeks": mx, **r.metrics})
    sweep = pd.DataFrame(sweep)
    log("holdout replay: hand rule and sweep simulated")

    # 3. Production plan: keep the segment, model choice and error estimate from the
    #    pre-holdout backtest (checked out of sample above), refit on all history.
    seg, sel, sigma = seg_h, sel_h, sigma_h
    backtest_tbl = F.summary_table(scores_h, seg_h)
    fc = F.forecast(demand, seg, sel, m, T)
    log(f"production: chosen models refit on all {d['n_weeks']} weeks, {T}-week forecast done")
    fw = wide(fc, items)
    pol = policy.build_policy(items, sigma, fw, pcfg["service_level"], pcfg["review_period_weeks"])
    plan_start = last_week + pd.Timedelta(weeks=1)
    snap = snapshot.set_index("unique_id").loc[items["unique_id"]]
    due = ((snap["on_order_due_week"] - plan_start).dt.days // 7).fillna(-1).astype(int).to_numpy()
    qty = snap["on_order_qty"].to_numpy().astype(int)
    on_hand = snap["on_hand"].to_numpy().astype(int) + np.where((qty > 0) & (due <= 0), qty, 0)
    pipeline = np.zeros((len(items), T), dtype=int)
    for i in np.nonzero((qty > 0) & (due >= 1) & (due < T))[0]:
        pipeline[i, due[i]] += qty[i]
    pol["on_hand"] = on_hand
    pol["open_po_qty"] = pipeline.sum(axis=1)
    res = solve_plan(pol, fw.to_numpy(), pipeline, cfg)
    log(f"production: CP-SAT plan {res.status}, objective ${res.objective_usd:,.0f}, "
        f"bound ${res.bound_usd:,.0f}, {res.wall_time_s:.1f}s")

    # 4. Outputs.
    week_date = lambda k: (plan_start + pd.Timedelta(weeks=int(k))).date().isoformat()
    plan = res.orders.merge(items[["unique_id", "description", "supplier", "unit_cost_usd", "moq", "pallet_qty"]],
                            on="unique_id")
    plan = plan.merge(pol[["unique_id", "safety_stock", "on_hand"]], on="unique_id")
    plan.insert(0, "order_week_start", plan["order_week"].map(week_date))
    plan.insert(1, "arrival_week_start", plan["arrival_week"].map(week_date))
    plan["safety_stock"] = plan["safety_stock"].round(1)
    plan = plan.sort_values(["dc", "order_week", "sku"]).drop(columns=["unique_id"])
    plan.to_csv(out_dir / "plan.csv", index=False)
    trucks = res.trucks.copy()
    trucks.insert(1, "week_start", trucks["week"].map(week_date))
    trucks.to_csv(out_dir / "truck_plan.csv", index=False)
    pol_out = pol.merge(sel.rename("model"), left_on="unique_id", right_index=True)
    pol_out = pol_out.merge(seg[["unique_id", "sbc_class", "segment"]], on="unique_id")
    pol_out.round(2).to_csv(out_dir / "policy.csv", index=False)
    backtest_tbl.to_csv(out_dir / "backtest_summary.csv", index=False)
    holdout_acc.to_csv(out_dir / "holdout_accuracy.csv", index=False)
    sim_tbl = pd.DataFrame([{"policy": s.name, **s.metrics} for s in (sim_opt, sim_hand)])
    sim_tbl.to_csv(out_dir / "simulation_summary.csv", index=False)
    sweep.to_csv(out_dir / "hand_rule_sweep.csv", index=False)
    pd.DataFrame(solver_log).to_csv(out_dir / "solver_log.csv", index=False)

    sel_row = backtest_tbl.set_index("model")
    hold_row = holdout_acc.set_index("model")
    headline = {
        "series": int(len(items)), "weeks": int(d["n_weeks"]), "holdout_weeks": int(H),
        "backtest_wape_selected": float(sel_row.at["Selected", "wape_all"]),
        "backtest_wape_seasonal_naive": float(sel_row.at[F.BASELINE, "wape_all"]),
        "backtest_mase_selected": float(sel_row.at["Selected", "mase_all"]),
        "backtest_mase_seasonal_naive": float(sel_row.at[F.BASELINE, "mase_all"]),
        "holdout_wape_selected": float(hold_row.at["Selected", "wape_all"]),
        "holdout_wape_seasonal_naive": float(hold_row.at[F.BASELINE, "wape_all"]),
        "holdout_mase_selected": float(hold_row.at["Selected", "mase_all"]),
        "holdout_mase_seasonal_naive": float(hold_row.at[F.BASELINE, "mase_all"]),
        "sim_optimized": sim_opt.metrics, "sim_hand_rule": sim_hand.metrics,
        "plan_status": res.status, "plan_objective_usd": res.objective_usd, "plan_bound_usd": res.bound_usd,
        "plan_solve_s": res.wall_time_s, "plan_order_lines": int(len(res.orders)),
        "plan_value_usd": float(res.orders["value_usd"].sum()),
        "runtime_s": time.perf_counter() - _T0,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(headline, indent=2, default=float))
    report.write_report(out_dir / "report.html", cfg=cfg, cleaning=cleaning, items=items, demand=demand,
                        rolling=rolling, fc=fc, selection=sel, holdout_start=holdout_start,
                        backtest_tbl=backtest_tbl, holdout_acc=holdout_acc, sim=(sim_opt, sim_hand),
                        sweep=sweep, plan=plan, trucks=trucks, plan_result=res, policy_tbl=pol_out,
                        solver_log=pd.DataFrame(solver_log), headline=headline)
    log(f"wrote {out_dir / 'plan.csv'} ({len(plan)} order lines) and {out_dir / 'report.html'}")
    return headline


def print_headline(h: dict) -> None:
    o, hr = h["sim_optimized"], h["sim_hand_rule"]
    print("\nForecast accuracy (WAPE / mean MASE, lower is better)")
    print(f"  backtest before holdout : selected {h['backtest_wape_selected']:.3f} / {h['backtest_mase_selected']:.3f}"
          f"   seasonal naive {h['backtest_wape_seasonal_naive']:.3f} / {h['backtest_mase_seasonal_naive']:.3f}")
    print(f"  holdout, {h['holdout_weeks']} weeks       : selected {h['holdout_wape_selected']:.3f} / {h['holdout_mase_selected']:.3f}"
          f"   seasonal naive {h['holdout_wape_seasonal_naive']:.3f} / {h['holdout_mase_seasonal_naive']:.3f}")
    print(f"\nHoldout replay on synthetic demand, {h['series']} SKU-DCs x {h['holdout_weeks']} weeks")
    print(f"  {'':28s}{'optimized':>14s}{'hand rule':>14s}")
    for key, label, fmt in [("fill_rate_all", "fill rate", "{:.1%}"), ("fill_rate_fast", "  fast movers", "{:.1%}"),
                            ("fill_rate_slow", "  slow movers", "{:.1%}"),
                            ("stockout_weeks_all", "stockout SKU-DC-weeks", "{:d}"),
                            ("avg_inventory_value_all", "avg inventory value", "${:,.0f}"),
                            ("avg_inventory_value_fast", "  fast movers", "${:,.0f}"),
                            ("avg_inventory_value_slow", "  slow movers", "${:,.0f}"),
                            ("total_cost_usd", "holding+ordering+shortage", "${:,.0f}"),
                            ("dock_weeks_over_capacity", "dock-weeks over capacity", "{:d}"),
                            ("weeks_over_budget", "weeks over budget", "{:d}")]:
        print(f"  {label:28s}{fmt.format(o[key]):>14s}{fmt.format(hr[key]):>14s}")
    print(f"\nPlan: {h['plan_order_lines']} order lines, ${h['plan_value_usd']:,.0f}, CP-SAT {h['plan_status']} "
          f"in {h['plan_solve_s']:.1f}s; total runtime {h['runtime_s']:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "out"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    h = run(load_config(args.config), Path(args.data), Path(args.out))
    print_headline(h)


if __name__ == "__main__":
    main()
