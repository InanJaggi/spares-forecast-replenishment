import math

import numpy as np
import pandas as pd
import pytest

from spares import adapter, generate, policy
from spares.config import load_config
from spares.optimize import integer_demand, solve_plan


def small_cfg(**data):
    return load_config(overrides={"data": {"n_skus": 4, "n_weeks": 60, **data},
                                  "forecast": {"holdout_weeks": 4, "cv_windows": 2},
                                  "hand_rule": {"sweep_min_weeks": [4]},
                                  "plan": {"solver_time_limit_s": 3.0, "sim_time_limit_s": 1.0}})


def test_generator_is_deterministic():
    cfg = small_cfg()
    a, b = generate.generate(cfg), generate.generate(cfg)
    pd.testing.assert_frame_equal(a.erp_export, b.erp_export)
    pd.testing.assert_frame_equal(a.sku_master, b.sku_master)
    c = generate.generate(small_cfg(seed=7))
    assert not a.truth["y"].equals(c.truth["y"])


def test_adapter_recovers_clean_demand(tmp_path):
    cfg = small_cfg()
    data = generate.generate(cfg)
    generate.write(data, tmp_path)
    master, _ = adapter.load_sku_master(tmp_path / "sku_master.csv")
    d, log = adapter.load_demand(tmp_path / "erp_demand_export.csv", master, cfg["data"]["dcs"],
                                 cfg["data"]["start_week"], cfg["data"]["n_weeks"])
    merged = d.merge(data.truth, on=["unique_id", "ds"], suffixes=("", "_true"))
    assert len(merged) == len(data.truth)
    assert (merged["y"] == merged["y_true"]).all()
    assert adapter.normalize_dc("dc cmh ", ["DC-CMH"]) == "DC-CMH"


def test_policy_math():
    z = policy.z_value(0.95)
    assert z == pytest.approx(1.6449, abs=1e-3)
    ss = policy.safety_stock(sigma=10, lead_time_weeks=3, review_weeks=1, service_level=0.95)
    assert ss == pytest.approx(z * 10 * 2)
    assert policy.reorder_point(np.array([5, 5, 5, 5, 9]), 3, 1, ss) == pytest.approx(20 + ss)
    assert policy.safety_stock(0, 3, 1, 0.95) == 0
    with pytest.raises(ValueError):
        policy.z_value(1.0)


def test_integer_demand_keeps_total():
    f = np.array([0.3, 0.3, 0.3, 0.3, 2.6])
    d = integer_demand(f)
    assert d.sum() == round(f.sum()) and (d >= 0).all()


def test_optimizer_respects_moq_capacity_budget():
    cfg = load_config(overrides={"capacity": {"weekly_budget_usd": 3000.0, "truck_pallets": 2,
                                              "max_trucks_per_week": {"DC-CMH": 1, "DC-DFW": 1, "DC-ATL": 1}},
                                 "plan": {"order_weeks": 3, "solver_time_limit_s": 5.0}})
    items = pd.DataFrame({
        "unique_id": ["A|DC-CMH", "B|DC-CMH", "C|DC-DFW"], "sku": ["A", "B", "C"],
        "dc": ["DC-CMH", "DC-CMH", "DC-DFW"], "unit_cost_usd": [20.0, 150.0, 60.0],
        "pallet_qty": [50, 10, 20], "lead_time_weeks": [1, 2, 1], "moq": [40, 5, 30],
        "safety_stock": [30.0, 6.0, 20.0], "on_hand": [10, 0, 5]})
    demand = np.array([[30.0] * 5, [4.0] * 5, [15.0] * 5])
    pipeline = np.zeros((3, 5), dtype=int)
    res = solve_plan(items, demand, pipeline, cfg)
    o = res.orders
    assert len(o) > 0
    moq = items.set_index("unique_id")["moq"]
    assert (o["qty"] >= o["unique_id"].map(moq)).all()
    spend = o.groupby("order_week")["value_usd"].sum()
    assert (spend <= 3000.0 + 1e-6).all()
    pal = o.groupby(["dc", "arrival_week"])["pallets"].sum()
    assert (pal <= 2).all()
    ps = items.set_index("unique_id")["pallet_qty"]
    assert (o["pallets"] == [math.ceil(q / ps[u]) for q, u in zip(o["qty"], o["unique_id"])]).all()


def test_end_to_end_tiny(tmp_path):
    from spares.pipeline import run
    h = run(small_cfg(), tmp_path / "data", tmp_path / "out")
    for f in ["plan.csv", "report.html", "simulation_summary.csv", "backtest_summary.csv"]:
        assert (tmp_path / "out" / f).exists()
    assert 0 <= h["sim_optimized"]["fill_rate_all"] <= 1
    assert h["sim_optimized"]["weeks_over_budget"] == 0
    assert h["sim_optimized"]["dock_weeks_over_capacity"] == 0
