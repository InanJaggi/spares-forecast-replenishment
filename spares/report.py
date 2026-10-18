"""Static HTML report with matplotlib charts embedded as PNG (no server needed)."""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LIGHT = "#c9c8c3"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "legend.frameon": False, "axes.titlesize": 9.5, "axes.titlecolor": INK, "axes.titleweight": "bold",
})


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chart_forecasts(demand, rolling, fc, selection, items, holdout_start) -> str:
    sel_items = items.assign(model=items["unique_id"].map(selection))
    vol = demand.groupby("unique_id")["y"].sum()
    fast = sel_items[sel_items["model"] == "AutoETS"].assign(v=lambda d: d["unique_id"].map(vol))
    slow = sel_items[sel_items["model"].isin(["CrostonOptimized", "ADIDA", "IMAPA"])].assign(
        v=lambda d: d["unique_id"].map(vol))
    def two(frame, second):
        ids = list(frame.sort_values("v", ascending=False)["unique_id"])
        return [ids[0], ids[min(second, len(ids) - 1)]] if ids else []
    picks = list(dict.fromkeys(two(fast, 3) + two(slow, 5) + list(items["unique_id"])))[:4]
    one_step = rolling[rolling["ds"] == rolling["origin"]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharex=True)
    start = holdout_start - pd.Timedelta(weeks=26)
    for ax, uid in zip(axes.flat, picks):
        h = demand[(demand["unique_id"] == uid) & (demand["ds"] >= start)]
        o = one_step[one_step["unique_id"] == uid]
        f = fc[fc["unique_id"] == uid]
        ax.plot(h["ds"], h["y"], color=MUTED, linewidth=1.5, label="actual demand")
        ax.plot(o["ds"], o["Selected"].clip(lower=0), color=BLUE, label=f"1-week-ahead, {selection[uid]}")
        ax.plot(o["ds"], o["SeasonalNaive"], color=ORANGE, linewidth=1.5, label="1-week-ahead, seasonal naive")
        ax.plot(f["ds"], f["Selected"].clip(lower=0), color=BLUE, alpha=0.45, label="forecast used in the plan")
        ax.axvline(holdout_start, color=LIGHT, linewidth=1)
        sku, dc = uid.split("|")
        ax.set_title(f"{sku} at {dc}", loc="left")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", labelrotation=0)
        ax.legend(fontsize=7.5, loc="upper left")
    for ax in axes[1]:
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    fig.supylabel("units per week", color=MUTED, fontsize=9)
    fig.tight_layout()
    return _png(fig)


def chart_tradeoff(sim_opt, sim_hand, sweep) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = sweep["avg_inventory_value_all"] / 1e6
    ax.plot(x, sweep["fill_rate_all"] * 100, color=ORANGE, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=2, label="hand rule, five min/max settings")
    for _, r in sweep.iterrows():
        ax.annotate(f"{r.min_weeks}/{r.max_weeks} wk", (r.avg_inventory_value_all / 1e6, r.fill_rate_all * 100),
                    textcoords="offset points", xytext=(6, -10), fontsize=7.5, color=MUTED)
    o = sim_opt.metrics
    ax.plot([o["avg_inventory_value_all"] / 1e6], [o["fill_rate_all"] * 100], color=BLUE, marker="o",
            markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none",
            label="optimized (forecast + safety stock + CP-SAT)")
    ax.annotate("optimized", (o["avg_inventory_value_all"] / 1e6, o["fill_rate_all"] * 100),
                textcoords="offset points", xytext=(8, 4), fontsize=8, color=INK)
    ax.set_xlabel("average inventory value, $M (end of week)")
    ax.set_ylabel("unit fill rate, %")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return _png(fig)


def chart_plan(trucks, cfg) -> str:
    dcs = cfg["data"]["dcs"]
    cap = cfg["capacity"]
    fig, axes = plt.subplots(1, len(dcs), figsize=(10, 2.8), sharey=True)
    weeks = sorted(trucks["week"].unique()) if len(trucks) else [1]
    for ax, dc in zip(np.atleast_1d(axes), dcs):
        t = trucks[trucks["dc"] == dc].set_index("week").reindex(weeks, fill_value=0)
        ax.bar(t.index, t["pallets_open_po"], width=0.6, color=LIGHT, label="open POs")
        ax.bar(t.index, t["pallets_new"], bottom=t["pallets_open_po"], width=0.6, color=BLUE, label="new orders")
        limit = cap["max_trucks_per_week"][dc] * cap["truck_pallets"]
        ax.axhline(limit, color=INK, linewidth=1)
        ax.text(weeks[0] - 0.4, limit, f"dock limit {limit}", fontsize=7.5, color=MUTED, va="bottom")
        ax.set_title(dc, loc="left")
        ax.set_xlabel("arrival week (1 = next week)")
        ax.set_xticks(weeks)
    np.atleast_1d(axes)[0].set_ylabel("inbound pallets")
    np.atleast_1d(axes)[0].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    return _png(fig)


def _table(df: pd.DataFrame, fmt: dict | None = None, cls: str = "") -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            s = fmt[c](v) if c in fmt and pd.notna(v) else ("" if pd.isna(v) else str(v))
            num = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            cells.append(f"<td class='{'n' if num else ''}'>{html.escape(s)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _acc_table(t: pd.DataFrame) -> str:
    t = t.copy()
    t["model"] = t["model"].replace({"Selected": "Chosen per series"})
    t = t[["model", "series", "wape_all", "mase_all", "wape_fast", "mase_fast", "wape_slow", "mase_slow"]]
    t.columns = ["model", "series", "WAPE all", "MASE all", "WAPE fast", "MASE fast", "WAPE slow", "MASE slow"]
    f = {c: (lambda v: f"{v:.3f}") for c in t.columns[2:]}
    return _table(t, f)


def write_report(path: Path, *, cfg, cleaning, items, demand, rolling, fc, selection, holdout_start,
                 backtest_tbl, holdout_acc, sim, sweep, plan, trucks, plan_result, policy_tbl, solver_log,
                 headline) -> None:
    sim_opt, sim_hand = sim
    o, hr = sim_opt.metrics, sim_hand.metrics
    last = demand["ds"].max()
    usd = lambda v: f"${v:,.0f}"
    pct = lambda v: f"{v:.1%}"

    comp_rows = [
        ("Unit fill rate, all", pct(o["fill_rate_all"]), pct(hr["fill_rate_all"])),
        ("Unit fill rate, fast movers", pct(o["fill_rate_fast"]), pct(hr["fill_rate_fast"])),
        ("Unit fill rate, slow movers", pct(o["fill_rate_slow"]), pct(hr["fill_rate_slow"])),
        ("Stockout SKU-DC-weeks (of %d)" % o["item_weeks"], str(o["stockout_weeks_all"]), str(hr["stockout_weeks_all"])),
        ("Average inventory value, all", usd(o["avg_inventory_value_all"]), usd(hr["avg_inventory_value_all"])),
        ("Average inventory value, fast movers", usd(o["avg_inventory_value_fast"]), usd(hr["avg_inventory_value_fast"])),
        ("Average inventory value, slow movers", usd(o["avg_inventory_value_slow"]), usd(hr["avg_inventory_value_slow"])),
        ("Holding + ordering + shortage cost", usd(o["total_cost_usd"]), usd(hr["total_cost_usd"])),
        ("Purchase spend in the window", usd(o["purchase_spend_usd"]), usd(hr["purchase_spend_usd"])),
        ("Order lines / pallets", f"{o['order_lines']} / {o['pallets']}", f"{hr['order_lines']} / {hr['pallets']}"),
        ("Dock-weeks over truck capacity", str(o["dock_weeks_over_capacity"]), str(hr["dock_weeks_over_capacity"])),
        ("Order weeks over budget", str(o["weeks_over_budget"]), str(hr["weeks_over_budget"])),
    ]
    comp = pd.DataFrame(comp_rows, columns=["metric", sim_opt.name, sim_hand.name])

    clean = pd.DataFrame([(k.replace("_", " "), v) for k, v in cleaning.items()], columns=["step", "count"])
    plan_sum = plan.groupby(["dc", "order_week_start"]).agg(
        lines=("sku", "size"), units=("qty", "sum"), pallets=("pallets", "sum"), value=("value_usd", "sum")
    ).reset_index()
    plan_view = plan[["dc", "order_week_start", "arrival_week_start", "sku", "description", "supplier", "qty",
                      "pallets", "value_usd", "moq", "on_hand", "safety_stock"]]
    choice = selection.value_counts().rename_axis("model").reset_index(name="series")
    pol_view = policy_tbl.sort_values("safety_stock_value_usd", ascending=False).head(12)[
        ["sku", "dc", "segment", "model", "lead_time_weeks", "mean_weekly_forecast", "sigma", "safety_stock",
         "reorder_point", "safety_stock_value_usd"]]
    gaps = (solver_log["objective_usd"] - solver_log["bound_usd"]) / solver_log["objective_usd"].abs()
    res = plan_result
    n2 = lambda v: f"{v:,.1f}"

    body = f"""
<header>
  <p class="kicker">Corvane Industrial · spare parts · synthetic data</p>
  <h1>Spares forecast and replenishment run</h1>
  <p class="meta">History {demand['ds'].min().date()} to {last.date()} ({demand['ds'].nunique()} weeks),
  {len(items)} SKU-DC series across {', '.join(cfg['data']['dcs'])}. Plan for the {cfg['plan']['order_weeks']}
  order weeks starting {(last + pd.Timedelta(weeks=1)).date()}. Runtime {headline['runtime_s']:.0f}s.</p>
</header>

<section class="summary">
  <p>Forecasts: per-series model choice cut the backtest WAPE from {headline['backtest_wape_seasonal_naive']:.3f}
  (seasonal naive) to {headline['backtest_wape_selected']:.3f}, and the holdout WAPE from
  {headline['holdout_wape_seasonal_naive']:.3f} to {headline['holdout_wape_selected']:.3f}.</p>
  <p>Holdout replay ({headline['holdout_weeks']} weeks, simulated on synthetic demand): unit fill rate
  {pct(o['fill_rate_all'])} with the optimized policy against {pct(hr['fill_rate_all'])} with the hand rule;
  stockout SKU-DC-weeks {o['stockout_weeks_all']} against {hr['stockout_weeks_all']}; average inventory value
  {usd(o['avg_inventory_value_all'])} against {usd(hr['avg_inventory_value_all'])}.</p>
  <p>Plan: {len(plan)} order lines worth {usd(plan['value_usd'].sum())}; CP-SAT status {res.status},
  objective {usd(res.objective_usd)} against a proven bound of {usd(res.bound_usd)}.</p>
</section>

<h2>1. Data adapter</h2>
<p>The ERP export is order lines. The adapter normalizes DC codes, drops re-exported lines and return credits,
fills weeks without shipments with zero and fills blank pallet quantities from the category median.</p>
{_table(clean, {}, 'narrow')}

<h2>2. Forecasts</h2>
<p>Series are split by average demand interval (Syntetos-Boylan): fast movers get SeasonalNaive and AutoETS
(13-week season), slow movers get SeasonalNaive, CrostonOptimized, ADIDA and IMAPA. The model with the lowest
backtest RMSE is chosen per series. WAPE is pooled absolute error over pooled demand; MASE is each series' MAE
over its in-sample one-step naive MAE, averaged across series.</p>
<h3>Backtest on the weeks before {holdout_start.date()} (statsforecast cross_validation,
{cfg['forecast']['cv_windows']} windows x {cfg['forecast']['cv_horizon']} weeks). Sets the model choice and the
safety-stock sigma used in the replay and in the plan.</h3>
{_acc_table(backtest_tbl)}
<p class="note">"Chosen per series" is scored on the same windows it was chosen on, so it is optimistic. The
holdout table below is the out-of-sample check.</p>
<h3>Holdout: {headline['holdout_weeks']} weeks from {holdout_start.date()}, 1 to {cfg['forecast']['cv_horizon']}
weeks ahead, models chosen on earlier weeks only</h3>
{_acc_table(holdout_acc)}
<div class="row">{_table(choice, {}, 'narrow')}</div>
<figure><img src="{chart_forecasts(demand, rolling, fc, selection, items, holdout_start)}" alt="Forecast versus actual for four SKU-DC series">
<figcaption>Two fast and two slow movers. Gray: actual weekly demand. Left of the vertical line is history;
from {holdout_start.date()} the lines are 1-week-ahead forecasts made each week; after {last.date()} the faint
line is the forecast the plan uses.</figcaption></figure>

<h2>3. Inventory policy</h2>
<p>Safety stock = z x sigma x sqrt(lead time + review week), with z for a
{cfg['policy']['service_level']:.0%} cycle service level and sigma the chosen model's backtest RMSE. Reorder
point = forecast over lead time + review week + safety stock. Total safety stock value
{usd(policy_tbl['safety_stock_value_usd'].sum())}. Largest twelve:</p>
{_table(pol_view, {'mean_weekly_forecast': n2, 'sigma': n2, 'safety_stock': n2, 'reorder_point': n2,
                   'safety_stock_value_usd': usd})}

<h2>4. Replenishment plan (CP-SAT)</h2>
<p>One model for all three DCs: holding + order-line + freight-per-pallet + shortage cost, plus a soft penalty below safety
stock, subject to MOQ, pallets per box truck ({cfg['capacity']['truck_pallets']}), dock limits per DC-week and a
{usd(cfg['capacity']['weekly_budget_usd'])} purchase budget per order week. Status {res.status} in
{res.wall_time_s:.1f}s, gap to bound {(res.objective_usd - res.bound_usd) / res.objective_usd:.1%}.
Cost split: holding {usd(res.costs['holding_usd'])}, shortage {usd(res.costs['shortage_usd'])}, below safety
stock {usd(res.costs['below_ss_usd'])}, order lines {usd(res.costs['order_lines_usd'])}, freight
{usd(res.costs['freight_usd'])}. Much of the shortage cost is in weeks before any new order can arrive.</p>
{_table(plan_sum, {'value': usd, 'units': lambda v: f'{v:,.0f}'}, 'narrow')}
<figure><img src="{chart_plan(trucks, cfg)}" alt="Inbound pallets per DC and week against dock limits">
<figcaption>Inbound pallets per DC and arrival week: open POs already on the way (gray) and new orders from
this plan (blue). The black line is the dock limit (trucks x pallets per truck).</figcaption></figure>
<details><summary>All {len(plan)} order lines</summary>
{_table(plan_view, {'value_usd': usd, 'safety_stock': n2}, 'small')}
</details>

<h2>5. Holdout replay (simulated, synthetic demand)</h2>
<p>The last {headline['holdout_weeks']} weeks of synthetic demand replayed week by week. Both policies start from
the same stock (trailing 26-week average x (lead time + 2) weeks) with no open POs. The optimized policy
re-forecasts and re-solves CP-SAT each week and places only that week's orders. The hand rule orders up to max
when stock plus open POs falls to min, rounded up to MOQ, with no capacity or budget check. Lost sales, no
backorders. Costs use the same rates as the optimizer.</p>
{_table(comp)}
<figure><img src="{chart_tradeoff(sim_opt, sim_hand, sweep)}" alt="Fill rate versus inventory value">
<figcaption>Fill rate against average inventory value. Orange: the hand rule at five min/max settings (weeks of
trailing average demand). Blue: the optimized policy.</figcaption></figure>
<p class="note">Weekly CP-SAT re-plans: {len(solver_log)} solves, {(solver_log['status'] == 'OPTIMAL').sum()}
reached the 1% gap target, worst gap {gaps.max():.1%}, mean solve {solver_log['wall_time_s'].mean():.1f}s
({cfg['plan']['solver_workers']} workers, {cfg['plan']['sim_time_limit_s']}s limit).</p>

<h2>6. Assumptions and limits</h2>
<ul>
<li>All demand, costs, lead times and stock levels are synthetic, generated with seed {cfg['data']['seed']}.</li>
<li>Lead times are fixed; real supplier lead times vary and would need their own safety stock term.</li>
<li>Safety stock assumes independent, roughly normal weekly errors. For lumpy slow movers that understates risk.</li>
<li>CP-SAT runs parallel workers under a time limit, so repeated runs can differ slightly.</li>
<li>Costs (carrying rate, shortage penalty, truck and order-line cost, budget) are my assumptions in
config/corvane.toml, not Corvane figures.</li>
</ul>
"""
    css = """
:root{--surface:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--accent:#2a78d6}
body{background:var(--surface);color:var(--ink);font:15px/1.55 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;
margin:0;padding:40px 16px 80px}
main{max-width:1000px;margin:0 auto}
h1{font-size:28px;margin:4px 0 8px;font-weight:650;letter-spacing:-.01em}
h2{font-size:19px;margin:44px 0 8px;padding-top:18px;border-top:1px solid var(--line);font-weight:650}
h3{font-size:14px;margin:22px 0 6px;color:var(--muted);font-weight:600}
.kicker{color:var(--muted);font-size:13px;margin:0}.meta{color:var(--muted);margin:0}
.summary{margin-top:22px;padding:14px 18px;border-left:3px solid var(--accent);background:#f4f6f9}
.summary p{margin:6px 0}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px;font-variant-numeric:tabular-nums}
table.narrow{width:auto;min-width:340px}table.small{font-size:12px}
th{text-align:left;font-weight:600;color:var(--muted);border-bottom:1px solid var(--ink);padding:5px 8px}
td{border-bottom:1px solid var(--line);padding:4px 8px}td.n{text-align:right}
figure{margin:18px 0}figure img{max-width:100%;height:auto;display:block}
figcaption,.note{color:var(--muted);font-size:13px}
details{margin:12px 0}summary{cursor:pointer;color:var(--accent)}
.row{display:flex;gap:24px;flex-wrap:wrap}
@media (max-width:640px){body{padding:24px 16px}table{display:block;overflow-x:auto}}
"""
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
           f"<title>Corvane spares run</title><style>{css}</style></head><body><main>{body}</main></body></html>")
    Path(path).write_text(doc, encoding="utf-8")
