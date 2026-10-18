# STATUS: spares-forecast-replenishment

## Progress
- Done: upstream smoke runs, pipeline, full run (120 s), 6 tests passing, report screenshot, README,
  NOTICE, local commits. Nothing pushed.

## Stages
| Stage | Status | Evidence |
| --- | --- | --- |
| Cloned | n/a (yes as dependencies) | Upstreams are pip packages, pinned in `requirements.txt`: statsforecast 2.1.1, ortools 9.15.6755. No upstream code copied. |
| Runnable | yes | `docs/upstream_smoke_statsforecast.txt` (README quick start, AutoARIMA on AirPassengers), `docs/upstream_smoke_ortools.txt` (official CP-SAT `simple_sat_program.py`) |
| Customized | yes | Corvane data generator, ERP adapter, forecasting/selection, safety stock, CP-SAT plan, holdout simulation, HTML report. `docs/pipeline_run.txt`, `docs/sample_output/`, `docs/report_screenshot.png` |
| Tested | yes | `docs/pytest_output.txt`: 6 passed in 30 s (generator determinism, adapter round trip, policy math, integer demand rounding, optimizer MOQ/capacity/budget, end-to-end tiny run) |
| Published | no | Local git only |

## Commands
```
cd fde-projects/spares-forecast-replenishment
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python upstream_smoke/statsforecast_quickstart.py
.venv/Scripts/python upstream_smoke/ortools_simple_sat_program.py
.venv/Scripts/python -m spares.pipeline      # about 2 min; writes data/, out/plan.csv, out/report.html
.venv/Scripts/python -m pytest -q
```

## Files I added
`spares/{__init__,config,generate,adapter,forecast,policy,optimize,simulate,report,pipeline}.py`,
`config/corvane.toml`, `tests/test_spares.py`, `upstream_smoke/*.py` (docs examples), `data/*.csv`
(generated, seed 42), `docs/*`, `README.md`, `NOTICE`, `requirements.txt`, `.gitignore`, `STATUS.md`.
No upstream files changed.

## Results for the portfolio site (all from `python -m spares.pipeline`, `docs/pipeline_run.txt`)
- Forecast WAPE on a 13-week out-of-sample holdout: 0.368 with per-series model choice vs 0.472 for
  seasonal naive (22% lower); mean MASE 0.949 vs 1.090. Source: `docs/sample_output/holdout_accuracy.csv`.
- Backtest WAPE (6 x 4-week windows): 0.361 vs 0.470. Source: `docs/sample_output/backtest_summary.csv`.
- Holdout replay on synthetic demand, 120 SKU-DCs x 13 weeks: the optimized policy held 40% less
  inventory ($528,040 vs $885,697 average) with 0 budget or dock-capacity breaches (hand rule: 5 and 2),
  at 95.1% vs 97.9% fill rate. Slow-mover fill rate 83.3% vs 73.8%. Source:
  `docs/sample_output/simulation_summary.csv`.
- Against the hand rule at a similar inventory level (2/8 weeks: 87.9% fill, $584,085) the optimized
  policy is better on both fill rate and inventory. Source: `docs/sample_output/hand_rule_sweep.csv`.
- Production plan: 121 order lines, $704,854, CP-SAT 5.6% from proven bound in 10 s. Source: `docs/sample_output/plan.csv`.
- Whole pipeline runs in about 2 minutes on a 16 GB laptop without GPU.

Do not claim the optimized policy beats the 4/10 hand rule overall: it has lower fill rate, more
stockout weeks (94 vs 78) and higher total cost ($275,916 vs $200,551) under my shortage penalty.

## Open issues
- Weekly CP-SAT re-plans in the replay hit the 3 s limit with a mean 43% gap to the bound (all FEASIBLE,
  none proven within 1%). More solve time or a tighter formulation would likely improve the replay.
  I compared HiGHS, SCIP and CBC through pywraplp on the same model; CP-SAT was best at 10 s.
- Cost parameters (shortage penalty, safety-stock penalty, freight per pallet, budget) are my
  assumptions and drive the fill-rate vs inventory trade-off; not tuned.
- CP-SAT with parallel workers and a time limit is not bit-for-bit reproducible across runs.
- Report screenshot is cut before the replay table (the full page is `docs/sample_output/report.html`).
- No license chosen for my own code yet; owner decides.
- Optional vehicle-routing step skipped.
