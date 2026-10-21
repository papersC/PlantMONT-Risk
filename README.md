# PlantMONT-Risk simulator

Slot-level simulator and experiment scripts for the paper
"Risk-Aware Event-Triggered Scheduling and Routing for Energy-Neutral Terahertz Plant Nanosensor Networks"
(N. Ababneh and K. Almi'ani).

## Contents

| File | Purpose |
|---|---|
| `sim.py` | Simulator: plant topology, parametric THz link model, moisture/blockage/harvest processes, VOC burst workload, all seven policies, risk-weighted greedy solver |
| `run_experiments.py` | The seven campaigns of the paper (default, moisture, harvest, load, Pareto, stress, sensitivity), 30 seeds each |
| `gap.py` | Optimality gap of the greedy solver against exhaustive enumeration on small instances |
| `scaling.py` | Gateway runtime versus network size |
| `make_outputs.py` | Regenerates the pgfplots figures and LaTeX tables in `generated/` and prints every statistic quoted in the paper |
| `test_sim.py` | Determinism, common-random-number, energy-causality and conflict-freedom tests |
| `results/` | Raw per-seed results used in the paper (JSON) |
| `generated/` | Figure and table snippets inserted in `main.tex` |

## Policies

`TDMA`, `PAFS` (plant-aware frequency selection), `ERA` (event-only random access), `EAMAC` (energy-aware MAC),
`ETEV` (proposed scheduler without CVaR and chance-constraint penalty), `RISK` (PlantMONT-Risk),
`RISK_NOM` (PlantMONT-Risk without moisture information).

## Usage

```sh
pip install -r requirements.txt
python -m pytest -q                       # about 10 s
python -c "from sim import run_one; print(run_one(({}, 'RISK', 0)))"   # one 500-epoch run
python make_outputs.py                    # rebuild figures/tables from the stored results
./reproduce.sh                            # full rerun, about 2.5 h on 2 cores
```

Any default parameter (see `DEFAULT` in `sim.py` and Table 3 of the paper) can be overridden through the
configuration dictionary, for example `run_one(({'mbar': 0.8, 'lam0': 0.24}, 'ETEV', 3))`.

## Notes

- Tested with Python 3.13, NumPy 2.5 and SciPy 1.18.
- Results are deterministic for a given seed; only the wall-clock runtime metrics (`rt_ms`, `rt_p95_ms`) vary.
- The link model uses study parameters, not calibrated plant or device coefficients (see the Limitations paragraph of the paper).
