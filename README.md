# DO-IQS (Inverse Optimal Stopping via IQ-Learning)

Final implementation of inverse optimal stopping algorithms based on IQ‑Learning and variants. This repository includes several modeling choices, environments, and utilities to run ablation studies and export plots/tables.

## Features

Algorithms (binary actions):
- IQ‑Learning
- IQ‑Learning + SMOTE oversampling
- IQ‑Learning + Confidence‑Score SMOTE (CS‑SMOTE)
- Model‑based IQ‑Learning (dynamics approximation)
- Model‑based IQ‑Learning + SMOTE / CS‑SMOTE
- DO‑IQS, DO‑IQS‑LB
- Classifier, Classifier‑SMOTE

Environments:
- Car environment (toy optimal stopping)
- 2D Brownian motion datasets
- Bessel(2)
- STAR and RADIAL
- Change‑point detection examples

All models are trained offline by design. They can be extended to online training (e.g., actor‑critic), but the focus here is safe offline imitation where online interaction may be infeasible.

## Quick Start

Install dependencies (suggested Python 3.9+):

```
pip install -r requirements.txt
```

Run a small synthetic experiment via the unified entrypoint:

```
python run_experiment.py --variant base --epochs 1 --synthetic
```

Load an offline memory dataset from NPZ and train a conservative agent:

```
python run_experiment.py --variant conserv --memory-npz path/to/memory.npz --epochs 5
```

Expected NPZ keys: `state_mem`, `next_state_mem`, `action_mem`, `done_mem`, `path_ids`, `time_ids`.

### Typical Examples and Defaults

The authors typically run multi‑seed ablations with the following defaults:

- seeds: 5
- epochs: 250
- batch size: 128
- learning rates: `q_lr=0.01`, `env_lr=0.01`, `g_lr=0.001`
- thresholds: `--out-thresh 0.0`, `--out-thresh-alt 0.005`
- SMOTE K: 12

Examples:

```
# Base and Conservative families for Azure (offline OTI) with defaults
python run_ablation.py --examples azure --seeds 5 --epochs 250 --batch-size 128 \
  --q-lr 0.01 --env-lr 0.01 --g-lr 0.001 --out-thresh 0.0 --out-thresh-alt 0.005

# NASA Turbofan
python run_ablation.py --examples nasa_turbofan --seeds 5 --epochs 250 --batch-size 128 \
  --q-lr 0.01 --env-lr 0.01 --g-lr 0.001 --out-thresh 0.0 --out-thresh-alt 0.005

# Change the output location for plots and tables
python run_ablation.py --examples azure nasa_turbofan --seeds 5 --epochs 250 \
  --plots-dir outputs/ablation_runs
```

## Unified API

Use the single entry surface for all IQ‑Learn variants:

```python
from inverse_opt_stopping.iq_learn_unified import IQ_Agent, plot_st_reg_car

agent = IQ_Agent(variant='base', obs_dim=2, action_dim=2, approx_dynamics=False, approx_g=False)
```

Variants: `base`, `conserv`, `car`. The wrapper proxies to the canonical implementations under `inverse_opt_stopping/`.

## Ablation Runner (one‑shot)

Generate ablation results for multiple variants and examples in a single pass, including MTTE–MEMR plots and BA/MTTE/MEMR tables.

```
python run_ablation.py --examples azure nasa_turbofan --seeds 3 --epochs 50 --print-tables
```

Outputs:
- Base: `outputs/<example>_balanced_acc.npy`, `outputs/<example>_mtte.npy`, `outputs/<example>_memr.npy`
- Conservative: `outputs/<example>_balanced_acc_fix_conserv.npy`, `outputs/<example>_mtte_fix_conserv.npy`, `outputs/<example>_memr_fix_conserv.npy`
- Plots/Tables: `outputs/ablation/<example>[_fix_conserv]_mtte_memr.png`, `outputs/ablation/<example>[_fix_conserv]_tables.txt`

Useful flags:
- `--base` or `--conservative` to select families (defaults to both)
- `--plots-dir` to change plots/tables location
- `--no-plots`, `--no-tables` to disable saving

## Tests

Run unit tests (uses the local virtual environment if present):

```
pytest -q
# or
python -m pytest -q
```

The tests validate the unified wrapper, network shapes, a training smoke pass, the vectorized Double‑Q next‑state logic, and the NPZ loader.

## Notes

- Canonical implementations live in `inverse_opt_stopping/iq_learn_base.py`, `iq_learn_conserv.py`, and `iq_learn_baase_car.py`. Other `iq_learn*.py` variants are deprecated and intentionally raise at import time.
- The ablation and experiment runners now route through the unified API to avoid duplication while preserving previous behavior.
