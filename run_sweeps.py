"""
Unified hyperparameter sweep runner for IQ-Learn.

This script consolidates the ad-hoc run_* sweep scripts (e.g., epsilon, cs,
cs-decay, number of paths) into a single, parameterized CLI. It reuses the
unified IQ-Learn import surface and mirrors outputs similar to prior scripts.

Key capabilities
- Sweep over a single hyperparameter: one of {eps, cs, cs_decay, eps_decay_rate,
  gamma, n_paths, out_thresh}.
- Choose example/problem: {radial, star, CP1, CP2, CP3, CP1M, CP2M, CP3M,
  bmG, bmgG, car, azure, nasa_turbofan}.
- Run base or conservative variant via the unified agent wrapper.
- Optionally use synthetic data (fast) instead of environment simulation.

Outputs
- Saves BA/MTTE/MEMR arrays under `outputs/sweeps/` when `--save` is set.
- Prints a concise summary table (mean +/- std) to stdout when `--print` is set.

Notes
- Default configuration keeps runtime small; adjust `--epochs`/`--seeds` for
  production experiments.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from inverse_opt_stopping.iq_learn_unified import IQ_Agent, IQ_AgentSuperset


# ------------------------------- Data helpers -------------------------------

def make_synthetic_memory(n_paths: int = 6, steps_per_path: int = 10, obs_dim: int = 2, seed: int = 0) -> Dict[str, np.ndarray]:
    """Create a lightweight synthetic offline memory dict.

    Shapes
    - state_mem: [N, obs_dim]
    - next_state_mem: [N, obs_dim]
    - action_mem: [N]
    - done_mem: [N]
    - path_ids: [N]
    - time_ids: [N]

    The memory dict schema is preserved to match environments/ and agents.
    """
    rng = np.random.default_rng(seed)
    N = n_paths * steps_per_path
    state = rng.normal(size=(N, obs_dim)).astype(np.float32)
    next_state = state + 0.05 * rng.normal(size=(N, obs_dim)).astype(np.float32)
    actions = np.array([0, 1] * (N // 2) + ([0] if N % 2 else []), dtype=np.int32)
    rng.shuffle(actions)
    path_ids = np.repeat(np.arange(n_paths), steps_per_path).astype(np.int32)
    time_ids = np.tile(np.arange(steps_per_path), n_paths).astype(np.int32)
    return {
        'state_mem': state,
        'next_state_mem': next_state,
        'action_mem': actions,
        'done_mem': (actions == 0).astype(np.int32),
        'path_ids': path_ids,
        'time_ids': time_ids,
    }


def simulate_example(example: str, episodes_train: int, episodes_test: int, max_path_length: int) -> Tuple[dict, dict, int]:
    """Simulate train/test memory via the repo's forward algorithms layer.

    Returns (train_mem, test_mem, t_downsample) where t_downsample mirrors prior
    scripts' per-example time downsampling (10 for azure/turbofan).
    """
    from forward_algorithms.simulate_expert_data import Simulation

    problem = {
        'radial': 'radial', 'star': 'star',
        'CP1': 'CP1', 'CP2': 'CP2', 'CP3': 'CP3', 'CP1M': 'CP1M', 'CP2M': 'CP2M', 'CP3M': 'CP3M',
        'bmG': 'symm_bm_G', 'bmgG': 'symm_bm_gG', 'car': 'car', 'azure': 'azure', 'nasa_turbofan': 'nasa_turbofan',
    }[example]
    t_downsample = 10 if example in {'azure', 'nasa_turbofan'} else 1
    sim = Simulation(problem=problem, total_n=episodes_train)
    train = sim.simulate_expert(episodes=episodes_train, max_path_length=max_path_length)
    test = sim.simulate_test(episodes=episodes_test, max_path_length=max_path_length)
    return train, test, t_downsample


# ------------------------------ Core sweep logic ----------------------------

@dataclass
class SweepResult:
    values: List[float]
    ba: np.ndarray  # [S, V]
    mtte: np.ndarray  # [S, V]
    memr: np.ndarray  # [S, V]


def _build_agent(use_superset: bool, conservative: bool, obs_dim: int, action_dim: int,
                 approx_dynamics: bool, approx_g: bool, device: str,
                 base_kwargs: Dict[str, Any]) -> Any:
    """Factory for IQ agents (superset by default).

    base_kwargs include training-time hyperparameters passed to canonical impls
    by the superset delegate (e.g., epsilon, cs, cs_decay, eps_decay_rate,
    oversampling strategy toggles). This preserves behavior of public APIs.
    """
    if use_superset:
        return IQ_AgentSuperset(
            obs_dim=obs_dim,
            action_dim=action_dim,
            approx_dynamics=approx_dynamics,
            approx_g=approx_g,
            device=device,
            conservative=conservative,
            pre_oversample=True,
            use_native=True,
            **base_kwargs,
        )
    variant = 'conserv' if conservative else 'base'
    return IQ_Agent(
        variant=variant,
        obs_dim=obs_dim,
        action_dim=action_dim,
        approx_dynamics=approx_dynamics,
        approx_g=approx_g,
        device=device,
        **base_kwargs,
    )


def run_param_sweep(
    *,
    example: str,
    param: str,
    values: Sequence[float],
    seeds: Sequence[int],
    n_epochs: int = 5,
    batch_size: int = 64,
    conservative: bool = False,
    approx_dynamics: bool = False,
    approx_g: bool = False,
    device: str = 'cpu',
    use_superset: bool = True,
    use_synthetic: bool = False,
    synthetic_paths: int = 6,
    synthetic_steps: int = 10,
    obs_dim: int = 2,
    action_dim: int = 2,
    episodes_train: int = 250,
    episodes_test: int = 100,
    max_path_length: int = 100,
    base_kwargs: Dict[str, Any] | None = None,
) -> SweepResult:
    """Run a 1D sweep over a selected hyperparameter.

    Parameters that can be swept via `param`:
      - 'eps', 'cs', 'cs_decay', 'eps_decay_rate', 'gamma', 'n_paths', 'out_thresh'.

    Returns arrays shaped [num_seeds, num_values] for BA/MTTE/MEMR.
    """
    base_kwargs = dict(base_kwargs or {})

    # Prepare data
    if use_synthetic:
        train_mem = make_synthetic_memory(synthetic_paths, synthetic_steps, obs_dim, seed=0)
        test_mem = make_synthetic_memory(synthetic_paths // 2 or 1, synthetic_steps, obs_dim, seed=1)
        _ = 1  # t_downsample placeholder
    else:
        train_mem, test_mem, _ = simulate_example(example, episodes_train, episodes_test, max_path_length)

    S, V = len(seeds), len(values)
    ba = np.zeros((S, V), dtype=np.float32)
    mtte = np.zeros((S, V), dtype=np.float32)
    memr = np.zeros((S, V), dtype=np.float32)

    sweep_keys = {'eps', 'cs', 'cs_decay', 'eps_decay_rate', 'gamma', 'n_paths', 'out_thresh', 'q_lr', 'env_lr', 'g_lr'}
    if param not in sweep_keys:
        raise ValueError(f"Unsupported sweep param: {param}. Choose from {sorted(sweep_keys)}")

    for si, seed in enumerate(seeds):
        for vi, val in enumerate(values):
            kwargs = dict(base_kwargs)
            # Map param to constructor/train kwargs. 'n_paths' changes dataset size.
            if param == 'n_paths' and not use_synthetic:
                # Re-simulate with different total episodes
                train_mem, test_mem, _ = simulate_example(example, int(val), int(max(10, val * 0.4)), max_path_length)
            elif param == 'n_paths' and use_synthetic:
                train_mem = make_synthetic_memory(int(val), synthetic_steps, obs_dim, seed=seed)
                test_mem = make_synthetic_memory(max(1, int(val) // 2), synthetic_steps, obs_dim, seed=seed + 1)
            elif param in {'eps', 'cs', 'cs_decay', 'eps_decay_rate', 'gamma', 'out_thresh', 'q_lr', 'env_lr', 'g_lr'}:
                kwargs[param] = float(val)

            agent = _build_agent(use_superset, conservative, obs_dim, action_dim,
                                 approx_dynamics, approx_g, device, kwargs)

            # Train and evaluate
            agent.train(train_mem, batch_size=batch_size, n_epoches=n_epochs, verbose=0)
            agent.test(test_mem, from_grid=False)

            # Collect metrics (fall back to NaNs if missing)
            ba[si, vi] = getattr(agent, 'balanced_accuracy', np.nan)
            mtte[si, vi] = getattr(agent, 'mtte', np.nan)
            memr[si, vi] = getattr(agent, 'memr', np.nan)

    return SweepResult(values=list(map(float, values)), ba=ba, mtte=mtte, memr=memr)


# ----------------------------------- CLI -----------------------------------

def _array_to_str(x: np.ndarray) -> str:
    m, s = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    return ' | '.join([f"{mi:.4f} +/- {si:.4f}" for mi, si in zip(m, s)])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Unified hyperparameter sweeps for IQ-Learn')
    p.add_argument('--example', default='star',
                   choices=['radial','star','CP1','CP2','CP3','CP1M','CP2M','CP3M','bmG','bmgG','car','azure','nasa_turbofan'])
    p.add_argument('--param', required=True,
                   choices=['eps','cs','cs_decay','eps_decay_rate','gamma','n_paths','out_thresh','q_lr','env_lr','g_lr'])
    p.add_argument('--values', nargs='+', type=float, required=True,
                   help='List of values for the sweep (space-separated).')
    p.add_argument('--seeds', type=int, default=1)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--conservative', action='store_true')
    p.add_argument('--approx-dynamics', action='store_true')
    p.add_argument('--approx-g', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--legacy', action='store_true', help='Use canonical variant wrapper instead of superset')

    # Data generation
    p.add_argument('--synthetic', action='store_true', help='Use small synthetic memory for fast runs')
    p.add_argument('--synthetic-paths', type=int, default=6)
    p.add_argument('--synthetic-steps', type=int, default=10)
    p.add_argument('--obs-dim', type=int, default=2)
    p.add_argument('--action-dim', type=int, default=2)
    p.add_argument('--episodes-train', type=int, default=250)
    p.add_argument('--episodes-test', type=int, default=100)
    p.add_argument('--max-path-length', type=int, default=100)

    # Training hyperparameters forwarded to agent
    p.add_argument('--q-lr', type=float, default=0.01)
    p.add_argument('--env-lr', type=float, default=0.01)
    p.add_argument('--g-lr', type=float, default=0.001)
    p.add_argument('--eps', type=float, default=0.1)
    p.add_argument('--cs', type=float, default=0.999)
    p.add_argument('--cs-decay', type=float, default=0.99)
    p.add_argument('--eps-decay-rate', type=float, default=1.0)
    p.add_argument('--smote-k', type=int, default=12)
    p.add_argument('--is-cs', action='store_true')
    p.add_argument('--oversampling', default='SMOTE', choices=['none','SMOTE','CS-SMOTE','LSMOTE','CS-LSMOTE'],
                   help='Oversampling strategy (default SMOTE). Use "none" to disable.')
    p.add_argument('--classify', action='store_true', help='Train classifier instead of full IQS loss')
    p.add_argument('--out-thresh', type=float, default=0.0)

    # Output controls
    p.add_argument('--save', action='store_true', help='Save BA/MTTE/MEMR to outputs/sweeps')
    p.add_argument('--print', dest='do_print', action='store_true', help='Print mean +/- std summary table')
    p.add_argument('--plot', action='store_true', help='Save simple BA/MTTE/MEMR vs param plots')
    # Multi-model overlays
    p.add_argument('--models', nargs='+', default=None,
                   choices=['classifier','iqs','iqs-smote','model-based','model-based-smote','do-iqs','do-iqs-lb'],
                   help='Run multiple model presets and overlay results with legend')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(range(args.seeds))
    cv_splits = 1 if args.conservative else 2
    if args.synthetic:
        cv_splits = 1
    base_kwargs: Dict[str, Any] = dict(
        divergence='hellinger',
        cross_val_splits=cv_splits,
        q_lr=args.q_lr, env_lr=args.env_lr, g_lr=args.g_lr,
        epsilon=args.eps, cs=args.cs, cs_decay=args.cs_decay, eps_decay_rate=args.eps_decay_rate,
        SMOTE_K=args.smote_k, is_cs=args.is_cs, out_thresh=args.out_thresh,
        oversampling=(None if args.oversampling.lower() == 'none' else args.oversampling),
        classify=bool(args.classify),
    )

    # Multi-model branch: run a sweep per requested model and overlay plots
    if args.models:
        results: List[Tuple[str, SweepResult]] = []
        for model in args.models:
            overrides: Dict[str, Any] = {}
            approx_dyn = False
            approx_g = False
            if model == 'classifier':
                overrides.update(oversampling=None, classify=True)
            elif model == 'iqs':
                overrides.update(oversampling=None, classify=False)
            elif model == 'iqs-smote':
                overrides.update(oversampling='SMOTE', classify=False)
            elif model == 'model-based':
                overrides.update(oversampling=None, classify=False)
                approx_dyn = True
            elif model == 'model-based-smote':
                overrides.update(oversampling='SMOTE', classify=False)
                approx_dyn = True
            elif model == 'do-iqs':
                overrides.update(oversampling=None, classify=False)
                approx_dyn = True
                approx_g = True
            elif model == 'do-iqs-lb':
                overrides.update(oversampling='CS-LSMOTE', classify=False, is_cs=True)
                approx_dyn = True
                approx_g = True

            model_kwargs = dict(base_kwargs)
            # only override keys that exist in base_kwargs
            for k, v in overrides.items():
                if k in model_kwargs:
                    model_kwargs[k] = v

            res = run_param_sweep(
                example=args.example,
                param=args.param,
                values=args.values,
                seeds=seeds,
                n_epochs=args.epochs,
                batch_size=args.batch_size,
                conservative=args.conservative,
                approx_dynamics=approx_dyn,
                approx_g=approx_g,
                device=args.device,
                use_superset=not args.legacy,
                use_synthetic=args.synthetic,
                synthetic_paths=args.synthetic_paths,
                synthetic_steps=args.synthetic_steps,
                obs_dim=args.obs_dim,
                action_dim=args.action_dim,
                episodes_train=args.episodes_train,
                episodes_test=args.episodes_test,
                max_path_length=args.max_path_length,
                base_kwargs=model_kwargs,
            )
            results.append((model, res))

        suf = ("_fix_conserv" if args.conservative else "")
        stem = f"{args.example}{suf}_{args.param}"
        if args.do_print:
            print(f"Param sweep (multi-model): {args.param} = {results[0][1].values}")
            for label, r in results:
                print(f"[{label}] BA   :", _array_to_str(r.ba))
                print(f"[{label}] MTTE :", _array_to_str(r.mtte))
                print(f"[{label}] MEMR :", _array_to_str(r.memr))

        if args.save:
            os.makedirs('outputs/sweeps', exist_ok=True)
            np.save(f"outputs/sweeps/{stem}_models.npy", np.array([m for m, _ in results], dtype=object))
            np.save(f"outputs/sweeps/{stem}_values.npy", np.asarray(results[0][1].values, dtype=np.float32))
            for label, r in results:
                np.save(f"outputs/sweeps/{stem}_ba_{label}.npy", r.ba)
                np.save(f"outputs/sweeps/{stem}_mtte_{label}.npy", r.mtte)
                np.save(f"outputs/sweeps/{stem}_memr_{label}.npy", r.memr)

        if args.plot:
            try:
                import matplotlib.pyplot as plt
                os.makedirs('outputs/sweeps', exist_ok=True)
                x = np.asarray(results[0][1].values, dtype=float)
                for metric in ['ba', 'mtte', 'memr']:
                    plt.figure(figsize=(7,4.5))
                    for label, r in results:
                        arr = getattr(r, metric)
                        m, s = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
                        plt.plot(x, m, marker='o', label=label)
                        plt.fill_between(x, m - s, m + s, alpha=0.12)
                    plt.xlabel(args.param)
                    plt.ylabel(metric.upper())
                    plt.title(f"{args.example} {metric.upper()} vs {args.param}")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"outputs/sweeps/{stem}_{metric}_multi.png", dpi=150)
                    plt.close()
            except Exception as e:
                print(f"Multi-plotting failed: {e}")
        return

    res = run_param_sweep(
        example=args.example,
        param=args.param,
        values=args.values,
        seeds=seeds,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        conservative=args.conservative,
        approx_dynamics=args.approx_dynamics,
        approx_g=args.approx_g,
        device=args.device,
        use_superset=not args.legacy,
        use_synthetic=args.synthetic,
        synthetic_paths=args.synthetic_paths,
        synthetic_steps=args.synthetic_steps,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        episodes_train=args.episodes_train,
        episodes_test=args.episodes_test,
        max_path_length=args.max_path_length,
        base_kwargs=base_kwargs,
    )

    if args.do_print:
        print(f"Param sweep: {args.param} = {res.values}")
        print("BA    :", _array_to_str(res.ba))
        print("MTTE  :", _array_to_str(res.mtte))
        print("MEMR  :", _array_to_str(res.memr))

    if args.save:
        os.makedirs('outputs/sweeps', exist_ok=True)
        suf = ("_fix_conserv" if args.conservative else "")
        stem = f"{args.example}{suf}_{args.param}"
        np.save(f"outputs/sweeps/{stem}_values.npy", np.asarray(res.values, dtype=np.float32))
        np.save(f"outputs/sweeps/{stem}_ba.npy", res.ba)
        np.save(f"outputs/sweeps/{stem}_mtte.npy", res.mtte)
        np.save(f"outputs/sweeps/{stem}_memr.npy", res.memr)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            os.makedirs('outputs/sweeps', exist_ok=True)
            suf = ("_fix_conserv" if args.conservative else "")
            stem = f"{args.example}{suf}_{args.param}"
            x = np.asarray(res.values, dtype=float)
            for name, arr in [('ba', res.ba), ('mtte', res.mtte), ('memr', res.memr)]:
                m, s = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
                plt.figure(figsize=(6,4))
                plt.plot(x, m, marker='o')
                plt.fill_between(x, m - s, m + s, alpha=0.2)
                plt.xlabel(args.param)
                plt.ylabel(name.upper())
                plt.title(f"{args.example} {name.upper()} vs {args.param}")
                plt.grid(True, alpha=0.3)
                outp = f"outputs/sweeps/{stem}_{name}.png"
                plt.tight_layout()
                plt.savefig(outp, dpi=150)
                plt.close()
        except Exception as e:
            print(f"Plotting failed: {e}")


if __name__ == '__main__':
    main()
