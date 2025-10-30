"""
Unified experiment runner for IQ-Learn variants.

This script provides a single CLI to run base, conservative, or car variants of
IQ-Learn via the unified wrapper. It is non-invasive (keeps existing run_*.py).

Examples:
  python run_experiment.py --variant base --epochs 1 --synthetic
  python run_experiment.py --variant conserv --epochs 5 --synthetic --paths 8 --steps 12
"""

import argparse
import numpy as np
from pathlib import Path

from inverse_opt_stopping.iq_learn_unified import IQ_Agent, IQ_AgentSuperset, plot_st_reg_car


def make_synthetic_memory(n_paths=6, steps_per_path=10, obs_dim=2, seed=0):
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


def parse_args():
    p = argparse.ArgumentParser(description="Unified IQ-Learn experiment runner")
    p.add_argument('--variant', choices=['base', 'conserv', 'car'], default='base')
    p.add_argument('--obs-dim', type=int, default=2)
    p.add_argument('--action-dim', type=int, default=2)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cpu')
    p.add_argument('--approx-dynamics', action='store_true')
    p.add_argument('--approx-g', action='store_true')
    p.add_argument('--synthetic', action='store_true', help='Use synthetic memory')
    p.add_argument('--paths', type=int, default=6)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--memory-npz', type=str, help='Path to .npz with offline memory dict keys')
    p.add_argument('--legacy', action='store_true', help='Use legacy wrapper instead of superset agent')
    return p.parse_args()


def main():
    args = parse_args()
    if args.legacy:
        agent = IQ_Agent(
            variant=args.variant,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            approx_dynamics=args.approx_dynamics,
            approx_g=args.approx_g,
            device=args.device,
        )
    else:
        # Use superset agent (native where possible; otherwise delegates)
        agent = IQ_AgentSuperset(
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            approx_dynamics=args.approx_dynamics,
            approx_g=args.approx_g,
            device=args.device,
            conservative=(args.variant == 'conserv'),
            pre_oversample=True,
            use_native=True,
        )

    mem = None
    if args.memory_npz:
        mem = load_memory_npz(args.memory_npz)
    elif args.synthetic:
        mem = make_synthetic_memory(args.paths, args.steps, args.obs_dim)

    if mem is not None:
        print(f"Training {args.variant} variant on memory: {len(mem['state_mem'])} samples")
        agent.train(mem, batch_size=args.batch_size, n_epoches=args.epochs, verbose=1)
    else:
        print("No dataset provided. Use --memory-npz <path> or --synthetic.")

    print("Done.")


if __name__ == '__main__':
    main()


def load_memory_npz(path):
    """Load offline memory from an .npz file.

    Expected keys:
      - state_mem: float array [N, D]
      - next_state_mem: float array [N, D]
      - action_mem: int array [N]
      - done_mem: int/bool array [N]
      - path_ids: int array [N]
      - time_ids: int array [N]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Memory file not found: {path}")
    with np.load(path, allow_pickle=False) as npz:
        required = ['state_mem', 'next_state_mem', 'action_mem', 'done_mem', 'path_ids', 'time_ids']
        missing = [k for k in required if k not in npz]
        if missing:
            raise KeyError(f"Missing keys in {path}: {missing}. Expected {required}")
        mem = {
            'state_mem': np.array(npz['state_mem'], dtype=np.float32),
            'next_state_mem': np.array(npz['next_state_mem'], dtype=np.float32),
            'action_mem': np.array(npz['action_mem'], dtype=np.int32),
            'done_mem': np.array(npz['done_mem'], dtype=np.int32),
            'path_ids': np.array(npz['path_ids'], dtype=np.int32),
            'time_ids': np.array(npz['time_ids'], dtype=np.int32),
        }
    # Basic validation
    N = mem['state_mem'].shape[0]
    for k, v in mem.items():
        if v.shape[0] != N:
            raise ValueError(f"Inconsistent length for {k}: {v.shape[0]} != {N}")
    if mem['state_mem'].shape != mem['next_state_mem'].shape:
        raise ValueError("state_mem and next_state_mem must have the same shape")
    return mem
