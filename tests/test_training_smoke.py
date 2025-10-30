import numpy as np
import torch

from inverse_opt_stopping.iq_learn_unified import IQ_Agent


def make_synthetic_memory(n_paths=6, steps_per_path=10, obs_dim=2):
    rng = np.random.default_rng(0)
    N = n_paths * steps_per_path
    state = rng.normal(size=(N, obs_dim)).astype(np.float32)
    next_state = state + 0.05 * rng.normal(size=(N, obs_dim)).astype(np.float32)
    # Balanced binary actions
    actions = np.array([0, 1] * (N // 2) + ([0] if N % 2 else []), dtype=np.int32)
    rng.shuffle(actions)
    # Path/time indices
    path_ids = np.repeat(np.arange(n_paths), steps_per_path).astype(np.int32)
    time_ids = np.tile(np.arange(steps_per_path), n_paths).astype(np.int32)

    mem = {
        'state_mem': state,
        'next_state_mem': next_state,
        'action_mem': actions,
        'done_mem': (actions == 0).astype(np.int32),
        'path_ids': path_ids,
        'time_ids': time_ids,
    }
    return mem


def test_training_smoke_runs_one_epoch():
    mem = make_synthetic_memory()
    agent = IQ_Agent(
        variant='base',
        obs_dim=2,
        action_dim=2,
        approx_dynamics=False,
        approx_g=False,
        device='cpu',
    )
    # One epoch, small batch size
    agent.train(mem, batch_size=16, n_epoches=1, verbose=0)
    # Basic post-condition: attributes filled
    assert isinstance(agent.epoch_balanced_accuracy, list)
    assert len(agent.epoch_balanced_accuracy) >= 1
