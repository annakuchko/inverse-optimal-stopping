import numpy as np
from pathlib import Path

from run_experiment import load_memory_npz


def test_load_memory_npz_roundtrip(tmp_path: Path):
    N, D = 12, 3
    state = np.random.randn(N, D).astype(np.float32)
    next_state = state + 0.1 * np.random.randn(N, D).astype(np.float32)
    action = (np.arange(N) % 2).astype(np.int32)
    done = (action == 0).astype(np.int32)
    path_ids = np.repeat(np.arange(3), N // 3).astype(np.int32)
    time_ids = np.tile(np.arange(N // 3), 3).astype(np.int32)
    p = tmp_path / 'mem.npz'
    np.savez(p,
             state_mem=state,
             next_state_mem=next_state,
             action_mem=action,
             done_mem=done,
             path_ids=path_ids,
             time_ids=time_ids)

    mem = load_memory_npz(p)
    assert mem['state_mem'].shape == (N, D)
    assert mem['next_state_mem'].shape == (N, D)
    assert mem['action_mem'].shape == (N,)
    assert mem['done_mem'].shape == (N,)
    assert mem['path_ids'].shape == (N,)
    assert mem['time_ids'].shape == (N,)
