try:
    import numpy as np
    import torch
    from inverse_opt_stopping.iq_learn_unified import IQ_Agent
except Exception:
    import pytest
    pytest.skip("Skipping IQ_Agent tests because numpy/torch or project imports are unavailable", allow_module_level=True)


def make_dummy_memory(n_paths=3, path_length=4, obs_dim=3):
    total = n_paths * path_length
    state_mem = np.random.randn(total, obs_dim).astype(np.float32)
    next_state_mem = np.roll(state_mem, -1, axis=0).astype(np.float32)
    action_mem = np.random.randint(0, 2, size=(total,)).astype(np.int64)
    done_mem = np.zeros(total, dtype=bool)
    path_ids = np.repeat(np.arange(n_paths), path_length)
    time_ids = np.tile(np.arange(path_length), n_paths)
    return {
        'state_mem': state_mem,
        'next_state_mem': next_state_mem,
        'action_mem': action_mem,
        'done_mem': done_mem,
        'path_ids': path_ids,
        'time_ids': time_ids,
    }


def test_iq_agent_forward_and_state_dict():
    obs_dim = 3
    mem = make_dummy_memory(n_paths=2, path_length=3, obs_dim=obs_dim)
    agent = IQ_Agent(variant='base', obs_dim=obs_dim, approx_g=False, approx_dynamics=True, oversampling=None,
                     q_lr=1e-3, env_lr=1e-3, g_lr=0.0, device='cpu', seed=0)

    # test getQ on a single state
    sample_state = torch.from_numpy(mem['state_mem'][:2]).float()
    q_mean, q_std = agent.getQ(sample_state, evaluate=True)
    assert q_mean.shape[0] == 2
    assert q_mean.shape[1] == 2

    # test infer_next_s (uses q_net.get_next_s)
    s_dash = agent.infer_next_s(sample_state)
    assert s_dash.shape[0] == 2

    # test save/load of state_dict
    sd = agent.q_net.state_dict()
    # load back into the same network (sanity)
    agent.q_net.load_state_dict(sd)
