import numpy as np

import run_sweeps as sweeps


def test_param_sweep_synthetic_eps_smoke():
    # Two values, one seed, synthetic memory to keep it fast
    res = sweeps.run_param_sweep(
        example='star',
        param='eps',
        values=[0.1, 0.2],
        seeds=[0],
        n_epochs=1,
        batch_size=16,
        conservative=False,
        approx_dynamics=False,
        approx_g=False,
        device='cpu',
        use_superset=True,
        use_synthetic=True,
        synthetic_paths=4,
        synthetic_steps=5,
        obs_dim=2,
        action_dim=2,
        base_kwargs={'oversampling': None, 'classify': False},
    )
    assert res.ba.shape == (1, 2)
    assert res.mtte.shape == (1, 2)
    assert res.memr.shape == (1, 2)
    # Allow NaNs if metrics are not populated; but arrays must exist
    assert isinstance(res.values, list) and len(res.values) == 2


def test_param_sweep_invalid_param_raises():
    try:
        sweeps.run_param_sweep(
            example='star',
            param='invalid_param',
            values=[1, 2],
            seeds=[0],
            use_synthetic=True,
        )
    except ValueError as e:
        assert 'Unsupported sweep param' in str(e)
    else:
        assert False, 'Expected ValueError for unsupported param'


def test_param_sweep_q_lr_smoke():
    # Sweep over q_lr to ensure learning-rate params are accepted
    res = sweeps.run_param_sweep(
        example='star',
        param='q_lr',
        values=[0.001, 0.01],
        seeds=[0],
        n_epochs=1,
        batch_size=16,
        conservative=False,
        approx_dynamics=False,
        approx_g=False,
        device='cpu',
        use_superset=True,
        use_synthetic=True,
        synthetic_paths=4,
        synthetic_steps=5,
        obs_dim=2,
        action_dim=2,
        base_kwargs={'oversampling': None, 'classify': False},
    )
    assert res.ba.shape == (1, 2)
