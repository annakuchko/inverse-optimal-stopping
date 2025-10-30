import types

import inverse_opt_stopping.iq_learn_unified as unified


def test_unified_imports_available():
    assert hasattr(unified, 'IQ_Agent')
    assert hasattr(unified, 'plot_st_reg_car')


def test_unified_base_instantiation_and_plot_dispatch(monkeypatch):
    # Build a lightweight base agent via unified wrapper; avoid training.
    agent = unified.IQ_Agent(
        variant='base',
        obs_dim=2,
        action_dim=2,
        approx_dynamics=False,  # keep it simple
        approx_g=False,
        device='cpu',
    )
    # Sanity: wrapper exposes underlying impl
    assert hasattr(agent, 'impl')
    assert hasattr(agent.impl, 'q_net')

    # Stub out the base plot function inside the unified module to avoid heavy plotting
    called = {}

    def _stub(a, tm):
        called['args'] = (a, tm)
        return 'ok-base'

    monkeypatch.setattr(unified, '_plot_st_reg_car_base', _stub, raising=True)
    # Dispatch should unwrap to underlying base agent and hit our stub
    out = unified.plot_st_reg_car(agent, {'dummy': 1})
    assert out == 'ok-base'
    assert 'args' in called
