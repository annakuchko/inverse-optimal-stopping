import torch

from inverse_opt_stopping.q_networks import (
    OfflineQNetwork_orig,
    OfflineQNetwork,
    DoubleOfflineQNetwork,
    RNN_QNetwork,
)


def test_offline_q_network_orig_shapes():
    net = OfflineQNetwork_orig(obs_dim=3, action_dim=2, gamma=0.99, device='cpu')
    x = torch.randn(4, 3)
    q = net.get_Q(x)
    assert q.shape == (4, 2)
    # Sigmoid output in (0,1)
    assert torch.all(q >= 0) and torch.all(q <= 1)


def test_offline_q_network_with_next_state():
    net = OfflineQNetwork(obs_dim=3, action_dim=2, gamma=0.99, device='cpu')
    x = torch.randn(5, 3)
    q = net.get_Q(x)
    s_next = net.get_next_s(x)
    assert q.shape == (5, 2)
    assert s_next.shape == (5, 3)


def test_double_offline_q_network_min_selection():
    net = DoubleOfflineQNetwork(obs_dim=3, action_dim=2, gamma=0.99, device='cpu')
    x = torch.randn(6, 3)
    q = net.get_Q(x)
    s_next = net.get_next_s(x)
    assert q.shape == (6, 2)
    assert s_next.shape == (6, 3)


def test_rnn_q_network_shapes():
    net = RNN_QNetwork(obs_dim=3, action_dim=2, gamma=0.99, device='cpu')
    x = torch.randn(2, 4, 3)  # [B=2, T=4, D=3]
    q = net.get_Q(x)
    assert q.shape == (2, 2)
