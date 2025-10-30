import torch

from inverse_opt_stopping.q_networks import DoubleOfflineQNetwork


def test_double_next_state_vectorized_matches_manual():
    torch.manual_seed(0)
    net = DoubleOfflineQNetwork(obs_dim=3, action_dim=2, gamma=0.99, device='cpu')
    net.eval()
    x = torch.randn(8, 3)
    with torch.no_grad():
        Q1, Q2, st1, st2 = net.forward(x)
        # Manual selection using original logic
        qmin = torch.minimum(Q1, Q2)
        manual = torch.zeros_like(st1)
        for i, el in enumerate(qmin):
            if el[0] == Q1[i][0] and el[1] == Q1[i][1]:
                manual[i] = st1[i]
            else:
                manual[i] = st2[i]

        # Vectorized method
        out = net.get_next_s(x)
        assert torch.allclose(out, manual)
