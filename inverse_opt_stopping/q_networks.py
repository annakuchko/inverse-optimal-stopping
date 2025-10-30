import torch
import torch.nn as nn
from torch.autograd import Variable, grad
#  Collection of ANN/RNN for soft Q-function approximation

class SoftQNetwork(nn.Module):
    """Base class for soft Q-function style networks.

    Parameters
    - obs_dim: input feature dimension for observations
    - action_dim: number of actions (size of Q output)
    - gamma: discount factor carried by networks that predict next state/value
    - device: torch device string

    Subclasses must implement ``_forward`` and may override helpers.
    """
    def __init__(self, obs_dim, action_dim, gamma, device='cpu'):
        super(SoftQNetwork, self).__init__()
        self.gamma = gamma
        self.device = device
        self.method_tanh = True
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def _forward(self, x,):
        return NotImplementedError

    def forward(self, x, both=False):
        out = self._forward(x)
        return out

    def jacobian(self, outputs, inputs):
        """Compute Jacobian d(outputs)/d(inputs) for batched inputs.

        Notes: This forms per-output gradients in a loop over ``output_dim``.
        This is suitable for small output dimensions, which is the case for
        Q-vectors over few discrete actions.
        """
        batch_size, output_dim = outputs.shape
        jacobian = []
        for i in range(output_dim):
            v = torch.zeros_like(outputs)
            v[:, i] = 1.
            dy_i_dx = grad(outputs,
                           inputs,
                           grad_outputs=v,
                           retain_graph=True,
                           create_graph=True)[0]  # shape [B, N]
            jacobian.append(dy_i_dx)

        jacobian = torch.stack(jacobian, dim=-1).requires_grad_()
        return jacobian

    def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
        """Gradient penalty on input sensitivity.

        Computes a penalty proportional to ||∂f/∂x||_2 − 1 on interpolated
        inputs. Intended as a regularizer; usages in this repo keep batch sizes
        and output dims small.
        """
        expert_data = obs1
        policy_data = obs2
        batch_size = expert_data.size()[0]

        # Calculate interpolation
        if expert_data.ndim == 4:
            alpha = torch.rand(batch_size, 1, 1, 1)  # B, C, H, W input
        else:
            alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        interpolated = alpha * expert_data #+ (1 - alpha) * policy_data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.to(expert_data.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.forward(interpolated)
        # Calculate gradients of probabilities with respect to examples
        gradients = self.jacobian(prob_interpolated, interpolated)

        # Gradients have shape (batch_size, input_dim, output_dim)
        out_size = gradients.shape[-1]
        gradients_norm = gradients.reshape([batch_size, -1, out_size]).norm(2, dim=1)

        # Return gradient penalty
        return lambda_ * ((gradients_norm - 1) ** 2).mean()

class gNetworkRNN(nn.Module):
    """Recurrent approximator for g(s) along a sequence.

    Expects input of shape [B, T, obs_dim] and returns per-step g estimates.
    """
    def __init__(self, obs_dim, action_dim, gamma, device):
        super(gNetworkRNN, self).__init__()
        self.gamma = gamma
        self.hn_size = 128
        self.lstm0 = nn.LSTM(input_size=obs_dim, 
                             hidden_size=self.hn_size,
                             num_layers=16,
                             batch_first=True,
                             dropout=0.0)
        self.fc0 = nn.Linear(self.hn_size, 256)
        self.inorm0 = nn.BatchNorm1d(256)
        self.dp0 = nn.Dropout(0.0)
        self.elu = nn.SiLU() # nn.ELU()
        
        self.fc_out = nn.Linear(256, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        out = self._forward(x)
        return out

    def _forward(self, x):
        # print(f'x.shape: {x.shape}')
        output, (h_n,c_n) = self.lstm0(x.to(torch.float32))
        # h_n = h_n[-1,:,:].reshape(-1,self.hn_size)
        h_n = output
        # print(f'output.shape: {output.shape}')
        x = self.fc0(h_n)
        x = self.dp0(x)
        x = self.elu(x)
        
        x = self.fc_out(x)
        q = self.out(x) #.reshape(-1,1)
        # q = x.reshape(-1,self.action_dim)
        return q

    def get_g(self, state):
        g_approx = self.forward(state)
        return g_approx

    def weights_init_uniform(self, m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1 or classname.find('Conv') != -1:
                m.weight.data.normal_(0.001, 1.0)
                m.bias.data.fill_(0.001)

class gNetwork(nn.Module):
    """Feed-forward approximator for g(s).

    Maps observation vectors to a scalar in [0, 1] using a few MLP layers.
    """
    def __init__(self, obs_dim, action_dim, gamma, device):
        super(gNetwork, self).__init__()

        self.fc1_g = nn.Linear(obs_dim, 32)
        self.bn1 =  nn.BatchNorm1d(32)
        self.dp1_g = nn.Dropout(0.0)

        self.fc2_g = nn.Linear(32, 64)
        self.bn2 =  nn.BatchNorm1d(64)
        self.dp2_g = nn.Dropout(0.0)
        
        self.fc3_g = nn.Linear(64, 128)
        self.bn3 =  nn.BatchNorm1d(128)
        self.dp3_g = nn.Dropout(0.0)

        self.fc_out_g = nn.Linear(128, 1)
        self.bn_out =  nn.BatchNorm1d(1)
        self.out_g = nn.Sigmoid()
        self.elu = nn.SiLU() #  nn.ELU()

    def forward(self, x):
        out = self._forward(x)
        return out

    def _forward(self, x):
        # approxiate g fun separately
        x = x.to(torch.float32)
        if len(x.size())==1:
            x = x.unsqueeze(0)
        g_approx = self.fc1_g(x)
        g_approx = self.bn1(g_approx)
        g_approx = self.dp1_g(g_approx)
        g_approx = self.elu(g_approx)

        
# 
        g_approx = self.fc2_g(g_approx)
        g_approx = self.bn2(g_approx)
        g_approx = self.dp2_g(g_approx)
        g_approx = self.elu(g_approx)
        
        # g_approx, hn = self.rnn_g(g_approx)
        
        g_approx = self.fc3_g(g_approx)
        g_approx = self.bn3(g_approx)
        g_approx = self.dp3_g(g_approx)
        g_approx = self.elu(g_approx)
        

        g_approx = self.fc_out_g(g_approx)
        g_approx = self.bn_out(g_approx)
        g_approx = self.out_g(g_approx)

        return g_approx

    def get_g(self, state):
        g_approx = self.forward(state)
        return g_approx

    def weights_init_uniform(self, m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1 or classname.find('Conv') != -1:
                m.weight.data.normal_(0.001, 1.0)
                m.bias.data.fill_(0.001)

class DoubleOfflineQNetwork(SoftQNetwork):
    """Double Q-network with shared API.

    Forward returns (q1, q2, next_state_1, next_state_2) from two underlying
    ``OfflineQNetwork`` instances. ``get_Q`` and ``get_next_s`` select a
    conservative minimum as used by callers in IQ-Learn variants.
    """
    def __init__(self, obs_dim, action_dim, gamma, device='cpu'):
        super(DoubleOfflineQNetwork, self).__init__(obs_dim, action_dim, gamma, device)
        self.q_net1 = OfflineQNetwork(obs_dim, action_dim, gamma, device)
        self.q_net2 = OfflineQNetwork(obs_dim, action_dim, gamma, device)
        
    def _forward(self, x):
        q1, st1 = self.q_net1.forward(x)
        q2, st2 = self.q_net2.forward(x)
        # print(f'st1.shape: {st1.shape}')
        # print(f'st2.shape: {st2.shape}')
        # print(f'torch.stack([st1,st2]).shape: {torch.stack([st1,st2]).shape}')
        # print(f'torch.mean(torch.stack([st1,st2])).shape: {torch.mean(torch.stack([st1,st2]),0).shape}')
        return q1, q2, st1, st2

    def get_Q(self, state):
        """Return conservative Q = min(Q1, Q2)."""
        Q1, Q2, st1, st2 = self.forward(state)
        # return torch.minimum(torch.stack([Q1,Q2]), 0)
        return torch.minimum(Q1,Q2)

    def get_next_s(self, state):
        """Return next-state prediction corresponding to the min-Q branch."""
        Q1, Q2, st1, st2 = self.forward(state)
        # Choose st1 when Q1 <= Q2 element-wise for all actions; otherwise st2.
        mask = (Q1 <= Q2).all(dim=1)  # [B]
        st = torch.where(mask[:, None], st1, st2)
        return st
    def weights_init_uniform(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1 or classname.find('Conv') != -1:
            m.weight.data.normal_(0.001, 1.0)
            m.bias.data.fill_(0.001)

class OfflineQNetwork(SoftQNetwork):
    """Q-network with a parallel head for next-state prediction.

    Forward returns (q, next_state). ``get_Q``/``get_next_s`` expose heads.
    """
    def __init__(self, obs_dim, action_dim, gamma, device='cpu', q_entropy=False):
        super(OfflineQNetwork, self).__init__(obs_dim, action_dim, gamma, device)
        self.obs_dim = obs_dim
        self.gamma = gamma
        if q_entropy:
            dp = 0.25
        else:
            dp=0.0

        self.fc1 = nn.Linear(obs_dim, 32)
        # self.inorm1 = nn.LazyBatchNorm1d()
        self.inorm1 = nn.BatchNorm1d(32)
        self.dp1 = nn.Dropout(dp)

        self.elu = nn.SiLU() # nn.ELU()

        self.fc2 = nn.Linear(32, 64)
        # self.inorm2 = nn.LazyBatchNorm1d()
        self.inorm2 = nn.BatchNorm1d(64)
        self.dp2 = nn.Dropout(dp)

        # self.rnn = nn.RNN(64,64,3)
        # self.rnn_st = nn.RNN(64,64,3)
        self.state_pred = nn.Linear(64, 128)
        self.state_pred_inorm = nn.BatchNorm1d(128)
        # self.state_pred_act = 
        self.state_pred_out = nn.Linear(128, obs_dim)


        self.fc3 = nn.Linear(64, 128)
        # self.inorm3 = nn.LazyBatchNorm1d()
        self.inorm3 = nn.BatchNorm1d(128)
        self.dp3 = nn.Dropout(dp)
        
        self.fc_out = nn.Linear(128, action_dim)
        # self.inorm_out = nn.LazyBatchNorm1d()
        self.inorm_out = nn.BatchNorm1d(action_dim)
        # self.dp_out = nn.Dropout(0.25)
        self.out = nn.Sigmoid()
        # self.inorm4 = nn.LazyBatchNorm1d()
        # self.inorm5 = nn.LazyBatchNorm1d()

        # self.out = nn.Threshold(0,0)
        self.fc1_output = None
        self.fc2_output = None
        self.fc_out_output = None

    def _forward(self, x):
        # approxiate g fun separately

        x = x.to(torch.float32)
        x = self.fc1(x)

        self.fc1_output = x.clone()
        x = self.inorm1(x)
        x = self.dp1(x)
        x = self.elu(x)
        
        x = self.fc2(x)
        self.fc2_output = x.clone()
        x = self.inorm2(x)
        x = self.dp2(x)
        x = self.elu(x)

        next_state = self.state_pred(x.clone())
        next_state = self.state_pred_inorm(next_state)
        # next_state = self.dp3(next_state)
        next_state = self.elu(next_state) 
        next_state = self.state_pred_out(next_state)
        
        x = self.fc3(x)
        x = self.inorm3(x)
        x = self.dp3(x)
        x = self.elu(x)

        x = self.fc_out(x)
        self.fc_out_output = x.clone()
        x = self.inorm_out(x)
        q = self.out(x)
        return q, next_state

    def get_Q(self, state):
        """Return Q-values with shape [B, action_dim]."""
        Q, next_state = self.forward(state)
        return Q

    def get_next_s(self, state):
        """Return next-state prediction with shape [B, obs_dim]."""
        Q, next_state = self.forward(state)
        return next_state

    def get_layers_output(self):
        return self.fc1_output, self.fc2_output, self.fc_out_output, self.rnn_x_st, self.rnn_hn_st, self.rnn_x_q, self.rnn_hn_q

    def weights_init_uniform(self, m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1 or classname.find('Conv') != -1:
                m.weight.data.normal_(0.001, 1.0)
                m.bias.data.fill_(0.001)


class DoubleOfflineQNetwork_orig(SoftQNetwork):
    """Double Q-network without next-state head.

    Keeps two ``OfflineQNetwork_orig`` instances and returns the minimum Q.
    """
    def __init__(self, obs_dim, action_dim, gamma, device='cpu'):
        super(DoubleOfflineQNetwork_orig, self).__init__(obs_dim, action_dim, gamma, device)
        self.q_net1 = OfflineQNetwork_orig(obs_dim, action_dim, gamma, device)
        self.q_net2 = OfflineQNetwork_orig(obs_dim, action_dim, gamma, device)

    def _forward(self, x):
        q1 = self.q_net1.forward(x)
        q2 = self.q_net2.forward(x)
        return q1,q2

    def get_Q(self, state):
        """Return conservative Q = min(Q1, Q2)."""
        Q1, Q2 = self.forward(state)
        # return torch.minimum(torch.stack([Q1,Q2]),0)
        return torch.minimum(Q1,Q2)

    def weights_init_uniform(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1 or classname.find('Conv')!=-1:
            m.weight.data.normal_(0.001, 1.0)
            m.bias.data.fill_(0.001)

class OfflineQNetwork_orig(SoftQNetwork):
    """Simple feed-forward Q-network (no next-state head)."""
    def __init__(self, obs_dim, action_dim, gamma, device='cpu', q_entropy=False):
        super(OfflineQNetwork_orig, self).__init__(obs_dim, action_dim, gamma, device)
        self.gamma = gamma
        if q_entropy:
            dp = 0.25
        else:
            dp = 0.0
        self.fc1 = nn.Linear(obs_dim, 32)
        # self.inorm1 = nn.LazyBatchNorm1d()
        self.inorm1 = nn.BatchNorm1d(32)
        self.dp1 = nn.Dropout(dp)

        self.elu = nn.SiLU() # nn.ELU()
        self.fc2 = nn.Linear(32, 64)
        self.inorm2 = nn.BatchNorm1d(64)
        self.dp2 = nn.Dropout(dp)

        self.fc3 = nn.Linear(64, 128)
        self.inorm3 = nn.BatchNorm1d(128)
        self.dp3 = nn.Dropout(dp)

        self.fc_out = nn.Linear(128, action_dim)
        # self.inorm_out = nn.LazyBatchNorm1d()
        self.inorm_out = nn.BatchNorm1d(action_dim)
        self.out = nn.Sigmoid()

    def _forward(self, x):
        x = self.fc1(x.to(torch.float32))
        self.fc1_output = x.clone()
        x = self.inorm1(x)
        x = self.dp1(x)
        x = self.elu(x)

        # x, hn = self.rnn1(x)
        # self.rnn_x_1 = x.clone()
        # self.rnn_hn_1 = hn.clone()

        x = self.fc2(x)
        self.fc2_output = x.clone()
        x = self.inorm2(x)
        x = self.dp2(x)
        x = self.elu(x)

        x = self.fc3(x)
        x = self.inorm3(x)
        x = self.dp3(x)
        x = self.elu(x)

        x = self.fc_out(x)
        self.fc_out_output = x.clone()
        x = self.inorm_out(x)
        q = self.out(x)
        return q

    def get_Q(self, state):
        """Return Q-values with shape [B, action_dim]."""
        Q = self.forward(state)
        return Q

    def weights_init_uniform(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1 or classname.find('Conv')!=-1:
            m.weight.data.normal_(0.001, 1.0)
            m.bias.data.fill_(0.001)

    def get_layers_output(self):
        return self.fc1_output, self.fc2_output, self.fc_out_output, self.rnn_x_1, self.rnn_hn_1, self.rnn_x_2, self.rnn_hn_2


class RNN_QNetwork(SoftQNetwork):
    """Recurrent Q-network over sequences.

    Expects [B, T, obs_dim] and outputs Q-values at the last step [B, A].
    """
    def __init__(self, obs_dim, action_dim, gamma, device='cpu'):
        super(RNN_QNetwork, self).__init__(obs_dim, action_dim, gamma, device)
        self.gamma = gamma
        self.action_dim = action_dim
        self.hn_size = 128
        self.lstm0 = nn.GRU(input_size=obs_dim, 
                             hidden_size=self.hn_size,
                             num_layers=16,
                             batch_first=True,
                             dropout=0.0)
        self.fc0 = nn.Linear(self.hn_size, 256)
        self.inorm0 = nn.BatchNorm1d(256)
        self.dp0 = nn.Dropout(0.0)
        self.elu = nn.SiLU() # nn.ELU()
        
        self.fc_out = nn.Linear(256, action_dim)
        self.out = nn.Sigmoid()

    def _forward(self, x):
        # print(f'x.shape: {x.shape}')
        output, h_n = self.lstm0(x.to(torch.float32))
        # print(f'h_n.shape: {h_n.shape}')
        # h_n = h_n[-1,:,:].reshape(-1,self.hn_size)
        h_n = output[:,-1,:].reshape(-1,self.hn_size)
        # print(f'output.shape: {output.shape}')
        
        # print(f'c_n.shape: {c_n.shape}')
        x = self.fc0(h_n)
        # x = self.inorm0(x)
        x = self.dp0(x)
        x = self.elu(x)
        
        x = self.fc_out(x)
        q = self.out(x).reshape(-1,self.action_dim)
        # q = x.reshape(-1,self.action_dim)
        # print(f'q.shape: {q.shape}')
        return q

    def get_Q(self, state):
        """Return Q-values with shape [B, action_dim]."""
        Q = self.forward(state)
        return Q

    def weights_init_uniform(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1 or classname.find('Conv')!=-1:
            m.weight.data.normal_(0.001, 1.0)
            m.bias.data.fill_(0.001)

    def get_layers_output(self):
        return self.fc1_output, self.fc2_output, self.fc_out_output, self.rnn_x_1, self.rnn_hn_1, self.rnn_x_2, self.rnn_hn_2

# class Generator:
#     def __init__(self):
#         pass
    
# class Discriminator:
#     def __init__(self):
#         pass
    
    
# class GAN:
#     def __init__(self,):
#         pass
#     def sample_fake(self,):
#         pass
#     def training_step(self,):
#         pass
#     def train(self,):
#         pass
#     def fit_resample(self,dataframe, class_ids):
#         minority_class = 0
#         self.train(dataframe, class_ids, minority_class)
#         df_res, y_resampled =  = self.resample(dataframe, class_ids)
#         return df_res, y_resampled
    

class Classifier(SoftQNetwork):
    """Binary classifier head returning P(action=1 | s) for action_dim==2.

    For action_dim > 2, returns logits/sigmoids for action_dim-1 classes.
    """
    def __init__(self, obs_dim, action_dim, gamma, device='cpu', q_entropy=False):
        super(Classifier, self).__init__(obs_dim, action_dim, gamma, device)
        self.gamma = gamma
        if q_entropy:
            dp = 0.25
        else:
            dp = 0.0
        self.fc1 = nn.Linear(obs_dim, 32)
        # self.inorm1 = nn.LazyBatchNorm1d()
        self.inorm1 = nn.BatchNorm1d(32)
        self.dp1 = nn.Dropout(dp)

        self.elu = nn.SiLU() # nn.ELU()
        self.fc2 = nn.Linear(32, 64)
        self.inorm2 = nn.BatchNorm1d(64)
        self.dp2 = nn.Dropout(dp)

        self.fc3 = nn.Linear(64, 128)
        self.inorm3 = nn.BatchNorm1d(128)
        self.dp3 = nn.Dropout(dp)

        self.fc_out = nn.Linear(128, action_dim-1)
        # self.inorm_out = nn.LazyBatchNorm1d()
        self.inorm_out = nn.BatchNorm1d(action_dim-1)
        self.out = nn.Sigmoid()

    def _forward(self, x):
        x = self.fc1(x.to(torch.float32))
        self.fc1_output = x.clone()
        x = self.inorm1(x)
        x = self.dp1(x)
        x = self.elu(x)

        # x, hn = self.rnn1(x)
        # self.rnn_x_1 = x.clone()
        # self.rnn_hn_1 = hn.clone()

        x = self.fc2(x)
        self.fc2_output = x.clone()
        x = self.inorm2(x)
        x = self.dp2(x)
        x = self.elu(x)

        x = self.fc3(x)
        x = self.inorm3(x)
        x = self.dp3(x)
        x = self.elu(x)

        x = self.fc_out(x)
        self.fc_out_output = x.clone()
        x = self.inorm_out(x)
        q = self.out(x)
        return q

    def get_Q(self, state):
        """Return class probabilities/logits as configured by the head."""
        Q = self.forward(state)
        return Q

    def weights_init_uniform(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1 or classname.find('Conv')!=-1:
            m.weight.data.normal_(0.001, 1.0)
            m.bias.data.fill_(0.001)

    def get_layers_output(self):
        return self.fc1_output, self.fc2_output, self.fc_out_output, self.rnn_x_1, self.rnn_hn_1, self.rnn_x_2, self.rnn_hn_2


import torch.nn.functional as F

class NonNegLinear(nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight.clamp(max=0.))
        # return F.linear(input, -4*F.softplus(self.weight), 4*F.softplus(self.bias))
import math
class NonPositiveLinear(nn.Module):
    """
    y = x @ W^T + b    with W_{ij} <= 0  (enforced via softplus)

    Parameters
    ----------
    in_features : int
    out_features : int
    bias : bool
        If True, adds learnable bias.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # unconstrained parameter
        self.U = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming-type init on U so that W starts around 0
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.U)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    # ------------------------------------------------------------
    @property
    def weight(self):
        """Return the *constrained* non-positive weight matrix."""
        return -F.softplus(self.U)      # <= 0 element-wise

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return f'in_features={self.U.shape[1]}, out_features={self.U.shape[0]}, bias={self.bias is not None}'

class DegreeNetwork0(nn.Module):
    """Legacy degree network (unused). Kept for reference."""
    def __init__(self):
        super(DegreeNetwork, self).__init__()
        
        # self.fc1 = NonNegLinear(1, 1)
        self.fc1 = NonPositiveLinear(1,1)
        self.act = nn.Sigmoid() # nn.ELU()
     
    def forward(self, x):
        return 1*self.act(self.fc1(x.reshape(-1,1).detach()))
    
    def weights_init_uniform(self, m):
        classname = m.__class__.__name__
        # if classname.find('Linear') != -1 or classname.find('Conv')!=-1:
            
            # m.weight.data.fill_(-2)
            # m.bias.data.fill_(-10)
class DegreeNetwork(nn.Module):
    """Smooth indicator for conservative thresholding.

    Returns sigma((tau - r)/t) with learnable temperature t = softplus(log_t).
    """
    def __init__(self, tau=0.3, log_t=0.0):  # t = softplus(log_t)
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(tau)))
        self.log_t = nn.Parameter(torch.tensor(float(log_t)))
    def forward(self, r):  # r in [0,1], no grad if ROD is external
        t = F.softplus(self.log_t) + 1e-6
        return torch.sigmoid((self.tau - r.reshape(-1,1).detach()) / t)
