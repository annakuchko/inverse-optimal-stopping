"""
Unified IQ-Learn entrypoint

This module centralizes the remaining valid IQ-Learn implementations into a
single import surface. It provides a thin wrapper class that delegates to the
canonical implementations while letting callers choose the variant at runtime.

Variants supported:
  - base:        inverse_opt_stopping.iq_learn_base.IQ_Agent
  - conserv:     inverse_opt_stopping.iq_learn_conserv.IQ_Agent
  - car:         inverse_opt_stopping.iq_learn_baase_car.IQ_Agent

Usage:
  from inverse_opt_stopping.iq_learn_unified import IQ_Agent, plot_st_reg_car

  agent = IQ_Agent(variant="base", obs_dim=..., ...)
  # or agent = IQ_Agent("conserv", ...)

Notes:
  - This file intentionally keeps logic minimal by proxying to the canonical
    modules. It avoids duplicating long implementations while still offering a
    single, consistent import path for users and scripts.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
import numpy as np

# Optional superset implementation pieces
import torch
from torch import nn
from typing import Tuple
try:
    from .q_networks import (
        OfflineQNetwork,
        OfflineQNetwork_orig,
        DoubleOfflineQNetwork,
        gNetwork,
        Classifier,
        DegreeNetwork,
    )
except Exception:
    # Leave these optional so that importing the wrapper still works
    OfflineQNetwork = OfflineQNetwork_orig = DoubleOfflineQNetwork = None  # type: ignore
    gNetwork = Classifier = DegreeNetwork = None  # type: ignore

try:
    from .helpers_oversampling import oversample_memory as _oversample_memory
except Exception:
    _oversample_memory = None  # type: ignore

try:
    from .helpers_conservative import apply_degree_threshold as _apply_degree_threshold
except Exception:
    _apply_degree_threshold = None  # type: ignore

# Import canonical variants
try:
    from .iq_learn_base import IQ_Agent as _IQ_Agent_Base, plot_st_reg_car as _plot_st_reg_car_base
except Exception as e:
    _IQ_Agent_Base = None  # type: ignore
    _plot_st_reg_car_base = None  # type: ignore
    _base_import_error = e
else:
    _base_import_error = None

try:
    from .iq_learn_conserv import IQ_Agent as _IQ_Agent_Conserv, plot_st_reg_car as _plot_st_reg_car_conserv
except Exception as e:
    _IQ_Agent_Conserv = None  # type: ignore
    _plot_st_reg_car_conserv = None  # type: ignore
    _conserv_import_error = e
else:
    _conserv_import_error = None

try:
    from .iq_learn_baase_car import IQ_Agent as _IQ_Agent_Car, plot_st_reg_car as _plot_st_reg_car_car
except Exception as e:
    _IQ_Agent_Car = None  # type: ignore
    _plot_st_reg_car_car = None  # type: ignore
    _car_import_error = e
else:
    _car_import_error = None


Variant = Literal["base", "conserv", "car"]


class IQ_Agent:
    """
    Unified IQ-Learn agent wrapper. Dispatches to the chosen canonical variant.

    Parameters
    - variant: one of {"base", "conserv", "car"}. Defaults to "base".
    - All other keyword arguments are passed to the underlying implementation.
    """

    def __init__(
        self,
        variant: Variant = "base",
        **kwargs: Any,
    ) -> None:
        self.variant: Variant = variant
        self._impl = self._make_impl(variant=variant, **kwargs)

    @staticmethod
    def _make_impl(variant: Variant, **kwargs: Any):  # type: ignore[no-untyped-def]
        if variant == "base":
            if _IQ_Agent_Base is None:
                raise ImportError(
                    f"Failed to import base variant: {_base_import_error}"
                )
            return _IQ_Agent_Base(**kwargs)
        elif variant == "conserv":
            if _IQ_Agent_Conserv is None:
                raise ImportError(
                    f"Failed to import conserv variant: {_conserv_import_error}"
                )
            return _IQ_Agent_Conserv(**kwargs)
        elif variant == "car":
            if _IQ_Agent_Car is None:
                raise ImportError(
                    f"Failed to import car variant: {_car_import_error}"
                )
            return _IQ_Agent_Car(**kwargs)
        else:
            raise ValueError(
                f"Unknown variant '{variant}'. Expected one of 'base', 'conserv', 'car'."
            )

    # ------------- proxy all attribute access to underlying implementation ----
    def __getattr__(self, name: str) -> Any:
        # Called only if attribute not found on wrapper.
        return getattr(self._impl, name)

    # Expose a way to get the underlying implementation directly when needed.
    @property
    def impl(self):
        return self._impl


def plot_st_reg_car(iq_agent: IQ_Agent | Any, test_memory: Any):  # type: ignore[no-untyped-def]
    """
    Dispatch to the correct plot function for the agent's variant.

    If a raw underlying agent instance from a canonical module is passed, this
    will try to infer the variant by type.
    """
    # If caller passed the unified wrapper, unwrap to the impl for type checks.
    agent = getattr(iq_agent, "impl", iq_agent)

    # Try dispatch by instance type first.
    if _IQ_Agent_Conserv is not None and isinstance(agent, _IQ_Agent_Conserv):
        if _plot_st_reg_car_conserv is None:
            raise ImportError(
                f"Conservative plot function unavailable: {_conserv_import_error}"
            )
        return _plot_st_reg_car_conserv(agent, test_memory)
    if _IQ_Agent_Car is not None and isinstance(agent, _IQ_Agent_Car):
        if _plot_st_reg_car_car is None:
            raise ImportError(
                f"Car plot function unavailable: {_car_import_error}"
            )
        return _plot_st_reg_car_car(agent, test_memory)
    if _IQ_Agent_Base is not None and isinstance(agent, _IQ_Agent_Base):
        if _plot_st_reg_car_base is None:
            raise ImportError(
                f"Base plot function unavailable: {_base_import_error}"
            )
        return _plot_st_reg_car_base(agent, test_memory)

    # If types are inconclusive, fall back on variant hint if present.
    variant_hint: Optional[str] = getattr(iq_agent, "variant", None)  # type: ignore[attr-defined]
    if variant_hint == "conserv":
        if _plot_st_reg_car_conserv is None:
            raise ImportError(
                f"Conservative plot function unavailable: {_conserv_import_error}"
            )
        return _plot_st_reg_car_conserv(agent, test_memory)
    if variant_hint == "car":
        if _plot_st_reg_car_car is None:
            raise ImportError(
                f"Car plot function unavailable: {_car_import_error}"
            )
        return _plot_st_reg_car_car(agent, test_memory)
    # Default to base
    if _plot_st_reg_car_base is None:
        raise ImportError(
            f"Base plot function unavailable: {_base_import_error}"
        )
    return _plot_st_reg_car_base(agent, test_memory)


__all__ = [
    "IQ_Agent",
    "plot_st_reg_car",
]


class IQ_AgentSuperset:
    """
    Single-file superset IQ-Learn agent.

    Unifies the distinct features of base, conservative, and car variants under one
    implementation surface. For training/testing, this class can delegate to the
    canonical implementations to preserve exact behavior, and then mirror the
    learned weights into its own networks.

    Parameters (selected):
    - obs_dim, action_dim: input and action sizes
    - approx_g: approximate g(s) with a separate network
    - approx_dynamics: use a next-state head in Q-network
    - conservative: enable conservative degree model logic
    - q_entropy: enable dropout/ensembling in Q for uncertainty
    - classify: train binary classifier head instead of Q
    - oversampling: one of {None, 'SMOTE', 'CS-SMOTE', 'LSMOTE'}
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        *,
        divergence: str = "kl_fix",
        approx_g: bool = True,
        approx_dynamics: bool = True,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        q_lr: float = 1e-3,
        env_lr: float = 1e-3,
        g_lr: float = 1e-3,
        device: str = "cpu",
        dt: float = 1.0,
        oversampling: Optional[str] = None,
        plot_against_time: bool = False,
        seed: Optional[int] = None,
        cross_val_splits: int = 2,
        conservative: bool = False,
        discretiser: Optional[Any] = None,
        out_thresh: float = 0.005,
        q_entropy: bool = False,
        SMOTE_K: int = 12,
        is_cs: bool = False,
        classify: bool = False,
        cs: float = 0.999,
        cs_decay: float = 0.99,
        eps_decay_rate: float = 0.99,
        variant_hint: Optional[Variant] = None,
        pre_oversample: bool = False,
        use_native: bool = False,
        **kwargs: Any,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.divergence = divergence
        self.gamma = gamma
        self.epsilon = epsilon
        self.dt = dt
        self.oversampling = oversampling
        self.plot_against_time = plot_against_time
        self.cross_val_splits = cross_val_splits
        self.conservative = conservative
        self.discretiser = discretiser
        self.out_thresh = out_thresh
        self.q_entropy = q_entropy
        self.SMOTE_K = SMOTE_K
        self.IS_CS = is_cs
        self.classify = classify
        self.cs = cs
        self.cs_decay = cs_decay
        self.eps_decay_rate = eps_decay_rate
        self.approx_g = approx_g
        self.approx_dynamics = approx_dynamics
        self.variant_hint = variant_hint
        self.pre_oversample = pre_oversample
        self.use_native = use_native

        # Build networks (lightweight; actual training may be delegated)
        if classify and Classifier is not None:
            # Native classifier head for binary action policy
            self.q_net = Classifier(obs_dim=obs_dim, action_dim=action_dim, gamma=gamma, device=device, q_entropy=q_entropy)  # type: ignore
            self.g_net = None
        else:
            if approx_dynamics:
                # When approx_g is used, some pipelines concatenate y -> obs (+1)
                q_obs_dim = obs_dim + (1 if approx_g else 0)
                if OfflineQNetwork is None:
                    raise ImportError("q_networks unavailable for superset")
                self.q_net = OfflineQNetwork(obs_dim=q_obs_dim, action_dim=action_dim, gamma=gamma, device=device, q_entropy=q_entropy)  # type: ignore
                if approx_g and gNetwork is not None:
                    self.g_net = gNetwork(obs_dim=obs_dim, action_dim=action_dim, gamma=gamma, device=device)  # type: ignore
                else:
                    self.g_net = None
            else:
                if OfflineQNetwork_orig is None:
                    raise ImportError("q_networks unavailable for superset")
                self.q_net = OfflineQNetwork_orig(obs_dim=obs_dim, action_dim=action_dim, gamma=gamma, device=device, q_entropy=q_entropy)  # type: ignore
                self.g_net = None

        if conservative and DegreeNetwork is not None:
            self.degree_model = DegreeNetwork()
        else:
            self.degree_model = None

        self.q_net = self.q_net.to(self.device)
        if self.g_net is not None:
            self.g_net = self.g_net.to(self.device)
        if self.degree_model is not None:
            self.degree_model = self.degree_model.to(self.device)

        # Optimizers/schedulers (placeholders; exact behavior preserved when delegating)
        self.critic_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=q_lr)
        if self.g_net is not None:
            self.g_optimizer = torch.optim.Adam(self.g_net.parameters(), lr=g_lr)

        self.best_q_net = None
        self.best_g_net = None
        self.epoch = 0

        # delegate can be used to preserve exact train/test logic
        self._delegate: Optional[IQ_Agent] = None

    # ----------------------------- Inference helpers -------------------------
    def getQ(self, obs: torch.Tensor, evaluate: bool = False, mc_estimate: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        self.q_net.eval() if evaluate else self.q_net.train()
        with torch.no_grad() if evaluate else torch.enable_grad():
            if evaluate and self.q_entropy and mc_estimate:
                qs = [self.q_net.get_Q(obs.clone().to(self.device)) for _ in range(5)]
                q = torch.stack(qs, dim=-1).mean(dim=-1)
                q_std = torch.stack(qs, dim=-1).std(dim=-1)
            elif evaluate and self.q_entropy:
                qs = [self.q_net.get_Q(obs.clone().to(self.device)) for _ in range(3)]
                q = torch.stack(qs, dim=-1).mean(dim=-1)
                q_std = torch.stack(qs, dim=-1).std(dim=-1)
            else:
                q = self.q_net.get_Q(obs.clone().to(self.device))
                q_std = torch.zeros_like(q)
        return q.reshape(-1, self.action_dim), q_std.reshape(-1, self.action_dim)

    def getQ_s_a(self, states: torch.Tensor, actions: torch.Tensor, evaluate: bool = False, mc_estimate: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        Qs, qs_var = self.getQ(states, evaluate=evaluate, mc_estimate=mc_estimate)
        idx = actions.to(torch.int64).reshape(-1, 1).to(Qs.device)
        return torch.gather(Qs, dim=-1, index=idx), torch.gather(qs_var, dim=-1, index=idx)

    def getS_dash(self, states: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        self.q_net.eval() if evaluate else self.q_net.train()
        with torch.no_grad():
            if hasattr(self.q_net, 'get_next_s'):
                s_dash = self.q_net.get_next_s(states.clone().to(self.device)).cpu() + states.clone()
            else:
                s_dash = states.clone()
        return s_dash

    def getV(self, obs: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        q, _ = self.getQ(obs, evaluate=evaluate)
        return self.epsilon * torch.logsumexp(q / self.epsilon, dim=1, keepdim=True)

    def infer_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q, _ = self.getQ_s_a(state, action, evaluate=True)
        return q

    def infer_next_s(self, state: torch.Tensor) -> torch.Tensor:
        return self.getS_dash(state, evaluate=True)

    # ------------------------------ Train/Test (delegate) --------------------
    def _ensure_delegate(self):
        if self._delegate is not None:
            return
        # choose canonical variant to delegate
        variant: Variant = 'conserv' if self.conservative else 'base'
        self._delegate = IQ_Agent(variant=variant,
                                  obs_dim=self.obs_dim,
                                  action_dim=self.action_dim,
                                  divergence=self.divergence,
                                  approx_g=self.approx_g,
                                  approx_dynamics=self.approx_dynamics,
                                  gamma=self.gamma,
                                  epsilon=self.epsilon,
                                  q_lr=self.critic_optimizer.param_groups[0]['lr'],
                                  env_lr=1e-3,
                                  g_lr=1e-3,
                                  device=self.device,
                                  dt=self.dt,
                                  oversampling=self.oversampling,
                                  plot_against_time=self.plot_against_time,
                                  cross_val_splits=self.cross_val_splits,
                                  conservative=self.conservative,
                                  discretiser=self.discretiser,
                                  out_thresh=self.out_thresh,
                                  q_entropy=self.q_entropy,
                                  SMOTE_K=self.SMOTE_K,
                                  is_cs=self.IS_CS,
                                  classify=self.classify,
                                  cs=self.cs,
                                  cs_decay=self.cs_decay,
                                  eps_decay_rate=self.eps_decay_rate)

    def train(self, mem: dict, *, batch_size: int = 128, n_epoches: int = 1, verbose: int = 1, **kwargs: Any):
        # Optional pre-oversample pass
        if self.pre_oversample and self.oversampling and _oversample_memory is not None:
            try:
                # Normalize a couple of alias strings for oversampling strategies
                strategy = self.oversampling
                if isinstance(strategy, str) and strategy.upper() in {"CS-LSMOTE", "CS_LSMOTE"}:
                    strategy = "LSMOTE"
                mem = _oversample_memory(mem, strategy=strategy, smote_k=self.SMOTE_K, approx_g=self.approx_g, is_cs=self.IS_CS)
            except Exception:
                pass

        native_classifier = self.use_native and self.classify and not self.approx_dynamics
        if native_classifier:
            # Native training path
            self._native_train_classifier(mem, batch_size=batch_size, n_epoches=n_epoches, verbose=verbose)
            # Prepare delegate for downstream test() using our trained weights
            self._ensure_delegate()
            assert self._delegate is not None
            try:
                self._delegate.q_net.load_state_dict(self.q_net.state_dict())
                if getattr(self._delegate, 'g_net', None) is not None and self.g_net is not None:
                    self._delegate.g_net.load_state_dict(self.g_net.state_dict())
            except Exception:
                pass
        else:
            # Delegate full training to canonical implementation
            self._ensure_delegate()
            assert self._delegate is not None
            self._delegate.train(mem, batch_size=batch_size, n_epoches=n_epoches, verbose=verbose)
            # Mirror learned weights into this instance's networks when shapes match
            try:
                self.q_net.load_state_dict(self._delegate.q_net.state_dict())
                if getattr(self._delegate, 'g_net', None) is not None and self.g_net is not None:
                    self.g_net.load_state_dict(self._delegate.g_net.state_dict())
            except Exception:
                pass
            # Copy metrics if present
            for attr in ['mtte', 'memr', 'f1', 'pr_auc', 'balanced_accuracy', 'epoch_balanced_accuracy', 'epoch_loss']:
                if hasattr(self._delegate, attr):
                    setattr(self, attr, getattr(self._delegate, attr))

    def test(self, test_memory: dict, *, from_grid: bool = False, validation: bool = False):
        # Always use delegate for test to leverage canonical metrics computation
        self._ensure_delegate()
        assert self._delegate is not None
        # If we trained natively, ensure delegate has our weights
        try:
            # Load weights into underlying implementation
            impl = getattr(self._delegate, 'impl', self._delegate)
            impl.q_net.load_state_dict(self.q_net.state_dict())
            if getattr(impl, 'g_net', None) is not None and self.g_net is not None:
                impl.g_net.load_state_dict(self.g_net.state_dict())
        except Exception:
            pass
        # Ensure delegate impl has fallback best_* snapshots if expected by canonical test
        try:
            impl = getattr(self._delegate, 'impl', self._delegate)
            if getattr(impl, 'best_q_net', None) is None:
                impl.best_q_net = impl.q_net.state_dict()
            if getattr(impl, 'g_net', None) is not None and getattr(impl, 'best_g_net', None) is None:
                impl.best_g_net = impl.g_net.state_dict()
        except Exception:
            pass
        out = self._delegate.test(test_memory, from_grid=from_grid, validation=validation)
        # Mirror metrics back to superset for uniform access
        for attr in ['mtte', 'memr', 'f1', 'pr_auc', 'balanced_accuracy']:
            if hasattr(self._delegate, attr):
                setattr(self, attr, getattr(self._delegate, attr))
        return out

    # ------------------------------ Native trainer (limited) -----------------
    def _native_train_classifier(self, mem: dict, *, batch_size: int, n_epoches: int, verbose: int):
        self.q_net.train()
        opt = self.critic_optimizer
        loss_fn = nn.BCELoss()
        X = torch.tensor(np.asarray(mem['state_mem']), dtype=torch.float32)
        y = torch.tensor(np.asarray(mem['action_mem']).reshape(-1, 1), dtype=torch.float32)
        N = X.shape[0]
        for epoch in range(n_epoches):
            perm = torch.randperm(N)
            for i in range(0, N, batch_size):
                idx = perm[i:i+batch_size]
                xb = X[idx].to(self.device)
                yb = y[idx].to(self.device)
                opt.zero_grad()
                out = self.q_net(xb)
                loss = loss_fn(out.float(), yb)
                loss.backward()
                opt.step()
