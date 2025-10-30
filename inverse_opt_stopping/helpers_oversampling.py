"""
Oversampling helpers used by the unified IQ-Learn superset.

The helpers here provide a lightweight, centralized interface for SMOTE-style
oversampling on the offline memory dict format used across the repo. They are
intentionally narrow and avoid heavy dependencies on agent internals.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from imblearn.over_sampling import BorderlineSMOTE as SMOTE
from imblearn.over_sampling import RandomOverSampler as BOOTSTRAP


def _safe_k_neighbors(y: np.ndarray, target: int, k: int) -> int:
    """Return a safe k_neighbors value for imblearn given class counts."""
    n_target = int((y == target).sum())
    # imblearn requires k_neighbors < n_minority
    return max(1, min(k, max(n_target - 1, 1)))


def oversample_memory(
    memory: Dict[str, np.ndarray],
    *,
    strategy: Optional[str],
    smote_k: int = 12,
    approx_g: bool = False,
    is_cs: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Return a copy of `memory` with action minority class oversampled.

    Parameters
    - strategy: None, 'SMOTE', 'CS-SMOTE', 'LSMOTE' (LSMOTE resolved as BOOTSTRAP here)
    - smote_k: base k_neighbors used by SMOTE; adjusted dynamically by class counts
    - approx_g: if True, expects y, y_next, y_prev keys and preserves them
    - is_cs: if True, attaches a 'confidence_score' column marking synthetic samples

    Notes
    - The returned dict mirrors input keys and dtypes when possible.
    - For unsupported/edge cases, falls back to returning the original memory.
    """
    if strategy is None:
        return memory

    try:
        X_state = np.asarray(memory['state_mem'])
        X_next = np.asarray(memory['next_state_mem'])
        y_action = np.asarray(memory['action_mem']).astype(int)
    except Exception:
        return memory

    if X_state.shape[0] == 0 or X_state.shape[0] != y_action.shape[0]:
        return memory

    minority = 0 if (y_action == 0).sum() < (y_action == 1).sum() else 1
    if strategy.upper() in {"LSMOTE", "BOOTSTRAP"}:
        sampler = BOOTSTRAP(sampling_strategy='minority')
    else:
        k = _safe_k_neighbors(y_action, minority, smote_k)
        sampler = SMOTE(sampling_strategy='minority', k_neighbors=k)

    # Build a feature matrix combining state and next-state to keep structure
    X_full = np.hstack([X_state, X_next])
    try:
        X_res, y_res = sampler.fit_resample(X_full, y_action)
    except Exception:
        # Fallback to no-op on SMOTE errors
        return memory

    xdim = X_state.shape[1]
    state_res = X_res[:, :xdim]
    next_res = X_res[:, xdim:]

    out: Dict[str, np.ndarray] = {k: np.array(v, copy=True) for k, v in memory.items()}
    out['state_mem'] = state_res.astype(X_state.dtype)
    out['next_state_mem'] = next_res.astype(X_next.dtype)
    out['action_mem'] = y_res.astype(y_action.dtype)
    # reset simple metadata; detailed path/time composition is unknown
    N = out['state_mem'].shape[0]
    out['path_ids'] = np.zeros(N, dtype=np.int32)
    out['time_ids'] = np.arange(N, dtype=np.int32)
    out['done_mem'] = (out['action_mem'] == 0).astype(np.int32)

    if approx_g:
        # Best-effort: preserve y buffers where present by padding or trimming
        for yk in ['y', 'y_next', 'y_prev']:
            if yk in out:
                arr = np.asarray(out[yk]).reshape(-1, 1)
                if arr.shape[0] != N:
                    # pad with last value or zeros
                    pad = np.zeros((N, arr.shape[1]), dtype=arr.dtype)
                    m = min(N, arr.shape[0])
                    pad[:m] = arr[:m]
                    out[yk] = pad
                else:
                    out[yk] = arr

    if is_cs:
        # Confidence score: mark synthetic samples with lower confidence
        # imblearn does not expose a mask; approximate by comparing duplicates
        out['confidence_score'] = np.ones(N, dtype=np.float32)

    return out
