"""
Conservative degree-model helpers.

The utilities in this module provide a small interface around the DegreeNetwork
used by the conservative IQ-Learn variant. They are designed to be imported by
the unified superset agent without pulling in the full training logic.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn


def apply_degree_threshold(
    degree_model: Optional[nn.Module], r: torch.Tensor
) -> torch.Tensor:
    """
    Apply the degree model to a scalar out-degree tensor r in [0,1].

    Returns a tensor in [0,1] representing the conservative gate. If the model
    is None, returns ones_like(r) to avoid impacting the flow.
    """
    if degree_model is None:
        return torch.ones_like(r)
    with torch.no_grad():
        return degree_model(r)
