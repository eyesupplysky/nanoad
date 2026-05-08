"""LayerNorm: per-token normalization over the last axis with learnable affine.

Composed entirely from public ops (``mean``, ``sub``, ``mul``, ``power``, ``div``, ``add``),
so backward is HOA-correct for free — no dedicated op required.
"""

from __future__ import annotations

import numpy as np

from nanoad.nn.module import Module
from nanoad.tensor import Tensor


class LayerNorm(Module):
    """Layer normalization over the last axis.

    For input of shape ``(..., num_features)``, normalizes each "token" (slice over the
    last axis) to zero mean and unit variance, then applies a learnable per-feature affine
    ``gamma * x_hat + beta``. The reduction axis is fixed to ``-1`` because that is the
    universal transformer convention; multi-axis normalization is out of scope.
    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features: int = num_features
        self.eps: float = eps
        self.gamma = Tensor(np.ones(num_features))
        self.beta = Tensor(np.zeros(num_features))

    def forward(self, x: Tensor) -> Tensor:
        if x.data.shape[-1] != self.num_features:
            raise ValueError(
                f"LayerNorm expected last-axis size {self.num_features}; got input shape {x.shape}"
            )
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        var = (centered * centered).mean(axis=-1, keepdims=True)
        x_hat = centered / (var + self.eps) ** 0.5
        return self.gamma * x_hat + self.beta
