"""BatchNorm2d: normalize across (N, H, W) per channel; learnable affine; running stats."""

from __future__ import annotations

import numpy as np

from nanoad.nn.module import Module
from nanoad.tensor import Tensor


class BatchNorm2d(Module):
    """Batch-norm over (N, C, H, W) input, normalizing per channel across (N, H, W)."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.num_features: int = num_features
        self.eps: float = eps
        self.momentum: float = momentum
        self.gamma = Tensor(np.ones(num_features))
        self.beta = Tensor(np.zeros(num_features))
        self.running_mean = np.zeros(num_features, dtype=np.float64)
        self.running_var = np.ones(num_features, dtype=np.float64)

    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim != 4:
            raise ValueError(f"BatchNorm2d expects 4-d input (N,C,H,W); got {x.shape}")
        c = self.num_features
        if x.data.shape[1] != c:
            raise ValueError(
                f"BatchNorm2d expected C={c} channels; got {x.shape[1]} from input shape {x.shape}"
            )

        if self.training:
            mean = x.mean(axis=(0, 2, 3), keepdims=True)
            centered = x - mean
            var = (centered * centered).mean(axis=(0, 2, 3), keepdims=True)
            # Update running stats from batch stats (no autograd through .data).
            self.running_mean = (
                (1.0 - self.momentum) * self.running_mean
                + self.momentum * mean.data.reshape(c)
            )
            self.running_var = (
                (1.0 - self.momentum) * self.running_var
                + self.momentum * var.data.reshape(c)
            )
            x_hat = centered / (var + self.eps) ** 0.5
        else:
            mean_const = Tensor(self.running_mean.reshape(1, c, 1, 1))
            var_const = Tensor(self.running_var.reshape(1, c, 1, 1))
            x_hat = (x - mean_const) / (var_const + self.eps) ** 0.5

        return self.gamma.reshape(1, c, 1, 1) * x_hat + self.beta.reshape(1, c, 1, 1)
