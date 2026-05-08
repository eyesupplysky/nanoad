"""Linear (fully-connected) layer."""

from __future__ import annotations

import numpy as np

from nanoad.nn.module import Module
from nanoad.ops.bmm import bmm
from nanoad.tensor import Tensor


class Linear(Module):
    """Affine map y = x @ W + b with He-initialized weights. Input may be 2-d or higher."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        std = float(np.sqrt(2.0 / in_features))
        self.weight = Tensor(np.random.randn(in_features, out_features) * std)
        self.bias = Tensor(np.zeros(out_features))

    def forward(self, x: Tensor) -> Tensor:
        return bmm(x, self.weight) + self.bias
