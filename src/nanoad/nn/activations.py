"""Module wrappers for activation functions."""

from __future__ import annotations

import numpy as np

from nanoad.nn.module import Module
from nanoad.ops.activations import relu as _relu
from nanoad.ops.activations import tanh as _tanh
from nanoad.tensor import Tensor


class ReLU(Module):
    """Module wrapper for nanoad.ops.activations.relu."""

    def forward(self, x: Tensor) -> Tensor:
        return _relu(x)


class Tanh(Module):
    """Module wrapper for nanoad.ops.activations.tanh."""

    def forward(self, x: Tensor) -> Tensor:
        return _tanh(x)


class GELU(Module):
    """Gaussian-Error-Linear-Unit activation, tanh approximation. Composed from public ops."""

    _COEFF: float = float(np.sqrt(2.0 / np.pi))

    def forward(self, x: Tensor) -> Tensor:
        inner = self._COEFF * (x + 0.044715 * (x ** 3))
        return 0.5 * x * (1.0 + _tanh(inner))
