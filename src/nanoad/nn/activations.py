"""Module wrappers for activation functions."""

from __future__ import annotations

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
