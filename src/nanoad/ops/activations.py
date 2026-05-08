"""Activation functions: relu, tanh."""

from __future__ import annotations

import numpy as np

from nanoad.tensor import Tensor


def relu(x: Tensor) -> Tensor:
    """Rectified linear unit, elementwise. Sub-gradient at 0 is taken to be 0 by convention."""
    out = Tensor(np.maximum(x.data, 0.0), _prev=(x,), _op="relu")

    def _backward() -> None:
        x.grad += (x.data > 0.0) * out.grad

    out._backward = _backward
    return out


def tanh(x: Tensor) -> Tensor:
    """Hyperbolic tangent, elementwise. d(tanh x)/dx = 1 - tanh^2 x."""
    t = np.tanh(x.data)
    out = Tensor(t, _prev=(x,), _op="tanh")

    def _backward() -> None:
        x.grad += (1.0 - t * t) * out.grad

    out._backward = _backward
    return out
