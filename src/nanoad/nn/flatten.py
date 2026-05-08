"""Flatten Module: reshape (N, ...) -> (N, prod(rest))."""

from __future__ import annotations

from nanoad.nn.module import Module
from nanoad.tensor import Tensor


class Flatten(Module):
    """Collapse all non-batch dims into one. Output shape (N, -1)."""

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], -1)
