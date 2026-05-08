"""MaxPool2d Module wrapper around the max_pool2d op."""

from __future__ import annotations

from nanoad.nn.module import Module
from nanoad.ops.pool import max_pool2d as _max_pool2d
from nanoad.tensor import Tensor


class MaxPool2d(Module):
    """Non-overlapping max pool by default (stride=kernel). No padding."""

    def __init__(self, kernel_size: int, stride: int | None = None) -> None:
        super().__init__()
        self.kernel_size: int = kernel_size
        self.stride: int | None = stride

    def forward(self, x: Tensor) -> Tensor:
        return _max_pool2d(x, self.kernel_size, self.stride)
