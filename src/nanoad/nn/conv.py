"""Conv2d Module wrapper around the conv2d op."""

from __future__ import annotations

import numpy as np

from nanoad.nn.module import Module
from nanoad.ops.conv import conv2d as _conv2d
from nanoad.tensor import Tensor


class Conv2d(Module):
    """2-D cross-correlation layer. Layout NCHW. He-initialized weight."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        fan_in = in_channels * kernel_size * kernel_size
        std = float(np.sqrt(2.0 / fan_in))
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * std
        )
        self.bias: Tensor | None = Tensor(np.zeros(out_channels)) if bias else None
        self.stride: int = stride
        self.padding: int = padding

    def forward(self, x: Tensor) -> Tensor:
        return _conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
