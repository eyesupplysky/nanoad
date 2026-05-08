"""Sequential composition of Modules."""

from __future__ import annotations

from nanoad.nn.module import Module
from nanoad.tensor import Tensor


class Sequential(Module):
    """Apply layers in order; each layer's output feeds the next."""

    def __init__(self, *layers: Module) -> None:
        self.layers: list[Module] = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
