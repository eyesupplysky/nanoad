"""Stochastic gradient descent."""

from __future__ import annotations

from collections.abc import Iterable

from nanoad.tensor import Tensor


class SGD:
    """Plain SGD: p.data <- p.data - lr * p.grad. No momentum, no weight decay."""

    def __init__(self, params: Iterable[Tensor], lr: float) -> None:
        self.params: list[Tensor] = list(params)
        self.lr: float = lr

    def step(self) -> None:
        """Update every parameter by one SGD step using its current .grad."""
        for p in self.params:
            if p.grad is None:
                continue
            p.data = p.data - self.lr * p.grad.data

    def zero_grad(self) -> None:
        """Reset .grad on every managed parameter."""
        for p in self.params:
            p.zero_grad()
