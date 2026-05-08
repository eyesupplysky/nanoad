"""Adam optimizer with bias-corrected moment estimates."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from nanoad.tensor import Tensor


class Adam:
    """Adam (Kingma & Ba, 2014). Per-param first/second moment EMAs with bias correction."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.params: list[Tensor] = list(params)
        self.lr: float = lr
        self.b1: float = betas[0]
        self.b2: float = betas[1]
        self.eps: float = eps
        self.t: int = 0
        self.m: list[NDArray[np.float64]] = [np.zeros_like(p.data) for p in self.params]
        self.v: list[NDArray[np.float64]] = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        """Update every parameter by one Adam step using its current .grad."""
        self.t += 1
        b1_correction = 1.0 - self.b1**self.t
        b2_correction = 1.0 - self.b2**self.t
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)
            m_hat = self.m[i] / b1_correction
            v_hat = self.v[i] / b2_correction
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        """Reset .grad on every managed parameter."""
        for p in self.params:
            p.zero_grad()
