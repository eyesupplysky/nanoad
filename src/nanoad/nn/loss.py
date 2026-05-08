"""Loss function modules."""

from __future__ import annotations

from numpy.typing import ArrayLike

from nanoad.nn.module import Module
from nanoad.ops.softmax import cross_entropy as _cross_entropy
from nanoad.tensor import Tensor


class CrossEntropy(Module):
    """Cross-entropy with integer targets — Module wrapper around the fused op."""

    def forward(self, logits: Tensor, targets: ArrayLike) -> Tensor:
        return _cross_entropy(logits, targets)
