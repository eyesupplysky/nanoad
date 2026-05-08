"""Softmax and fused cross-entropy ops."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nanoad.tensor import Tensor


def _softmax_array(x: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Numerically stable softmax: subtract max before exp, then normalize."""
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    out: NDArray[np.float64] = exp / exp.sum(axis=axis, keepdims=True)
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable softmax along axis. Returns probabilities summing to 1."""
    s = _softmax_array(x.data, axis)
    out = Tensor(s, _prev=(x,), _op="softmax")

    def _backward() -> None:
        # dL/dx = s * (dL/ds - sum(s * dL/ds, axis, keepdims))
        weighted = s * out.grad
        dot = weighted.sum(axis=axis, keepdims=True)
        x.grad += s * (out.grad - dot)

    out._backward = _backward
    return out


def cross_entropy(logits: Tensor, targets: ArrayLike) -> Tensor:
    """Fused log-softmax + negative log-likelihood. Returns scalar mean loss across the batch.

    Backward gives the famous closed form: (softmax(logits) - one_hot(targets)) / batch_size.
    """
    if logits.data.ndim != 2:
        raise ValueError(f"cross_entropy expects 2-d logits; got shape {logits.shape}")
    targets_arr = np.asarray(targets, dtype=np.int64)
    batch_size = logits.data.shape[0]
    if targets_arr.ndim != 1 or targets_arr.shape[0] != batch_size:
        raise ValueError(
            f"targets must be 1-d with batch size {batch_size}; got shape {targets_arr.shape}"
        )

    shifted = logits.data - logits.data.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    sum_exp = exp.sum(axis=-1, keepdims=True)
    log_probs = shifted - np.log(sum_exp)
    s = exp / sum_exp

    nll_per_sample = -log_probs[np.arange(batch_size), targets_arr]
    out = Tensor(nll_per_sample.mean(), _prev=(logits,), _op="cross_entropy")

    def _backward() -> None:
        grad = s.copy()
        grad[np.arange(batch_size), targets_arr] -= 1.0
        grad /= batch_size
        logits.grad += grad * out.grad

    out._backward = _backward
    return out
