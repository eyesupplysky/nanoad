"""Softmax and fused cross-entropy ops.

``softmax`` caches its forward probabilities and uses them to build a Tensor-valued VJP.
``cross_entropy`` is a fused log-softmax + NLL — its VJP returns the closed-form
``(softmax(x) - one_hot) / N``, scaled by ``out_grad``. The fusion keeps first-order
gradients exact and stable; second-order gradients through the fused VJP treat the cached
probabilities as constants (use unfused softmax + NLL if exact second derivatives are
needed).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nanoad._engine import register_vjp
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
    return Tensor(s, _prev=(x,), _op="softmax", _fwd_ctx={"axis": axis})


def cross_entropy(logits: Tensor, targets: ArrayLike) -> Tensor:
    """Fused log-softmax + negative log-likelihood. Returns scalar mean loss across the batch.

    Backward gives the famous closed form: ``(softmax(logits) - one_hot(targets)) / batch_size``.
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
    return Tensor(
        nll_per_sample.mean(),
        _prev=(logits,),
        _op="cross_entropy",
        _fwd_ctx={"probs": s, "targets": targets_arr, "batch_size": batch_size},
    )


@register_vjp("softmax")
def _vjp_softmax(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    assert fwd_ctx is not None
    (x,) = parents
    axis = fwd_ctx["axis"]
    # Recompute via the public op so x stays in the gradient graph for second-order autograd.
    # dL/dx = s * (dL/ds - sum(s * dL/ds, axis, keepdims))
    s = softmax(x, axis)
    weighted = s * out_grad
    dot = weighted.sum(axis=axis, keepdims=True)
    return (s * (out_grad - dot),)


@register_vjp("cross_entropy")
def _vjp_cross_entropy(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    assert fwd_ctx is not None
    s = fwd_ctx["probs"]
    targets = fwd_ctx["targets"]
    batch_size = fwd_ctx["batch_size"]
    grad_logits = s.copy()
    grad_logits[np.arange(batch_size), targets] -= 1.0
    grad_logits /= batch_size
    # Multiply by upstream scalar out_grad through a public op so the chain is intact.
    return (Tensor(grad_logits) * out_grad,)
