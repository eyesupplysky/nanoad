"""Embedding lookup: gather rows of a weight matrix by integer indices.

Forward gathers rows; backward scatters the upstream gradient back into a sparse-style
matrix shaped like the weight. The gather and scatter ops are each other's VJP, mirroring
the ``unbroadcast`` / ``broadcast_to`` pattern — so the gradient DAG stays tape-aware
and second-order autograd works through the embedding.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nanoad._engine import register_vjp
from nanoad.tensor import Tensor


def embedding(weight: Tensor, indices: ArrayLike) -> Tensor:
    """Gather rows of ``weight`` indexed by ``indices``.

    For ``weight`` of shape ``(vocab_size, embed_dim)`` and integer ``indices`` of
    arbitrary shape ``S``, the output has shape ``S + (embed_dim,)``.
    """
    if weight.data.ndim != 2:
        raise ValueError(f"embedding weight must be 2-d (vocab, dim); got shape {weight.shape}")
    indices_arr = np.asarray(indices, dtype=np.int64)
    return Tensor(
        weight.data[indices_arr],
        _prev=(weight,),
        _op="embedding",
        _fwd_ctx={"indices": indices_arr, "vocab_size": weight.data.shape[0]},
    )


def _embedding_scatter(
    grad: Tensor,
    indices: NDArray[np.int64],
    vocab_size: int,
) -> Tensor:
    """Sparse scatter-add: build ``(vocab_size, embed_dim)`` by adding rows of grad at indices.

    The transpose (and mutual VJP) of ``embedding``'s gather. Repeated indices produce
    accumulation through ``np.add.at`` rather than overwrite.
    """
    embed_dim = grad.data.shape[-1]
    out = np.zeros((vocab_size, embed_dim), dtype=np.float64)
    np.add.at(out, indices, grad.data)
    return Tensor(
        out,
        _prev=(grad,),
        _op="embedding_scatter",
        _fwd_ctx={"indices": indices},
    )


@register_vjp("embedding")
def _vjp_embedding(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    assert fwd_ctx is not None
    return (_embedding_scatter(out_grad, fwd_ctx["indices"], fwd_ctx["vocab_size"]),)


@register_vjp("embedding_scatter")
def _vjp_embedding_scatter(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    assert fwd_ctx is not None
    indices = fwd_ctx["indices"]
    # Transpose of scatter is gather: pick rows by indices, kept tape-aware via the public
    # ``embedding`` op key so a higher-order backward routes back through scatter again.
    return (
        Tensor(
            out_grad.data[indices],
            _prev=(out_grad,),
            _op="embedding",
            _fwd_ctx={"indices": indices, "vocab_size": out_grad.data.shape[0]},
        ),
    )
