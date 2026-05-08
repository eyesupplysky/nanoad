"""Embedding nn module: forward shape, parameter discovery, gradient correctness."""

from __future__ import annotations

import numpy as np

from nanoad import Tensor
from nanoad.nn import Embedding


def test_embedding_module_forward_shape() -> None:
    emb = Embedding(num_embeddings=10, embedding_dim=4)
    indices = np.array([[3, 1, 7], [0, 2, 9]])
    out = emb(indices)
    assert out.shape == (2, 3, 4)


def test_embedding_module_weight_is_a_parameter() -> None:
    emb = Embedding(num_embeddings=5, embedding_dim=3)
    params = list(emb.parameters())
    assert len(params) == 1
    assert params[0].shape == (5, 3)


def test_embedding_module_grad_check(grad_check) -> None:
    """grad of (loss = emb(indices).sum()) w.r.t. the weight matches finite differences."""
    indices = np.array([[1, 0], [2, 1]])

    def fn(weight: Tensor) -> Tensor:
        local = Embedding(5, 3)
        local.weight = weight
        return local(indices)

    grad_check(fn, np.random.randn(5, 3))
