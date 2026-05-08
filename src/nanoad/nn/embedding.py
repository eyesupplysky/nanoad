"""Embedding: a learnable lookup table from integer indices to vectors."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from nanoad.nn.module import Module
from nanoad.ops.embedding import embedding
from nanoad.tensor import Tensor


class Embedding(Module):
    """Map integer indices to learnable embedding vectors.

    Forward accepts an integer-valued ArrayLike of arbitrary shape ``S`` and returns a
    Tensor of shape ``S + (embedding_dim,)``. Standard transformer init: ``N(0, 1)``.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        self.weight = Tensor(np.random.randn(num_embeddings, embedding_dim))

    def forward(self, indices: ArrayLike) -> Tensor:
        return embedding(self.weight, indices)
