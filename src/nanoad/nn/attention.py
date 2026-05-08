"""MultiHeadAttention: scaled dot-product attention with optional causal mask.

Q/K/V/output are ``(D, D)`` linear projections over the embedding axis. Heads are split
via reshape + transpose so the per-head dot-product runs as batched matmul over
``(B, H, T, Dh)``. The causal mask, when enabled, adds a constant ``-1e9`` Tensor to the
upper triangle of the score matrix; softmax then drives those positions to ~0 with finite
gradients.
"""

from __future__ import annotations

import numpy as np

from nanoad.nn.module import Module
from nanoad.ops.bmm import bmm
from nanoad.ops.softmax import softmax
from nanoad.tensor import Tensor


class MultiHeadAttention(Module):
    """Scaled dot-product multi-head self-attention with optional causal masking."""

    def __init__(self, embed_dim: int, num_heads: int, causal: bool = False) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.head_dim: int = embed_dim // num_heads
        self.causal: bool = causal
        std = float(np.sqrt(2.0 / embed_dim))
        self.q_weight = Tensor(np.random.randn(embed_dim, embed_dim) * std)
        self.k_weight = Tensor(np.random.randn(embed_dim, embed_dim) * std)
        self.v_weight = Tensor(np.random.randn(embed_dim, embed_dim) * std)
        self.out_weight = Tensor(np.random.randn(embed_dim, embed_dim) * std)

    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim != 3:
            raise ValueError(f"MultiHeadAttention expects 3-d input (B, T, D); got {x.shape}")
        b, t, d = x.shape
        if d != self.embed_dim:
            raise ValueError(
                f"MultiHeadAttention expected embed_dim={self.embed_dim}; got {d}"
            )
        h, dh = self.num_heads, self.head_dim

        q = bmm(x, self.q_weight).reshape(b, t, h, dh).transpose(0, 2, 1, 3)
        k = bmm(x, self.k_weight).reshape(b, t, h, dh).transpose(0, 2, 1, 3)
        v = bmm(x, self.v_weight).reshape(b, t, h, dh).transpose(0, 2, 1, 3)

        scale = 1.0 / float(np.sqrt(dh))
        scores = bmm(q, k.transpose(0, 1, 3, 2)) * scale  # (B, H, T, T)

        if self.causal:
            mask = np.triu(np.full((t, t), -1e9), k=1)
            scores = scores + Tensor(mask)

        attn = softmax(scores, axis=-1)
        out_per_head = bmm(attn, v)  # (B, H, T, Dh)
        merged = out_per_head.transpose(0, 2, 1, 3).reshape(b, t, d)
        return bmm(merged, self.out_weight)
