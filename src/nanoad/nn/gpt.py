"""GPT: a decoder-only transformer composed from existing nanoad primitives.

Architecture: token + learned positional embeddings, ``num_layers`` transformer blocks
with pre-LayerNorm causal self-attention and a feed-forward (4x) inner block, a final
LayerNorm, and an unembedding projection to vocabulary logits. No dropout — this is a
teaching reference, not a regularization study.

Loss is left to the caller: reshape ``logits`` to ``(B*T, V)`` and pair with
``targets.reshape(B*T)`` through ``nanoad.cross_entropy``. Generation lives in
``nanoad.nn.gpt.generate`` (separate slice).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nanoad.nn.activations import GELU
from nanoad.nn.attention import MultiHeadAttention
from nanoad.nn.embedding import Embedding
from nanoad.nn.layernorm import LayerNorm
from nanoad.nn.linear import Linear
from nanoad.nn.module import Module
from nanoad.tensor import Tensor


class _MLP(Module):
    """Position-wise feed-forward block: Linear → GELU → Linear with 4x inner width."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.fc1 = Linear(embed_dim, 4 * embed_dim)
        self.act = GELU()
        self.fc2 = Linear(4 * embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _TransformerBlock(Module):
    """Pre-LN transformer block: x + attn(LN(x)); x + MLP(LN(x))."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, causal=True)
        self.ln2 = LayerNorm(embed_dim)
        self.mlp = _MLP(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(Module):
    """Decoder-only transformer with learned positional embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        block_size: int,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.vocab_size: int = vocab_size
        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.num_layers: int = num_layers
        self.block_size: int = block_size
        self.tok_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = Embedding(block_size, embed_dim)
        self.blocks: list[_TransformerBlock] = [
            _TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size)

    def forward(self, idx: ArrayLike) -> Tensor:
        idx_arr = np.asarray(idx, dtype=np.int64)
        if idx_arr.ndim != 2:
            raise ValueError(
                f"GPT expects 2-d (B, T) integer indices; got shape {idx_arr.shape}"
            )
        _, t = idx_arr.shape
        if t > self.block_size:
            raise ValueError(
                f"sequence length {t} exceeds block_size {self.block_size}"
            )
        tok = self.tok_emb(idx_arr)
        pos = self.pos_emb(np.arange(t, dtype=np.int64))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def generate(
        self,
        idx: ArrayLike,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> NDArray[np.int64]:
        """Autoregressive sampling. Returns ``(B, T_init + max_new_tokens)`` int64 ids."""
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive; got {temperature}")
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative; got {max_new_tokens}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive when set; got {top_k}")
        out = np.asarray(idx, dtype=np.int64)
        if out.ndim != 2:
            raise ValueError(f"GPT.generate expects 2-d (B, T) indices; got shape {out.shape}")
        for _ in range(max_new_tokens):
            cond = out[:, -self.block_size :]
            logits = self(cond).data[:, -1, :] / float(temperature)
            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                threshold = np.sort(logits, axis=-1)[:, -k][:, None]
                logits = np.where(logits < threshold, -np.inf, logits)
            probs = _softmax_rows(logits)
            next_ids = np.array(
                [np.random.choice(probs.shape[-1], p=row) for row in probs],
                dtype=np.int64,
            )
            out = np.concatenate([out, next_ids[:, None]], axis=1)
        return out


def _softmax_rows(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable row-wise softmax over the last axis."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
