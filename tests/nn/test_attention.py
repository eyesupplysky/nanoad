"""MultiHeadAttention: forward shape, causal masking, and end-to-end grad check."""

from __future__ import annotations

import numpy as np
import pytest

from nanoad import Tensor
from nanoad.nn import MultiHeadAttention


def test_mha_forward_shape() -> None:
    mha = MultiHeadAttention(embed_dim=8, num_heads=2)
    x = Tensor(np.random.randn(3, 5, 8))
    out = mha(x)
    assert out.shape == (3, 5, 8)


def test_mha_rejects_indivisible_num_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(embed_dim=8, num_heads=3)


def test_mha_rejects_wrong_embed_dim() -> None:
    mha = MultiHeadAttention(embed_dim=8, num_heads=2)
    x = Tensor(np.random.randn(3, 5, 4))
    with pytest.raises(ValueError, match="embed_dim=8"):
        mha(x)


def test_mha_rejects_non_3d_input() -> None:
    mha = MultiHeadAttention(embed_dim=8, num_heads=2)
    x = Tensor(np.random.randn(3, 8))
    with pytest.raises(ValueError, match="3-d"):
        mha(x)


def test_mha_parameters_lists_four_weights() -> None:
    mha = MultiHeadAttention(embed_dim=8, num_heads=2)
    params = list(mha.parameters())
    assert len(params) == 4
    assert all(p.shape == (8, 8) for p in params)


def test_mha_causal_mask_blocks_future_positions() -> None:
    """Causal mode: output position t depends only on input positions [:t+1].

    Perturb future input positions and check earlier outputs are byte-identical.
    """
    np.random.seed(42)
    mha = MultiHeadAttention(embed_dim=8, num_heads=2, causal=True)
    x_a = np.random.randn(1, 4, 8)
    out_a = mha(Tensor(x_a))

    x_b = x_a.copy()
    x_b[0, 2:] += 1.0  # perturb positions 2 and 3 (the "future")
    out_b = mha(Tensor(x_b))

    # Positions 0 and 1 must be unaffected by changes at 2 and 3.
    np.testing.assert_allclose(out_a.data[0, 0], out_b.data[0, 0], atol=1e-10)
    np.testing.assert_allclose(out_a.data[0, 1], out_b.data[0, 1], atol=1e-10)
    # Position 2 should change (it sees its own perturbation).
    assert not np.allclose(out_a.data[0, 2], out_b.data[0, 2], atol=1e-6)


def test_mha_non_causal_mixes_all_positions() -> None:
    """Without the mask, output position 0 sees future positions."""
    np.random.seed(42)
    mha = MultiHeadAttention(embed_dim=8, num_heads=2, causal=False)
    x_a = np.random.randn(1, 4, 8)
    out_a = mha(Tensor(x_a))

    x_b = x_a.copy()
    x_b[0, 3] += 5.0  # perturb just the last position
    out_b = mha(Tensor(x_b))
    assert not np.allclose(out_a.data[0, 0], out_b.data[0, 0], atol=1e-6)


def test_mha_grad_check(grad_check) -> None:
    """End-to-end grad check on a small MHA instance with pinned weights."""
    np.random.seed(0)
    embed_dim, num_heads = 4, 2

    def fn(
        x: Tensor,
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        out_weight: Tensor,
    ) -> Tensor:
        local = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        local.q_weight = q_weight
        local.k_weight = k_weight
        local.v_weight = v_weight
        local.out_weight = out_weight
        return local(x)

    grad_check(
        fn,
        np.random.randn(2, 3, embed_dim) * 0.5,
        np.random.randn(embed_dim, embed_dim) * 0.5,
        np.random.randn(embed_dim, embed_dim) * 0.5,
        np.random.randn(embed_dim, embed_dim) * 0.5,
        np.random.randn(embed_dim, embed_dim) * 0.5,
    )
