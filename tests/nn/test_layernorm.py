"""LayerNorm: forward shape, normalization correctness, and finite-difference grad check."""

from __future__ import annotations

import numpy as np
import pytest

from nanoad import Tensor
from nanoad.nn import LayerNorm


def test_layernorm_forward_shape() -> None:
    ln = LayerNorm(8)
    x = Tensor(np.random.randn(2, 5, 8))
    out = ln(x)
    assert out.shape == (2, 5, 8)


def test_layernorm_zero_mean_unit_var_per_token() -> None:
    """With gamma=1, beta=0, output per token has ~0 mean and ~1 var across the last axis."""
    ln = LayerNorm(16)
    x = Tensor(np.random.randn(4, 7, 16) * 3.0 + 1.0)
    y = ln(x)
    per_token_mean = y.data.mean(axis=-1)
    per_token_var = y.data.var(axis=-1)
    np.testing.assert_allclose(per_token_mean, 0.0, atol=1e-6)
    np.testing.assert_allclose(per_token_var, 1.0, atol=1e-3)


def test_layernorm_affine_applied() -> None:
    """Constant gamma scales variance by gamma**2; constant beta shifts the mean."""
    ln = LayerNorm(8)
    ln.gamma = Tensor(np.full(8, 2.0))
    ln.beta = Tensor(np.full(8, -3.0))
    x = Tensor(np.random.randn(5, 8))
    y = ln(x)
    np.testing.assert_allclose(y.data.mean(axis=-1), np.full(5, -3.0), atol=1e-6)
    np.testing.assert_allclose(y.data.var(axis=-1), np.full(5, 4.0), atol=1e-3)


def test_layernorm_rejects_wrong_last_dim() -> None:
    ln = LayerNorm(8)
    x = Tensor(np.random.randn(2, 5, 4))
    with pytest.raises(ValueError, match="last-axis size 8"):
        ln(x)


def test_layernorm_grad_check(grad_check) -> None:
    """Gradients of (loss = ln(x).sum()) w.r.t. x, gamma, beta match finite differences."""

    def fn(x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        local = LayerNorm(4)
        local.gamma = gamma
        local.beta = beta
        return local(x)

    grad_check(
        fn,
        np.random.randn(3, 4),
        np.array([1.5, 0.5, 1.0, 0.8]),
        np.array([0.1, -0.2, 0.05, 0.0]),
    )


def test_layernorm_parameters_lists_gamma_and_beta() -> None:
    ln = LayerNorm(5)
    params = list(ln.parameters())
    assert len(params) == 2
    shapes = sorted(tuple(p.shape) for p in params)
    assert shapes == [(5,), (5,)]
