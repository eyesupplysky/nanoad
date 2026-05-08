"""M8: ``hessian`` — full Hessian materialized via stacked HVPs over basis vectors."""

from __future__ import annotations

import numpy as np
import pytest

from nanoad import Tensor
from nanoad.functional import hessian


def test_hessian_of_quadratic_form_is_symmetrized_matrix() -> None:
    """For f(x) = xᵀAx, H = A + Aᵀ."""
    n = 4
    rng = np.random.default_rng(0)
    A_data = rng.standard_normal((n, n))
    x_data = rng.standard_normal((n, 1))
    A = Tensor(A_data)

    def f(x: Tensor) -> Tensor:
        return ((x.transpose() @ A) @ x).sum()

    H = hessian(f)(x_data)
    np.testing.assert_allclose(H.data, A_data + A_data.T, rtol=1e-10)


def test_hessian_of_sum_of_squares_is_two_identity() -> None:
    """For f(x) = Σ xᵢ², H = 2I."""
    n = 5
    rng = np.random.default_rng(1)
    x_data = rng.standard_normal(n)
    H = hessian(lambda x: (x * x).sum())(x_data)
    np.testing.assert_allclose(H.data, 2.0 * np.eye(n), rtol=1e-12)


def test_hessian_of_scalar_input_is_one_by_one() -> None:
    """For f(x) = x³, H at x=2 is 6x = 12. Shape is (1, 1)."""
    H = hessian(lambda x: x**3)(2.0)
    assert H.shape == (1, 1)
    np.testing.assert_allclose(H.data, 12.0)


def test_hessian_returns_tensor() -> None:
    """Hessian's return type is Tensor."""
    H = hessian(lambda x: (x * x).sum())(np.array([1.0, 2.0]))
    assert isinstance(H, Tensor)


def test_hessian_argnum_negative_resolves_from_end() -> None:
    """argnum=-1 picks the last argument."""
    rng = np.random.default_rng(2)
    A_data = rng.standard_normal((3, 3))
    x_data = rng.standard_normal((3,))
    y_data = rng.standard_normal((3,))
    A = Tensor(A_data)

    def f(x: Tensor, y: Tensor) -> Tensor:
        # f(x, y) = xᵀ A x + yᵀ y; H wrt y is 2I.
        xv = x.reshape(3, 1)
        return ((xv.transpose() @ A) @ xv).sum() + (y * y).sum()

    H = hessian(f, argnum=-1)(x_data, y_data)
    np.testing.assert_allclose(H.data, 2.0 * np.eye(3), rtol=1e-12)


def test_hessian_argnum_tuple_returns_per_arg_blocks() -> None:
    """tuple argnum returns a tuple of diagonal Hessian blocks (no cross-arg blocks)."""
    n_x, n_y = 3, 2
    rng = np.random.default_rng(3)
    A_data = rng.standard_normal((n_x, n_x))
    A = Tensor(A_data)

    def f(x: Tensor, y: Tensor) -> Tensor:
        xv = x.reshape(n_x, 1)
        return ((xv.transpose() @ A) @ xv).sum() + 3.0 * (y * y).sum()

    H_x, H_y = hessian(f, argnum=(0, 1))(np.zeros(n_x), np.zeros(n_y))
    np.testing.assert_allclose(H_x.data, A_data + A_data.T, rtol=1e-10)
    np.testing.assert_allclose(H_y.data, 6.0 * np.eye(n_y), rtol=1e-12)


def test_hessian_of_matrix_arg_flattens_input() -> None:
    """Matrix-shaped argument: H is (prod(shape), prod(shape)). For f(W) = sum(W²), H = 2I."""
    rows, cols = 2, 3
    n = rows * cols
    W_data = np.zeros((rows, cols))
    H = hessian(lambda W: (W * W).sum())(W_data)
    assert H.shape == (n, n)
    np.testing.assert_allclose(H.data, 2.0 * np.eye(n), rtol=1e-12)


def test_hessian_argnum_out_of_range_raises() -> None:
    """argnum=3 with two args is an IndexError."""
    with pytest.raises(IndexError, match="out of range"):
        hessian(lambda x, y: (x + y).sum(), argnum=3)(1.0, 1.0)
