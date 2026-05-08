"""Shared test fixtures: rng seeding and finite-difference gradient check."""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import pytest

from nanoad import Tensor


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Reseed Python's random module and numpy before every test for reproducibility."""
    random.seed(0xDEADBEEF)
    np.random.seed(0xDEADBEEF)


@pytest.fixture
def grad_check() -> Callable[..., None]:
    """Central-difference gradient check supporting n-d tensor inputs.

    Computes analytic gradients via .backward() (after summing to scalar if needed),
    then perturbs every element of every input and compares against finite differences.
    """

    def assert_grad_close(
        fn: Callable[..., Tensor],
        *inputs: object,
        eps: float = 1e-5,
        tol: float = 1e-4,
    ) -> None:
        base = [np.array(x, dtype=np.float64) for x in inputs]

        tensors = [Tensor(arr) for arr in base]
        out = fn(*tensors)
        if not isinstance(out, Tensor):
            raise TypeError("fn must return a Tensor")
        scalar = out if out.data.ndim == 0 else out.sum()
        scalar.backward()
        analytics = [t.grad.copy() for t in tensors]

        for i, x_arr in enumerate(base):
            numeric = np.zeros_like(x_arr)
            for idx in np.ndindex(x_arr.shape):
                plus = [arr.copy() for arr in base]
                minus = [arr.copy() for arr in base]
                plus[i][idx] += eps
                minus[i][idx] -= eps

                f_plus = fn(*[Tensor(v) for v in plus])
                f_minus = fn(*[Tensor(v) for v in minus])
                f_plus_val = float(f_plus.data.sum())
                f_minus_val = float(f_minus.data.sum())
                numeric[idx] = (f_plus_val - f_minus_val) / (2.0 * eps)

            diff = np.abs(numeric - analytics[i])
            scale = np.maximum(np.maximum(np.abs(numeric), np.abs(analytics[i])), 1.0)
            rel_max = float((diff / scale).max())
            assert rel_max < tol, (
                f"grad mismatch at input {i}: max rel diff {rel_max}\n"
                f"analytic={analytics[i]}\nnumeric={numeric}"
            )

    return assert_grad_close
