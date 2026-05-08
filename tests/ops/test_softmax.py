"""Forward and backward tests for softmax and cross_entropy."""

import numpy as np
import pytest

from nanoad import Tensor
from nanoad.ops.softmax import cross_entropy, softmax


def test_softmax_forward_sums_to_one():
    x = Tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    s = softmax(x, axis=-1)
    assert np.allclose(s.data.sum(axis=-1), 1.0)


def test_softmax_forward_uniform_input():
    x = Tensor([[0.0, 0.0, 0.0, 0.0]])
    s = softmax(x, axis=-1)
    assert np.allclose(s.data, 0.25)


def test_softmax_grads(grad_check):
    grad_check(lambda x: softmax(x, axis=-1), [[1.0, 0.5, -0.3], [0.2, 0.7, -0.1]])


def test_softmax_axis_zero(grad_check):
    grad_check(lambda x: softmax(x, axis=0), [[1.0, 2.0], [3.0, 4.0]])


def test_softmax_stability_with_large_inputs():
    """Should not overflow even for very large values."""
    x = Tensor([[1000.0, 1001.0, 1002.0]])
    s = softmax(x, axis=-1)
    assert np.all(np.isfinite(s.data))
    assert np.allclose(s.data.sum(), 1.0)


def test_cross_entropy_forward_perfect_prediction():
    """Logits with overwhelming mass on the target give near-zero loss."""
    logits = Tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
    loss = cross_entropy(logits, [0, 1])
    assert float(loss.data) < 1e-10


def test_cross_entropy_forward_uniform_logits():
    """Uniform logits over k classes give loss = log(k)."""
    n_classes = 5
    logits = Tensor(np.zeros((3, n_classes)))
    loss = cross_entropy(logits, [0, 1, 2])
    assert np.isclose(float(loss.data), np.log(n_classes))


def test_cross_entropy_grads_match_finite_diffs():
    """Analytic backward agrees with central differences for cross_entropy."""
    rng = np.random.default_rng(0)
    logits_data = rng.standard_normal((3, 4))
    targets = np.array([0, 2, 1])

    logits = Tensor(logits_data)
    loss = cross_entropy(logits, targets)
    loss.backward()
    analytic = logits.grad.copy()

    eps = 1e-5
    numeric = np.zeros_like(logits_data)
    for i in range(logits_data.shape[0]):
        for j in range(logits_data.shape[1]):
            plus = logits_data.copy()
            plus[i, j] += eps
            minus = logits_data.copy()
            minus[i, j] -= eps
            l_plus = float(cross_entropy(Tensor(plus), targets).data)
            l_minus = float(cross_entropy(Tensor(minus), targets).data)
            numeric[i, j] = (l_plus - l_minus) / (2 * eps)

    assert np.allclose(analytic, numeric, atol=1e-5)


def test_cross_entropy_rejects_non_2d_logits():
    logits = Tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="2-d logits"):
        cross_entropy(logits, [0])


def test_cross_entropy_rejects_target_size_mismatch():
    logits = Tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="batch size"):
        cross_entropy(logits, [0, 1, 0])
