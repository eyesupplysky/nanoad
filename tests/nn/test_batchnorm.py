"""BatchNorm2d: train-mode grad check, eval-mode determinism, running stats EMA."""

from __future__ import annotations

import numpy as np

from nanoad import Tensor
from nanoad.nn import BatchNorm2d


def test_batchnorm_forward_shape() -> None:
    bn = BatchNorm2d(3)
    x = Tensor(np.random.randn(2, 3, 4, 4))
    out = bn(x)
    assert out.shape == (2, 3, 4, 4)


def test_batchnorm_train_mode_zero_mean_unit_var() -> None:
    """With gamma=1, beta=0, output should have ~0 mean and ~1 var per channel."""
    bn = BatchNorm2d(3)
    x = Tensor(np.random.randn(8, 3, 5, 5) * 4.0 + 2.0)
    y = bn(x)
    per_channel_mean = y.data.mean(axis=(0, 2, 3))
    per_channel_var = y.data.var(axis=(0, 2, 3))
    np.testing.assert_allclose(per_channel_mean, 0.0, atol=1e-6)
    np.testing.assert_allclose(per_channel_var, 1.0, atol=1e-3)


def test_batchnorm_running_stats_ema_update() -> None:
    """After several forward passes, running_mean tracks an EMA of batch means."""
    bn = BatchNorm2d(2, momentum=0.5)
    starting_mean = bn.running_mean.copy()
    np.testing.assert_allclose(starting_mean, np.zeros(2))

    x = Tensor(np.ones((4, 2, 3, 3)) * 2.0)  # batch mean = 2.0 per channel
    bn(x)
    np.testing.assert_allclose(bn.running_mean, np.full(2, 1.0))  # 0.5*0 + 0.5*2

    bn(x)
    np.testing.assert_allclose(bn.running_mean, np.full(2, 1.5))  # 0.5*1 + 0.5*2


def test_batchnorm_eval_mode_uses_running_stats() -> None:
    """eval() makes the layer deterministic across different batch contents."""
    bn = BatchNorm2d(2, momentum=1.0)
    x_train = Tensor(np.random.randn(4, 2, 3, 3) + 5.0)
    bn(x_train)  # populates running stats from this batch
    bn.eval()
    x_eval_a = Tensor(np.ones((1, 2, 3, 3)))
    x_eval_b = Tensor(np.ones((1, 2, 3, 3)) * 3.0)
    out_a = bn(x_eval_a)
    out_b = bn(x_eval_b)
    # Running stats are fixed in eval mode, so the output is a deterministic
    # affine of the input: out_b - out_a equals (x_eval_b - x_eval_a) * scale.
    diff = out_b.data - out_a.data
    expected_scale = bn.gamma.data.reshape(1, 2, 1, 1) / np.sqrt(
        bn.running_var.reshape(1, 2, 1, 1) + bn.eps
    )
    expected = np.broadcast_to(2.0 * expected_scale, diff.shape)
    np.testing.assert_allclose(diff, expected)


def test_batchnorm_train_mode_grad_check(grad_check) -> None:
    """Gradients of (loss = bn(x).sum()) w.r.t. x, gamma, beta match finite differences."""
    bn = BatchNorm2d(2)
    np.random.seed(0)
    bn.gamma = Tensor(np.array([1.5, 0.5]))
    bn.beta = Tensor(np.array([0.1, -0.2]))

    def fn(x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        # Rebind into a fresh BN to avoid running-stats drift across grad_check perturbations.
        local = BatchNorm2d(2)
        local.gamma = gamma
        local.beta = beta
        return local(x)

    grad_check(
        fn,
        np.random.randn(3, 2, 3, 3),
        np.array([1.5, 0.5]),
        np.array([0.1, -0.2]),
    )


def test_batchnorm_train_eval_toggle_via_module_train() -> None:
    """Module.train(False) propagates to BN inside a Sequential."""
    from nanoad.nn import Sequential

    bn = BatchNorm2d(2)
    model = Sequential(bn)
    assert bn.training is True
    model.eval()
    assert bn.training is False
    model.train()
    assert bn.training is True


def test_batchnorm_parameters_excludes_running_stats() -> None:
    """gamma and beta show up as parameters; running_mean/var do not."""
    bn = BatchNorm2d(3)
    params = list(bn.parameters())
    assert len(params) == 2
    shapes = sorted(tuple(p.shape) for p in params)
    assert shapes == [(3,), (3,)]
