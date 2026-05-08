"""SGD tests: convergence on a synthetic regression problem."""

import numpy as np

from nanoad import Tensor
from nanoad.nn import Linear
from nanoad.optim import SGD


def test_sgd_step_decreases_loss_on_linear_regression():
    """Train a single Linear layer to fit y = 2x + 3."""
    np.random.seed(0)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 1))
    y_true = 2.0 * x + 3.0

    layer = Linear(1, 1)
    opt = SGD(layer.parameters(), lr=0.1)

    initial_loss = float(((layer(Tensor(x)).data - y_true) ** 2).mean())

    for _ in range(200):
        pred = layer(Tensor(x))
        loss = ((pred - y_true) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    final_loss = float(((layer(Tensor(x)).data - y_true) ** 2).mean())
    assert final_loss < 0.01 * initial_loss
    assert abs(float(layer.weight.data[0, 0]) - 2.0) < 0.05
    assert abs(float(layer.bias.data[0]) - 3.0) < 0.05


def test_sgd_zero_grad_clears_all_params():
    np.random.seed(0)
    layer = Linear(2, 3)
    opt = SGD(layer.parameters(), lr=0.1)
    for p in layer.parameters():
        p.grad = p.grad + 1.0
    opt.zero_grad()
    for p in layer.parameters():
        assert (p.grad == 0).all()


def test_sgd_step_uses_lr_correctly():
    """Manual one-step check: with known grad and lr, the new param is grad-lr*update."""
    np.random.seed(0)
    layer = Linear(2, 1)
    layer.weight.data = np.array([[0.5], [0.5]])
    layer.bias.data = np.array([0.0])
    layer.weight.grad = np.array([[1.0], [2.0]])
    layer.bias.grad = np.array([3.0])

    opt = SGD(layer.parameters(), lr=0.1)
    opt.step()

    assert np.allclose(layer.weight.data, [[0.4], [0.3]])
    assert np.allclose(layer.bias.data, [-0.3])
