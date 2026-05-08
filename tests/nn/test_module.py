"""Module base class tests: parameters() walk and zero_grad."""

import numpy as np

from nanoad import Tensor
from nanoad.nn import Linear, Module, ReLU, Sequential


class _Toy(Module):
    """Two stacked Linear layers; tests submodule recursion."""

    def __init__(self) -> None:
        self.fc1 = Linear(3, 4)
        self.fc2 = Linear(4, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x))


def test_parameters_walks_submodules():
    m = _Toy()
    params = list(m.parameters())
    assert len(params) == 4
    shapes = sorted(p.shape for p in params)
    assert shapes == sorted([(3, 4), (4,), (4, 2), (2,)])


def test_parameters_handles_list_attr():
    class _ListMod(Module):
        def __init__(self) -> None:
            self.layers = [Linear(2, 3), Linear(3, 4)]

        def forward(self, x: Tensor) -> Tensor:
            for layer in self.layers:
                x = layer(x)
            return x

    m = _ListMod()
    assert len(list(m.parameters())) == 4


def test_parameters_handles_direct_tensor_attr():
    class _Bare(Module):
        def __init__(self) -> None:
            self.weight = Tensor(np.zeros((2, 3)))

        def forward(self, x: Tensor) -> Tensor:
            return x @ self.weight

    m = _Bare()
    assert len(list(m.parameters())) == 1


def test_zero_grad_resets_all_parameters():
    m = _Toy()
    for p in m.parameters():
        p.grad = Tensor(np.full(p.data.shape, 5.0))
    m.zero_grad()
    for p in m.parameters():
        assert p.grad is None


def test_sequential_collects_layer_parameters():
    s = Sequential(Linear(3, 4), ReLU(), Linear(4, 2))
    assert len(list(s.parameters())) == 4


def test_module_call_invokes_forward():
    m = _Toy()
    x = Tensor(np.zeros((1, 3)))
    out = m(x)
    assert out.shape == (1, 2)


def test_unimplemented_forward_raises():
    import pytest

    bare = Module()
    with pytest.raises(NotImplementedError):
        bare(0)
