"""Module.training flag and train()/eval() recursive toggle."""

from __future__ import annotations

from nanoad.nn import Linear, Module, ReLU, Sequential


def test_default_training_mode_is_true() -> None:
    layer = Linear(3, 2)
    assert layer.training is True


def test_eval_sets_training_false() -> None:
    layer = Linear(3, 2)
    layer.eval()
    assert layer.training is False


def test_train_after_eval_sets_training_true() -> None:
    layer = Linear(3, 2)
    layer.eval()
    layer.train()
    assert layer.training is True


def test_train_explicit_false_equivalent_to_eval() -> None:
    layer = Linear(3, 2)
    layer.train(False)
    assert layer.training is False


def test_train_recurses_into_sequential_layers() -> None:
    inner = Linear(2, 2)
    model = Sequential(inner, ReLU())
    model.eval()
    assert model.training is False
    assert inner.training is False


class _Nested(Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = Linear(3, 4)
        self.tail = Sequential(Linear(4, 4), ReLU(), Linear(4, 2))


def test_train_recurses_through_nested_attributes_and_lists() -> None:
    m = _Nested()
    m.eval()
    assert m.head.training is False
    for layer in m.tail.layers:
        assert layer.training is False
