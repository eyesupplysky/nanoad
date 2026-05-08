"""nanoad — from-scratch reverse-mode autograd and small neural-net framework."""

from nanoad.ops.activations import relu, tanh
from nanoad.ops.arithmetic import add, div, mul, power, sub
from nanoad.ops.linalg import matmul, reshape, transpose
from nanoad.ops.reductions import mean, sum
from nanoad.tensor import Tensor

__all__ = [
    "Tensor",
    "add",
    "div",
    "matmul",
    "mean",
    "mul",
    "power",
    "relu",
    "reshape",
    "sub",
    "sum",
    "tanh",
    "transpose",
]

__version__ = "0.0.1"
