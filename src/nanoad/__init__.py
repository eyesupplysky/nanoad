"""nanoad — from-scratch reverse-mode autograd and small neural-net framework."""

from nanoad.ops.activations import relu, tanh
from nanoad.ops.arithmetic import add, div, mul, power, sub
from nanoad.ops.bmm import bmm
from nanoad.ops.embedding import embedding
from nanoad.ops.linalg import matmul, reshape, transpose
from nanoad.ops.reductions import mean, sum
from nanoad.ops.softmax import cross_entropy, softmax
from nanoad.tensor import Tensor

from nanoad import functional

__all__ = [
    "Tensor",
    "add",
    "bmm",
    "cross_entropy",
    "div",
    "embedding",
    "functional",
    "matmul",
    "mean",
    "mul",
    "power",
    "relu",
    "reshape",
    "softmax",
    "sub",
    "sum",
    "tanh",
    "transpose",
]

__version__ = "0.0.1"
