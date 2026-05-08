"""2-D convolution: forward via im2col + matmul, backward via col2im scatter-add."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanoad.tensor import Tensor


def _im2col_indices(
    c_in: int,
    h: int,
    w: int,
    kh: int,
    kw: int,
    stride: int,
    padding: int,
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp], int, int]:
    """Build (k, i, j) gather indices and output (h_out, w_out) for an im2col window.

    The returned index arrays have shape (c_in*kh*kw, h_out*w_out) and gather windowed
    patches when used as `x_pad[:, k, i, j]`.
    """
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1
    if h_out <= 0 or w_out <= 0:
        raise ValueError(
            f"conv2d output shape would be non-positive: h_out={h_out}, w_out={w_out}"
        )

    i0 = np.tile(np.repeat(np.arange(kh), kw), c_in)
    i1 = stride * np.repeat(np.arange(h_out), w_out)
    j0 = np.tile(np.tile(np.arange(kw), kh), c_in)
    j1 = stride * np.tile(np.arange(w_out), h_out)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(c_in), kh * kw).reshape(-1, 1)
    return k, i, j, h_out, w_out


def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> Tensor:
    """2-D cross-correlation. Layout NCHW. Output shape (N, C_out, H_out, W_out)."""
    if x.data.ndim != 4:
        raise ValueError(f"conv2d expects 4-d x (N,C,H,W); got shape {x.shape}")
    if weight.data.ndim != 4:
        raise ValueError(f"conv2d expects 4-d weight (Co,Ci,kH,kW); got shape {weight.shape}")
    n, c_in, h, w = x.data.shape
    c_out, c_in_w, kh, kw = weight.data.shape
    if c_in != c_in_w:
        raise ValueError(
            f"conv2d in_channels mismatch: x has {c_in}, weight has {c_in_w}"
        )

    k, i, j, h_out, w_out = _im2col_indices(c_in, h, w, kh, kw, stride, padding)

    if padding > 0:
        x_pad = np.pad(x.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    else:
        x_pad = x.data
    cols = x_pad[:, k, i, j]  # (N, C_in*kH*kW, H_out*W_out)
    w_2d = weight.data.reshape(c_out, -1)  # (C_out, C_in*kH*kW)
    out_2d = np.einsum("ok,nkl->nol", w_2d, cols)  # (N, C_out, L)
    out_data = out_2d.reshape(n, c_out, h_out, w_out)
    if bias is not None:
        if bias.data.ndim != 1 or bias.data.shape[0] != c_out:
            raise ValueError(f"conv2d bias must be shape ({c_out},); got {bias.shape}")
        out_data = out_data + bias.data.reshape(1, c_out, 1, 1)

    parents: tuple[Tensor, ...] = (x, weight) if bias is None else (x, weight, bias)
    out = Tensor(out_data, _prev=parents, _op="conv2d")

    def _backward() -> None:
        # grad_out shape: (N, C_out, H_out, W_out)
        g_2d = out.grad.reshape(n, c_out, h_out * w_out)  # (N, C_out, L)

        # dW = sum_n grad_out[n] @ cols[n].T
        dw_2d = np.einsum("nol,nkl->ok", g_2d, cols)  # (C_out, C_in*kH*kW)
        weight.grad += dw_2d.reshape(weight.data.shape)

        if bias is not None:
            bias.grad += out.grad.sum(axis=(0, 2, 3))

        # dcols = w.T @ grad_out
        dcols = np.einsum("ok,nol->nkl", w_2d, g_2d)  # (N, C_in*kH*kW, L)

        dx_pad = np.zeros_like(x_pad)
        np.add.at(dx_pad, (slice(None), k, i, j), dcols)
        if padding > 0:
            x.grad += dx_pad[:, :, padding : padding + h, padding : padding + w]
        else:
            x.grad += dx_pad

    out._backward = _backward
    return out
