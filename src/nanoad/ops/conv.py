"""2-D convolution: forward via im2col + matmul, backward via col2im scatter-add.

The forward stashes the im2col gather indices and the columnized input in ``_fwd_ctx``;
the VJP composes Tensor-valued contributions for ``x``, ``weight``, and (optionally)
``bias``. The col2im scatter still uses ``np.add.at`` for performance — it's wrapped as a
leaf Tensor for tape-aware accumulation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanoad._engine import register_vjp
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
    patches when used as ``x_pad[:, k, i, j]``.
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
    fwd_ctx: dict[str, object] = {
        "k": k,
        "i": i,
        "j": j,
        "h_out": h_out,
        "w_out": w_out,
        "n": n,
        "c_in": c_in,
        "c_out": c_out,
        "h": h,
        "w": w,
        "kh": kh,
        "kw": kw,
        "stride": stride,
        "padding": padding,
        "cols": cols,
        "x_pad_shape": x_pad.shape,
        "has_bias": bias is not None,
    }
    return Tensor(out_data, _prev=parents, _op="conv2d", _fwd_ctx=fwd_ctx)


@register_vjp("conv2d")
def _vjp_conv2d(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    assert fwd_ctx is not None
    has_bias = fwd_ctx["has_bias"]
    if has_bias:
        x, weight, _bias = parents
    else:
        x, weight = parents

    n = fwd_ctx["n"]
    c_out = fwd_ctx["c_out"]
    h_out = fwd_ctx["h_out"]
    w_out = fwd_ctx["w_out"]
    h = fwd_ctx["h"]
    w = fwd_ctx["w"]
    padding = fwd_ctx["padding"]
    cols: NDArray = fwd_ctx["cols"]
    k = fwd_ctx["k"]
    i = fwd_ctx["i"]
    j = fwd_ctx["j"]
    x_pad_shape = fwd_ctx["x_pad_shape"]

    g_2d = out_grad.data.reshape(n, c_out, h_out * w_out)  # (N, C_out, L)

    # dW = sum_n grad_out[n] @ cols[n].T → (C_out, C_in*kH*kW), reshape to weight shape.
    dw_2d = np.einsum("nol,nkl->ok", g_2d, cols)
    dw_data = dw_2d.reshape(weight.data.shape)

    # dx via col2im scatter-add of (w_2d.T @ grad_out) into the padded input grid.
    w_2d = weight.data.reshape(c_out, -1)
    dcols = np.einsum("ok,nol->nkl", w_2d, g_2d)
    dx_pad = np.zeros(x_pad_shape, dtype=np.float64)
    np.add.at(dx_pad, (slice(None), k, i, j), dcols)
    if padding > 0:
        dx_data = dx_pad[:, :, padding : padding + h, padding : padding + w]
    else:
        dx_data = dx_pad

    # Wrap as leaf Tensors and combine with out_grad via public ops to keep the tape live.
    # The col2im scatter is itself a linear function of out_grad, so multiplying by 1 (via
    # broadcast through the constructor) is correct first-order; the dependency on out_grad
    # is captured by the fact that dx_data and dw_data were computed *from* out_grad.data.
    # To keep the gradient graph faithful for HOA we *also* attach the upstream out_grad
    # as a parent through a multiply by 1 — but for fused linear ops the simpler shape is
    # to wrap the linear results as leaves; second-order gradients through conv are out
    # of scope for M7.
    grads: list[Tensor] = [Tensor(dx_data), Tensor(dw_data)]
    if has_bias:
        # db = grad_out summed over (N, H_out, W_out)
        db_data = out_grad.data.sum(axis=(0, 2, 3))
        grads.append(Tensor(db_data))
    return tuple(grads)
