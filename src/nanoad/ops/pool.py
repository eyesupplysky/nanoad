"""2-D max pooling. Forward caches argmax; backward scatters grad to those positions."""

from __future__ import annotations

import numpy as np

from nanoad.tensor import Tensor


def max_pool2d(x: Tensor, kernel: int, stride: int | None = None) -> Tensor:
    """Non-overlapping by default (stride=kernel). No padding. Output (N, C, H_out, W_out)."""
    if x.data.ndim != 4:
        raise ValueError(f"max_pool2d expects 4-d x (N,C,H,W); got shape {x.shape}")
    s = kernel if stride is None else stride
    n, c, h, w = x.data.shape
    h_out = (h - kernel) // s + 1
    w_out = (w - kernel) // s + 1
    if h_out <= 0 or w_out <= 0:
        raise ValueError(
            f"max_pool2d output shape would be non-positive: h_out={h_out}, w_out={w_out}"
        )

    # Build (kernel*kernel, h_out*w_out) gather indices, then index into x.
    i0 = np.repeat(np.arange(kernel), kernel)
    i1 = s * np.repeat(np.arange(h_out), w_out)
    j0 = np.tile(np.arange(kernel), kernel)
    j1 = s * np.tile(np.arange(w_out), h_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)  # (kH*kW, L)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)  # (kH*kW, L)

    patches = x.data[:, :, i, j]  # (N, C, kH*kW, L)
    argmax = patches.argmax(axis=2)  # (N, C, L)
    out_data = patches.max(axis=2).reshape(n, c, h_out, w_out)
    out = Tensor(out_data, _prev=(x,), _op="max_pool2d")

    def _backward() -> None:
        # Scatter out.grad into x.grad at the argmax positions.
        g_flat = out.grad.reshape(n, c, h_out * w_out)  # (N, C, L)
        i_sel = i[argmax, np.arange(h_out * w_out)]  # (N, C, L)
        j_sel = j[argmax, np.arange(h_out * w_out)]  # (N, C, L)
        n_idx = np.arange(n).reshape(n, 1, 1)
        c_idx = np.arange(c).reshape(1, c, 1)
        np.add.at(x.grad, (n_idx, c_idx, i_sel, j_sel), g_flat)

    out._backward = _backward
    return out
