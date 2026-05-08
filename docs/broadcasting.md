# Broadcasting in the backward pass

NumPy's broadcasting rules let `a + b` work even when `a.shape != b.shape`, by virtually expanding size-1 dimensions and prepending size-1 axes. Forward broadcasting is invisible to user code. Backward broadcasting is the part everyone gets wrong on the first try.

## The asymmetry

If `a.shape = (3, 1)` and `b.shape = (3, 4)`, then `a + b` produces output of shape `(3, 4)`. NumPy effectively pretended `a` had shape `(3, 4)` by tiling its single column four times.

When the backward pass arrives with `out.grad` of shape `(3, 4)`, you can't just write `a.grad += out.grad` — that's a shape mismatch, NumPy will either raise or (worse) broadcast assignment in surprising ways. The right answer is: `a` *was* tiled across axis 1, so `a.grad` should sum the upstream gradient along that axis to get back to shape `(3, 1)`.

The rule, in one sentence: **the backward pass is the transpose of the forward pass, and the transpose of "tile along an axis" is "sum along that axis".**

## `unbroadcast`, the chokepoint

Every elementwise op in nanoad routes its parent grads through one helper:

```python
# src/nanoad/ops/_broadcast.py
def unbroadcast(grad: NDArray, target_shape: tuple[int, ...]) -> NDArray:
    extra = grad.ndim - len(target_shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    axes_to_sum = tuple(
        i
        for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape, strict=True))
        if t_dim == 1 and g_dim != 1
    )
    if axes_to_sum:
        grad = grad.sum(axis=axes_to_sum, keepdims=True)
    return grad
```

Two reductions, in order:

1. **Strip extra leading axes.** NumPy broadcasting can prepend axes (e.g. shape `(4,)` broadcasts up against shape `(3, 4)`). The backward needs to *remove* those by summing — `grad.shape[:extra]` get reduced away.
2. **Collapse size-1 dims that were broadcast up.** For each remaining axis, if the target had size 1 but the grad has size > 1, that axis was virtually tiled forward. Sum it back down with `keepdims=True` so the result aligns with the target shape exactly.

After both passes, `grad.shape == target_shape` and we can `+=` it into the leaf's `.grad`.

## Worked example: bias addition

Bias addition is the most common broadcast in neural nets:

```python
W = Tensor(np.zeros((128, 10)))   # logits for batch 128, 10 classes
b = Tensor(np.zeros((10,)))       # per-class bias
y = W + b                          # output shape (128, 10)
```

Forward: `b` is virtually tiled across the batch axis to match `W`. Backward: `out.grad` is shape `(128, 10)`.

- `unbroadcast(out.grad, (128, 10))` → no change (target matches).
- `unbroadcast(out.grad, (10,))` → `extra = 1`, sum over axis 0 → shape `(10,)`. The bias gradient is the per-class sum across the batch, exactly as expected.

## Worked example: scale by a scalar

```python
x = Tensor(np.ones((3, 4)))
s = Tensor(2.0)            # 0-d scalar
y = x * s
```

Forward: `s` broadcasts up to shape `(3, 4)`. Backward gives `out.grad` of shape `(3, 4)` with value `s` (since `d(x*s)/dx = s`).

For `x.grad`: `unbroadcast(s.data * out.grad, (3, 4))` → no change.

For `s.grad`: `unbroadcast(x.data * out.grad, ())` → `extra = 2`, sum over axes (0, 1) → 0-d scalar. The scalar's gradient is the sum of `x * out.grad` over every element it touched.

## When `unbroadcast` does nothing

If shapes already match, both reduction steps are no-ops and `grad` is returned unchanged. This is the fast path and the common case in non-broadcast ops (e.g. `add(a, b)` where `a.shape == b.shape == out.shape`).

## Why a single chokepoint

Routing all elementwise backward grads through `unbroadcast` means broadcasting bugs surface in exactly one place. `add`, `sub`, `mul`, `div` all delegate the shape-fixing work, so adding a new elementwise op is just "compute the local derivative, hand it to `unbroadcast` per parent, accumulate."
