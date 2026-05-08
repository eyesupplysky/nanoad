# How to add a new op

Every primitive op in nanoad follows the same shape: a forward function that builds a new `Tensor`, plus a `_backward` closure that accumulates parent gradients. The pattern is small and uniform — adding an op is mostly typing the local derivative.

## The op-pair recipe

Take `relu` (in `src/nanoad/ops/activations.py`):

```python
def relu(x: Tensor) -> Tensor:
    """Rectified linear unit, elementwise. Sub-gradient at 0 is taken to be 0."""
    out_data = np.maximum(x.data, 0.0)
    out = Tensor(out_data, _prev=(x,), _op="relu")

    def _backward() -> None:
        x.grad += (x.data > 0).astype(np.float64) * out.grad

    out._backward = _backward
    return out
```

That's the whole pattern. Five things to get right:

1. **Forward computes `out_data`** with NumPy. No `Tensor` arithmetic — that would re-enter the autograd graph and create spurious nodes.
2. **Construct `out` with `_prev` set to every parent tensor** the gradient depends on. Forgetting a parent silently breaks gradient flow into it.
3. **Inside `_backward`, accumulate with `+=`** into each parent's `.grad`. Never overwrite.
4. **Use `out.grad`, not a captured argument**, as the upstream gradient. The closure runs after `backward()` has populated `out.grad`.
5. **Assign `out._backward = _backward`** before returning. The default `_backward` is a no-op for leaves; opting in is your job.

## Implementing the local derivative

The math you need is the *vector–Jacobian product* (VJP): given upstream `dL/dy = out.grad`, compute `dL/dx = (dy/dx) · dL/dy` for each parent.

For elementwise ops the Jacobian is diagonal — multiply pointwise. For `relu`: `dy/dx = 1 where x > 0, else 0`, so `dL/dx = (x > 0) * out.grad`.

For ops with shape changes (matmul, conv2d) the VJP is a matmul or convolution itself. See `src/nanoad/ops/linalg.py` for `matmul`'s backward (a transposed matmul each side) and `src/nanoad/ops/conv.py` for `conv2d`'s im2col / col2im pair.

For elementwise ops with broadcasting, route each parent's grad through `unbroadcast` so shape differences from forward broadcasting are reduced away. See [broadcasting.md](broadcasting.md).

## Test the gradient

`tests/conftest.py` provides a `gradcheck` fixture that compares analytic gradients against finite differences. Add a test in `tests/ops/`:

```python
def test_relu_grad(gradcheck):
    x = Tensor(np.array([-1.0, 0.5, 2.0]))
    gradcheck(lambda t: relu(t).sum(), x)
```

If the analytic backward matches central finite differences within ~1e-5, your op is wired correctly. Failing this check is by far the most common way an op bug shows up — gradients are silently wrong otherwise, since shape errors are caught but value errors aren't.

## Wire into the package

If the op should be importable from the top-level `nanoad` namespace:

1. Add it to `src/nanoad/ops/<group>.py` — pick the file by family (`activations.py`, `arithmetic.py`, etc.).
2. Re-export from `src/nanoad/__init__.py` and add the name to `__all__`.

If it's a layer (composition of ops with parameters), wrap it as a `Module` subclass under `src/nanoad/nn/`. Read [tape.md](tape.md) and the existing layers — `Linear` is the smallest example.

## Common pitfalls

- **Overwriting instead of accumulating.** A parent used by two ops will silently lose one contribution if either closure does `parent.grad = ...` instead of `+=`.
- **Forgetting `unbroadcast` on an elementwise op.** Will work if all parents have the same shape as the output; will silently miscompute the moment broadcasting kicks in.
- **Capturing the wrong variable in the closure.** Python closures bind by name. If you capture `grad` from an outer loop variable instead of `out.grad`, you'll get the wrong gradient or a stale one.
- **Using `Tensor` arithmetic in the forward.** `out = Tensor(a.data + b.data, ...)` is the right shape; `out = a + b` re-enters the tape and creates extra nodes.
- **Mutating `.data` in place.** Several backward closures read parent `.data` after the forward has returned (`mul`, `div`, `power`). In-place mutation between forward and backward will corrupt those gradients without raising.
