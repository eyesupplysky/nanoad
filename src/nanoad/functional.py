"""Functional autograd transforms: grad, vjp, jvp, hvp, hessian.

JAX-style functional API on top of nanoad's tape-aware backward. No new engine
machinery — every transform reduces to one or two calls to ``Tensor.backward()``
on cleverly composed scalars:

- ``vjp`` reduces to ``(out * cotangent).sum().backward()``.
- ``hvp`` adds a second backward through ``(grad * vector).sum()``.
- ``jvp`` flips ``vjp`` via the standard transposition trick — build
  ``phi(u) = (out * u).sum()``, double-backward yields ``J · v``.
- ``hessian`` stacks ``hvp`` with basis-vector tangents.

These transforms inherit M7's HOA limitations: ``hvp`` and ``hessian`` (which
need second-order grads) silently degrade to zero through ``relu`` / ``conv2d``
/ ``max_pool2d`` / ``cross_entropy`` because their VJPs cache non-differentiable
state in ``_fwd_ctx``. ``grad``, ``vjp``, and ``jvp`` are first-order and remain
correct through every op. Use ``tanh``, ``softmax``, arithmetic, ``matmul``, and
reductions if you need HOA-correct second derivatives.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from nanoad.tensor import Tensor

__all__ = ["grad", "hessian", "hvp", "jvp", "vjp"]


def _as_leaf(value: ArrayLike | Tensor) -> Tensor:
    """Wrap ``value`` as a fresh leaf Tensor, copying out of any prior tape."""
    if isinstance(value, Tensor):
        return Tensor(value.data)
    return Tensor(value)


def _check_scalar(out: Tensor, op_name: str) -> None:
    """Reject non-scalar fn outputs with a message that names the offending transform."""
    if out.ndim != 0:
        raise ValueError(
            f"{op_name} requires scalar-output fn; got shape {out.shape}. "
            "Reduce to a scalar (e.g., .sum()) before differentiating."
        )


def _normalize_argnum(argnum: int, n_args: int) -> int:
    """Resolve negative argnum and bounds-check; raise IndexError on out-of-range."""
    k = argnum + n_args if argnum < 0 else argnum
    if not 0 <= k < n_args:
        raise IndexError(
            f"argnum {argnum} out of range for fn with {n_args} arguments"
        )
    return k


def _read_leaf_grad(leaf: Tensor) -> Tensor:
    """Return ``leaf.grad`` or a zero Tensor of the right shape if disconnected."""
    if leaf.grad is None:
        return Tensor(np.zeros_like(leaf.data))
    return leaf.grad


def _check_shapes(name: str, leaves: list[Tensor], directions: list[Tensor]) -> None:
    """Fail fast if a tangent / cotangent / vector doesn't match its primal's shape."""
    for i, (leaf, direction) in enumerate(zip(leaves, directions, strict=True)):
        if leaf.shape != direction.shape:
            raise ValueError(
                f"{name}: direction[{i}] shape {direction.shape} "
                f"!= primal[{i}] shape {leaf.shape}"
            )


def _sum_terms(terms: list[Tensor]) -> Tensor:
    """Reduce a non-empty list of scalar Tensors via the public ``+`` op."""
    accumulator = terms[0]
    for term in terms[1:]:
        accumulator = accumulator + term
    return accumulator


def grad(
    fn: Callable[..., Tensor],
    argnum: int | tuple[int, ...] = 0,
) -> Callable[..., Tensor | tuple[Tensor, ...]]:
    """Return a function that computes the gradient of ``fn`` at its arguments.

    ``argnum`` is the index (or indices) of the argument(s) to differentiate
    against. A scalar ``argnum`` returns a single Tensor; a tuple returns a tuple
    of Tensors. ``fn`` must return a scalar Tensor.
    """

    def grad_fn(*args: ArrayLike | Tensor) -> Tensor | tuple[Tensor, ...]:
        leaves = [_as_leaf(a) for a in args]
        out = fn(*leaves)
        _check_scalar(out, "grad")
        out.backward()
        if isinstance(argnum, int):
            return _read_leaf_grad(leaves[_normalize_argnum(argnum, len(leaves))])
        return tuple(
            _read_leaf_grad(leaves[_normalize_argnum(k, len(leaves))]) for k in argnum
        )

    return grad_fn


def vjp(
    fn: Callable[..., Tensor],
    *primals: ArrayLike | Tensor,
) -> tuple[Tensor, Callable[[ArrayLike | Tensor], tuple[Tensor, ...]]]:
    """Eagerly run ``fn(*primals)`` and return ``(out, vjp_fn)``.

    Calling ``vjp_fn(cotangent)`` returns the tuple ``(J^T cotangent)_i``, one
    entry per primal. The closure reuses the captured forward graph; safe to
    call multiple times with different cotangents.
    """
    leaves = [_as_leaf(p) for p in primals]
    out = fn(*leaves)

    def vjp_fn(cotangent: ArrayLike | Tensor) -> tuple[Tensor, ...]:
        ct = _as_leaf(cotangent)
        if ct.shape != out.shape:
            raise ValueError(
                f"vjp: cotangent shape {ct.shape} != out shape {out.shape}"
            )
        for leaf in leaves:
            leaf.zero_grad()
        seed = (out * ct).sum()
        seed.backward()
        return tuple(_read_leaf_grad(leaf) for leaf in leaves)

    return out, vjp_fn


def jvp(
    fn: Callable[..., Tensor],
    primals: Sequence[ArrayLike | Tensor],
    tangents: Sequence[ArrayLike | Tensor],
) -> tuple[Tensor, Tensor]:
    """Compute ``fn(*primals)`` and the directional derivative ``J · tangents``.

    Implementation: build ``phi(u) = sum(out * u)``; backward yields each leaf's
    gradient as a tape-aware function of ``u``. A second backward of
    ``sum_i sum(leaf_grad_i * tangent_i)`` wrt ``u`` recovers ``J · v`` — the
    standard "VJP-of-VJP" transposition trick (JAX's strategy when only
    reverse-mode rules are defined).
    """
    if len(primals) != len(tangents):
        raise ValueError(
            f"jvp: primals and tangents must have the same length "
            f"(got {len(primals)} and {len(tangents)})"
        )
    leaves = [_as_leaf(p) for p in primals]
    out = fn(*leaves)

    tan_tensors = [_as_leaf(t) for t in tangents]
    _check_shapes("jvp", leaves, tan_tensors)

    u = Tensor(np.zeros_like(out.data))
    phi = (out * u).sum()
    phi.backward()

    leaf_grads = [_read_leaf_grad(leaf) for leaf in leaves]
    contraction_terms = [
        (leaf_grads[i] * tan_tensors[i]).sum() for i in range(len(leaves))
    ]
    psi = _sum_terms(contraction_terms)

    u.zero_grad()
    psi.backward()
    if u.grad is None:
        return out, Tensor(np.zeros_like(u.data))
    return out, u.grad


def hvp(
    fn: Callable[..., Tensor],
    primals: Sequence[ArrayLike | Tensor],
    vector: Sequence[ArrayLike | Tensor],
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Compute ``fn(*primals)`` (scalar) and ``H · vector``.

    Returns ``(out, hvp_value)`` where ``hvp_value`` is a tuple matching
    ``primals`` length, each entry shape-matched to the corresponding primal.
    Implementation: backward once for gradients ``g``; backward again on
    ``sum_i sum(g_i * vector_i)`` to populate the second-order ``H @ v`` into
    each leaf's grad.
    """
    if len(primals) != len(vector):
        raise ValueError(
            f"hvp: primals and vector must have the same length "
            f"(got {len(primals)} and {len(vector)})"
        )
    leaves = [_as_leaf(p) for p in primals]
    out = fn(*leaves)
    _check_scalar(out, "hvp")

    vec_tensors = [_as_leaf(v) for v in vector]
    _check_shapes("hvp", leaves, vec_tensors)

    out.backward()
    grads = [_read_leaf_grad(leaf) for leaf in leaves]
    contraction_terms = [(grads[i] * vec_tensors[i]).sum() for i in range(len(leaves))]
    contraction = _sum_terms(contraction_terms)

    for leaf in leaves:
        leaf.zero_grad()
    contraction.backward()
    return out, tuple(_read_leaf_grad(leaf) for leaf in leaves)


def hessian(
    fn: Callable[..., Tensor],
    argnum: int | tuple[int, ...] = 0,
) -> Callable[..., Tensor | tuple[Tensor, ...]]:
    """Return a function that computes the Hessian of ``fn`` at its arguments.

    A scalar ``argnum`` returns a single ``(n, n)`` Hessian Tensor where ``n``
    is the flattened size of that argument. A tuple ``argnum`` returns a tuple
    of per-argument diagonal Hessian blocks — cross-argument blocks
    ``∂²f/∂a_i ∂a_j`` are not computed (use ``grad`` + manual ``hvp`` if you
    need them). ``fn`` must return a scalar Tensor.
    """

    def hess_fn(*args: ArrayLike | Tensor) -> Tensor | tuple[Tensor, ...]:
        if isinstance(argnum, int):
            return _hessian_block(fn, args, _normalize_argnum(argnum, len(args)))
        return tuple(
            _hessian_block(fn, args, _normalize_argnum(k, len(args))) for k in argnum
        )

    return hess_fn


def _hessian_block(
    fn: Callable[..., Tensor],
    args: tuple[ArrayLike | Tensor, ...],
    k: int,
) -> Tensor:
    """Materialize the diagonal Hessian block for arg ``k`` via n basis-vector HVPs."""
    target = args[k]
    target_data = (
        target.data if isinstance(target, Tensor)
        else np.asarray(target, dtype=np.float64)
    )
    base_shape = target_data.shape
    n = int(np.prod(base_shape)) if base_shape else 1

    # Pre-build zero vectors for the non-target args; these stay constant across rows.
    zero_vectors: list[np.ndarray] = []
    for j, arg in enumerate(args):
        if j == k:
            zero_vectors.append(np.zeros(0))  # placeholder, replaced per row
            continue
        arg_data = (
            arg.data if isinstance(arg, Tensor)
            else np.asarray(arg, dtype=np.float64)
        )
        zero_vectors.append(np.zeros_like(arg_data))

    rows: list[np.ndarray] = []
    for i in range(n):
        e_flat = np.zeros(n)
        e_flat[i] = 1.0
        e_shaped = e_flat.reshape(base_shape) if base_shape else float(e_flat[0])
        vector_seq = list(zero_vectors)
        vector_seq[k] = e_shaped
        _, hv = hvp(fn, args, tuple(vector_seq))
        rows.append(np.asarray(hv[k].data).flatten())

    return Tensor(np.stack(rows))
