"""Newton step on a small PSD quadratic via conjugate gradient using only ``hvp``.

The quadratic ``f(x) = (1/2) xᵀ A x - bᵀ x`` has Hessian ``A`` and unique
minimizer ``x* = A⁻¹ b``. Starting from ``x₀ = 0``, the Newton update is
``x* - x₀ = A⁻¹ b``. We solve the linear system ``A v = b`` via the conjugate
gradient method, materializing the matrix-vector products with ``hvp(f, x, v)``
— the Hessian itself is never built.

This is the M8 demo: it shows that ``hvp`` is enough to do Newton-type
optimization on a strictly-convex problem without ever computing or storing the
Hessian explicitly. Run from the project root after ``pip install -e .[dev]``.
"""

from __future__ import annotations

import numpy as np

from nanoad import Tensor
from nanoad.functional import hvp


def cg_solve(
    matvec,
    b: np.ndarray,
    *,
    tol: float = 1e-8,
    max_iters: int | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Conjugate gradient on ``matvec(v) = A v`` for symmetric PSD ``A`` and target ``b``.

    Returns ``(x, residual_history)``. The history is the L2 norm of the
    residual after each iteration.
    """
    n = b.size
    if max_iters is None:
        max_iters = n
    x = np.zeros_like(b)
    r = b - matvec(x)
    p = r.copy()
    rs_old = float(r @ r)
    history: list[float] = [float(np.sqrt(rs_old))]
    for _ in range(max_iters):
        Ap = matvec(p)
        alpha = rs_old / float(p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        history.append(float(np.sqrt(rs_new)))
        if np.sqrt(rs_new) < tol:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    return x, history


def main() -> None:
    rng = np.random.default_rng(42)
    n = 20

    M = rng.standard_normal((n, n))
    A_data = M.T @ M + 1.0 * np.eye(n)  # SPD by construction
    b_data = rng.standard_normal(n)
    A = Tensor(A_data)
    b = Tensor(b_data)

    def quadratic(x: Tensor) -> Tensor:
        # 0.5 xᵀ A x - bᵀ x. Reshape x to a column for matmul, then unwrap with .sum().
        x_col = x.reshape(n, 1)
        quad = ((x_col.transpose() @ A) @ x_col).sum()
        linear = (b * x).sum()
        return 0.5 * quad - linear

    def matvec(v: np.ndarray) -> np.ndarray:
        # The Hessian of `quadratic` is A; hvp gives Av.
        x_zero = np.zeros(n)
        _, hv = hvp(quadratic, (x_zero,), (v,))
        return np.asarray(hv[0].data)

    print(f"Solving Ax = b with n = {n} via CG using only hvp(quadratic, x, v).")
    x_solution, history = cg_solve(matvec, b_data, tol=1e-10)

    closed_form = np.linalg.solve(A_data, b_data)
    error = float(np.linalg.norm(x_solution - closed_form))

    print(f"\nCG residuals (first / last): {history[0]:.4e} -> {history[-1]:.4e}")
    print(f"CG iterations:               {len(history) - 1}")
    print(f"|x_cg - x_closed_form|:      {error:.3e}")

    assert error < 1e-4, f"CG-Newton did not converge to closed-form solution (err={error})"
    print("\nNewton step recovered via Hessian-vector products only - no Hessian materialized.")


if __name__ == "__main__":
    main()
