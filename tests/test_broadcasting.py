"""Broadcasting backward correctness — the part the README warns is easy to get wrong.

Each test exercises a different shape pair and uses grad_check to verify that
the analytic gradient (via Tensor.backward()) matches finite differences.
"""

import numpy as np


def test_add_vector_plus_scalar(grad_check):
    grad_check(lambda a, b: a + b, [1.0, 2.0, 3.0], 5.0)


def test_add_matrix_plus_row(grad_check):
    grad_check(
        lambda a, b: a + b,
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [10.0, 20.0, 30.0],
    )


def test_add_matrix_plus_col(grad_check):
    grad_check(
        lambda a, b: a + b,
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[10.0], [20.0]],
    )


def test_add_outer_broadcast(grad_check):
    """(2, 1) + (1, 3) → (2, 3)."""
    grad_check(
        lambda a, b: a + b,
        [[1.0], [2.0]],
        [[10.0, 20.0, 30.0]],
    )


def test_add_three_d_with_vector(grad_check):
    """(4, 1, 3) + (3,) → (4, 1, 3)."""
    a = np.arange(12, dtype=np.float64).reshape(4, 1, 3)
    b = np.array([100.0, 200.0, 300.0])
    grad_check(lambda x, y: x + y, a, b)


def test_mul_matrix_with_row(grad_check):
    grad_check(
        lambda a, b: a * b,
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [10.0, 20.0, 30.0],
    )


def test_mul_outer_broadcast(grad_check):
    grad_check(
        lambda a, b: a * b,
        [[1.0], [2.0], [3.0]],
        [[10.0, 20.0]],
    )


def test_sub_matrix_minus_row(grad_check):
    grad_check(
        lambda a, b: a - b,
        [[5.0, 10.0], [15.0, 20.0]],
        [1.0, 2.0],
    )


def test_div_matrix_by_col(grad_check):
    grad_check(
        lambda a, b: a / b,
        [[2.0, 4.0], [6.0, 8.0]],
        [[2.0], [4.0]],
    )


def test_compound_broadcast_chain(grad_check):
    """((a + b) * c).sum() — chained broadcasts through three operands."""
    a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    b = [10.0, 20.0, 30.0]
    c = [[100.0], [200.0]]
    grad_check(lambda x, y, z: (x + y) * z, a, b, c)


def test_relu_two_d(grad_check):
    from nanoad import relu

    grad_check(lambda x: relu(x), [[-1.0, 2.0], [3.0, -4.0]])


def test_tanh_two_d(grad_check):
    from nanoad import tanh

    grad_check(lambda x: tanh(x), [[0.1, -0.5], [1.0, -1.5]])
