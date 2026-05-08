"""Finite-difference gradient checks for arithmetic ops."""


def test_add(grad_check):
    grad_check(lambda a, b: a + b, 2.0, 3.0)


def test_sub(grad_check):
    grad_check(lambda a, b: a - b, 5.0, 2.0)


def test_mul(grad_check):
    grad_check(lambda a, b: a * b, 4.0, 3.0)


def test_div(grad_check):
    grad_check(lambda a, b: a / b, 6.0, 2.5)


def test_pow(grad_check):
    grad_check(lambda x: x**3, 2.0)


def test_neg(grad_check):
    grad_check(lambda x: -x, 4.0)


def test_radd_with_python_scalar(grad_check):
    grad_check(lambda x: 5.0 + x, 3.0)


def test_rsub_with_python_scalar(grad_check):
    grad_check(lambda x: 10.0 - x, 4.0)


def test_rmul_with_python_scalar(grad_check):
    grad_check(lambda x: 2.0 * x, 3.0)


def test_rtruediv_with_python_scalar(grad_check):
    grad_check(lambda x: 10.0 / x, 2.0)


def test_compound_expression(grad_check):
    """f(a, b) = (a + b)^2 * a — chained backward through several ops."""
    grad_check(lambda a, b: (a + b) ** 2 * a, 1.5, 2.0)
