"""Forward/backward time and peak-memory benchmark for nanoad ops."""

from __future__ import annotations

import argparse
import statistics
import tracemalloc
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nanoad import Tensor
from nanoad.ops.activations import relu, tanh
from nanoad.ops.arithmetic import add, div, mul, power, sub
from nanoad.ops.conv import conv2d
from nanoad.ops.linalg import matmul, reshape, transpose
from nanoad.ops.pool import max_pool2d
from nanoad.ops.reductions import mean as mean_op
from nanoad.ops.reductions import sum as sum_op
from nanoad.ops.softmax import cross_entropy, softmax


class BenchCase:
    """Fixture for one benchmarkable op: (name, shape label, input factory, op application)."""

    def __init__(
        self,
        name: str,
        shape_label: str,
        make_inputs: Callable[[], tuple[Any, ...]],
        apply: Callable[..., Tensor],
    ) -> None:
        self.name = name
        self.shape_label = shape_label
        self.make_inputs = make_inputs
        self.apply = apply


class Result:
    """One row of bench output: median forward/backward us and peak Python heap KiB."""

    def __init__(
        self,
        name: str,
        shape_label: str,
        forward_us: float,
        backward_us: float,
        peak_kib: float,
    ) -> None:
        self.name = name
        self.shape_label = shape_label
        self.forward_us = forward_us
        self.backward_us = backward_us
        self.peak_kib = peak_kib


def _to_scalar(out: Tensor) -> Tensor:
    """Reduce a non-scalar Tensor to a scalar via sum so backward() can run."""
    if out.data.ndim == 0:
        return out
    return out.sum()


def measure(case: BenchCase, *, warmup: int, repeat: int) -> Result:
    """Time forward and backward for one bench case, plus peak Python heap usage."""
    for _ in range(warmup):
        leaves = case.make_inputs()
        out = case.apply(*leaves)
        _to_scalar(out).backward()

    fwd_times: list[float] = []
    bwd_times: list[float] = []
    for _ in range(repeat):
        leaves = case.make_inputs()
        t0 = perf_counter()
        out = case.apply(*leaves)
        t1 = perf_counter()
        loss = _to_scalar(out)
        t2 = perf_counter()
        loss.backward()
        t3 = perf_counter()
        fwd_times.append(t1 - t0)
        bwd_times.append(t3 - t2)

    tracemalloc.start()
    leaves = case.make_inputs()
    out = case.apply(*leaves)
    _to_scalar(out).backward()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return Result(
        name=case.name,
        shape_label=case.shape_label,
        forward_us=statistics.median(fwd_times) * 1e6,
        backward_us=statistics.median(bwd_times) * 1e6,
        peak_kib=peak_bytes / 1024.0,
    )


def _rand_tensor(shape: tuple[int, ...], rng: np.random.Generator) -> Tensor:
    """Build a fresh leaf Tensor of standard-normal samples with the given shape."""
    return Tensor(rng.standard_normal(shape))


def _rand_targets(batch: int, classes: int, rng: np.random.Generator) -> NDArray[np.int64]:
    """Sample integer class labels uniformly in [0, classes) for cross-entropy."""
    return rng.integers(0, classes, size=batch, dtype=np.int64)


def build_cases(rng: np.random.Generator) -> list[BenchCase]:
    """Build the standard bench fixture list. Shapes mirror MLP/CNN workloads at batch 128."""
    big = (128, 64, 14, 14)
    big_label = "(128, 64, 14, 14)"
    mat_x, mat_w = (128, 784), (784, 128)
    logits_shape = (128, 10)
    conv_x, conv_w = (128, 8, 14, 14), (16, 8, 3, 3)
    pool_x = (128, 16, 14, 14)

    return [
        BenchCase(
            name="add",
            shape_label=big_label,
            make_inputs=lambda: (_rand_tensor(big, rng), _rand_tensor(big, rng)),
            apply=lambda a, b: add(a, b),
        ),
        BenchCase(
            name="sub",
            shape_label=big_label,
            make_inputs=lambda: (_rand_tensor(big, rng), _rand_tensor(big, rng)),
            apply=lambda a, b: sub(a, b),
        ),
        BenchCase(
            name="mul",
            shape_label=big_label,
            make_inputs=lambda: (_rand_tensor(big, rng), _rand_tensor(big, rng)),
            apply=lambda a, b: mul(a, b),
        ),
        BenchCase(
            name="div",
            shape_label=big_label,
            make_inputs=lambda: (
                _rand_tensor(big, rng),
                Tensor(rng.standard_normal(big) + 2.0),
            ),
            apply=lambda a, b: div(a, b),
        ),
        BenchCase(
            name="power",
            shape_label=big_label,
            make_inputs=lambda: (_rand_tensor(big, rng),),
            apply=lambda x: power(x, 3.0),
        ),
        BenchCase(
            name="matmul",
            shape_label=f"{mat_x} @ {mat_w}",
            make_inputs=lambda: (_rand_tensor(mat_x, rng), _rand_tensor(mat_w, rng)),
            apply=lambda a, b: matmul(a, b),
        ),
        BenchCase(
            name="transpose",
            shape_label="(128, 16, 7, 7)",
            make_inputs=lambda: (_rand_tensor((128, 16, 7, 7), rng),),
            apply=lambda x: transpose(x, axes=(0, 2, 3, 1)),
        ),
        BenchCase(
            name="reshape",
            shape_label="(128, 784) -> (128, 1, 28, 28)",
            make_inputs=lambda: (_rand_tensor((128, 784), rng),),
            apply=lambda x: reshape(x, (128, 1, 28, 28)),
        ),
        BenchCase(
            name="sum",
            shape_label=f"{big_label} axis=(2,3)",
            make_inputs=lambda: (_rand_tensor(big, rng),),
            apply=lambda x: sum_op(x, axis=(2, 3)),
        ),
        BenchCase(
            name="mean",
            shape_label=f"{big_label} axis=(2,3)",
            make_inputs=lambda: (_rand_tensor(big, rng),),
            apply=lambda x: mean_op(x, axis=(2, 3)),
        ),
        BenchCase(
            name="relu",
            shape_label=big_label,
            make_inputs=lambda: (_rand_tensor(big, rng),),
            apply=lambda x: relu(x),
        ),
        BenchCase(
            name="tanh",
            shape_label=big_label,
            make_inputs=lambda: (_rand_tensor(big, rng),),
            apply=lambda x: tanh(x),
        ),
        BenchCase(
            name="softmax",
            shape_label=f"{logits_shape} axis=1",
            make_inputs=lambda: (_rand_tensor(logits_shape, rng),),
            apply=lambda x: softmax(x, axis=1),
        ),
        BenchCase(
            name="cross_entropy",
            shape_label=f"logits {logits_shape}, targets (128,)",
            make_inputs=lambda: (
                _rand_tensor(logits_shape, rng),
                _rand_targets(128, 10, rng),
            ),
            apply=lambda logits, targets: cross_entropy(logits, targets),
        ),
        BenchCase(
            name="conv2d",
            shape_label=f"x{conv_x} w{conv_w} pad=1",
            make_inputs=lambda: (
                _rand_tensor(conv_x, rng),
                _rand_tensor(conv_w, rng),
                _rand_tensor((conv_w[0],), rng),
            ),
            apply=lambda x, w, b: conv2d(x, w, b, stride=1, padding=1),
        ),
        BenchCase(
            name="max_pool2d",
            shape_label=f"{pool_x} kernel=2",
            make_inputs=lambda: (_rand_tensor(pool_x, rng),),
            apply=lambda x: max_pool2d(x, kernel=2),
        ),
    ]


def _format_table(results: Sequence[Result]) -> str:
    """Render results as a GitHub-flavored markdown table."""
    name_w = max(len("op"), *(len(r.name) for r in results))
    shape_w = max(len("shape"), *(len(r.shape_label) for r in results))

    header = (
        f"| {'op':<{name_w}} | {'shape':<{shape_w}} "
        f"| {'fwd us':>10} | {'bwd us':>10} | {'peak KiB':>10} |"
    )
    sep = (
        f"| {'-' * name_w} | {'-' * shape_w} "
        f"| {'-' * 10} | {'-' * 10} | {'-' * 10} |"
    )
    rows = [
        f"| {r.name:<{name_w}} | {r.shape_label:<{shape_w}} "
        f"| {r.forward_us:>10.1f} | {r.backward_us:>10.1f} | {r.peak_kib:>10.1f} |"
        for r in results
    ]
    return "\n".join([header, sep, *rows])


def main() -> None:
    """Bench CLI entry point."""
    parser = argparse.ArgumentParser(description="nanoad op benchmarks")
    parser.add_argument("--warmup", type=int, default=2, help="warmup iterations per case")
    parser.add_argument("--repeat", type=int, default=10, help="measured iterations per case")
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="run only cases whose name contains this substring",
    )
    parser.add_argument("--seed", type=int, default=0, help="rng seed for input generation")
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    rng = np.random.default_rng(args.seed)
    cases = build_cases(rng)
    if args.filter is not None:
        cases = [c for c in cases if args.filter in c.name]
        if not cases:
            parser.error(f"no cases matched filter {args.filter!r}")

    results: list[Result] = []
    for case in cases:
        print(f"  running {case.name}...", flush=True)
        results.append(measure(case, warmup=args.warmup, repeat=args.repeat))

    print()
    print(_format_table(results))


if __name__ == "__main__":
    main()
