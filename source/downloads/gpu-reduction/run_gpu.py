"""Numerical acceptance and complete-reduction timing, including every reduction level."""

import argparse
import importlib
import importlib.metadata
import json
import math
import statistics
from pathlib import Path

import torch


def reference(x, y, operation: str) -> tuple:
    """FP64 CPU reference plus absolute contribution sum for error normalization."""
    a, b = x.cpu().double(), y.cpu().double()
    if operation == "max":
        return (a.max().item(), 0.0) if a.numel() else (-math.inf, 0.0)
    terms = a * b if operation == "dot" else a
    return terms.sum().item(), terms.abs().sum().item()


def validate(launch, x, y, out, operation: str) -> dict:
    """Check cancellation-aware error, repeated calls and unchanged inputs."""
    a, b = x.clone(), y.clone()
    expected, magnitude = reference(x, y, operation)
    observed = []
    for _ in range(2):
        out.fill_(float("nan"))
        launch(x, y, out)
        torch.cuda.synchronize()
        value = out.item()
        if operation == "max":
            assert value == expected, (value, expected)
        else:
            # Conservative teaching tolerance, not a universal numerical error theorem.
            tolerance = 1e-6 + 2e-6 * magnitude
            assert math.isfinite(value) and abs(value - expected) <= tolerance, (value, expected, tolerance)
        observed.append(value)
    torch.testing.assert_close(x, a, atol=0, rtol=0)
    torch.testing.assert_close(y, b, atol=0, rtol=0)
    error = abs(observed[-1] - expected) if math.isfinite(expected) else 0.0
    return dict(
        correctness="passed",
        reference_fp64=expected,
        result=observed[-1],
        absolute_error=error,
        contribution_sum=magnitude,
        scaled_error=error / max(magnitude, 1e-30),
        repeat_equal=observed[0] == observed[1],
    )


def time_launch(launch, x, y, out) -> dict:
    """CUDA event timing excludes JIT and allocations; includes all reduction levels."""
    for _ in range(10):
        launch(x, y, out)
    torch.cuda.synchronize()
    values = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            launch(x, y, out)
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / 50)
    return dict(median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values), samples_ms=values)


def main() -> None:
    """Run finite float32 inputs; empty max is defined as negative infinity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["triton", "tilelang"], required=True)
    parser.add_argument("--impl", choices=["reference", "student"], default="reference")
    parser.add_argument("--operation", choices=["sum", "max", "dot"], default="sum")
    parser.add_argument("--mode", choices=["check", "bench", "profile"], default="check")
    parser.add_argument("--n", type=int, default=1003)
    parser.add_argument("--block", type=int, choices=[128, 256, 512, 1024], default=256)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.cuda is None:
        parser.error("Requires CUDA PyTorch and a compatible NVIDIA GPU")
    if args.n < 0 or (args.mode != "check" and args.n == 0):
        parser.error("n>=0; use check mode for the empty case")
    workers = args.workers or (4 if args.backend == "triton" else 128)
    if workers not in ((4, 8) if args.backend == "triton" else (128, 256)):
        parser.error("workers must be 4/8 warps (Triton) or 128/256 threads (TileLang)")
    module = importlib.import_module(f"{args.impl}_{args.backend}")
    sizes = (
        sorted(set([0, 1, 31, 32, 33, 127, 128, 129, 255, 256, 257, 520, args.block**2 + 1, args.n]))
        if args.mode == "check"
        else [args.n]
    )
    torch.manual_seed(20261029)
    rows = []
    for n in sizes:
        launch = module.prepare(n, args.block, workers, args.operation)
        patterns = ["random", "ones", "negative", "cancellation"] if args.mode == "check" else ["random"]
        for pattern in patterns:
            x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            if pattern == "ones":
                x.fill_(1)
                y.fill_(1)
            elif pattern == "negative":
                x = -x.abs() - 1
                y.fill_(1)
            elif pattern == "cancellation":
                x[::2] = 1
                x[1::2] = -1
                y.fill_(1)
            out = torch.empty(1, device="cuda")
            row = dict(n=n, pattern=pattern, **validate(launch, x, y, out, args.operation))
            if args.mode == "bench":
                row.update(time_launch(launch, x, y, out))
                row["input_gbs"] = (8 if args.operation == "dot" else 4) * n / (row["median_ms"] * 1e6)
            elif args.mode == "profile":
                launch(x, y, out)
                torch.cuda.synchronize()
            if n <= 1003 and hasattr(launch, "partials"):
                row["first_partials"] = launch.partials[0].cpu().tolist() if launch.partials else []
            print(json.dumps(row, allow_nan=True), flush=True)
            rows.append(row)
        del launch
    report = dict(
        gpu=torch.cuda.get_device_name(),
        backend=args.backend,
        backend_version=importlib.metadata.version(args.backend),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        operation=args.operation,
        block=args.block,
        workers=workers,
        dtype="float32",
        cache_policy="repeated buffers; no flush",
        rows=rows,
    )
    if args.output:
        # JSON has no infinity literal; represent empty max explicitly as a string.
        def clean(value):
            if isinstance(value, float) and not math.isfinite(value):
                return str(value)
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, list):
                return [clean(v) for v in value]
            return value

        args.output.write_text(json.dumps(clean(report), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
