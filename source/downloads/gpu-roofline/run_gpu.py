"""Validate and benchmark learning kernels; never equate requested bandwidth with DRAM."""

import argparse
import importlib
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch

ALPHA = 1.25


def expected(x, y, operation: str):
    """Evaluate the numerical contract using a PyTorch reference."""
    return torch.add(x, y) if operation == "add" else torch.add(y, x, alpha=ALPHA)


def check_launch(launch, tensors, operation: str) -> None:
    """Check every output and ensure input buffers are unchanged."""
    for x, y, out in tensors:
        x_before, y_before = x.clone(), y.clone()
        out.fill_(float("nan"))
        launch(x, y, out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected(x_before, y_before, operation), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(x, x_before, rtol=0, atol=0)
        torch.testing.assert_close(y, y_before, rtol=0, atol=0)


def allocate(n: int, buffers: int):
    """Allocate fixed float32 buffers before compilation and timing."""
    return [
        (torch.randn(n, device="cuda"), torch.randn(n, device="cuda"), torch.empty(n, device="cuda"))
        for _ in range(buffers)
    ]


def benchmark(launch, tensors, warmup: int = 10, repeats: int = 50, samples: int = 7) -> dict:
    """Measure batches with CUDA events; repeated buffers may remain cache-resident."""
    for i in range(warmup):
        launch(*tensors[i % len(tensors)])
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(repeats):
            launch(*tensors[i % len(tensors)])
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / repeats)
    return dict(
        median_ms=statistics.median(values),
        min_ms=min(values),
        max_ms=max(values),
        samples_ms=values,
        warmup=warmup,
        repeats=repeats,
        timing="CUDA events per batch; host submission gaps may remain",
    )


def version(name: str) -> str:
    """Return installed package version without importing the other backend."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    """Run checks, a one-launch profiling target, or a benchmark sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["triton", "tilelang", "torch"], required=True)
    parser.add_argument("--impl", choices=["reference", "student"], default="reference")
    parser.add_argument("--operation", choices=["add", "axpy"], default="add")
    parser.add_argument("--mode", choices=["check", "bench", "profile"], default="check")
    parser.add_argument("--n", type=int, default=1048576)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--buffers", type=int, default=1)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dump-source", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.cuda is None:
        parser.error("This exercise requires a CUDA-enabled PyTorch installation and NVIDIA GPU")
    if args.n < 0 or args.buffers <= 0 or args.block not in (128, 256, 512, 1024):
        parser.error("n >= 0, buffers > 0, block in {128,256,512,1024} required")
    workers = args.workers or (4 if args.backend == "triton" else 128)
    allowed = (4, 8) if args.backend == "triton" else (128, 256)
    if workers not in allowed:
        parser.error(f"workers for {args.backend}: {allowed}")
    if args.mode != "check" and args.n == 0:
        parser.error("n=0 is a correctness case, not a benchmark")
    torch.manual_seed(20261029)
    module = None if args.backend == "torch" else importlib.import_module(f"{args.impl}_{args.backend}")
    if module is None and args.impl == "student":
        parser.error("student implementation must use triton or tilelang")
    configs = [(b, w) for b in (128, 256, 512, 1024) for w in allowed] if args.sweep else [(args.block, workers)]
    rows = []
    for block, worker_count in configs:
        sizes = (
            sorted(set([0, 1, 31, 32, 33, 127, 128, 129, block - 1, block, block + 1, 1003, args.n]))
            if args.mode == "check"
            else [args.n]
        )
        for n in sizes:
            tensors = allocate(n, args.buffers)
            if module is None:

                def launch(x, y, out):
                    if args.operation == "add":
                        torch.add(x, y, out=out)
                    else:
                        torch.add(y, x, alpha=ALPHA, out=out)
            else:
                launch = module.prepare(n, block, worker_count, args.operation)
            check_launch(launch, tensors, args.operation)  # Includes JIT, outside timing.
            row = dict(n=n, block=block, workers=worker_count, correctness="passed")
            if args.dump_source and n:
                if args.backend == "tilelang" and hasattr(launch, "get_kernel_source"):
                    args.dump_source.write_text(launch.get_kernel_source())
                else:
                    parser.error("--dump-source is for TileLang callables exposing get_kernel_source()")
            if args.mode == "bench":
                row.update(benchmark(launch, tensors))
                seconds = row["median_ms"] * 1e-3
                row.update(
                    logical_bytes=12 * n,
                    requested_gbs=12 * n / seconds / 1e9,
                    useful_gflops=(n if args.operation == "add" else 2 * n) / seconds / 1e9,
                    buffer_footprint_bytes=12 * n * args.buffers,
                    cache_policy="reuse allocated buffers; no cache flush",
                )
            elif args.mode == "profile":
                launch(*tensors[0])  # One extra matched launch after correctness calls.
                torch.cuda.synchronize()
            print(json.dumps(row), flush=True)
            rows.append(row)
            del launch, tensors
    device = torch.cuda.get_device_properties(0)
    report = dict(
        backend=args.backend,
        implementation=args.impl,
        operation=args.operation,
        gpu=device.name,
        capability=list(torch.cuda.get_device_capability()),
        sm_count=device.multi_processor_count,
        dtype="float32",
        cuda=torch.version.cuda,
        torch=torch.__version__,
        triton=version("triton"),
        tilelang=version("tilelang"),
        rows=rows,
    )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
