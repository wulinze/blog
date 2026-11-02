"""Pure Python teaching model; values are estimates, never GPU measurements."""

import argparse
import json
import math


def model(flops: float, byte_count: float, peak_tflops: float, bandwidth_gbs: float) -> dict:
    """Compute a two-resource roof using decimal SI units."""
    if any(not math.isfinite(v) or v <= 0 for v in (flops, byte_count, peak_tflops, bandwidth_gbs)):
        raise ValueError("work, bytes, peak and bandwidth must be finite and positive")
    peak = peak_tflops * 1e12
    bandwidth = bandwidth_gbs * 1e9
    ai = flops / byte_count
    compute_us = flops / peak * 1e6
    memory_us = byte_count / bandwidth * 1e6
    return dict(
        ai=ai,
        ridge=peak / bandwidth,
        roof_tflops=min(peak, ai * bandwidth) / 1e12,
        compute_us=compute_us,
        memory_us=memory_us,
        lower_bound_us=max(compute_us, memory_us),
    )


def work(operation: str, m: int, n: int, k: int, input_bytes: int = 4, output_bytes: int = 4) -> tuple:
    """Count useful work and logical bytes; GEMM uses the conventional 2MNK count."""
    if min(m, n, k, input_bytes, output_bytes) <= 0:
        raise ValueError("dimensions and element widths must be positive")
    if operation == "add":
        return n, (2 * input_bytes + output_bytes) * n
    if operation == "axpy":
        return 2 * n, (2 * input_bytes + output_bytes) * n
    if operation == "gemv":
        return 2 * m * k, input_bytes * (m * k + k) + output_bytes * m
    if operation == "gemm":
        return 2 * m * n * k, input_bytes * (m * k + k * n) + output_bytes * m * n
    raise ValueError(f"unknown operation: {operation}")


def occupancy(
    threads: int,
    registers: int,
    shared: int,
    *,
    max_threads: int = 2048,
    max_warps: int = 64,
    max_blocks: int = 32,
    registers_sm: int = 65536,
    shared_sm: int = 65536,
) -> dict:
    """Simplified capacity model; excludes allocation granularity and partitions."""
    if threads <= 0 or registers <= 0 or shared < 0:
        raise ValueError("invalid per-block resources")
    warps = (threads + 31) // 32
    limits = dict(
        threads=max_threads // threads,
        warps=max_warps // warps,
        blocks=max_blocks,
        registers=registers_sm // (threads * registers),
        shared=shared_sm // shared if shared else max_blocks,
    )
    blocks = min(limits.values())
    return dict(
        resident_blocks=blocks, resident_warps=blocks * warps, occupancy=blocks * warps / max_warps, limits=limits
    )


def self_check() -> None:
    """Check independently derived examples and invalid-input handling."""
    assert work("add", 1, 12, 1) == (12, 144)
    assert work("gemm", 2, 3, 4) == (48, 104)
    assert work("gemm", 2, 3, 4, 2, 4) == (48, 64)
    assert math.isclose(model(*work("gemm", 4096, 4096, 4096), 60, 1500)["ai"], 4096 / 6)
    r = model(12000, 144000, 60, 1500)
    assert math.isclose(r["ridge"], 40)
    assert math.isclose(r["lower_bound_us"], 0.096)
    assert occupancy(256, 64, 0)["resident_blocks"] == 4
    assert occupancy(256, 64, 32768)["resident_blocks"] == 2
    assert occupancy(33, 1, 0)["resident_warps"] == 64
    assert occupancy(4096, 1, 0)["resident_blocks"] == 0
    for invalid in [0, -1, float("nan"), float("inf")]:
        try:
            model(1, invalid, 60, 1500)
        except ValueError:
            continue
        raise AssertionError("invalid model input accepted")
    print("PASS: arithmetic, mixed widths, units, resource limits and input guards")


def main() -> None:
    """Print estimates; defaults describe a hypothetical device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=["add", "axpy", "gemv", "gemm"], default="gemm")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--input-bytes", type=int, default=4)
    parser.add_argument("--output-bytes", type=int, default=4)
    parser.add_argument("--peak-tflops", type=float, default=60)
    parser.add_argument("--bandwidth-gbs", type=float, default=1500)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    flops, byte_count = work(args.operation, args.m, args.n, args.k, args.input_bytes, args.output_bytes)
    print(
        json.dumps(
            dict(
                kind="model-not-measurement",
                operation=args.operation,
                flops=flops,
                logical_bytes=byte_count,
                **model(flops, byte_count, args.peak_tflops, args.bandwidth_gbs),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
