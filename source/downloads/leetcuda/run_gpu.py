"""Verify 4 boundary cases, then optionally measure full baseline composition.

Requires Linux NVIDIA GPU + CUDA PyTorch + selected DSL. No silent CPU fallback.
"""

import argparse
import importlib
import importlib.metadata
import json
import statistics
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter", type=int, required=True, choices=range(4, 41))
    parser.add_argument("--backend", choices=["triton", "tilelang"], required=True)
    parser.add_argument("--student", help="Python module name, e.g. student_triton")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--output", default="result.json")
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        print("SKIP: NVIDIA CUDA GPU unavailable; no correctness or timing claim.", file=sys.stderr)
        return 77
    from exercises import expected, make_problem, run_reference

    backend = importlib.import_module("reference_" + args.backend)
    student = importlib.import_module(args.student) if args.student else None
    versions = {name: importlib.metadata.version(name) for name in ["torch", args.backend]}
    record = {
        "chapter": args.chapter,
        "backend": args.backend,
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "versions": versions,
        "scope": "unfused semantic baseline, not architecture-specific optimization",
        "cases": [],
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    for variant in range(4):
        p = make_problem(args.chapter, variant)
        originals = {k: v.clone() for k, v in p.items() if isinstance(v, torch.Tensor)}

        def launch():
            return student.solve(p, backend) if student else run_reference(p, backend)

        actual = launch()
        torch.cuda.synchronize()
        reference = expected(p)
        if not torch.isfinite(actual).all():
            raise AssertionError("nonfinite output")
        # Quantized probability can cross a code boundary after FP32 softmax rounding.
        atol = 3e-2 if args.chapter == 39 else 2e-3
        torch.testing.assert_close(actual.cpu().double(), reference.double(), atol=atol, rtol=2e-3)
        for key, value in originals.items():
            torch.testing.assert_close(p[key], value, atol=0, rtol=0)
        delta = actual.cpu().double() - reference.double()
        item = {
            "variant": variant,
            "status": "PASS",
            "shape": list(actual.shape),
            "max_abs": delta.abs().max().item(),
            "relative_l2": (delta.norm() / reference.double().norm().clamp_min(1e-12)).item(),
        }
        if args.benchmark:
            for _ in range(5):
                launch()
            torch.cuda.synchronize()
            samples = []
            for _ in range(20):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                launch()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            item["composition_ms_median"] = statistics.median(samples)
            item["composition_ms_samples"] = samples
        record["cases"].append(item)
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
