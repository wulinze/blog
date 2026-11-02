"""Execute the exact upstream lab for this chapter in an existing checkout.

Does not clone, alter the checkout, install dependencies, or claim SKIP is PASS.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--arch", default="sm_120a")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("upstream.json"))
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if args.arch not in {"sm_80", "sm_86", "sm_89", "sm_90a", "sm_120a", "sm_120f"}:
        raise SystemExit("unsupported architecture argument")
    config = json.loads(args.config.read_text())
    repo = args.repo.resolve()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != config["commit"]:
        raise SystemExit(f"checkout mismatch: expected {config['commit']}, got {actual}")
    kind = config["kind"]
    if kind == "book-test":
        if config.get("requires_arch") and args.arch != config["requires_arch"]:
            print("SKIP: this lab requires " + config["requires_arch"])
            return 77
        commands = [["bash", "build_tests.sh", "--arch", args.arch, "--ch", config["test"]]]
        cwd = repo / "kernels/interview/book/tests"
        if config.get("requires_arch") == "sm_90a":
            nvcc = os.environ.get("NVCC") or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
            source = cwd / "ch13_hgemm_wgmma.cu"
            build = cwd / ".build"
            if not args.print_only:
                build.mkdir(exist_ok=True)
            binary = build / "chapter_hopper_wgmma"
            commands = [
                [
                    nvcc,
                    "-gencode",
                    "arch=compute_90a,code=sm_90a",
                    "-std=c++20",
                    "-O2",
                    "--expt-relaxed-constexpr",
                    "-I" + str(repo / "third-party/cutlass/include"),
                    "-I../..",
                    str(source),
                    "-o",
                    str(binary),
                ],
                [str(binary)],
            ]
    elif kind == "fp4-gemm":
        if args.arch not in {"sm_120a", "sm_120f"}:
            print("SKIP: this FP4 lab requires the SM120 build target")
            return 77
        commands = [
            ["bash", "build.sh", "--arch", args.arch, "--jobs", "4"],
            [f"./bin/leetcuda_bench_{args.arch.replace('_', '')}.bin", "--fp4-gemm"],
        ]
        cwd = repo / "kernels/interview"
    else:
        commands = [
            [
                sys.executable,
                "-m",
                "ffpa_attn.bench",
                "--fwd-backend",
                "cuda",
                "--cuda-impl",
                config["implementation"],
                "--B",
                "1",
                "--H",
                "1",
                "--N",
                "128",
                "--D",
                str(config.get("head_dim", 128)),
                "--dtype",
                "bf16",
                "--tasks",
                "self-attn,causal",
                "--no-bwd",
                "--show-allclose",
            ]
        ]
        cwd = repo
    for command in commands:
        print("cwd:", cwd, "\nargv:", json.dumps(command))
        if args.print_only:
            continue
        completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(completed.stdout)
        if completed.returncode:
            return completed.returncode
        if "SKIP" in completed.stdout or "skipped:" in completed.stdout or "no test sources found" in completed.stdout:
            print("SKIP detected: inspect individual results; not a completed GPU validation.")
            return 77
    print("Commands finished. Inspect numerical PASS/FAIL and error report; exit code alone is not accuracy evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
