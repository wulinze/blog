"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json
import math


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab09(case):
    k = [3, 8, 17, 33][case]
    a = [[(r + 1) * (i % 5 - 2) for i in range(k)] for r in range(3)]
    x = [i % 3 - 1 for i in range(k)]
    lanes = min(k, 8)
    partial = [[sum(row[i] * x[i] for i in range(denominator, k, lanes)) for denominator in range(lanes)] for row in a]
    out = [sum(p) for p in partial]
    ref = [dot(row, x) for row in a]
    assert out == ref
    return result(
        f"M=3,K={k}，展示 {lanes} 个逻辑 lane",
        [
            step("向量 x", x, "为便于显示使用最多 8 个逻辑 lane；不是修改 CUDA warp 宽度。"),
            step("各行 lane partial", partial, "同一行内归约，不跨行混合。"),
            step("每行结果", out, "只有该行 owner 写出。"),
        ],
        "分工归约与独立点积相同，包括非整除 K。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab09(args.case), ensure_ascii=False, indent=2, allow_nan=False))
