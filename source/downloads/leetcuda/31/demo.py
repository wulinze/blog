"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json
import math


def close(a, b, tol=1e-9):
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), (a, b)


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab31(case):
    scales = [(1, 1), (1, 4), (2, 0.5), (0.25, 8)][case]
    parts = [3, 5]
    correct = sum(p * s for p, s in zip(parts, scales))
    wrong = sum(parts) * scales[0]
    explicit = dot([1 * scales[0], 2 * scales[0], 2 * scales[1], 3 * scales[1]], [1, 1, 1, 1])
    close(correct, explicit)
    return result(
        f"两个 K 组 scale 乘积={scales}",
        [
            step("未缩放 partial", parts, "每个 partial 属于不同 K 组。"),
            step(
                "分别缩放后相加",
                {"correct": correct, "dequant_reference": explicit},
                "K 方向变化的 scale 不能移到全部归约之后。",
            ),
            step(
                "错误外提的反例",
                {"wrong": wrong, "difference": wrong - correct},
                "scale 相同的 case 不会暴露错误，必须测试不同 scale。",
            ),
        ],
        "按组缩放与显式反量化点积一致。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab31(args.case), ensure_ascii=False, indent=2, allow_nan=False))
