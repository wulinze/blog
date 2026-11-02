"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json
import math


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def matmul(a, b):
    return [[dot(row, col) for col in zip(*b)] for row in a]


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab10(case):
    bk = [1, 2, 3, 4][case]
    a, b = [[1, 2, 3, 4, 5], [2, 0, 1, -1, 3]], [[1, 2], [0, 1], [2, 0], [1, -1], [3, 2]]
    c = [[0, 0], [0, 0]]
    steps = [step("输入 A/B", {"A": a, "B": b}, "A[2,5] @ B[5,2]；K=5 用于覆盖尾块。")]
    for start in range(0, 5, bk):
        for i in range(2):
            for j in range(2):
                c[i][j] += sum(a[i][k] * b[k][j] for k in range(start, min(start + bk, 5)))
        steps.append(
            step(
                f"K tile [{start},{min(start + bk, 5)}) 后 C",
                [row[:] for row in c],
                "输出累加器跨 K tile 保留；尾部只计有效 k。",
            )
        )
    assert c == matmul(a, b)
    return result(f"BK={bk}，K=5", steps, "每种 BK 都与完整矩阵乘一致。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab10(args.case), ensure_ascii=False, indent=2, allow_nan=False))
