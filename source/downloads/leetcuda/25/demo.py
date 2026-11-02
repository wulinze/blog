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


def lab25(case):
    bias = [-10, -2, 0, 5][case]
    a = [[1, 2], [3, 4]]
    b = [[2, -1], [1, 2]]
    c = matmul(a, b)
    out = [[max(0, v + bias) for v in row] for row in c]
    expected = [[max(0, 4 + bias), max(0, 3 + bias)], [max(0, 10 + bias), max(0, 5 + bias)]]
    assert out == expected
    return result(
        f"GEMM + bias({bias}) + ReLU",
        [
            step("GEMM 累加器", c, "先验证纯 GEMM。"),
            step("bias 后", [[v + bias for v in row] for row in c], "此小例子用同一 bias，GPU 练习要求逐列 bias。"),
            step("ReLU 后", out, "激活属于 epilogue，必须计入所需延迟。"),
        ],
        "完整 epilogue 与手算四个输出一致。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab25(args.case), ensure_ascii=False, indent=2, allow_nan=False))
