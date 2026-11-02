"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json
import math


def close(a, b, tol=1e-9):
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), (a, b)


def softmax(x):
    if not x:
        return []
    m = max(x)
    e = [math.exp(v - m) for v in x]
    z = math.fsum(e)
    return [v / z for v in e]


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab38(case):
    a = [0, 1, 3, 10][case]
    q = [a + 1, a + 2]
    k = [[1, 3], [2, -1], [4, 2]]
    b = [sum(row[d] for row in k) / len(k) for d in range(2)]
    qp = [v - a for v in q]
    kp = [[v - b[d] for d, v in enumerate(row)] for row in k]
    original = [dot(q, row) for row in k]
    partial = [dot(qp, row) for row in kp]
    delta = [dot([a, a], row) for row in kp]
    fixed = [s + t for s, t in zip(partial, delta)]
    for x, y in zip(softmax(original), softmax(fixed)):
        close(x, y)
    return result(
        f"Q 块均值示例 a=[{a},{a}]",
        [
            step("原始 scores", original, "单 query，D=2；为专注恒等式省略共同 sqrt(D)。"),
            step("平滑 Q/K 后", {"scores_prime": partial, "deltaS": delta}, "deltaS 随 key 变化，不能消掉。"),
            step(
                "概率对照",
                {"original": softmax(original), "without_delta": softmax(partial), "with_delta": softmax(fixed)},
                "只差每行常数时概率相同。",
            ),
        ],
        "补回 deltaS 后概率等价；非零均值可暴露漏补错误。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab38(args.case), ensure_ascii=False, indent=2, allow_nan=False))
