"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json
import math


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


def e4m3_values():
    """Positive E4M3FN codes 0..126. Code 127 is NaN, not finite."""
    values = []
    for code in range(127):
        exponent, mantissa = divmod(code, 8)
        value = mantissa * 2.0**-9 if exponent == 0 else (1 + mantissa / 8) * 2.0 ** (exponent - 7)
        values.append(value)
    assert values[-1] == 448
    return values


FP8 = e4m3_values()


FP4 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


def nearest(value, table):
    """Nearest representable finite value; tie uses even encoding LSB."""
    return min(range(len(table)), key=lambda i: (abs(table[i] - value), i % 2, i))


def quant8(x):
    assert all(math.isfinite(v) for v in x)
    scale = max(map(abs, x), default=0) / 448 or 1.0
    codes = [nearest(abs(v) / scale, FP8) | (128 if math.copysign(1, v) < 0 else 0) for v in x]
    restored = [(-1 if c & 128 else 1) * FP8[c & 127] * scale for c in codes]
    return restored, scale, codes


def lab34(case):
    scores = [[0, 0, 0, 0], [0, 0, 0, 8], [1, 2, 3, 4], [-10, 0, 10, 1]][case]
    values = [1, 3, -2, 6]
    p = softmax(scores)
    out = dot(p, values)
    sq, _, _ = quant8(scores)
    vq, _, _ = quant8(values)
    pq = softmax(sq)
    output_qk = dot(pq, values)
    output_v = dot(p, vq)
    ess = 1 / sum(x * x for x in p)
    assert 1 - 1e-9 <= ess <= len(p) + 1e-9
    return result(
        f"scores={scores}",
        [
            step("高精度概率与 ESS", {"p": p, "ESS": ess, "O": out}, "均匀时 ESS 大，尖峰时接近 1。"),
            step(
                "只量化分数",
                {"scores_hat": sq, "O": output_qk, "error": output_qk - out},
                "为隔离机制直接量化 score；不冒充真实 Q/K 的全部误差。",
            ),
            step("只量化 V", {"V_hat": vq, "O": output_v, "error": output_v - out}, "各阶段单独注入误差，再研究组合。"),
        ],
        "ESS 在合法范围；分别记录两种误差来源。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab34(args.case), ensure_ascii=False, indent=2, allow_nan=False))
