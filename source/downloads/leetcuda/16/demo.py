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


def online(scores, values, block):
    m, total, accum = -math.inf, 0.0, 0.0
    snapshots = []
    for start in range(0, len(scores), block):
        s, v = scores[start : start + block], values[start : start + block]
        new_m = max(m, max(s))
        alpha = math.exp(m - new_m) if total else 0.0
        p = [math.exp(x - new_m) for x in s]
        total, accum = alpha * total + sum(p), alpha * accum + dot(p, v)
        m = new_m
        partial_ref = dot(softmax(scores[: start + block]), values[: start + block])
        close(accum / total, partial_ref)
        snapshots.append(
            step(
                f"处理 key [{start},{min(start + block, len(scores))})",
                {
                    "scores": s,
                    "m": m,
                    "alpha": alpha,
                    "denominator": total,
                    "u": accum,
                    "O": accum / total,
                },
                "旧 denominator 和旧 u 同乘 alpha；本块 p 以新 m 为指数基准。",
            )
        )
    return (accum / total if total else 0.0), snapshots


def lab16(case):
    scores = [[0, 0], [1, 2, 3, 4], [0, -2, 8, 1], [5]][case]
    values = [2 + 2 * i for i in range(len(scores))]
    out, snap = online(scores, values, 2)
    close(out, dot(softmax(scores), values))
    return result(f"scores={scores},V={values}", snap, "每一个 KV 前缀状态与完整前缀 attention 相同。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab16(args.case), ensure_ascii=False, indent=2, allow_nan=False))
