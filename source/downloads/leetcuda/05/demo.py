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


def lab05(case):
    scores = [[1, 2, 3, 4], [1000, 1001, 1002], [-9, -9, -9], [2, -100, 4, 1, 8]][case]
    block = [1, 2, 2, 3][case]
    _, snapshots = online(scores, [1] * len(scores), block)
    p = softmax(scores)
    close(sum(p), 1)
    for a, b in zip(p, softmax([s + 100 for s in scores])):
        close(a, b)
    return result(
        f"logits={scores}，block={block}",
        snapshots + [step("最终概率", p, "先有最终状态，才能输出各位置最终概率。")],
        "行和为 1；加常数不改变概率；每块在线分母通过独立参考。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab05(args.case), ensure_ascii=False, indent=2, allow_nan=False))
