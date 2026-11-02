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


def state(scores, values):
    if not scores:
        return 0.0, -math.inf
    m = max(scores)
    return dot(softmax(scores), values), m + math.log(sum(math.exp(s - m) for s in scores))


def merge(a, b):
    oa, la = a
    ob, lb = b
    if la == -math.inf:
        return b
    if lb == -math.inf:
        return a
    c = max(la, lb)
    wa, wb = math.exp(la - c), math.exp(lb - c)
    return (wa * oa + wb * ob) / (wa + wb), c + math.log(wa + wb)


def lab26(case):
    scores = [-2, 1, 0, 4, 3][: case + 2]
    values = list(range(2, 2 + len(scores)))
    direct = dot(softmax(scores), values)
    one, _ = online(scores, values, 2)
    two = merge(state(scores[::2], values[::2]), state(scores[1::2], values[1::2]))[0]
    close(direct, one)
    close(direct, two)
    return result(
        f"N={len(scores)}，同一输入三种数学组织",
        [
            step("完整参考", {"O": direct}, "materialize 的数学参考。"),
            step("单状态在线", {"O": one}, "对应统一/WS 流水共同的数学状态。"),
            step("双状态 + merge", {"O": two}, "这里只验证组织等价，不模拟 CUDA 性能。"),
        ],
        "三种组织与独立完整参考对齐。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab26(args.case), ensure_ascii=False, indent=2, allow_nan=False))
