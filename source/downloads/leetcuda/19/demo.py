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


def lab19(case):
    count = [1, 2, 3, 5][case]
    scores = [i % 3 - 1 for i in range(count)]
    values = [i + 2 for i in range(count)]
    a = state(scores[::2], values[::2])
    b = state(scores[1::2], values[1::2])
    out = merge(a, b)[0]
    close(out, dot(softmax(scores), values))
    return result(
        f"{count} 个 KV tile，用单元素 tile 演示",
        [
            step("奇偶路由", {"A": list(range(0, count, 2)), "B": list(range(1, count, 2))}, "一组为空也必须结束。"),
            step("两组 O/LSE", {"A": [a[0], str(a[1])], "B": [b[0], str(b[1])]}, "每组有自己的分母。"),
            step("最终 merge", {"O": out, "reference": dot(softmax(scores), values)}, "不能直接把 O_A 与 O_B 相加。"),
        ],
        "单 tile、奇数 tile 与空 consumer 均正确。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab19(args.case), ensure_ascii=False, indent=2, allow_nan=False))
