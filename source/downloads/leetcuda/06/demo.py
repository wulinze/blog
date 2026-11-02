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


def lab06(case):
    scores, values = [0, 1, 3, -2], [10, 2, 4, 8]
    cuts = [1, 2, 0, 4]
    cut = cuts[case]
    a, b = state(scores[:cut], values[:cut]), state(scores[cut:], values[cut:])
    out, lse = merge(a, b)
    ref, ref_lse = state(scores, values)
    close(out, ref)
    close(lse, ref_lse)
    assert merge((0, -math.inf), (0, -math.inf)) == (0, -math.inf)
    return result(
        f"分界 key={cut}（0/4 会产生空段）",
        [
            step("原始 scores / V", {"scores": scores, "V": values}, "两段必须是不同 key 子集。"),
            step(
                "两段独立终态",
                {"A_O": a[0], "A_LSE": str(a[1]), "B_O": b[0], "B_LSE": str(b[1])},
                "空段 LSE=-inf，合并时先分支。",
            ),
            step("稳定合并", {"O": out, "LSE": lse, "full_reference": ref}, "按指数质量加权，不是简单平均。"),
        ],
        "不同分界和空段均等于完整参考；双空返回空状态。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab06(args.case), ensure_ascii=False, indent=2, allow_nan=False))
