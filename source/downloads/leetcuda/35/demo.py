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


def lab35(case):
    bias = [0, 3, 10, 100][case]
    scores = [1 + bias, 2 + bias, -1 + bias]
    v = [2, 5, 8]
    mu = sum(v) / len(v)
    a = dot(softmax(scores), v)
    b = dot(softmax([s - bias for s in scores]), [x - mu for x in v]) + mu
    close(a, b)
    return result(
        f"每行 logit 公共偏置={bias}",
        [
            step("平滑前", {"scores": scores, "V": v, "O": a}, "这里把 K 平滑造成的行常数直接展示为 score 偏置。"),
            step(
                "平滑后",
                {"scores": [s - bias for s in scores], "V_centered": [x - mu for x in v]},
                "V 中心化后输出需加回均值。",
            ),
            step("修正输出", {"mean_V": mu, "O_corrected": b}, "未量化代数等价；加入量化后需另测误差。"),
        ],
        "行常数消除与 V 均值加回保持完整输出。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab35(args.case), ensure_ascii=False, indent=2, allow_nan=False))
