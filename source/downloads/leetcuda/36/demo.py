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


def lab36(case):
    scale = [0.25, 1, 2, 4][case]
    raw = [-1, 0, 2]
    v = [1, 4, 8]
    correct = dot(softmax([s * scale for s in raw]), v)
    wrong = dot(softmax(raw), v) * scale
    exp2 = [2 ** ((s - max(raw)) * scale * math.log2(math.e)) for s in raw]
    via_exp2 = dot([p / sum(exp2) for p in exp2], v)
    close(correct, via_exp2)
    return result(
        f"QK scale 乘积 / sqrt(D)={scale}",
        [
            step("正确 score 缩放", [s * scale for s in raw], "先恢复 logits 尺度，再 softmax。"),
            step("自然指数与 exp2", {"O_exp": correct, "O_exp2": via_exp2}, "log2(e) 只转换指数底，不改变数学。"),
            step(
                "错误把 scale 移出 softmax",
                {"wrong": wrong, "difference": wrong - correct},
                "scale=1 恰好不暴露错误，必须多 case。",
            ),
        ],
        "exp 与 exp2 路径等价；演示非法外提反例。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab36(args.case), ensure_ascii=False, indent=2, allow_nan=False))
