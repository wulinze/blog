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


def lab20(case):
    chunk = [1, 2, 3, 4][case]
    q = [1, 2, -1, 3, 2]
    keys = [[2, 0, 1, -1, 2], [0, 2, 1, 1, -1]]
    scores = [0.0, 0.0]
    snaps = []
    for start in range(0, len(q), chunk):
        for j, k in enumerate(keys):
            scores[j] += dot(q[start : start + chunk], k[start : start + chunk])
        snaps.append(
            step(f"D chunk [{start},{min(start + chunk, len(q))})", scores[:], "仍是 partial score；尚不可 softmax。")
        )
    ref = [dot(q, k) for k in keys]
    assert scores == ref
    p = softmax([s / math.sqrt(len(q)) for s in scores])
    snaps.append(step("完整 D 后的概率", p, "缩放使用完整 D，不是 chunk 宽度。"))
    return result(f"D=5,D_chunk={chunk}", snaps, "所有 chunk 方案等于完整点积，尾 chunk 正确。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab20(args.case), ensure_ascii=False, indent=2, allow_nan=False))
