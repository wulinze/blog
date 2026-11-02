"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab22(case):
    n = [1, 29, 32, 33][case]
    threads, width = 8, 4
    rounds = (n + threads * width - 1) // (threads * width)
    seen = []
    snaps = []
    for r in range(rounds):
        rows = []
        for t in range(threads):
            ids = [r * threads * width + t * width + v for v in range(width)]
            seen.extend(i for i in ids if i < n)
            rows.append([i if i < n else "mask" for i in ids])
        snaps.append(step(f"round {r}: thread × value", rows, "每行是一条逻辑线程的四个值；mask 只排除无效元素。"))
    assert sorted(seen) == list(range(n)) and len(seen) == len(set(seen))
    return result(f"N={n},8 逻辑线程×4 values", snaps, "每个有效坐标恰好搬一次，包括第二轮尾部。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab22(args.case), ensure_ascii=False, indent=2, allow_nan=False))
