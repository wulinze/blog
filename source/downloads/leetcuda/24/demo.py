"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab24(case):
    row = [0, 1, 3, 7][case]
    source = list(range(8))
    physical = [None] * 8
    for col, value in enumerate(source):
        physical[col ^ row] = value
    restored = [physical[col ^ row] for col in range(8)]
    assert restored == source
    return result(
        f"教学 swizzle：row={row}",
        [
            step("global 逻辑顺序", source, "本模型不是某个硬件 descriptor 的逐位实现。"),
            step("shared 物理顺序", physical, "生产者按映射存储。"),
            step("消费者按匹配布局读取", restored, "错误地按线性顺序读会看到第二步的置换。"),
        ],
        "匹配布局 round-trip 逐元素相同。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab24(args.case), ensure_ascii=False, indent=2, allow_nan=False))
