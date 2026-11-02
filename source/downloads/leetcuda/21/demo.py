"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab21(case):
    sr, sc = [(3, 1), (4, 1), (1, 2), (0, 1)][case]
    offsets = [[r * sr + c * sc for c in range(3)] for r in range(2)]
    flat = sum(offsets, [])
    unique = len(set(flat))
    return result(
        f"shape=(2,3),stride=({sr},{sc})",
        [
            step("坐标 → offset", offsets, "shape 定义合法坐标，stride 决定地址。"),
            step(
                "覆盖情况",
                {"logical_elements": 6, "unique_offsets": unique, "span": max(flat) + 1, "aliases": 6 - unique},
                "alias 布局不能直接当唯一写者的 copy 输出布局。",
            ),
        ],
        "枚举直接显示连续、padding、列主序和别名四种情形。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab21(args.case), ensure_ascii=False, indent=2, allow_nan=False))
