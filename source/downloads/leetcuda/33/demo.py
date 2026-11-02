"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json
import math


def close(a, b, tol=1e-9):
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), (a, b)


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab33(case):
    sf = [1, 2, 4, 8]
    shift = case
    data = [1, 2, 3, 4]
    aligned = sum(16 * x * s for x, s in zip(data, sf))
    misplaced = sum(16 * x * sf[(g + shift) % 4] for g, x in enumerate(data))
    close(aligned, 784)
    return result(
        f"把 SF 错移 {shift} 组，观察错误",
        [
            step(
                "data / scale 按组",
                {"data_values": data, "scale": sf, "elements_per_group": 16},
                "每组重复同一个 data 值，方便手算。",
            ),
            step("正确配对", {"C": aligned}, "sum_g(16*data_g*scale_g)。"),
            step("错位 SF", {"C_wrong": misplaced, "error": misplaced - aligned}, "只改 scale 位置就足以破坏结果。"),
        ],
        "正确结果 784；非均匀测试能揭露 scale 错位。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab33(args.case), ensure_ascii=False, indent=2, allow_nan=False))
