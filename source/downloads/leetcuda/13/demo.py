"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab13(case):
    row = case * 2
    physical = [c ^ row for c in range(8)]
    recovered = [c ^ row for c in physical]
    assert recovered == list(range(8))
    order = [(r, c) for g in range(0, 4, 2) for c in range(4) for r in range(g, g + 2)]
    assert len(set(order)) == 16
    return result(
        f"教学 XOR 行号={row}",
        [
            step("逻辑列 → 物理列", physical, "8×8 小模型；不声称这就是某个硬件 SW128。"),
            step("再异或一次", recovered, "可逆性保证地址不丢失，不保证实际事务无 bank conflict。"),
            step("grouped CTA 坐标顺序", order, "ID 映射不等于硬件承诺的执行顺序。"),
        ],
        "XOR 逆映射和 CTA 一一对应检查通过。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab13(args.case), ensure_ascii=False, indent=2, allow_nan=False))
