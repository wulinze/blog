"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab17(case):
    rows = 8
    groups = [1, 2, 4, 8][case]
    owners = [r // (rows // groups) for r in range(rows)]
    assert all(owners.count(g) == rows // groups for g in range(groups))
    return result(
        f"8 query 行，{groups} 个逻辑计算组",
        [
            step("每行 owner", owners, "这表示逻辑分工，不能直接推断 CUDA warp 排布。"),
            step("每组状态行数", [owners.count(g) for g in range(groups)], "每行 m/denominator/u 只在一个组维护，无跨组行合并。"),
            step(
                "共享与私有",
                {"shared": "K/V tile", "private": "各组 Q 行及 m/denominator/u", "tradeoff": "更多组不减少输出元素总数"},
                "P fragment 仍要匹配 PV 输入布局。",
            ),
        ],
        "所有 query 行恰好属于一个组。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab17(args.case), ensure_ascii=False, indent=2, allow_nan=False))
