"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def mma_coords(lane):
    g, t = divmod(lane, 4)
    return [[g, 2 * t], [g, 2 * t + 1], [g + 8, 2 * t], [g + 8, 2 * t + 1]]


def lab12(case):
    lane = [0, 5, 16, 31][case]
    all_coords = [tuple(c) for denominator in range(32) for c in mma_coords(denominator)]
    assert len(set(all_coords)) == 128 and set(all_coords) == {(i, j) for i in range(16) for j in range(8)}
    board = [["·"] * 8 for _ in range(16)]
    for i, j in mma_coords(lane):
        board[i][j] = lane
    return result(
        f"m16n8k16 FP32 C fragment，lane={lane}",
        [
            step("lane 的 4 个 C 坐标", mma_coords(lane), "仅适用本文指定的 PTX C/D 布局，不能套到 A/B。"),
            step("16×8 输出归属", board, "数字位置属于所选 lane，其余由其他 lane 持有。"),
            step(
                "全集覆盖",
                {"lanes": 32, "values_per_lane": 4, "unique_output_coordinates": 128},
                "穷举证明无遗漏无重叠。",
            ),
        ],
        "128 个输出坐标唯一覆盖；不是设备指令执行测试。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab12(args.case), ensure_ascii=False, indent=2, allow_nan=False))
