"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def protocol(stages, tiles):
    status = ["empty"] * stages
    snapshots = []
    for t in range(tiles):
        slot = t % stages
        phase = (t // stages) % 2
        assert status[slot] == "empty"
        for event in ["loading", "full", "reading", "empty"]:
            if event == "empty":
                assert status[slot] == "reading"
            status[slot] = event
            snapshots.append(
                step(
                    f"tile {t}: {event}",
                    {"slot": slot, "phase": phase, "states": status[:]},
                    "教学状态机：只有读者完成后才能重新写入；不是并发设备模拟器。",
                )
            )
    assert all(s == "empty" for s in status)
    return snapshots


def lab14(case):
    stages = [1, 2, 2, 3][case]
    tiles = [1, 3, 5, 7][case]
    return result(
        f"stages={stages},tiles={tiles}", protocol(stages, tiles), "每次覆盖都发生在 empty；绕环后 phase 正确翻转。"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab14(args.case), ensure_ascii=False, indent=2, allow_nan=False))
