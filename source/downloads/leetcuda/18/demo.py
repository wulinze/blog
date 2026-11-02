"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab18(case):
    sk, sv = [(1, 1), (2, 1), (2, 2), (3, 2)][case]
    snaps = []
    for t in range(5):
        snaps.append(
            step(
                f"KV tile {t} 的两套槽",
                {"K_slot": t % sk, "K_phase": (t // sk) % 2, "V_slot": t % sv, "V_phase": (t // sv) % 2},
                "QK 消费 K，softmax 后 PV 消费 V；释放时刻分别判断。",
            )
        )
    return result(f"K stages={sk},V stages={sv}", snaps, "两套环形编号按各自 stage 数独立更新。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab18(args.case), ensure_ascii=False, indent=2, allow_nan=False))
