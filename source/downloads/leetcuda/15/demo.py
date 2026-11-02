"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab15(case):
    consumers = [128, 256, 256, 384][case]
    rc = [128, 128, 192, 192][case]
    rp, p = 32, 128
    total = p * rp + consumers * rc
    budget = 65536
    return result(
        f"producer={p}×{rp},consumer={consumers}×{rc}",
        [
            step(
                "角色寄存器概念预算",
                {"producer": p * rp, "consumer": consumers * rc, "total": total},
                "未计分配粒度；预算 65536 仅为演示参数，不代表所有 GPU。",
            ),
            step(
                "是否超过教学预算",
                {"budget": budget, "fits": total <= budget, "remaining": budget - total},
                "超预算意味着需要改配置；不代表编译器必然采取哪种处理。",
            ),
            step(
                "buffer 协议仍需独立满足",
                {"producer": "wait empty → TMA", "consumer": "wait full → read/MMA → release"},
                "寄存器够用不保证没有同步死锁。",
            ),
        ],
        "按角色计算预算；超额配置明确标出，不当成可运行硬件配置。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab15(args.case), ensure_ascii=False, indent=2, allow_nan=False))
