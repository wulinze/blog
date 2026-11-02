"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json
import math
import statistics


def close(a, b, tol=1e-9):
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), (a, b)


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab40(case):
    prep = [0, 0.2, 0.4, 0.8][case]
    kernel_speedup = 2
    total = prep + (1 - prep) / kernel_speedup
    samples = [total * x for x in [1.02, 0.99, 1, 1.05, 1.01]]
    close(1 / total, 1 / (prep + (1 - prep) / 2))
    return result(
        f"未加速时间占比 f={prep}",
        [
            step("基线总时间=1", {"prepare": prep, "kernel": 1 - prep}, "所有时间是构造数据，不是设备 benchmark。"),
            step(
                "主 kernel 加速 2 倍后",
                {"prepare": prep, "kernel": (1 - prep) / 2, "total": total, "e2e_speedup": 1 / total},
                "不能把 kernel-only 的 2 倍写成端到端 2 倍。",
            ),
            step(
                "构造的重复测量",
                {"samples": samples, "median": statistics.median(samples), "minimum": min(samples)},
                "真实实验保存完整分布与环境。",
            ),
        ],
        "端到端加速符合 Amdahl 推导；示例时间明确标为构造。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab40(args.case), ensure_ascii=False, indent=2, allow_nan=False))
