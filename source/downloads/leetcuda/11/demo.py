"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab11(case):
    count, load, compute = 4, [1, 3, 5, 8][case], 4
    serial = count * (load + compute)
    ideal = load + compute + (count - 1) * max(load, compute)
    timeline = []
    for t in range(count):
        start = load + t * max(load, compute)
        timeline.append([t, start, start + compute])
    assert ideal <= serial and timeline[-1][-1] == ideal
    v = 1.0003
    rounded = round(v * 1024) / 1024  # local binade [1,2), educational RNE
    return result(
        f"搬运 L={load}，计算 C={compute}，4 tiles",
        [
            step("串行时间", {"time": serial}, "单位是教学时间单位，不是 GPU 周期。"),
            step(
                "理想独立流水 compute 区间 [tile,start,end]", timeline, "假设可提前搬运、无资源争用；真实硬件需另测。"
            ),
            step(
                "精度与时间分别记录",
                {"ideal_time": ideal, "x": v, "10_fraction_bit_value": rounded, "input_error": rounded - v},
                "该舍入示例只覆盖 [1,2)；不是完整 TF32 编码器。",
            ),
        ],
        "理想时间线符合推导，且未将模型速度当实测。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab11(args.case), ensure_ascii=False, indent=2, allow_nan=False))
