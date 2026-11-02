"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab37(case):
    d = [128, 256, 512, 1024][case]
    a = 128 * d // 256
    b = 64 * d // 256
    assert a == 2 * b
    return result(
        f"Dv={d},consumer_threads=256",
        [
            step("Br=128 的累加器下界", {"values_per_thread": a}, "忽略全部其它临时寄存器。"),
            step("Br=64 的累加器下界", {"values_per_thread": b}, "下降来自实际输出块减小。"),
            step(
                "代价表",
                {"Q_CTA_count_ratio": 2, "new_cost": "跨组统计 / P shared 交接 / 更多任务"},
                "仅更改布局名称并不改变元素总数。",
            ),
        ],
        "寄存器逻辑下界按真实 tile 计算，代价单独列出。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab37(args.case), ensure_ascii=False, indent=2, allow_nan=False))
