"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab29(case):
    chunks = [1, 3, 5, 3][case]
    stages = [2, 2, 3, 3][case]
    seq = []
    for kv in range(2):
        for c in range(chunks):
            i = kv * chunks + c
            seq.append([kv, c, i, i % stages, (i // stages) % 2])
    assert [r[2] for r in seq] == list(range(2 * chunks))
    return result(
        f"2 KV tiles,{chunks} chunks,{stages} stages",
        [
            step("[KV,chunk,sequence,slot,phase]", seq, "slot/phase 绑定搬运序号，不只绑定 KV tile。"),
            step(
                "释放条件",
                {
                    "K": "最后一次 QK 读取结束",
                    "V": "最后一次 PV 读取结束",
                    "warning": "独立 barrier 集，不能混用完成事件",
                },
                "表只列编号；实际异步时序由设备协议保证。",
            ),
        ],
        "跨 KV 边界的搬运序号与环形相位连续。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab29(args.case), ensure_ascii=False, indent=2, allow_nan=False))
