"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab04(case):
    stride = [1, 2, 4, 8][case]
    addr = [4 * lane * stride for lane in range(32)]
    sectors = [a // 32 for a in addr]
    expected = [4, 8, 16, 32][case]
    assert len(set(sectors)) == expected
    keys = [i % 4 if case < 2 else 0 for i in range(32)]
    hist = [keys.count(i) for i in range(4)]
    assert sum(hist) == len(keys)
    return result(
        f"stride={stride}，{'均匀' if case < 2 else '热点'} histogram",
        [
            step("32 lane 地址", addr, "每 lane 读一个 4 字节 float；base=0，按 32B 对齐。"),
            step("对应 sector 编号", sectors, "同编号合成请求；不是 DRAM 实测流量。"),
            step(
                "请求效率",
                {"sectors": expected, "useful_bytes": 128, "request_bytes": 32 * expected, "efficiency": 4 / expected},
                "向量宽度变化时必须重算分子分母。",
            ),
            step("直方图与竞争", hist, "计数总和必须为 32；热点使同址原子集中。"),
        ],
        "sector 覆盖与独立地址集合一致；hist 总数守恒。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab04(args.case), ensure_ascii=False, indent=2, allow_nan=False))
