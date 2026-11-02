"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab27(case):
    d = [64, 192, 512, 1024][case]
    br, bc, chunk, threads = 64, 64, 64, 128
    regs = br * d // threads
    shared = (br + bc) * chunk * 2
    return result(
        f"D={d},chunk=64,Br=64,threads=128",
        [
            step(
                "S 与 O 元素",
                {"S": br * bc, "O": br * d, "D_chunks": d // chunk},
                "S 跨 chunk 累加；O 各 chunk 是不同输出。",
            ),
            step(
                "预算分别看",
                {"one_QK_chunk_bytes": shared, "O_values_per_thread_average": regs},
                "shared chunk 不随 D 增长，但完整 O 随 D 增长。",
            ),
        ],
        "元素数量直接推导，未把逻辑平均值冒充编译器寄存器报告。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab27(args.case), ensure_ascii=False, indent=2, allow_nan=False))
