"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab28(case):
    tasks = [1, 2, 3, 10][case]
    workers = 3
    assignment = [list(range(p, tasks, workers)) for p in range(workers)]
    flat = sum(assignment, [])
    assert sorted(flat) == list(range(tasks)) and len(flat) == len(set(flat))
    return result(
        f"tasks={tasks},workers={workers}",
        [
            step("静态任务分配", assignment, "第 p 行 worker 处理 p,p+P,...。"),
            step(
                "每 worker 任务数", list(map(len, assignment)), "真实 causal 任务耗时可能不同，数量均匀不等于时间均匀。"
            ),
            step("覆盖验收", sorted(flat), "每个任务恰好一次；新任务必须重置 m/denominator/u。"),
        ],
        "T<P、T=P 和不整除尾波均覆盖正确。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab28(args.case), ensure_ascii=False, indent=2, allow_nan=False))
