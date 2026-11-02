"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def mma_coords(lane):
    g, t = divmod(lane, 4)
    return [[g, 2 * t], [g, 2 * t + 1], [g + 8, 2 * t], [g + 8, 2 * t + 1]]


def lab23(case):
    am, an = [(1, 1), (2, 1), (1, 2), (2, 2)][case]
    bm, bn = 16 * am, 8 * an
    coords = []
    for m in range(am):
        for n in range(an):
            coords.extend((r + 16 * m, c + 8 * n) for lane in range(32) for r, c in mma_coords(lane))
    assert len(set(coords)) == bm * bn
    return result(
        f"atom 排布 M={am},N={an}",
        [
            step(
                "输出 tile",
                {"BM": bm, "BN": bn, "atoms": am * an, "elements": len(coords)},
                "这里只铺输出 M/N，不额外沿 K 复制输出。",
            ),
            step(
                "每 atom 的起点",
                [[16 * m, 8 * n] for m in range(am) for n in range(an)],
                "每个 atom 内采用前文的 C fragment 映射。",
            ),
            step(
                "完整覆盖",
                {"unique": len(set(coords)), "expected": bm * bn},
                "真实 TiledMMA 的线程复用方式还需具体类型验证。",
            ),
        ],
        "atom 输出组合无重叠无遗漏。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab23(args.case), ensure_ascii=False, indent=2, allow_nan=False))
