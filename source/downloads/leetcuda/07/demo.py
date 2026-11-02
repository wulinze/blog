"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json
import math


def close(a, b, tol=1e-9):
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), (a, b)


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab07(case):
    x = [[1, 2, 3, 4], [0, 0, 0, 0], [1000, 1000.01, 999.99, 1000], [3, 3, 3, 3]][case]
    eps = 1e-5
    mu = math.fsum(x) / len(x)
    var = math.fsum((v - mu) ** 2 for v in x) / len(x)
    mean_sq = math.fsum(v * v for v in x) / len(x)
    ln = [(v - mu) / math.sqrt(var + eps) for v in x]
    rms = [v / math.sqrt(mean_sq + eps) for v in x]
    close(sum(ln), 0, 1e-8)
    close(sum(v * v for v in rms) / len(x), mean_sq / (mean_sq + eps))
    return result(
        f"x={x}",
        [
            step("输入", x, "gamma=1,beta=0；使用总体方差。"),
            step(
                "统计量",
                {"mean": mu, "variance": var, "mean_square": mean_sq, "eps": eps},
                "中心化方差用两遍式，不用大数相减。",
            ),
            step("LayerNorm", ln, "减均值，eps 在根号内。"),
            step("RMSNorm", rms, "不减均值，输出不要求零均值。"),
        ],
        "LN 中心化和为 0；RMS 的平方均值符合含 eps 的公式。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab07(args.case), ensure_ascii=False, indent=2, allow_nan=False))
