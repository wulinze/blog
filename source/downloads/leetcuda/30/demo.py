"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json
import math


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def e4m3_values():
    """Positive E4M3FN codes 0..126. Code 127 is NaN, not finite."""
    values = []
    for code in range(127):
        exponent, mantissa = divmod(code, 8)
        value = mantissa * 2.0**-9 if exponent == 0 else (1 + mantissa / 8) * 2.0 ** (exponent - 7)
        values.append(value)
    assert values[-1] == 448
    return values


FP8 = e4m3_values()


FP4 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


def nearest(value, table):
    """Nearest representable finite value; tie uses even encoding LSB."""
    return min(range(len(table)), key=lambda i: (abs(table[i] - value), i % 2, i))


def quant8(x):
    assert all(math.isfinite(v) for v in x)
    scale = max(map(abs, x), default=0) / 448 or 1.0
    codes = [nearest(abs(v) / scale, FP8) | (128 if math.copysign(1, v) < 0 else 0) for v in x]
    restored = [(-1 if c & 128 else 1) * FP8[c & 127] * scale for c in codes]
    return restored, scale, codes


def lab30(case):
    x = [[0, 0, 0], [1, 2, 3, 4], [0.001, 1, 1000], [-2, -1, 0.5, 6]][case]
    restored, scale, codes = quant8(x)
    assert all(0 <= (c & 127) <= 126 for c in codes)
    return result(
        f"FP8 输入 {x}",
        [
            step("输入与 scale", {"x": x, "scale": scale}, "重建方向 x_hat=scale*q。"),
            step("E4M3FN 编码", codes, "最低 7 位为正幅度编码，最高位为符号。"),
            step(
                "反量化与误差",
                {"restored": restored, "error": [a - b for a, b in zip(restored, x)]},
                "零块采用 scale=1，不计算 0/0。",
            ),
        ],
        "编码均为有限 E4M3FN；软件码表最大值 448，自洽解码。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab30(args.case), ensure_ascii=False, indent=2, allow_nan=False))
