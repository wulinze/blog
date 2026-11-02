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


def quant4(x, global_scale=1.0):
    """Logical one-block NVFP4-like quantization, not hardware SF layout."""
    assert global_scale > 0 and all(math.isfinite(v) for v in x)
    ideal = max(map(abs, x), default=0) / (6 * global_scale)
    sf = FP8[nearest(ideal, FP8)] if ideal else 1.0
    sf = max(sf, FP8[1])  # explicit finite tiny-block policy
    codes = [nearest(abs(v) / (global_scale * sf), FP4) | (8 if math.copysign(1, v) < 0 else 0) for v in x]
    packed = [codes[i] | ((codes[i + 1] if i + 1 < len(codes) else 0) << 4) for i in range(0, len(codes), 2)]
    unpacked = [n for byte in packed for n in (byte & 15, byte >> 4)][: len(codes)]
    assert unpacked == codes
    restored = [(-1 if c & 8 else 1) * FP4[c & 7] * sf * global_scale for c in codes]
    return restored, sf, codes, packed


def lab32(case):
    x = [[0, 0, 0], [0.5, 1, 1.5, 2, 3, 4, 6], [-6, -2, -0.5, 0, 1], [0.1, 0.3, 0.7, 2.7, 9]][case]
    restored, sf, codes, packed = quant4(x)
    return result(
        f"FP4 输入 {x}",
        [
            step(
                "block scale 与编码",
                {"scale": sf, "codes": codes},
                "global scale=1；本例逻辑组不足 16 时仍使用同组 scale。",
            ),
            step("打包 byte", packed, "低 nibble 是偶数元素；奇数尾部高 nibble=0。"),
            step(
                "重建",
                {"x_hat": restored, "error": [a - b for a, b in zip(restored, x)]},
                "数值误差与字节 round-trip 是两项独立验收。",
            ),
        ],
        "所有 nibble pack/unpack 精确还原；实际硬件 SF layout 未模拟。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab32(args.case), ensure_ascii=False, indent=2, allow_nan=False))
