"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json
import math


def softmax(x):
    if not x:
        return []
    m = max(x)
    e = [math.exp(v - m) for v in x]
    z = math.fsum(e)
    return [v / z for v in e]


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


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


def lab39(case):
    scores = [[0, 0, 0, 0], [0, -3, -6, -9], [0, 1, 8, -1], [4, 3, 2, 1]][case]
    values = [1, 4, -2, 8]
    m = -math.inf
    denominator = 0.0
    u = 0.0
    snaps = []
    for start in range(0, 4, 2):
        s = scores[start : start + 2]
        v = values[start : start + 2]
        nm = max(m, max(s))
        alpha = math.exp(m - nm) if denominator else 0
        p = [math.exp(x - nm) for x in s]
        phat, sf, codes, _ = quant4([2688 * x for x in p])
        phat = [x / 2688 for x in phat]
        denominator = alpha * denominator + sum(p)
        u = alpha * u + dot(phat, v)
        m = nm
        snaps.append(
            step(
                f"KV tile {start // 2}",
                {
                    "m": m,
                    "alpha": alpha,
                    "p": p,
                    "p_hat": phat,
                    "SF": sf,
                    "codes": codes,
                    "l_exact": denominator,
                    "u_quantized": u,
                    "O_partial": u / denominator,
                },
                "分子用量化 P，分母保留定义中的未量化指数和。",
            )
        )
    ref = dot(softmax(scores), values)
    snaps.append(
        step(
            "最终误差",
            {"O": u / denominator, "reference": ref, "error": u / denominator - ref},
            "此差异是受控量化近似，不要求机器精度 allclose。",
        )
    )
    assert math.isfinite(u / denominator) and denominator > 0
    return result(f"scores={scores}，每 tile 2 个 key", snaps, "状态有限；记录每 tile 的 P 量化与最终参考误差。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab39(args.case), ensure_ascii=False, indent=2, allow_nan=False))
