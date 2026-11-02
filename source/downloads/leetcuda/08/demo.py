"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""


import argparse
import json
import math


def close(a, b, tol=1e-9):
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), (a, b)


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab08(case):
    theta = [0, math.pi / 4, math.pi / 2, math.pi][case]
    x, y = 3.0, 4.0
    out = [x * math.cos(theta) - y * math.sin(theta), x * math.sin(theta) + y * math.cos(theta)]
    close(dot(out, out), 25)
    stride = 32 if case % 2 == 0 else 33
    banks = [(lane * stride) % 32 for lane in range(32)]
    assert len(set(banks)) == (1 if stride == 32 else 32)
    return result(
        f"theta={theta:.3f}，shared stride={stride}",
        [
            step(
                "旋转前后",
                {"before": [x, y], "after": out, "norm_squared": dot(out, out)},
                "RoPE 单个坐标对；真实各对有各自频率。",
            ),
            step("列读取 bank", banks, "教学 float32、32 bank 模型，列号为 0。"),
            step(
                "不同 bank 数",
                {"unique_banks": len(set(banks)), "stride": stride},
                "padding 改变地址，不改变转置的数学。",
            ),
        ],
        "旋转范数保持；stride32/33 的 bank 覆盖符合手算。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab08(args.case), ensure_ascii=False, indent=2, allow_nan=False))
