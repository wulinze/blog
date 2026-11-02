"""CPU numerical/layout/protocol labs. Python 3.9+, standard library only.

These models do not execute CUDA instructions or predict hardware timings.
Run: python3 learning_lab.py --chapter 16 --case 2
Run every scenario: python3 learning_lab.py --self-test
"""

import argparse
import json
import math
import statistics


def close(a, b, tol=1e-9):
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), (a, b)


def softmax(x):
    if not x:
        return []
    m = max(x)
    e = [math.exp(v - m) for v in x]
    z = math.fsum(e)
    return [v / z for v in e]


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def matmul(a, b):
    return [[dot(row, col) for col in zip(*b)] for row in a]


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


def online(scores, values, block):
    m, total, accum = -math.inf, 0.0, 0.0
    snapshots = []
    for start in range(0, len(scores), block):
        s, v = scores[start : start + block], values[start : start + block]
        new_m = max(m, max(s))
        alpha = math.exp(m - new_m) if total else 0.0
        p = [math.exp(x - new_m) for x in s]
        total, accum = alpha * total + sum(p), alpha * accum + dot(p, v)
        m = new_m
        partial_ref = dot(softmax(scores[: start + block]), values[: start + block])
        close(accum / total, partial_ref)
        snapshots.append(
            step(
                f"处理 key [{start},{min(start + block, len(scores))})",
                {
                    "scores": s,
                    "m": m,
                    "alpha": alpha,
                    "denominator": total,
                    "u": accum,
                    "O": accum / total,
                },
                "旧 denominator 和旧 u 同乘 alpha；本块 p 以新 m 为指数基准。",
            )
        )
    return (accum / total if total else 0.0), snapshots


def state(scores, values):
    if not scores:
        return 0.0, -math.inf
    m = max(scores)
    return dot(softmax(scores), values), m + math.log(sum(math.exp(s - m) for s in scores))


def merge(a, b):
    oa, la = a
    ob, lb = b
    if la == -math.inf:
        return b
    if lb == -math.inf:
        return a
    c = max(la, lb)
    wa, wb = math.exp(la - c), math.exp(lb - c)
    return (wa * oa + wb * ob) / (wa + wb), c + math.log(wa + wb)


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


def lab05(case):
    scores = [[1, 2, 3, 4], [1000, 1001, 1002], [-9, -9, -9], [2, -100, 4, 1, 8]][case]
    block = [1, 2, 2, 3][case]
    _, snapshots = online(scores, [1] * len(scores), block)
    p = softmax(scores)
    close(sum(p), 1)
    for a, b in zip(p, softmax([s + 100 for s in scores])):
        close(a, b)
    return result(
        f"logits={scores}，block={block}",
        snapshots + [step("最终概率", p, "先有最终状态，才能输出各位置最终概率。")],
        "行和为 1；加常数不改变概率；每块在线分母通过独立参考。",
    )


def lab06(case):
    scores, values = [0, 1, 3, -2], [10, 2, 4, 8]
    cuts = [1, 2, 0, 4]
    cut = cuts[case]
    a, b = state(scores[:cut], values[:cut]), state(scores[cut:], values[cut:])
    out, lse = merge(a, b)
    ref, ref_lse = state(scores, values)
    close(out, ref)
    close(lse, ref_lse)
    assert merge((0, -math.inf), (0, -math.inf)) == (0, -math.inf)
    return result(
        f"分界 key={cut}（0/4 会产生空段）",
        [
            step("原始 scores / V", {"scores": scores, "V": values}, "两段必须是不同 key 子集。"),
            step(
                "两段独立终态",
                {"A_O": a[0], "A_LSE": str(a[1]), "B_O": b[0], "B_LSE": str(b[1])},
                "空段 LSE=-inf，合并时先分支。",
            ),
            step("稳定合并", {"O": out, "LSE": lse, "full_reference": ref}, "按指数质量加权，不是简单平均。"),
        ],
        "不同分界和空段均等于完整参考；双空返回空状态。",
    )


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


def lab09(case):
    k = [3, 8, 17, 33][case]
    a = [[(r + 1) * (i % 5 - 2) for i in range(k)] for r in range(3)]
    x = [i % 3 - 1 for i in range(k)]
    lanes = min(k, 8)
    partial = [[sum(row[i] * x[i] for i in range(denominator, k, lanes)) for denominator in range(lanes)] for row in a]
    out = [sum(p) for p in partial]
    ref = [dot(row, x) for row in a]
    assert out == ref
    return result(
        f"M=3,K={k}，展示 {lanes} 个逻辑 lane",
        [
            step("向量 x", x, "为便于显示使用最多 8 个逻辑 lane；不是修改 CUDA warp 宽度。"),
            step("各行 lane partial", partial, "同一行内归约，不跨行混合。"),
            step("每行结果", out, "只有该行 owner 写出。"),
        ],
        "分工归约与独立点积相同，包括非整除 K。",
    )


def lab10(case):
    bk = [1, 2, 3, 4][case]
    a, b = [[1, 2, 3, 4, 5], [2, 0, 1, -1, 3]], [[1, 2], [0, 1], [2, 0], [1, -1], [3, 2]]
    c = [[0, 0], [0, 0]]
    steps = [step("输入 A/B", {"A": a, "B": b}, "A[2,5] @ B[5,2]；K=5 用于覆盖尾块。")]
    for start in range(0, 5, bk):
        for i in range(2):
            for j in range(2):
                c[i][j] += sum(a[i][k] * b[k][j] for k in range(start, min(start + bk, 5)))
        steps.append(
            step(
                f"K tile [{start},{min(start + bk, 5)}) 后 C",
                [row[:] for row in c],
                "输出累加器跨 K tile 保留；尾部只计有效 k。",
            )
        )
    assert c == matmul(a, b)
    return result(f"BK={bk}，K=5", steps, "每种 BK 都与完整矩阵乘一致。")


def lab11(case):
    count, load, compute = 4, [1, 3, 5, 8][case], 4
    serial = count * (load + compute)
    ideal = load + compute + (count - 1) * max(load, compute)
    timeline = []
    for t in range(count):
        start = load + t * max(load, compute)
        timeline.append([t, start, start + compute])
    assert ideal <= serial and timeline[-1][-1] == ideal
    v = 1.0003
    rounded = round(v * 1024) / 1024  # local binade [1,2), educational RNE
    return result(
        f"搬运 L={load}，计算 C={compute}，4 tiles",
        [
            step("串行时间", {"time": serial}, "单位是教学时间单位，不是 GPU 周期。"),
            step(
                "理想独立流水 compute 区间 [tile,start,end]", timeline, "假设可提前搬运、无资源争用；真实硬件需另测。"
            ),
            step(
                "精度与时间分别记录",
                {"ideal_time": ideal, "x": v, "10_fraction_bit_value": rounded, "input_error": rounded - v},
                "该舍入示例只覆盖 [1,2)；不是完整 TF32 编码器。",
            ),
        ],
        "理想时间线符合推导，且未将模型速度当实测。",
    )


def mma_coords(lane):
    g, t = divmod(lane, 4)
    return [[g, 2 * t], [g, 2 * t + 1], [g + 8, 2 * t], [g + 8, 2 * t + 1]]


def lab12(case):
    lane = [0, 5, 16, 31][case]
    all_coords = [tuple(c) for denominator in range(32) for c in mma_coords(denominator)]
    assert len(set(all_coords)) == 128 and set(all_coords) == {(i, j) for i in range(16) for j in range(8)}
    board = [["·"] * 8 for _ in range(16)]
    for i, j in mma_coords(lane):
        board[i][j] = lane
    return result(
        f"m16n8k16 FP32 C fragment，lane={lane}",
        [
            step("lane 的 4 个 C 坐标", mma_coords(lane), "仅适用本文指定的 PTX C/D 布局，不能套到 A/B。"),
            step("16×8 输出归属", board, "数字位置属于所选 lane，其余由其他 lane 持有。"),
            step(
                "全集覆盖",
                {"lanes": 32, "values_per_lane": 4, "unique_output_coordinates": 128},
                "穷举证明无遗漏无重叠。",
            ),
        ],
        "128 个输出坐标唯一覆盖；不是设备指令执行测试。",
    )


def lab13(case):
    row = case * 2
    physical = [c ^ row for c in range(8)]
    recovered = [c ^ row for c in physical]
    assert recovered == list(range(8))
    order = [(r, c) for g in range(0, 4, 2) for c in range(4) for r in range(g, g + 2)]
    assert len(set(order)) == 16
    return result(
        f"教学 XOR 行号={row}",
        [
            step("逻辑列 → 物理列", physical, "8×8 小模型；不声称这就是某个硬件 SW128。"),
            step("再异或一次", recovered, "可逆性保证地址不丢失，不保证实际事务无 bank conflict。"),
            step("grouped CTA 坐标顺序", order, "ID 映射不等于硬件承诺的执行顺序。"),
        ],
        "XOR 逆映射和 CTA 一一对应检查通过。",
    )


def protocol(stages, tiles):
    status = ["empty"] * stages
    snapshots = []
    for t in range(tiles):
        slot = t % stages
        phase = (t // stages) % 2
        assert status[slot] == "empty"
        for event in ["loading", "full", "reading", "empty"]:
            if event == "empty":
                assert status[slot] == "reading"
            status[slot] = event
            snapshots.append(
                step(
                    f"tile {t}: {event}",
                    {"slot": slot, "phase": phase, "states": status[:]},
                    "教学状态机：只有读者完成后才能重新写入；不是并发设备模拟器。",
                )
            )
    assert all(s == "empty" for s in status)
    return snapshots


def lab14(case):
    stages = [1, 2, 2, 3][case]
    tiles = [1, 3, 5, 7][case]
    return result(
        f"stages={stages},tiles={tiles}", protocol(stages, tiles), "每次覆盖都发生在 empty；绕环后 phase 正确翻转。"
    )


def lab15(case):
    consumers = [128, 256, 256, 384][case]
    rc = [128, 128, 192, 192][case]
    rp, p = 32, 128
    total = p * rp + consumers * rc
    budget = 65536
    return result(
        f"producer={p}×{rp},consumer={consumers}×{rc}",
        [
            step(
                "角色寄存器概念预算",
                {"producer": p * rp, "consumer": consumers * rc, "total": total},
                "未计分配粒度；预算 65536 仅为演示参数，不代表所有 GPU。",
            ),
            step(
                "是否超过教学预算",
                {"budget": budget, "fits": total <= budget, "remaining": budget - total},
                "超预算意味着需要改配置；不代表编译器必然采取哪种处理。",
            ),
            step(
                "buffer 协议仍需独立满足",
                {"producer": "wait empty → TMA", "consumer": "wait full → read/MMA → release"},
                "寄存器够用不保证没有同步死锁。",
            ),
        ],
        "按角色计算预算；超额配置明确标出，不当成可运行硬件配置。",
    )


def lab16(case):
    scores = [[0, 0], [1, 2, 3, 4], [0, -2, 8, 1], [5]][case]
    values = [2 + 2 * i for i in range(len(scores))]
    out, snap = online(scores, values, 2)
    close(out, dot(softmax(scores), values))
    return result(f"scores={scores},V={values}", snap, "每一个 KV 前缀状态与完整前缀 attention 相同。")


def lab17(case):
    rows = 8
    groups = [1, 2, 4, 8][case]
    owners = [r // (rows // groups) for r in range(rows)]
    assert all(owners.count(g) == rows // groups for g in range(groups))
    return result(
        f"8 query 行，{groups} 个逻辑计算组",
        [
            step("每行 owner", owners, "这表示逻辑分工，不能直接推断 CUDA warp 排布。"),
            step("每组状态行数", [owners.count(g) for g in range(groups)], "每行 m/denominator/u 只在一个组维护，无跨组行合并。"),
            step(
                "共享与私有",
                {"shared": "K/V tile", "private": "各组 Q 行及 m/denominator/u", "tradeoff": "更多组不减少输出元素总数"},
                "P fragment 仍要匹配 PV 输入布局。",
            ),
        ],
        "所有 query 行恰好属于一个组。",
    )


def lab18(case):
    sk, sv = [(1, 1), (2, 1), (2, 2), (3, 2)][case]
    snaps = []
    for t in range(5):
        snaps.append(
            step(
                f"KV tile {t} 的两套槽",
                {"K_slot": t % sk, "K_phase": (t // sk) % 2, "V_slot": t % sv, "V_phase": (t // sv) % 2},
                "QK 消费 K，softmax 后 PV 消费 V；释放时刻分别判断。",
            )
        )
    return result(f"K stages={sk},V stages={sv}", snaps, "两套环形编号按各自 stage 数独立更新。")


def lab19(case):
    count = [1, 2, 3, 5][case]
    scores = [i % 3 - 1 for i in range(count)]
    values = [i + 2 for i in range(count)]
    a = state(scores[::2], values[::2])
    b = state(scores[1::2], values[1::2])
    out = merge(a, b)[0]
    close(out, dot(softmax(scores), values))
    return result(
        f"{count} 个 KV tile，用单元素 tile 演示",
        [
            step("奇偶路由", {"A": list(range(0, count, 2)), "B": list(range(1, count, 2))}, "一组为空也必须结束。"),
            step("两组 O/LSE", {"A": [a[0], str(a[1])], "B": [b[0], str(b[1])]}, "每组有自己的分母。"),
            step("最终 merge", {"O": out, "reference": dot(softmax(scores), values)}, "不能直接把 O_A 与 O_B 相加。"),
        ],
        "单 tile、奇数 tile 与空 consumer 均正确。",
    )


def lab20(case):
    chunk = [1, 2, 3, 4][case]
    q = [1, 2, -1, 3, 2]
    keys = [[2, 0, 1, -1, 2], [0, 2, 1, 1, -1]]
    scores = [0.0, 0.0]
    snaps = []
    for start in range(0, len(q), chunk):
        for j, k in enumerate(keys):
            scores[j] += dot(q[start : start + chunk], k[start : start + chunk])
        snaps.append(
            step(f"D chunk [{start},{min(start + chunk, len(q))})", scores[:], "仍是 partial score；尚不可 softmax。")
        )
    ref = [dot(q, k) for k in keys]
    assert scores == ref
    p = softmax([s / math.sqrt(len(q)) for s in scores])
    snaps.append(step("完整 D 后的概率", p, "缩放使用完整 D，不是 chunk 宽度。"))
    return result(f"D=5,D_chunk={chunk}", snaps, "所有 chunk 方案等于完整点积，尾 chunk 正确。")


def lab21(case):
    sr, sc = [(3, 1), (4, 1), (1, 2), (0, 1)][case]
    offsets = [[r * sr + c * sc for c in range(3)] for r in range(2)]
    flat = sum(offsets, [])
    unique = len(set(flat))
    return result(
        f"shape=(2,3),stride=({sr},{sc})",
        [
            step("坐标 → offset", offsets, "shape 定义合法坐标，stride 决定地址。"),
            step(
                "覆盖情况",
                {"logical_elements": 6, "unique_offsets": unique, "span": max(flat) + 1, "aliases": 6 - unique},
                "alias 布局不能直接当唯一写者的 copy 输出布局。",
            ),
        ],
        "枚举直接显示连续、padding、列主序和别名四种情形。",
    )


def lab22(case):
    n = [1, 29, 32, 33][case]
    threads, width = 8, 4
    rounds = (n + threads * width - 1) // (threads * width)
    seen = []
    snaps = []
    for r in range(rounds):
        rows = []
        for t in range(threads):
            ids = [r * threads * width + t * width + v for v in range(width)]
            seen.extend(i for i in ids if i < n)
            rows.append([i if i < n else "mask" for i in ids])
        snaps.append(step(f"round {r}: thread × value", rows, "每行是一条逻辑线程的四个值；mask 只排除无效元素。"))
    assert sorted(seen) == list(range(n)) and len(seen) == len(set(seen))
    return result(f"N={n},8 逻辑线程×4 values", snaps, "每个有效坐标恰好搬一次，包括第二轮尾部。")


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


def lab24(case):
    row = [0, 1, 3, 7][case]
    source = list(range(8))
    physical = [None] * 8
    for col, value in enumerate(source):
        physical[col ^ row] = value
    restored = [physical[col ^ row] for col in range(8)]
    assert restored == source
    return result(
        f"教学 swizzle：row={row}",
        [
            step("global 逻辑顺序", source, "本模型不是某个硬件 descriptor 的逐位实现。"),
            step("shared 物理顺序", physical, "生产者按映射存储。"),
            step("消费者按匹配布局读取", restored, "错误地按线性顺序读会看到第二步的置换。"),
        ],
        "匹配布局 round-trip 逐元素相同。",
    )


def lab25(case):
    bias = [-10, -2, 0, 5][case]
    a = [[1, 2], [3, 4]]
    b = [[2, -1], [1, 2]]
    c = matmul(a, b)
    out = [[max(0, v + bias) for v in row] for row in c]
    expected = [[max(0, 4 + bias), max(0, 3 + bias)], [max(0, 10 + bias), max(0, 5 + bias)]]
    assert out == expected
    return result(
        f"GEMM + bias({bias}) + ReLU",
        [
            step("GEMM 累加器", c, "先验证纯 GEMM。"),
            step("bias 后", [[v + bias for v in row] for row in c], "此小例子用同一 bias，GPU 练习要求逐列 bias。"),
            step("ReLU 后", out, "激活属于 epilogue，必须计入所需延迟。"),
        ],
        "完整 epilogue 与手算四个输出一致。",
    )


def lab26(case):
    scores = [-2, 1, 0, 4, 3][: case + 2]
    values = list(range(2, 2 + len(scores)))
    direct = dot(softmax(scores), values)
    one, _ = online(scores, values, 2)
    two = merge(state(scores[::2], values[::2]), state(scores[1::2], values[1::2]))[0]
    close(direct, one)
    close(direct, two)
    return result(
        f"N={len(scores)}，同一输入三种数学组织",
        [
            step("完整参考", {"O": direct}, "materialize 的数学参考。"),
            step("单状态在线", {"O": one}, "对应统一/WS 流水共同的数学状态。"),
            step("双状态 + merge", {"O": two}, "这里只验证组织等价，不模拟 CUDA 性能。"),
        ],
        "三种组织与独立完整参考对齐。",
    )


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


def lab31(case):
    scales = [(1, 1), (1, 4), (2, 0.5), (0.25, 8)][case]
    parts = [3, 5]
    correct = sum(p * s for p, s in zip(parts, scales))
    wrong = sum(parts) * scales[0]
    explicit = dot([1 * scales[0], 2 * scales[0], 2 * scales[1], 3 * scales[1]], [1, 1, 1, 1])
    close(correct, explicit)
    return result(
        f"两个 K 组 scale 乘积={scales}",
        [
            step("未缩放 partial", parts, "每个 partial 属于不同 K 组。"),
            step(
                "分别缩放后相加",
                {"correct": correct, "dequant_reference": explicit},
                "K 方向变化的 scale 不能移到全部归约之后。",
            ),
            step(
                "错误外提的反例",
                {"wrong": wrong, "difference": wrong - correct},
                "scale 相同的 case 不会暴露错误，必须测试不同 scale。",
            ),
        ],
        "按组缩放与显式反量化点积一致。",
    )


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


def lab33(case):
    sf = [1, 2, 4, 8]
    shift = case
    data = [1, 2, 3, 4]
    aligned = sum(16 * x * s for x, s in zip(data, sf))
    misplaced = sum(16 * x * sf[(g + shift) % 4] for g, x in enumerate(data))
    close(aligned, 784)
    return result(
        f"把 SF 错移 {shift} 组，观察错误",
        [
            step(
                "data / scale 按组",
                {"data_values": data, "scale": sf, "elements_per_group": 16},
                "每组重复同一个 data 值，方便手算。",
            ),
            step("正确配对", {"C": aligned}, "sum_g(16*data_g*scale_g)。"),
            step("错位 SF", {"C_wrong": misplaced, "error": misplaced - aligned}, "只改 scale 位置就足以破坏结果。"),
        ],
        "正确结果 784；非均匀测试能揭露 scale 错位。",
    )


def lab34(case):
    scores = [[0, 0, 0, 0], [0, 0, 0, 8], [1, 2, 3, 4], [-10, 0, 10, 1]][case]
    values = [1, 3, -2, 6]
    p = softmax(scores)
    out = dot(p, values)
    sq, _, _ = quant8(scores)
    vq, _, _ = quant8(values)
    pq = softmax(sq)
    output_qk = dot(pq, values)
    output_v = dot(p, vq)
    ess = 1 / sum(x * x for x in p)
    assert 1 - 1e-9 <= ess <= len(p) + 1e-9
    return result(
        f"scores={scores}",
        [
            step("高精度概率与 ESS", {"p": p, "ESS": ess, "O": out}, "均匀时 ESS 大，尖峰时接近 1。"),
            step(
                "只量化分数",
                {"scores_hat": sq, "O": output_qk, "error": output_qk - out},
                "为隔离机制直接量化 score；不冒充真实 Q/K 的全部误差。",
            ),
            step("只量化 V", {"V_hat": vq, "O": output_v, "error": output_v - out}, "各阶段单独注入误差，再研究组合。"),
        ],
        "ESS 在合法范围；分别记录两种误差来源。",
    )


def lab35(case):
    bias = [0, 3, 10, 100][case]
    scores = [1 + bias, 2 + bias, -1 + bias]
    v = [2, 5, 8]
    mu = sum(v) / len(v)
    a = dot(softmax(scores), v)
    b = dot(softmax([s - bias for s in scores]), [x - mu for x in v]) + mu
    close(a, b)
    return result(
        f"每行 logit 公共偏置={bias}",
        [
            step("平滑前", {"scores": scores, "V": v, "O": a}, "这里把 K 平滑造成的行常数直接展示为 score 偏置。"),
            step(
                "平滑后",
                {"scores": [s - bias for s in scores], "V_centered": [x - mu for x in v]},
                "V 中心化后输出需加回均值。",
            ),
            step("修正输出", {"mean_V": mu, "O_corrected": b}, "未量化代数等价；加入量化后需另测误差。"),
        ],
        "行常数消除与 V 均值加回保持完整输出。",
    )


def lab36(case):
    scale = [0.25, 1, 2, 4][case]
    raw = [-1, 0, 2]
    v = [1, 4, 8]
    correct = dot(softmax([s * scale for s in raw]), v)
    wrong = dot(softmax(raw), v) * scale
    exp2 = [2 ** ((s - max(raw)) * scale * math.log2(math.e)) for s in raw]
    via_exp2 = dot([p / sum(exp2) for p in exp2], v)
    close(correct, via_exp2)
    return result(
        f"QK scale 乘积 / sqrt(D)={scale}",
        [
            step("正确 score 缩放", [s * scale for s in raw], "先恢复 logits 尺度，再 softmax。"),
            step("自然指数与 exp2", {"O_exp": correct, "O_exp2": via_exp2}, "log2(e) 只转换指数底，不改变数学。"),
            step(
                "错误把 scale 移出 softmax",
                {"wrong": wrong, "difference": wrong - correct},
                "scale=1 恰好不暴露错误，必须多 case。",
            ),
        ],
        "exp 与 exp2 路径等价；演示非法外提反例。",
    )


def lab37(case):
    d = [128, 256, 512, 1024][case]
    a = 128 * d // 256
    b = 64 * d // 256
    assert a == 2 * b
    return result(
        f"Dv={d},consumer_threads=256",
        [
            step("Br=128 的累加器下界", {"values_per_thread": a}, "忽略全部其它临时寄存器。"),
            step("Br=64 的累加器下界", {"values_per_thread": b}, "下降来自实际输出块减小。"),
            step(
                "代价表",
                {"Q_CTA_count_ratio": 2, "new_cost": "跨组统计 / P shared 交接 / 更多任务"},
                "仅更改布局名称并不改变元素总数。",
            ),
        ],
        "寄存器逻辑下界按真实 tile 计算，代价单独列出。",
    )


def lab38(case):
    a = [0, 1, 3, 10][case]
    q = [a + 1, a + 2]
    k = [[1, 3], [2, -1], [4, 2]]
    b = [sum(row[d] for row in k) / len(k) for d in range(2)]
    qp = [v - a for v in q]
    kp = [[v - b[d] for d, v in enumerate(row)] for row in k]
    original = [dot(q, row) for row in k]
    partial = [dot(qp, row) for row in kp]
    delta = [dot([a, a], row) for row in kp]
    fixed = [s + t for s, t in zip(partial, delta)]
    for x, y in zip(softmax(original), softmax(fixed)):
        close(x, y)
    return result(
        f"Q 块均值示例 a=[{a},{a}]",
        [
            step("原始 scores", original, "单 query，D=2；为专注恒等式省略共同 sqrt(D)。"),
            step("平滑 Q/K 后", {"scores_prime": partial, "deltaS": delta}, "deltaS 随 key 变化，不能消掉。"),
            step(
                "概率对照",
                {"original": softmax(original), "without_delta": softmax(partial), "with_delta": softmax(fixed)},
                "只差每行常数时概率相同。",
            ),
        ],
        "补回 deltaS 后概率等价；非零均值可暴露漏补错误。",
    )


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


def lab40(case):
    prep = [0, 0.2, 0.4, 0.8][case]
    kernel_speedup = 2
    total = prep + (1 - prep) / kernel_speedup
    samples = [total * x for x in [1.02, 0.99, 1, 1.05, 1.01]]
    close(1 / total, 1 / (prep + (1 - prep) / 2))
    return result(
        f"未加速时间占比 f={prep}",
        [
            step("基线总时间=1", {"prepare": prep, "kernel": 1 - prep}, "所有时间是构造数据，不是设备 benchmark。"),
            step(
                "主 kernel 加速 2 倍后",
                {"prepare": prep, "kernel": (1 - prep) / 2, "total": total, "e2e_speedup": 1 / total},
                "不能把 kernel-only 的 2 倍写成端到端 2 倍。",
            ),
            step(
                "构造的重复测量",
                {"samples": samples, "median": statistics.median(samples), "minimum": min(samples)},
                "真实实验保存完整分布与环境。",
            ),
        ],
        "端到端加速符合 Amdahl 推导；示例时间明确标为构造。",
    )


LABS = {n: globals()[f"lab{n:02}"] for n in range(4, 41)}


def run(chapter, case=0):
    if chapter not in LABS or case not in range(4):
        raise ValueError("chapter must be 4..40; case must be 0..3")
    return LABS[chapter](case)


def self_test():
    count = 0
    for chapter in LABS:
        for case in range(4):
            report = run(chapter, case)
            assert report["steps"] and report["check"]
            json.dumps(report, allow_nan=False)
            count += 1
    # Format boundaries and ties, independently specified expected codes.
    assert FP8[1] == 2**-9 and FP8[56] == 1 and FP8[126] == 448
    assert nearest(1.0625, FP8) == 56
    assert nearest(1.1875, FP8) == 58
    assert nearest(0.75, FP4) == 2
    assert nearest(1.25, FP4) == 2
    for n in [1, 2, 15, 16, 17, 33]:
        _, _, codes, packed = quant4([(-1) ** i * (i % 7) for i in range(n)])
        assert [v for p in packed for v in (p & 15, p >> 4)][:n] == codes
    print(f"PASS: {count} CPU scenarios + format/packing boundary checks; GPU NOT RUN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter", type=int, default=4)
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        print(json.dumps(run(args.chapter, args.case), ensure_ascii=False, indent=2, allow_nan=False))
