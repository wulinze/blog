---
title: CUDA 算子学习（四十）：FP4 Attention：P 的两级量化
description: online 状态、P 量化与输出缩放的交互。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-fp4-attention-probability/
date: 2026-11-02 17:00:39
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-fp4-attention-preprocess/) · 下一篇：[（四十一）基准与精度：把学习结果变成可复现实验](/llm/cuda-benchmark-accuracy/)

P 在主循环中即时生成，分布从接近均匀到接近 one-hot 都可能出现。FP4 P 量化不仅舍入数据，还会舍入局部 scale。两级方案把局部 scale 放到更有分辨率的范围，但仍需验证整个 online 状态。

本篇接续系列第 40 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 39 章（PDF 阅读器第 554–560 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** online 状态、P 量化与输出缩放的交互。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| 未归一化 P | exp(score-running_max)，还未除最终 l |
| 一级 P scale | 把 P 的范围扩展到 data×SF 的联合域 |
| 二级 P scale | 每组局部 FP8 scale，再配合 E2M1 数据 |
| 在线重标定 | running max 变化时同步调整旧分子与分母 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 联合量程为什么出现 2688
E2M1 的最大有限幅度为 6，E4M3FN scale 的最大有限值为 448，乘积为 2688。以当前 running max 为基准，当前 tile 的指数值最多为 1，但不保证一定有值等于 1。把未归一化 P 乘 2688 后，再进行每组 data/SF 量化。

### 分母的定义不能悄悄变化
精确 softmax 的 l 来自未量化指数和。如果分子使用量化后的 P，分母仍是精确 l，那么量化使输出成为近似；不能为了好看而改成量化 P 的和，却仍声称与原接口一致。若实现采用不同域的共同缩放，要明确写出分子分母的同一比例。

### 逐 tile 检查状态
每次 running max 改变，旧 u 和 l 按 alpha 重缩放，新 P 量化后进入 PV。保存每块后的 m/l/u、局部 scale、饱和与下溢数，可定位长序列累计误差。只测一个 tile 无法覆盖重缩放问题。

## 公式：从定义到可核对的小例子

```text
p=exp(score-m_new), 0≤p≤1
p_scaled=2688*p
block_scale=round_fp8(max(p_scaled)/6)
p_hat=decode_fp4(q)*block_scale/2688
u_new=alpha*u_old+p_hat@V；l_new=alpha*l_old+sum(p)
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-fp4-attention-probability.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-fp4-attention-probability.svg" alt="FP4 Attention：P 的两级量化的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-fp4-attention-probability.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**更新 running max 与 alpha → 生成未归一化 P → 2688 放大 → 局部 SF + FP4 → PV 累加并缩放旧 u → 使用定义一致的 l 归一化**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/39/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`scores=[0, 0, 0, 0]，每 tile 2 个 key`。到最后一步应满足：状态有限；记录每 tile 的 P 量化与最终参考误差。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/39/cuda-fp4-attention-probability.zip) · [独立 CPU 源码](/downloads/leetcuda/39/demo.py) · [运行说明](/downloads/leetcuda/39/README.txt) · [练习说明](/downloads/leetcuda/39/EXERCISES.txt)

解压后先运行 CPU demo，不需安装第三方库，Python 3.9 及以上即可：

```bash
python3 demo.py --case 0
python3 demo.py --case 3
```

程序打印与网页相同的逐步状态，并执行模型内的断言。下面是**完整可独立运行**的本篇 CPU 文件；它验证数值、索引或协议，不代替 GPU 实现。

<details>
<summary>展开完整 Python demo（可直接复制运行）</summary>

```python
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
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** E2M1 量化 P 的物化 attention；CPU demo 另验证在线两级尺度，专用 CUDA demo 使用硬件 FP4。

```bash
python run_gpu.py --chapter 39 --backend triton
python run_gpu.py --chapter 39 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/vipshop/ffpa-attn/blob/861d75ee87824a14a9ebfe295d54c7dac3e3cabc/csrc/cuffpa/cute/fp4/sm_120/persist_d.cuh)。Linux NVIDIA SM120 环境；先按固定版本 README 准备 CUDA PyTorch、Python≥3.10、构建依赖和 CUDA 扩展。此入口运行整条量化 attention，前处理、主循环、误差报告一起观察。大 D 路径是否编入以构建的 headdim 配置为准。

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
git clone https://github.com/vipshop/ffpa-attn.git ffpa-attn-lab
git -C ffpa-attn-lab checkout --detach 861d75ee87824a14a9ebfe295d54c7dac3e3cabc
git -C ffpa-attn-lab submodule update --init --recursive
cd ffpa-attn-lab
# 在已装好匹配 CUDA PyTorch 的隔离环境中安装项目依赖
python -m pip install packaging ninja wheel setuptools setuptools-scm matplotlib
FFPA_SKIP_CUDA_EXT=1 python -m pip install -e ".[bench]" --no-build-isolation
bash build.sh --arch sm_120a --ext all --headdim 128,512 --jobs 4
cd ..
python3 run_upstream.py --repo ./ffpa-attn-lab --arch sm_120a
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** 量化 P 的逐阶段误差定位。

**TileLang 任务：** block-scaled PV 与 rescale 的接口。

Triton 先软件量化 P 并用高精度 PV 定位误差；TileLang 再接 block-scaled PV。测试后续 tile 才出现大最大值、均匀长行、causal 首行与极小概率，报告每个阶段的误差。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 39 --backend triton --student student_triton
python run_gpu.py --chapter 39 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 39 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 各 KV tile 后验证状态；短序列正确不能代替长序列稳定性。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

2688 是这套格式与尺度设计的值，不是任意 FP4 格式通用常量。当前 tile 最大指数不一定为 1，不能每块重新假设它达满量程。

定位性能时，回到 [第二篇的完整分析流程](/systems/cuda-overlap/#performance-analysis)：先定位值得优化的 kernel，再看计算/存储吞吐、eligible warp、stall 与资源限制，最后针对本章机制做单变量实验。stall 比例不等于可直接消除的时间，低命中率也不自动意味着缺少复用。

## 验证状态与延伸阅读

| 项目 | 本次交付状态 |
| --- | --- |
| 本篇 CPU 的 4 组案例 | 已执行，数值/布局/协议断言通过 |
| 网页交互 | 使用已校验快照，支持切换案例、前进、后退与重置 |
| Python GPU 文件 | 完整源码与统一入口，已做语法/静态检查 |
| CUDA / DSL 后端编译 | 当前编辑环境无 NVIDIA GPU，未做对应目标编译 |
| GPU 正确性与性能 | 未实测；不提供虚构加速比 |
| 原书专用实现 | 固定源码版本与运行入口，按硬件条件复现 |

主要学习来源是用户提供的 LeetCUDA 文档及 [LeetCUDA 项目](https://github.com/xlite-dev/LeetCUDA)、[FFPA 项目](https://github.com/vipshop/ffpa-attn)。这是围绕原文主题重新组织的学习讲解与实验，不直接复制整章。硬件指令语义核对 [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)；布局抽象参考 [CUTLASS CuTe 文档](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cpp/cute)；两种 DSL 继续参考 [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html) 与 [TileLang 文档](https://tilelang.com/)。

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（四十一）基准与精度：把学习结果变成可复现实验](/llm/cuda-benchmark-accuracy/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
