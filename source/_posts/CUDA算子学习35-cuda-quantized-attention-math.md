---
title: CUDA 算子学习（三十五）：量化 Attention：误差在哪里被放大
description: Q/K/V/P 的误差传播、平滑与不变性。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-quantized-attention-math/
date: 2026-11-02 17:00:34
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-block-scaled-mma/) · 下一篇：[（三十六）FP8 Attention 的前处理链](/llm/cuda-fp8-attention-preprocess/)

量化 attention 有多处损失来源，最终 allclose 只告诉你结果是否越过阈值。要学习如何改进，需分别测 QK、P、V 和累加的贡献，尤其检查 causal 早期行与尖峰分布。

本篇接续系列第 35 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 34 章（PDF 阅读器第 498–510 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** Q/K/V/P 的误差传播、平滑与不变性。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| logits 误差 | 量化 Q/K 后分数矩阵的误差 |
| 概率误差 | 分数扰动经 softmax 后的误差 |
| 输出误差 | 概率、V 量化和累加共同产生的最终误差 |
| ESS | 有效样本数，1/sum(p²)，描述权重是否集中 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### Softmax 对误差的敏感性依赖概率
小扰动下 delta_p_i≈p_i*(delta_s_i-sum(p*delta_s))。给整行分数加常数会抵消，改变相对分数才会改变概率。概率集中时输出更依赖少数 V，不能假设大量 key 总能平均掉误差。

### 平滑 K 的精确不变性
令 K′=K-mu_K，其中 mu_K 在 key 行方向是同一个向量。Q@K′.T=Q@K.T-Q@mu_K，每个 query 行减去同一标量，所以 softmax 不变。若还要输出 LSE，必须补回该行常数。

### 逐阶段注入量化
先只量化 Q/K，记录 logits 与概率差；再只量化 V；然后量化 P；最后组合。它们的误差可能相互放大或抵消，不能简单把各 max_abs 相加称为真实总误差。模型任务质量还需要超出单 kernel 的验证。

## 公式：从定义到可核对的小例子

```text
p=softmax(s)
delta_p_i≈p_i*(delta_s_i-sum_j(p_j*delta_s_j))
ESS=1/sum(p_i²)：均匀 n 项时为 n，one-hot 时为 1
O=p@V；比较 max_abs、相对 L2、逐行分布与失败输入
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-quantized-attention-math.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-quantized-attention-math.svg" alt="量化 Attention：误差在哪里被放大的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-quantized-attention-math.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**高精度 QK / P / O 参考 → 只量化 QK → 检查 logits → 只量化 V / P → 检查贡献 → 组合量化 → 逐行误差 → 尖峰 / causal / 长序列验收**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/34/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`scores=[0, 0, 0, 0]`。到最后一步应满足：ESS 在合法范围；分别记录两种误差来源。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/34/cuda-quantized-attention-math.zip) · [独立 CPU 源码](/downloads/leetcuda/34/demo.py) · [运行说明](/downloads/leetcuda/34/README.txt) · [练习说明](/downloads/leetcuda/34/EXERCISES.txt)

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


def quant8(x):
    assert all(math.isfinite(v) for v in x)
    scale = max(map(abs, x), default=0) / 448 or 1.0
    codes = [nearest(abs(v) / scale, FP8) | (128 if math.copysign(1, v) < 0 else 0) for v in x]
    restored = [(-1 if c & 128 else 1) * FP8[c & 127] * scale for c in codes]
    return restored, scale, codes


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab34(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** 单独量化 V 的 attention，帮助定位输出误差；多阶段开关是扩展题。

```bash
python run_gpu.py --chapter 34 --backend triton
python run_gpu.py --chapter 34 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/vipshop/ffpa-attn/blob/861d75ee87824a14a9ebfe295d54c7dac3e3cabc/csrc/cuffpa/cute/fp8/fp8_pscale.cuh)。Linux NVIDIA SM120 环境；先按固定版本 README 准备 CUDA PyTorch、Python≥3.10、构建依赖和 CUDA 扩展。此入口运行整条量化 attention，前处理、主循环、误差报告一起观察。大 D 路径是否编入以构建的 headdim 配置为准。

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

**Triton 任务：** 高精度参考对照分步量化误差。

**TileLang 任务：** 按阶段插入量化并保留误差日志。

两语言均提供开关只量化 Q/K、V 或 P，每组保存中间输出。输入包含均匀分数、单个尖峰、全负、长序列；输出按行报告误差，特别检查 causal 第一行。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 34 --backend triton --student student_triton
python run_gpu.py --chapter 34 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 34 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 区分 logits、概率、输出误差；极端分布/长序列都要覆盖。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

放宽容差不能修复布局错误。量化可接受的阈值来自应用需求与数据分布，不能照抄书中某张卡某组随机输入的 allclose 阈值。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（三十六）FP8 Attention 的前处理链](/llm/cuda-fp8-attention-preprocess/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
