---
title: CUDA 算子学习（三十六）：FP8 Attention 的前处理链
description: 融合、平滑、量化及 scale 生成。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-fp8-attention-preprocess/
date: 2026-11-02 17:00:35
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-quantized-attention-math/) · 下一篇：[（三十七）FP8 persist-D 与 scale 折叠](/llm/cuda-fp8-persist-scale/)

主 attention kernel 运行前，Q/K/V 可能已被平滑、旋转、量化和重排。前处理既决定精度，也决定短序列端到端成本，不能只优化最后的矩阵乘。

本篇接续系列第 36 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 35 章（PDF 阅读器第 511–523 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** 融合、平滑、量化及 scale 生成。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| smooth-K | 减去 key 方向公共均值向量 |
| Hadamard 旋转 | 成对作用于 Q/K 的正交变换，用于分散异常通道 |
| V 转置量化 | 为 PV 消费布局组织 V，同时生成 scale |
| 融合前处理 | 在一次遍历中做多项操作，减少中间读写 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 哪些变换保持数学
K 减公共均值不改变 softmax。Q/K 同乘正交矩阵 H，因 H@H.T=I，点积保持。V 减通道均值 mu_V 后，利用每行概率和为 1，有 P@(V-mu_V)=P@V-mu_V，输出需加回 mu_V。所有不变式都在量化前的实数计算中成立，量化后仍有误差。

### 格式和布局一起输出
前处理不仅输出低精度值，还输出 scale、padding 后形状与必要的均值。V 转置后量化组的含义可能变化，必须明确是沿 key、通道还是块分组，消费端采用相同约定。

### 融合先看依赖
全序列均值未得到前，不能随意在一个普通 CTA 中让所有 block 使用最终均值。多 kernel 基线先统计再转换；融合需要限定同步范围或重新组织算法。测量输出所有中间量的基线后，再减少不必要的落盘。

## 公式：从定义到可核对的小例子

```text
(Q@H)@(K@H).T = Q@K.T，H@H.T=I
softmax(Q@(K-mu_K).T)=softmax(Q@K.T)
P@(V-mu_V)+mu_V=P@V
T_e2e=T_statistics+T_quantize_layout+T_attention+T_epilogue
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-fp8-attention-preprocess.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-fp8-attention-preprocess.svg" alt="FP8 Attention 的前处理链的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-fp8-attention-preprocess.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**统计 K/V 均值与量化组 → 平滑 / 可选正交旋转 → 量化并生成 scale → V 转置与 padding → 主 kernel → 必要修正**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/35/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`每行 logit 公共偏置=0`。到最后一步应满足：行常数消除与 V 均值加回保持完整输出。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/35/cuda-fp8-attention-preprocess.zip) · [独立 CPU 源码](/downloads/leetcuda/35/demo.py) · [运行说明](/downloads/leetcuda/35/README.txt) · [练习说明](/downloads/leetcuda/35/EXERCISES.txt)

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


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab35(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** K 平滑后 attention；均值与量化融合、V 转置是扩展题。

```bash
python run_gpu.py --chapter 35 --backend triton
python run_gpu.py --chapter 35 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/vipshop/ffpa-attn/blob/861d75ee87824a14a9ebfe295d54c7dac3e3cabc/csrc/cuffpa/cute/fp8/quantize_fp8.cuh)。Linux NVIDIA SM120 环境；先按固定版本 README 准备 CUDA PyTorch、Python≥3.10、构建依赖和 CUDA 扩展。此入口运行整条量化 attention，前处理、主循环、误差报告一起观察。大 D 路径是否编入以构建的 headdim 配置为准。

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

**Triton 任务：** 融合前处理与多 kernel 基线。

**TileLang 任务：** 归约、转置与量化的 tile 融合。

Triton 先分别写均值、平滑、量化；TileLang 尝试把局部变换与转置融合。分别验证均值、重建 Q/K/V、最终 O；同时输出预处理占总时间比例。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 35 --backend triton --student student_triton
python run_gpu.py --chapter 35 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 35 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 各中间量均可核验，先测端到端占比再优化主 kernel。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

不能把预处理好的输入复用多次的时间，直接当成每次输入都变化的端到端延迟。旋转的维度和 padding 必须遵守正交变换约定。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（三十七）FP8 persist-D 与 scale 折叠](/llm/cuda-fp8-persist-scale/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
