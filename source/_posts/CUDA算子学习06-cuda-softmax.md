---
title: CUDA 算子学习（六）：稳定 Softmax：从整行到在线状态
description: 减最大值，推导分块 max 与 sum 的重缩放。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-softmax/
date: 2026-11-02 17:00:05
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-vector-atomic/) · 下一篇：[（七）LSE 与注意力状态合并](/llm/cuda-lse-merge/)

Softmax 的难点不是 exp 的写法，而是整行共享一个分母。先解决大分数溢出，再解决整行装不进一个 tile 的问题。在线状态能跨 tile 更新，但最终输出每个概率仍需要保留输入或重新读取输入。

本篇接续系列第 6 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 5 章（PDF 阅读器第 60–69 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** 减最大值，推导分块 max 与 sum 的重缩放。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| logit | 归一化前的实数分数 |
| 稳定 softmax | 先减行最大值再求指数和 |
| online 状态 | 已处理部分的最大值 m 和相对指数和 l |
| mask | 把不参与概率分布的位置排除，不能简单当成分数 0 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 为什么减最大值不改变概率
分子与分母同时乘 exp(-m)，比值不变。令 m 为最大 logit 后，所有指数的自变量不大于 0。比如 [1000,1001,1002] 直接求 exp 会溢出，而减去 1002 后变为 [-2,-1,0]，概率约 [0.0900,0.2447,0.6652]。

### 在线更新要重标定旧分母
旧状态的 l 以旧 m 为基准，新块最大值若更大，旧指数和也要乘 `exp(m_old-m_new)`。新块按 m_new 求指数和后相加。空状态单独处理为 m=-inf、l=0，避免计算 -inf-(-inf)。合并顺序在实数代数中等价，浮点结果允许有舍入差异。

### 两次归约不等于两次 DRAM 读取
短行可以先把全部输入载入寄存器，再求 max、sum、写结果，多个归约并不意味着重新访问 global。长行的在线第一遍只求状态，第二遍才用最终状态写概率。是否划算要看寄存器、行长度、行数和实际流量，而不是只数数学扫描次数。

## 公式：从定义到可核对的小例子

```text
m_new = max(m_old, max(block))
alpha = exp(m_old-m_new)
l_new = alpha*l_old + sum(exp(x-m_new) for x in block)
p_i = exp(x_i-m_final)/l_final
例：[1,2] 后接 [3]，旧 m=2，旧 l=1+exp(-1)
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-softmax.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-softmax.svg" alt="稳定 Softmax：从整行到在线状态的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-softmax.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**读一块 logits → 求块最大值 → 更新 m，缩放旧 l → 累加新指数和 → 最终状态 → 概率输出**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/05/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`logits=[1, 2, 3, 4]，block=1`。到最后一步应满足：行和为 1；加常数不改变概率；每块在线分母通过独立参考。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/05/cuda-softmax.zip) · [独立 CPU 源码](/downloads/leetcuda/05/demo.py) · [运行说明](/downloads/leetcuda/05/README.txt) · [练习说明](/downloads/leetcuda/05/EXERCISES.txt)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab05(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** 短行稳定 softmax；长行 online 两遍实现是扩展题。

```bash
python run_gpu.py --chapter 5 --backend triton
python run_gpu.py --chapter 5 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/xlite-dev/LeetCUDA/blob/3b52c5cd3332c5e992c7aef63087d43f67f23b8e/kernels/interview/book/tests/ch04_softmax.cu)。CUDA C++20、NVCC、对应目标 GPU 与 CUTLASS submodule。TMA 相关例子还需满足该版本头文件的 CUDA toolkit 要求。

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git LeetCUDA-lab
git -C LeetCUDA-lab checkout --detach 3b52c5cd3332c5e992c7aef63087d43f67f23b8e
git -C LeetCUDA-lab submodule update --init --recursive
python3 run_upstream.py --repo ./LeetCUDA-lab --arch sm_80
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** 单行 softmax；长行分块状态。

**TileLang 任务：** reduce_max / reduce_sum；online 两状态。

Triton 用 tl.max/tl.sum 做单行版本，尾部载入 -inf；TileLang 用二维 fragment 和 dim=1 归约。先验证宽度 1、17、127、129，再把极大偏置加到输入上，检查概率不变。扩展实现长行两遍 online，并把第二遍成本算入时间。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 5 --backend triton --student student_triton
python run_gpu.py --chapter 5 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 5 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 极大输入不溢出，行和接近 1；空行与全 mask 行有明确约定。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

本 demo 约定全 mask 行输出全零，普通有效行必须有限且行和约为 1。该约定需要接口显式说明，不能让全 -inf 行自然产生 NaN。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（七）LSE 与注意力状态合并](/llm/cuda-lse-merge/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
