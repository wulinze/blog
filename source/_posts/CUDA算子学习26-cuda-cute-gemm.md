---
title: CUDA 算子学习（二十六）：布局驱动的 GEMM 工程
description: 把数据布局、搬运划分、计算划分组合起来。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-cute-gemm/
date: 2026-11-02 17:00:25
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-cute-tma-layout/) · 下一篇：[（二十七）Attention 实现对照：统一流水、角色分工与双 consumer](/llm/cuda-cute-attention-variants/)

本章把前四篇 CuTe 抽象接成完整 GEMM。建议先跑通单 stage，再加入异步搬运，最后优化写回。每一步只改变一类机制，才能知道错误出在哪一层。

本篇接续系列第 26 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 25 章（PDF 阅读器第 362–377 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** 把数据布局、搬运划分、计算划分组合起来。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| 主循环 | 沿 K 搬运并累加 C 的过程 |
| epilogue | 完成缩放、偏置、激活或类型转换后写 C |
| 布局契约 | global、shared、fragment 三种映射的衔接 |
| staging | 为合并写回而临时存放输出 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 三种对象各有职责
Layout 描述坐标；Tensor 将布局绑到存储；TiledCopy 分配搬运；TiledMMA 分配计算。主循环先取 global tile，再按 copy 分工放入 shared，等待就绪后读取 operand fragment，gemm 累加到 C fragment。

### 流水协议独立于类型推导
布局正确不会自动保证生命周期正确。下一块的 producer 写入不能覆盖当前消费者仍在读的 shared；寄存器双缓冲也要防止覆盖尚未使用的 operand。类型系统能帮助检查形状，不能替代运行时同步证明。

### Epilogue 也要校验
C fragment 的持有顺序常不适合直接合并写出。可经 shared staging 重新组织，再向量化写回。加入 bias/activation 会改变数据读取和精度，计时必须包含实际部署的完整 epilogue。先与纯 GEMM 对齐，再单独打开每项。

## 公式：从定义到可核对的小例子

```text
C = alpha*(A@B) + beta*C_old + bias
若 activation=ReLU，则最后取 max(C,0)
本学习基线 alpha=1,beta=0，bias 按输出列广播
例：dot=11,bias=-12 → ReLU 输出 0
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-cute-gemm.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-cute-gemm.svg" alt="布局驱动的 GEMM 工程的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-cute-gemm.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**构造 global / shared layouts → TiledCopy 搬入并等待 → TiledMMA 沿 K 累加 → epilogue 缩放 / bias → staging → 合并写回**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/25/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`GEMM + bias(-10) + ReLU`。到最后一步应满足：完整 epilogue 与手算四个输出一致。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/25/cuda-cute-gemm.zip) · [独立 CPU 源码](/downloads/leetcuda/25/demo.py) · [运行说明](/downloads/leetcuda/25/README.txt) · [练习说明](/downloads/leetcuda/25/EXERCISES.txt)

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


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def matmul(a, b):
    return [[dot(row, col) for col in zip(*b)] for row in a]


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab25(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** GEMM + 逐列 bias + ReLU；staging 和向量写回是扩展题。

```bash
python run_gpu.py --chapter 25 --backend triton
python run_gpu.py --chapter 25 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/xlite-dev/LeetCUDA/blob/3b52c5cd3332c5e992c7aef63087d43f67f23b8e/kernels/interview/book/tests/ch24_hgemm_cute.cu)。CUDA C++20、NVCC、对应目标 GPU 与 CUTLASS submodule。TMA 相关例子还需满足该版本头文件的 CUDA toolkit 要求。

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git LeetCUDA-lab
git -C LeetCUDA-lab checkout --detach 3b52c5cd3332c5e992c7aef63087d43f67f23b8e
git -C LeetCUDA-lab submodule update --init --recursive
python3 run_upstream.py --repo ./LeetCUDA-lab --arch sm_80
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** 有 epilogue 的矩阵乘，完整基线对照。

**TileLang 任务：** 共享内存布局 + 流水 + 向量写回。

Triton tl.dot 后融合列 bias+ReLU；TileLang T.gemm 后显式 epilogue。先测非方阵和尾块，再比较是否 staging、不同 stages；保留原始输入，验证 beta 非零时没有错误覆盖。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 25 --backend triton --student student_triton
python run_gpu.py --chapter 25 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 25 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 核心、epilogue 与尾块逐层校验，源码/配置/版本一起保存。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

不要用某一份编译配置的良好结果宣称 CuTe 总比手写快。对照需要相同 dtype、累加类型、形状、tile 和计时范围。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（二十七）Attention 实现对照：统一流水、角色分工与双 consumer](/llm/cuda-cute-attention-variants/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
