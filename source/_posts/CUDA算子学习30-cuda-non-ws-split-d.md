---
title: CUDA 算子学习（三十）：大 D 的 non-WS 流水：让所有线程参与 MMA
description: chunk 解耦、跨 tile 预取、独立 barrier 集。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-non-ws-split-d/
date: 2026-11-02 17:00:29
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-persistent-attention/) · 下一篇：[（三十一）FP8 的格式与量化前处理](/llm/cuda-fp8-quantization/)

分工可以降低计算组的地址指令负担，也可能占用本来可用于 MMA 的线程和资源。non-WS 让更多线程参与计算，用明确的阶段协议安排搬运，适合与 WS 做同语义对照。

本篇接续系列第 30 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 29 章（PDF 阅读器第 429–445 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** chunk 解耦、跨 tile 预取、独立 barrier 集。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| non-WS | 本路径不专门预留一组只搬运的线程 |
| chunk 解耦 | QK 与 PV 使用各自 chunk 和 buffer 生命周期 |
| 独立 barrier 集 | 不同数据流不能意外共享完成计数 |
| 预取距离 | 发起搬运与真正消费之间的迭代间隔 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 全员计算仍需指定提交者
TMA 可以由选定线程提交，但其他线程并非整段 kernel 都处于 producer 角色。提交后大家继续各自计算。消费者必须等待对应数据，而提交者也不能阻塞在依赖自身未来动作的循环中。

### K/V 与 D chunk 的状态分开
一个 KV tile 内有多个 QK D chunk，之后还可能有多个 V 输出 chunk。slot 和 phase 应绑定实际的搬运序列编号，而不是随手复用外层 KV 索引。独立 barrier 集避免一个完成事件误唤醒另一数据流。

### More stages 的代价要量化
增加阶段可扩大预取窗口，也增加 shared、barrier 管理和临时寄存器。统一线程流中发射资源竞争依然存在。测量应同时记录 spill、等待原因与总时间，而不是只数活跃 warp。

## 公式：从定义到可核对的小例子

```text
load_sequence = kv_tile*chunks_per_tile + chunk_id
slot = load_sequence%stages
phase = floor(load_sequence/stages)%2
chunks 与 stages 不互为倍数时，更容易暴露错误相位
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-non-ws-split-d.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-non-ws-split-d.svg" alt="大 D 的 non-WS 流水：让所有线程参与 MMA的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-non-ws-split-d.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**选择提交线程并计算序号 → K chunk 发起 / 等待 → 所有计算线程做 QK → 独立 V 流水做 PV → 最后读完释放，预取后续**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/29/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`2 KV tiles,1 chunks,2 stages`。到最后一步应满足：跨 KV 边界的搬运序号与环形相位连续。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/29/cuda-non-ws-split-d.zip) · [独立 CPU 源码](/downloads/leetcuda/29/demo.py) · [运行说明](/downloads/leetcuda/29/README.txt) · [练习说明](/downloads/leetcuda/29/EXERCISES.txt)

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


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab29(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** 搬运序号分组的 gather；non-WS Split-D 的硬件实现见下面 CUDA demo。

```bash
python run_gpu.py --chapter 29 --backend triton
python run_gpu.py --chapter 29 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/xlite-dev/LeetCUDA/blob/3b52c5cd3332c5e992c7aef63087d43f67f23b8e/kernels/interview/book/tests/ch26c_ffpa_split_d_non_ws.cu)。CUDA C++20、NVCC、对应目标 GPU 与 CUTLASS submodule。TMA 相关例子还需满足该版本头文件的 CUDA toolkit 要求。

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git LeetCUDA-lab
git -C LeetCUDA-lab checkout --detach 3b52c5cd3332c5e992c7aef63087d43f67f23b8e
git -C LeetCUDA-lab submodule update --init --recursive
python3 run_upstream.py --repo ./LeetCUDA-lab --arch sm_120a
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** 全员计算路径与分工路径的调参对照。

**TileLang 任务：** 独立 K/V stages 与 non-WS 编排。

Triton 与 TileLang 分别保留普通流水和专门化基线。选择 chunk 数 1/3/5、stage 2/3 的组合做协议测试；记录每个 slot 的最近 writer 和最后 reader，不满足就阻止复用。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 29 --backend triton --student student_triton
python run_gpu.py --chapter 29 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 29 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 先画 buffer 生命周期，再测寄存器/等待/时间；更多 stage 不保证更快。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

把 producer 线程重新用于计算，未必减少整体指令或资源。比较需单独调参，否则一个方案的默认 tile 可能偏向另一方案。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（三十一）FP8 的格式与量化前处理](/llm/cuda-fp8-quantization/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
