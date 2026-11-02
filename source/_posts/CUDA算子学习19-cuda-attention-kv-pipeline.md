---
title: CUDA 算子学习（十九）：注意力的双流水：K、V 与角色分工
description: 不同数据的就绪时机，stage 占用和 full/empty 协议。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-attention-kv-pipeline/
date: 2026-11-02 17:00:18
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-attention-split-q/) · 下一篇：[（二十）多 consumer 与分段合并](/llm/cuda-attention-multi-consumer/)

K 与 V 的使用时刻不同：先用 K 生成分数，再做 softmax，最后用 V 更新输出。因此把 K/V 当一份同步搬运的黑盒，会隐藏本可重叠的部分。双流水要分别画出两类数据的生存期。

本篇接续系列第 19 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 18 章（PDF 阅读器第 227–244 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** 不同数据的就绪时机，stage 占用和 full/empty 协议。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| K pipeline | 供 QK 使用的数据槽与就绪事件 |
| V pipeline | 供 PV 使用的数据槽与就绪事件 |
| early release | 确认最后一次读已结束后提前释放某槽 |
| 反压 | 消费者未释放槽，生产者不能再推进 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 两条流水共享一个数学迭代
第 t 个 KV tile 的 K 用于 QK[t]，V 用于 PV[t]。producer 可提前装载后续 K/V，但不能覆盖仍在使用的 slot。K 最后被读取的时刻不一定等于 V 的最后读取时刻，释放点应由实际指令确定。

### stage 深度要匹配距离
更深的 V 预取可能覆盖 QK+softmax 的时间，K 预取则覆盖上一迭代剩余计算。两者需求不同。额外 stage 消耗 shared，导致较少 resident CTA 时，表面更深的流水也可能更慢。

### 短序列是协议测试
只有一个 tile 时，预取分支和排空路径最容易留下未完成等待；奇数 tile 会暴露环形索引错误。先验证这些边界，再测试长序列。每个等待必须对应一个必定发生的事件，不能靠“通常调度顺序”保证。

## 公式：从定义到可核对的小例子

```text
S_k ≈ stages_k*BK*D*element_bytes
S_v ≈ stages_v*BK*Dv*element_bytes
总 shared 还包括 Q、屏障、padding 与 staging
slot_k=t%stages_k，slot_v=t%stages_v：两套独立索引
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-attention-kv-pipeline.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-attention-kv-pipeline.svg" alt="注意力的双流水：K、V 与角色分工的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-attention-kv-pipeline.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**producer 等 K/V 的 empty → 分别提交 K / V 搬运 → K full → QK → softmax；V full → PV → 各自最后读完 → 独立释放**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/18/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`K stages=1,V stages=1`。到最后一步应满足：两套环形编号按各自 stage 数独立更新。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/18/cuda-attention-kv-pipeline.zip) · [独立 CPU 源码](/downloads/leetcuda/18/demo.py) · [运行说明](/downloads/leetcuda/18/README.txt) · [练习说明](/downloads/leetcuda/18/EXERCISES.txt)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab18(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** 小尺寸 attention 语义基线，包含明确的 top-left causal mask；各章专用流水、布局与分工见下面的固定版本 CUDA demo。

```bash
python run_gpu.py --chapter 18 --backend triton
python run_gpu.py --chapter 18 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/xlite-dev/LeetCUDA/blob/3b52c5cd3332c5e992c7aef63087d43f67f23b8e/kernels/interview/book/tests/ch17_fa2_tma_ws.cu)。CUDA C++20、NVCC、对应目标 GPU 与 CUTLASS submodule。TMA 相关例子还需满足该版本头文件的 CUDA toolkit 要求。

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git LeetCUDA-lab
git -C LeetCUDA-lab checkout --detach 3b52c5cd3332c5e992c7aef63087d43f67f23b8e
git -C LeetCUDA-lab submodule update --init --recursive
python3 run_upstream.py --repo ./LeetCUDA-lab --arch sm_120a
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** 比较编译器流水配置与支持的 warp specialization。

**TileLang 任务：** 分别调 K/V stages，追踪 consumer 释放。

两语言都先使用一条统一流水作为正确性基线。扩展到受支持的专门化路径时，独立扫 K/V stages；记录 single-tile、odd-tile 的通过状态及等待事件。浏览器中的时间线只表示依赖模型。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 18 --backend triton --student student_triton
python run_gpu.py --chapter 18 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 18 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 短序列、单 tile、尾 tile 先通过；生产/消费不平衡需要时间线证据。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

full/empty 计数不能从另一篇 kernel 照抄。producer 组 128 个线程与只有 1 个线程提交 TMA 是不同事实，实际 arrive 参与者必须数清。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（二十）多 consumer 与分段合并](/llm/cuda-attention-multi-consumer/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
