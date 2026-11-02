---
title: CUDA 算子学习（十三）：MMA 与 fragment：理解协作计算的数据归属
description: lane / fragment / ldmatrix 与多级流水。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-mma-fragments/
date: 2026-11-02 17:00:12
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-tf32-pipeline/) · 下一篇：[（十四）布局与调度：三种不同的 swizzle](/llm/cuda-swizzle/)

高层矩阵 tile 是一整块数学对象，硬件实际把它分散在多个 lane 的寄存器里。理解 fragment，首先要回答每个输出坐标由谁持有，再回答 A/B 如何到达正确的寄存器。

本篇接续系列第 13 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 12 章（PDF 阅读器第 129–141 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** lane / fragment / ldmatrix 与多级流水。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| MMA atom | warp 协作的最小矩阵乘加操作 |
| fragment | 每个 lane 持有的逻辑矩阵元素集合 |
| ldmatrix | 协作地把 shared 矩阵片段装入寄存器 |
| 累加类型 | C/D 的数值类型，和 A/B 输入类型分开选择 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 固定一条指令再谈布局
本例讨论 m16n8k16、FP16 A/B、FP32 C/D 的 warp MMA。A 有 256 个元素，B 有 128 个，C 有 128 个。32 lane 平分输出时每 lane 持有 4 个 FP32 累加值；这只是一个 atom，不是整个 CTA 的总寄存器数。

### 输出映射可逐点穷举
令 g=lane//4、t=lane%4。四个 C 元素位于 (g,2t)、(g,2t+1)、(g+8,2t)、(g+8,2t+1)。枚举所有 lane 可得到 16×8 的完整无重叠覆盖。A/B 的布局不同，不能把这个 C 公式直接用于加载 A/B。

### 装载地址与得到的数据不是同一映射
ldmatrix 中哪些 lane 提供行地址、哪些 lane 接收哪些寄存器字，要按 x1/x2/x4、是否 trans 和元素宽度分别查 PTX。mma 的 row/col 描述操作数解释，不等于全局内存一定使用哪种存储。先把 global→shared 的布局验证通过，再接 ldmatrix 与 MMA，最后验证 epilogue。

## 公式：从定义到可核对的小例子

```text
C[16,8] = A[16,16] @ B[16,8] + C[16,8]
每 atom FLOPs = 2*16*8*16 = 4096
每 lane 的 FP32 C 元素数 = 16*8/32 = 4
lane 5: g=1,t=1 → (1,2),(1,3),(9,2),(9,3)
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-mma-fragments.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-mma-fragments.svg" alt="MMA 与 fragment：理解协作计算的数据归属的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-mma-fragments.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**global 的逻辑 A/B → shared 的物理布局 → ldmatrix → A/B fragment → warp MMA → C fragment → 坐标映射 → 输出矩阵**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/12/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`m16n8k16 FP32 C fragment，lane=0`。到最后一步应满足：128 个输出坐标唯一覆盖；不是设备指令执行测试。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/12/cuda-mma-fragments.zip) · [独立 CPU 源码](/downloads/leetcuda/12/demo.py) · [运行说明](/downloads/leetcuda/12/README.txt) · [练习说明](/downloads/leetcuda/12/EXERCISES.txt)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab12(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** 完整矩阵乘语义；fragment 的精确 lane 映射由 CPU demo 与原书 CUDA 例子验证。

```bash
python run_gpu.py --chapter 12 --backend triton
python run_gpu.py --chapter 12 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/xlite-dev/LeetCUDA/blob/3b52c5cd3332c5e992c7aef63087d43f67f23b8e/kernels/interview/book/tests/ch11_hgemm_mma.cu)。CUDA C++20、NVCC、对应目标 GPU 与 CUTLASS submodule。TMA 相关例子还需满足该版本头文件的 CUDA toolkit 要求。

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git LeetCUDA-lab
git -C LeetCUDA-lab checkout --detach 3b52c5cd3332c5e992c7aef63087d43f67f23b8e
git -C LeetCUDA-lab submodule update --init --recursive
python3 run_upstream.py --repo ./LeetCUDA-lab --arch sm_80
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** 检查 dot 的 lowering；扩展题使用显式布局。

**TileLang 任务：** 用布局可视化追踪 A/B/C 的 fragment。

Triton 打印编译后的 PTX，识别 dot lowering 的实际形状；TileLang 查看生成的 CUDA，打印/可视化 fragment。先完成 128 个 C 坐标的唯一覆盖，再比较 M=16,N=8,K=16 和多个 atom 的结果。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 12 --backend triton --student student_triton
python run_gpu.py --chapter 12 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 12 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 独立核对每个输出坐标与覆盖；不能把高层 tile 当每线程存储量。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

WMMA 的不透明 fragment 与 PTX 指定的 fragment 不可混为一谈。原书某些路径采用 FP16 累加，本学习基线默认 FP32；必须注明差异再比较精度或性能。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（十四）布局与调度：三种不同的 swizzle](/llm/cuda-swizzle/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
