---
title: CUDA 算子学习（二十四）：MMA 的归属：从原子到输出 fragment
description: TiledMMA 分层与 A/B/C 的不同坐标映射。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-cute-tiled-mma/
date: 2026-11-02 17:00:23
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-cute-tiled-copy/) · 下一篇：[（二十五）Swizzle 与 TMA 的地址契约](/llm/cuda-cute-tma-layout/)

TiledCopy 解决搬运归属，TiledMMA 解决计算归属。它们可以覆盖同一个 tile，但不是同一种映射。连接两者时，重点检查操作数的每个值是否到达预期 lane/register。

本篇接续系列第 24 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 23 章（PDF 阅读器第 338–349 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** TiledMMA 分层与 A/B/C 的不同坐标映射。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| MMA atom | 固定形状和 TV 映射的协作计算单元 |
| TiledMMA | 用 atom 排布和目标 tile 扩展计算范围 |
| partition_A/B/C | 按照各自操作数的坐标映射分片 |
| accumulator | 跨 K 迭代保留的输出状态 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 从 atom 扩展到 tile
一个 m16n8 atom 有 128 个输出值。两个 M 方向 atom 覆盖 32 行，两个 N 方向 atom 覆盖 16 列；M/N 排布决定输出坐标，K 方向迭代则累加到已有输出，不新增输出面积。

### A、B、C 的分片形状不同
A 的逻辑轴是 M,K，B 是 N,K 或 K,N（取决于视图），C 是 M,N。相同的 thread ID 在三者中负责不同坐标集合。partition_fragment_C 得到的形状不是全局 C 的连续切片。

### Attention 的桥接最能暴露错误
QK 的 C 经 softmax 后作为 PV 的 A。可以零拷贝重解释的前提是两个 TV 映射对每个坐标都吻合；否则需要 shuffle/shared 等数据重排。元素总数一致只是必要条件，不是充分条件。

## 公式：从定义到可核对的小例子

```text
tile_output_elements = BM*BN
每线程平均累加值 = BM*BN/threads
例：BM=32,BN=16,threads=128 → 平均 4 个值
K atom 迭代增加计算量，不增加 C 元素数量
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-cute-tiled-mma.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-cute-tiled-mma.svg" alt="MMA 的归属：从原子到输出 fragment的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-cute-tiled-mma.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**选择 MMA atom → 按 M/N/K 定义 tile → 分别 partition A/B/C → copy fragment → gemm → 按 C 的 owner 写回**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/23/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`atom 排布 M=1,N=1`。到最后一步应满足：atom 输出组合无重叠无遗漏。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/23/cuda-cute-tiled-mma.zip) · [独立 CPU 源码](/downloads/leetcuda/23/demo.py) · [运行说明](/downloads/leetcuda/23/README.txt) · [练习说明](/downloads/leetcuda/23/EXERCISES.txt)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab23(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** 矩阵乘数值基线；TiledMMA 的 host 映射测试见原书专用程序。

```bash
python run_gpu.py --chapter 23 --backend triton
python run_gpu.py --chapter 23 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/xlite-dev/LeetCUDA/blob/3b52c5cd3332c5e992c7aef63087d43f67f23b8e/kernels/interview/book/tests/ch22_tiled_mma.cu)。此原书程序主要做 CuTe host 布局断言，仍使用 NVCC/CUTLASS 编译；它不是 GPU kernel 性能测试。

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git LeetCUDA-lab
git -C LeetCUDA-lab checkout --detach 3b52c5cd3332c5e992c7aef63087d43f67f23b8e
git -C LeetCUDA-lab submodule update --init --recursive
python3 run_upstream.py --repo ./LeetCUDA-lab --arch sm_80
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** dot 两端布局变化及编译器中间表示。

**TileLang 任务：** T.gemm 片段分配和 accumulator 可视化。

Triton 追踪 dot 前后布局转换的编译器 IR；TileLang 可视化 accumulator 和输入 fragment。用唯一编号填每个输入坐标，先验证小矩阵，再扩展到两种 M/N warp 排布。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 23 --backend triton --student student_triton
python run_gpu.py --chapter 23 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 23 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 元素计数守恒；点积输入布局与输出布局不能直接混用。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

平均寄存器数不包含临时量，也不保证均匀分配。改变 atom layout 后，softmax 的行归约范围和 P 交接都需要重新验证。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（二十五）Swizzle 与 TMA 的地址契约](/llm/cuda-cute-tma-layout/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
