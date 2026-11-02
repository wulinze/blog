---
title: CUDA 算子学习（二十三）：Copy 的归属：谁搬什么，搬到哪里
description: Tensor 视图与 TiledCopy 的 thread/value 划分。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-cute-tiled-copy/
date: 2026-11-02 17:00:22
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-cute-layout/) · 下一篇：[（二十四）MMA 的归属：从原子到输出 fragment](/llm/cuda-cute-tiled-mma/)

先回答“谁搬哪个元素”，再选择搬运指令。TiledCopy 把线程分工显式化，避免 global→shared、shared→register、register→global 三处各自写一套难以保持一致的索引。

本篇接续系列第 23 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 22 章（PDF 阅读器第 325–337 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** Tensor 视图与 TiledCopy 的 thread/value 划分。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| Tensor | 数据引擎/指针加上 layout 的视图 |
| Copy atom | 一次可选硬件搬运操作及其约束 |
| TV-layout | 从 thread/value 对到逻辑坐标的映射 |
| partition | 给某线程取出它负责的子视图，不等于发起搬运 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 线程与线程内 value 是两根轴
教学模型 8 个线程、每线程 4 个值，i=4*t+v，恰好覆盖 32 个元素。另一个映射 i=t+8*v 也覆盖全部，但每线程连续性不同。同一条 warp 指令的地址合并与单线程向量宽度要分别分析。

### 源和目标可以有不同布局
同一个逻辑元素在 global 行主序和 shared swizzle 中有不同 offset。copy 的职责是按逻辑对应关系把源值放入正确目标，不能把相同线性 offset 当作默认等价。

### Retile 不等于重新读内存
为了让 copy 的寄存器片段匹配 MMA fragment，可能需要重组织视图或实际数据交换。是否零代价取决于两个映射是否已经兼容。必须打印 TV-layout 和元素坐标；仅看类型能编译不够。

## 公式：从定义到可核对的小例子

```text
contiguous-per-thread: i=t*V+v
interleaved: i=t+T*v
T=8,V=4 → 32 个 (t,v) 覆盖 32 个逻辑元素
尾部 N=29 时 i>=29 不加载、不写回；它仍可能参与同步
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-cute-tiled-copy.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-cute-tiled-copy.svg" alt="Copy 的归属：谁搬什么，搬到哪里的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-cute-tiled-copy.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**逻辑 tile 与源 layout → thread/value 划分 → 源 partition 与目标 partition → 执行 copy / 等待完成 → round-trip 检查逐元素相等**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/22/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`N=1,8 逻辑线程×4 values`。到最后一步应满足：每个有效坐标恰好搬一次，包括第二轮尾部。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/22/cuda-cute-tiled-copy.zip) · [独立 CPU 源码](/downloads/leetcuda/22/demo.py) · [运行说明](/downloads/leetcuda/22/README.txt) · [练习说明](/downloads/leetcuda/22/EXERCISES.txt)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab22(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** 含尾部的双射 gather；改变 thread/value 分工与 shared 中转是扩展题。

```bash
python run_gpu.py --chapter 22 --backend triton
python run_gpu.py --chapter 22 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/xlite-dev/LeetCUDA/blob/3b52c5cd3332c5e992c7aef63087d43f67f23b8e/kernels/interview/book/tests/ch21_tiled_copy.cu)。此原书程序主要做 CuTe host 布局断言，仍使用 NVCC/CUTLASS 编译；它不是 GPU kernel 性能测试。

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git LeetCUDA-lab
git -C LeetCUDA-lab checkout --detach 3b52c5cd3332c5e992c7aef63087d43f67f23b8e
git -C LeetCUDA-lab submodule update --init --recursive
python3 run_upstream.py --repo ./LeetCUDA-lab --arch sm_80
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** 不同 tile 载入映射，跟踪 mask 与布局。

**TileLang 任务：** T.copy 的 global/shared/fragment 路径。

Triton 比较连续与交错索引载入；TileLang 实现 global→shared→global 往返。输入用 arange 编码位置以定位搬错，测试 N=1/29/32/33，确保不存在重复写和越界。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 22 --backend triton --student student_triton
python run_gpu.py --chapter 22 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 22 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 覆盖无遗漏、无非法重叠；对齐、向量宽度和尾部都可解释。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

copy atom 的对齐、字节宽度和参与线程限制不能靠高层类型名猜测。尾部无效元素与线程不参加 collective 是不同情况。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（二十四）MMA 的归属：从原子到输出 fragment](/llm/cuda-cute-tiled-mma/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
