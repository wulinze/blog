---
title: CUDA 算子学习（三十四）：Block-scaled MMA：scale 也是操作数
description: scale fragment、数据布局与流水的协同。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-block-scaled-mma/
date: 2026-11-02 17:00:33
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-nvfp4-format/) · 下一篇：[（三十五）量化 Attention：误差在哪里被放大](/llm/cuda-quantized-attention-math/)

低精度矩阵乘多了一条独立的数据流：scale。数据加载正确而 scale 放错位置，会得到形状正确但数值错误的结果。应把 SF 和 A/B 一样画进流水与资源预算。

本篇接续系列第 34 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 33 章（PDF 阅读器第 482–497 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** scale fragment、数据布局与流水的协同。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| SF | scale factor，矩阵指令的附加操作数 |
| scale fragment | 每个 lane 持有的 scale 元素集合 |
| zip | 把数值与其 scale 按契约配对，不是任意拼接 |
| block-scaled MMA | 沿约定块宽消费数据及对应 scale 的矩阵乘 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### Scale 的 K 分组参与归约
每个 K 组都有对应的 A/B scale。硬件 block-scaled 指令把它们作用到该组数据上再累加。外层 global scale 若对归约轴不变，才可在 epilogue 统一乘回。

### 四条搬运流都要就绪
A、B、SFA、SFB 可能各有独立 descriptor 和 shared 布局。consumer 必须等它要使用的所有数据完成，而不是只等待 A/B。事务字节数与释放计数应包含实际的 SF 搬运。

### 最有力的测试是给 scale 编码
让不同组 scale 故意不同，甚至选择 1、2、4 等容易手算的数；数据全 1。正确输出应等于每组元素数乘对应 scale 乘积之和。全同 scale 测试会掩盖布局置换错误。

## 公式：从定义到可核对的小例子

```text
C[m,n] = sum_g(sA[m,g]*sB[n,g]*sum_{k in g}(a[m,k]*b[k,n]))
每组 16 项全 1，scale 乘积分别 1、2 → C=16+32=48
shared_total = A+B+SFA+SFB+barriers+padding+staging
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-block-scaled-mma.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-block-scaled-mma.svg" alt="Block-scaled MMA：scale 也是操作数的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-block-scaled-mma.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**data 与 SF 按组对应 → 分别搬入 A/B/SFA/SFB → 等待四条依赖就绪 → 配对 fragment → scaled MMA → global scale / epilogue 写出**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/33/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`把 SF 错移 0 组，观察错误`。到最后一步应满足：正确结果 784；非均匀测试能揭露 scale 错位。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/33/cuda-block-scaled-mma.zip) · [独立 CPU 源码](/downloads/leetcuda/33/demo.py) · [运行说明](/downloads/leetcuda/33/README.txt) · [练习说明](/downloads/leetcuda/33/EXERCISES.txt)

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


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab33(case):
    sf = [1, 2, 4, 8]
    shift = case
    data = [1, 2, 3, 4]
    aligned = sum(16 * x * s for x, s in zip(data, sf))
    misplaced = sum(16 * x * sf[(g + shift) % 4] for g, x in enumerate(data))
    close(aligned, 784)
    return result(
        f"把 SF 错移 {shift} 组，观察错误",
        [
            step(
                "data / scale 按组",
                {"data_values": data, "scale": sf, "elements_per_group": 16},
                "每组重复同一个 data 值，方便手算。",
            ),
            step("正确配对", {"C": aligned}, "sum_g(16*data_g*scale_g)。"),
            step("错位 SF", {"C_wrong": misplaced, "error": misplaced - aligned}, "只改 scale 位置就足以破坏结果。"),
        ],
        "正确结果 784；非均匀测试能揭露 scale 错位。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab33(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** 逐 K 块 E2M1 重建后 GEMM；真实 block-scaled 指令与 SF 布局见专用 CUDA demo。

```bash
python run_gpu.py --chapter 33 --backend triton
python run_gpu.py --chapter 33 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/xlite-dev/LeetCUDA/blob/d32a58f53e4122dc68d660698ee025751a596fc5/kernels/interview/bench/fp4_gemm.cu)。这个 FP4 GEMM 在较新的固定提交中，不能用前面旧提交的测试编号寻找它。需要支持 SM120 FP4 的 CUDA toolkit、CUTLASS、cuDNN/cuBLAS 等该构建脚本依赖。运行 --fp4-gemm 会执行量化与主 GEMM 的组合验证。

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git LeetCUDA-lab
git -C LeetCUDA-lab checkout --detach d32a58f53e4122dc68d660698ee025751a596fc5
git -C LeetCUDA-lab submodule update --init --recursive
python3 run_upstream.py --repo ./LeetCUDA-lab --arch sm_120a
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** 目标架构支持的 block-scaled matmul。

**TileLang 任务：** 数据和 scale 的协作加载。

Triton 先用反量化 dot 参考，再接目标支持的 dot_scaled；TileLang 按目标后端的 block-scaled 示例实现。每组不同 scale、最后不足一组、数据/SF padding 都要验证；设备不支持时明确跳过硬件部分。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 33 --backend triton --student student_triton
python run_gpu.py --chapter 33 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 33 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 每个 scale 与正确数据块对应；对齐厂商库数值约定后再测。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

支持 FP4 存储不等于支持某一种 block-scaled 指令。SM100 与 SM120 的指令族及 scale layout 需要分别核对，不能直接复制模板参数。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（三十五）量化 Attention：误差在哪里被放大](/llm/cuda-quantized-attention-math/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
