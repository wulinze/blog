---
title: CUDA 算子学习（三十九）：FP4 Attention 的量化链
description: NVFP4 格式在 Q/K/V 预处理中的应用。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-fp4-attention-preprocess/
date: 2026-11-02 17:00:38
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-fp8-split-d-m4n2/) · 下一篇：[（四十）FP4 Attention：P 的两级量化](/llm/cuda-fp4-attention-probability/)

从 FP8 降到 FP4，格点更稀疏，平滑和 scale 管理更重要。特别是平滑 Q 与平滑 K 的作用不同：减 Q 均值通常产生随 key 变化的项，不能像 K 的行常数那样直接被 softmax 消掉。

本篇接续系列第 39 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 38 章（PDF 阅读器第 544–553 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** NVFP4 格式在 Q/K/V 预处理中的应用。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| smooth-Q | 本路径对一组 query 减去公共均值向量 |
| delta S | 补回随 key 变化的分数修正 |
| NVFP4 workspace | packed data、局部 SF、全局 scale 和布局元数据 |
| perm-aware mask | 按原始 key 坐标判断 causal，而非物理列号 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 展开平滑恒等式
令 Q′=Q-a，K′=K-b。原 QK.T=Q′K′.T+aK′.T+Q′b.T+ab.T。对某 query 行，后两项对所有 key 相同，可在 softmax 中消去；aK′.T 随 key 改变，必须补回 delta S。若 a 是 query 块均值，按块索引取对应修正。

### Packed 数据和 scale 同步重排
V 转置、列置换、padding 后，SF 必须跟随它代表的数据组。FP4 的逻辑元素指针不能简单按一个字节一个元素递增。先在 CPU 验证 pack/unpack，再检查设备 layout。

### Causal 使用逻辑位置
如果 key 列做了置换，物理第 j 列并不一定对应原 key j。mask 必须先恢复原始坐标再比较 query 位置。错误的 mask 在 dense 测试里不会暴露，因此两种模式都要测试。

## 公式：从定义到可核对的小例子

```text
Q=Q′+a，K=K′+b
softmax(QK.T)=softmax(Q′K′.T+aK′.T)（忽略每行常数）
deltaS=a@K′.T，再统一乘 1/sqrt(D)
FP4 byte_index=i//2；逻辑 i 与物理 byte 不能混同
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-fp4-attention-preprocess.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-fp4-attention-preprocess.svg" alt="FP4 Attention 的量化链的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-fp4-attention-preprocess.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**统计 Q 块 / K 均值 → 平滑并生成 delta S → 两级 scale + FP4 packing → 数据/SF 同步置换 → 消费时恢复逻辑 mask / 修正**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/38/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`Q 块均值示例 a=[0,0]`。到最后一步应满足：补回 deltaS 后概率等价；非零均值可暴露漏补错误。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/38/cuda-fp4-attention-preprocess.zip) · [独立 CPU 源码](/downloads/leetcuda/38/demo.py) · [运行说明](/downloads/leetcuda/38/README.txt) · [练习说明](/downloads/leetcuda/38/EXERCISES.txt)

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


def lab38(case):
    a = [0, 1, 3, 10][case]
    q = [a + 1, a + 2]
    k = [[1, 3], [2, -1], [4, 2]]
    b = [sum(row[d] for row in k) / len(k) for d in range(2)]
    qp = [v - a for v in q]
    kp = [[v - b[d] for d, v in enumerate(row)] for row in k]
    original = [dot(q, row) for row in k]
    partial = [dot(qp, row) for row in kp]
    delta = [dot([a, a], row) for row in kp]
    fixed = [s + t for s, t in zip(partial, delta)]
    for x, y in zip(softmax(original), softmax(fixed)):
        close(x, y)
    return result(
        f"Q 块均值示例 a=[{a},{a}]",
        [
            step("原始 scores", original, "单 query，D=2；为专注恒等式省略共同 sqrt(D)。"),
            step("平滑 Q/K 后", {"scores_prime": partial, "deltaS": delta}, "deltaS 随 key 变化，不能消掉。"),
            step(
                "概率对照",
                {"original": softmax(original), "without_delta": softmax(partial), "with_delta": softmax(fixed)},
                "只差每行常数时概率相同。",
            ),
        ],
        "补回 deltaS 后概率等价；非零均值可暴露漏补错误。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab38(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** Q/K 平滑 + deltaS 修正的未量化 attention；数据/SF packing 与置换见 CPU 和专用 CUDA demo。

```bash
python run_gpu.py --chapter 38 --backend triton
python run_gpu.py --chapter 38 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/vipshop/ffpa-attn/blob/861d75ee87824a14a9ebfe295d54c7dac3e3cabc/csrc/cuffpa/cute/fp4/quantize_fp4.cuh)。Linux NVIDIA SM120 环境；先按固定版本 README 准备 CUDA PyTorch、Python≥3.10、构建依赖和 CUDA 扩展。此入口运行整条量化 attention，前处理、主循环、误差报告一起观察。大 D 路径是否编入以构建的 headdim 配置为准。

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

**Triton 任务：** 软件 packed 参考与完整前处理。

**TileLang 任务：** 两级 scale 生成和打包融合。

Triton 和 TileLang 都先验证未量化的平滑恒等式，再插入 FP4。使用非零 Q/K 均值与 causal mask，故意省略 delta S 应被测试判错；然后验证数据与 SF 的置换还原。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 38 --backend triton --student student_triton
python run_gpu.py --chapter 38 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 38 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 数据/scale 同时对齐，精度与转换成本单独报告。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

平滑是某一实现的算法设计，不是 NVFP4 格式本身的强制要求。不要把原书实现的必须步骤泛化成所有 FP4 attention 的硬件规定。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（四十）FP4 Attention：P 的两级量化](/llm/cuda-fp4-attention-probability/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
