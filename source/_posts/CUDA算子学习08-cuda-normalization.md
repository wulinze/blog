---
title: CUDA 算子学习（八）：RMSNorm 与 LayerNorm：统计量和广播
description: 理解平方均值、中心化方差、eps 与稳定计算。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-normalization/
date: 2026-11-02 17:00:07
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-lse-merge/) · 下一篇：[（九）RoPE 与转置：数据重排为何影响性能](/llm/cuda-rope-transpose/)

RMSNorm 和 LayerNorm 都是“行统计 → 广播 → 逐元素变换”。相同的是执行骨架，不同的是统计量。把这一点弄清后，Softmax 的行归约经验就可以迁移到归一化算子。

本篇接续系列第 8 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 7 章（PDF 阅读器第 78–87 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** 理解平方均值、中心化方差、eps 与稳定计算。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| 均值 mu | 一行 N 个元素的算术平均 |
| 总体方差 var | sum((x-mu)^2)/N，不除以 N-1 |
| RMS | 平方均值开根，不先减均值 |
| epsilon | 放在根号内的稳定项，需和参考一致 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### RMSNorm 保留均值方向
RMSNorm 用平方均值缩放 x，不进行中心化。输入 [1,2,3,4] 的平方均值是 7.5。若 gamma 为 1，eps 忽略，输出为 x/sqrt(7.5)。给所有输入加同一个偏置，RMSNorm 输出一般会变。

### LayerNorm 先中心化
同一个例子均值 2.5，方差 1.25，中心化向量 [-1.5,-0.5,0.5,1.5]。真实接口中 gamma 和 beta 通常是长度 N 的通道向量，而不是一个共享标量。输出的零均值性质只适用于应用 gamma/beta 之前。

### 一遍公式可能损失小方差
数学上 `E[x²]-E[x]²` 等于方差，但两项都很大且接近时，浮点相减会丢有效位。两遍中心化方差或 Welford 更适合检查“大偏置、小波动”。不要仅靠 max(var,0) 掩盖数值问题。短行把输入保留在寄存器，可避免两遍统计对应两遍 global 读。

## 公式：从定义到可核对的小例子

```text
RMSNorm_i = x_i / sqrt(sum(x²)/N + eps) * gamma_i
LayerNorm_i = (x_i-mu)/sqrt(var+eps)*gamma_i+beta_i
mu=2.5, var=1.25 for x=[1,2,3,4]
所有统计使用实际有效 N，而不是补齐后的 tile 宽度
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-normalization.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-normalization.svg" alt="RMSNorm 与 LayerNorm：统计量和广播的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-normalization.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**读取一行与有效 mask → 求均值 / 平方均值 → LN 再求中心化方差 → 广播倒数平方根 → gamma / beta → 写回**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/07/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`x=[1, 2, 3, 4]`。到最后一步应满足：LN 中心化和为 0；RMS 的平方均值符合含 eps 的公式。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/07/cuda-normalization.zip) · [独立 CPU 源码](/downloads/leetcuda/07/demo.py) · [运行说明](/downloads/leetcuda/07/README.txt) · [练习说明](/downloads/leetcuda/07/EXERCISES.txt)

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


def lab07(case):
    x = [[1, 2, 3, 4], [0, 0, 0, 0], [1000, 1000.01, 999.99, 1000], [3, 3, 3, 3]][case]
    eps = 1e-5
    mu = math.fsum(x) / len(x)
    var = math.fsum((v - mu) ** 2 for v in x) / len(x)
    mean_sq = math.fsum(v * v for v in x) / len(x)
    ln = [(v - mu) / math.sqrt(var + eps) for v in x]
    rms = [v / math.sqrt(mean_sq + eps) for v in x]
    close(sum(ln), 0, 1e-8)
    close(sum(v * v for v in rms) / len(x), mean_sq / (mean_sq + eps))
    return result(
        f"x={x}",
        [
            step("输入", x, "gamma=1,beta=0；使用总体方差。"),
            step(
                "统计量",
                {"mean": mu, "variance": var, "mean_square": mean_sq, "eps": eps},
                "中心化方差用两遍式，不用大数相减。",
            ),
            step("LayerNorm", ln, "减均值，eps 在根号内。"),
            step("RMSNorm", rms, "不减均值，输出不要求零均值。"),
        ],
        "LN 中心化和为 0；RMS 的平方均值符合含 eps 的公式。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab07(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** RMSNorm / 两遍 LayerNorm；gamma、beta 和 Welford 是扩展题。

```bash
python run_gpu.py --chapter 7 --backend triton
python run_gpu.py --chapter 7 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/xlite-dev/LeetCUDA/blob/3b52c5cd3332c5e992c7aef63087d43f67f23b8e/kernels/interview/book/tests/ch06_norm.cu)。CUDA C++20、NVCC、对应目标 GPU 与 CUTLASS submodule。TMA 相关例子还需满足该版本头文件的 CUDA toolkit 要求。

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git LeetCUDA-lab
git -C LeetCUDA-lab checkout --detach 3b52c5cd3332c5e992c7aef63087d43f67f23b8e
git -C LeetCUDA-lab submodule update --init --recursive
python3 run_upstream.py --repo ./LeetCUDA-lab --arch sm_80
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** fused RMSNorm；LayerNorm 两遍统计。

**TileLang 任务：** fragment 归约；比较两遍方差与 Welford。

两种语言分别实现 RMSNorm 和两遍 LayerNorm，统计使用 float32。输入包含常数行、全零、小方差和 129 长度；gamma/beta 使用非均匀值，避免测试漏掉广播维度错误。扩展对照 Welford。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 7 --backend triton --student student_triton
python run_gpu.py --chapter 7 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 7 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 小方差、大偏置、非整除 hidden size；匹配 eps 和方差定义。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

补零可以用于 sum，但方差中无效位置不能参与 (0-mu)²；归约前必须再屏蔽。eps 既不是根号外相加，也不是每个元素平方后单独相加再除错次数。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（九）RoPE 与转置：数据重排为何影响性能](/llm/cuda-rope-transpose/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
