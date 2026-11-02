---
title: CUDA 算子学习（三十一）：FP8 的格式与量化前处理
description: e4m3、动态范围、per-row/per-block scale。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-fp8-quantization/
date: 2026-11-02 17:00:30
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-non-ws-split-d/) · 下一篇：[（三十二）FP8 GEMM：scale 如何进入累加](/llm/cuda-fp8-gemm/)

FP8 是浮点格式，不是把 float 截成一个整数。学习时先实现软件数轴与舍入，再把缩放、转换和布局放进 GPU。格式、scale 方向和分组必须明确，否则两份“FP8”数据不一定代表同一数值。

本篇接续系列第 31 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 30 章（PDF 阅读器第 446–454 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** e4m3、动态范围、per-row/per-block scale。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| E4M3FN | 有限值 FP8 格式，最大有限幅度 448，具有特殊编码约定 |
| scale s | 本篇用 x_hat=s*q 的反量化方向 |
| amax | 分组内 max(abs(x)) |
| 量化误差 | 原值与反量化重建值之差，不只是编码是否成功 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 量化由两次选择组成
先选 scale，使组内范围适配格式；再把 x/s 舍入到 E4M3FN 的可表示值。零块单独约定 s=1、q=0。舍入到最近偶数、饱和到有限值与 NaN 策略需要一致；本 demo 只接受有限输入。

### 大组节省 scale，小组保留分辨率
一个离群值会抬高整个组的 scale，使小值挤到更少的格点。按行、按块或按通道分组影响误差，也影响 scale 元数据的搬运和后续乘法。不能只看数据缩小到 1 字节。

### 前处理有真实成本
amax 归约、scale 写出、FP8 写出都需要执行。若量化结果只使用一次，必须把这些成本算入总时间；若复用很多次，则报告可摊销次数。读写融合需要验证中间量，而不是仅看最终矩阵乘。

## 公式：从定义到可核对的小例子

```text
s = amax/448 if amax>0 else 1
q = round_to_e4m3fn(x/s)
x_hat = s*q
abs_error=abs(x_hat-x)
例：同组 [1,2,448] 的 scale=1，小值与离群值共享同一格点系
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-fp8-quantization.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-fp8-quantization.svg" alt="FP8 的格式与量化前处理的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-fp8-quantization.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**按约定划分量化组 → 归约 amax / 处理零块 → 计算 scale 与归一值 → E4M3 舍入 / 饱和 → 输出 data+scale 并重建校验**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/30/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`FP8 输入 [0, 0, 0]`。到最后一步应满足：编码均为有限 E4M3FN；软件码表最大值 448，自洽解码。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/30/cuda-fp8-quantization.zip) · [独立 CPU 源码](/downloads/leetcuda/30/demo.py) · [运行说明](/downloads/leetcuda/30/README.txt) · [练习说明](/downloads/leetcuda/30/EXERCISES.txt)

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


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def e4m3_values():
    """Positive E4M3FN codes 0..126. Code 127 is NaN, not finite."""
    values = []
    for code in range(127):
        exponent, mantissa = divmod(code, 8)
        value = mantissa * 2.0**-9 if exponent == 0 else (1 + mantissa / 8) * 2.0 ** (exponent - 7)
        values.append(value)
    assert values[-1] == 448
    return values


FP8 = e4m3_values()


FP4 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


def nearest(value, table):
    """Nearest representable finite value; tie uses even encoding LSB."""
    return min(range(len(table)), key=lambda i: (abs(table[i] - value), i % 2, i))


def quant8(x):
    assert all(math.isfinite(v) for v in x)
    scale = max(map(abs, x), default=0) / 448 or 1.0
    codes = [nearest(abs(v) / scale, FP8) | (128 if math.copysign(1, v) < 0 else 0) for v in x]
    restored = [(-1 if c & 128 else 1) * FP8[c & 127] * scale for c in codes]
    return restored, scale, codes


def lab30(case):
    x = [[0, 0, 0], [1, 2, 3, 4], [0.001, 1, 1000], [-2, -1, 0.5, 6]][case]
    restored, scale, codes = quant8(x)
    assert all(0 <= (c & 127) <= 126 for c in codes)
    return result(
        f"FP8 输入 {x}",
        [
            step("输入与 scale", {"x": x, "scale": scale}, "重建方向 x_hat=scale*q。"),
            step("E4M3FN 编码", codes, "最低 7 位为正幅度编码，最高位为符号。"),
            step(
                "反量化与误差",
                {"restored": restored, "error": [a - b for a, b in zip(restored, x)]},
                "零块采用 scale=1，不计算 0/0。",
            ),
        ],
        "编码均为有限 E4M3FN；软件码表最大值 448，自洽解码。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab30(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** 逐行 E4M3FN 量化并反量化，返回 FP32 重建值；保留编码和分组 scale 是扩展题。

```bash
python run_gpu.py --chapter 30 --backend triton
python run_gpu.py --chapter 30 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/xlite-dev/LeetCUDA/blob/3b52c5cd3332c5e992c7aef63087d43f67f23b8e/kernels/interview/book/tests/ch34_fp8_quantize.cu)。CUDA C++20、NVCC、对应目标 GPU 与 CUTLASS submodule。TMA 相关例子还需满足该版本头文件的 CUDA toolkit 要求。

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git LeetCUDA-lab
git -C LeetCUDA-lab checkout --detach 3b52c5cd3332c5e992c7aef63087d43f67f23b8e
git -C LeetCUDA-lab submodule update --init --recursive
python3 run_upstream.py --repo ./LeetCUDA-lab --arch sm_120a
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** amax 与缩放量化；零块和离群点。

**TileLang 任务：** 归约 + 转换融合；scale 排布。

Triton 写 amax+缩放+转换，TileLang 融合同样前处理。与软件可表示值表逐项比较，覆盖零、次正规附近、舍入中点、448 和离群值。报告 max_abs 与饱和计数，不只 allclose。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 30 --backend triton --student student_triton
python run_gpu.py --chapter 30 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 30 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 重建误差与饱和计数；明确舍入和 scale 方向，先不要求 Tensor Core。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

FP8 E4M3 的不同变体有不同特殊值编码。软件模型明确使用 E4M3FN 有限数，不把通用 minifloat 或 INT8 当作 bit-exact FP8。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（三十二）FP8 GEMM：scale 如何进入累加](/llm/cuda-fp8-gemm/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
