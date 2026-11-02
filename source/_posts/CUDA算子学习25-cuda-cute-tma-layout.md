---
title: CUDA 算子学习（二十五）：Swizzle 与 TMA 的地址契约
description: 软件访问地址必须匹配 descriptor 生成的布局。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-cute-tma-layout/
date: 2026-11-02 17:00:24
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-cute-tiled-mma/) · 下一篇：[（二十六）布局驱动的 GEMM 工程](/llm/cuda-cute-gemm/)

软件写地址和硬件生成地址必须遵守同一份契约。TMA 自动 swizzle 后，消费者若仍按普通行主序读取，就会拿到“看起来范围合法、实际上坐标错误”的数据。

本篇接续系列第 25 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 24 章（PDF 阅读器第 350–361 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** 软件访问地址必须匹配 descriptor 生成的布局。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| 字节 offset | TMA 描述符与地址位操作使用的单位之一 |
| 元素 offset | CuTe 逻辑布局常用的元素索引单位 |
| Swizzle<B,M,S> | 对选定地址位段做 XOR 的布局变换 |
| round-trip | 搬入后按消费布局读出，恢复原始元素顺序 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 先统一字节和元素
half 一个元素 2 字节，FP8 一个元素 1 字节，FP4 一个字节装两个元素。对地址位做 XOR 时，改变元素位宽会改变位段的含义。必须写清 layout 输入输出的单位，不能复制同一组位移常量后直接换 dtype。

### 把硬件映射和软件逆映射配起来
教学图使用固定行的 XOR 列置换说明可逆性。实际 TMA 的 swizzle 模式、base 对齐和 box 大小有专门约束。按 descriptor 配置产生物理布局后，ldmatrix/TiledCopy 必须按对应布局读取，而不是再额外施加一次不匹配的 swizzle。

### 先测搬运再接计算
输入填 `1000*row+col`，搬入 shared，再按消费者视图读出写回 global。逐元素检查比最终矩阵乘误差更容易定位错误。改变 stage 偏移、base 地址和 dtype 后重复测试，再接 MMA。

## 公式：从定义到可核对的小例子

```text
byte_offset = element_offset * bytes_per_element（非 subbyte）
FP4: byte_index = element_index//2, nibble = element_index%2
教学置换 P(r,c)=(r,c XOR r)，P(P(r,c))=(r,c)
实际 TMA swizzle 以目标 PTX / CuTe layout 为准
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-cute-tma-layout.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-cute-tma-layout.svg" alt="Swizzle 与 TMA 的地址契约的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-cute-tma-layout.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**逻辑坐标 → 元素 offset → 换算字节 / packed 坐标 → descriptor 决定物理 swizzle → 消费者匹配布局读取 → 写回并逐坐标比对**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/24/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`教学 swizzle：row=0`。到最后一步应满足：匹配布局 round-trip 逐元素相同。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/24/cuda-cute-tma-layout.zip) · [独立 CPU 源码](/downloads/leetcuda/24/demo.py) · [运行说明](/downloads/leetcuda/24/README.txt) · [练习说明](/downloads/leetcuda/24/EXERCISES.txt)

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


def lab24(case):
    row = [0, 1, 3, 7][case]
    source = list(range(8))
    physical = [None] * 8
    for col, value in enumerate(source):
        physical[col ^ row] = value
    restored = [physical[col ^ row] for col in range(8)]
    assert restored == source
    return result(
        f"教学 swizzle：row={row}",
        [
            step("global 逻辑顺序", source, "本模型不是某个硬件 descriptor 的逐位实现。"),
            step("shared 物理顺序", physical, "生产者按映射存储。"),
            step("消费者按匹配布局读取", restored, "错误地按线性顺序读会看到第二步的置换。"),
        ],
        "匹配布局 round-trip 逐元素相同。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab24(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** 匹配 XOR 布局的 gather；真实 TMA round-trip 需要专用硬件路径。

```bash
python run_gpu.py --chapter 24 --backend triton
python run_gpu.py --chapter 24 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/xlite-dev/LeetCUDA/blob/3b52c5cd3332c5e992c7aef63087d43f67f23b8e/kernels/interview/book/tests/ch23_swizzle_tma.cu)。原书程序包含 host 布局断言和可用设备上的检查；无设备的分支会 SKIP，必须逐项阅读输出。

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git LeetCUDA-lab
git -C LeetCUDA-lab checkout --detach 3b52c5cd3332c5e992c7aef63087d43f67f23b8e
git -C LeetCUDA-lab submodule update --init --recursive
python3 run_upstream.py --repo ./LeetCUDA-lab --arch sm_80
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** descriptor 路径与普通 load 的等价验证。

**TileLang 任务：** 显式 swizzle 与 TMA copy 配对实验。

Triton descriptor copy 与普通 load 做等价验证；TileLang 显式 swizzle 和 TMA 搬运配对。小 tile、不同 stage、非零 base offset 各测一组。原书 ch23_swizzle_tma 是 host 布局断言，不能当硬件 copy 测试。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 24 --backend triton --student student_triton
python run_gpu.py --chapter 24 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 24 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 先小 tile round-trip，再接 MMA；分别核验对齐、base 与 phase。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

可逆 XOR 图只是布局直觉，不是 SW32/SW64/SW128 的逐位硬件实现。实际参数必须引用当前目标的 descriptor 和布局定义。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（二十六）布局驱动的 GEMM 工程](/llm/cuda-cute-gemm/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
