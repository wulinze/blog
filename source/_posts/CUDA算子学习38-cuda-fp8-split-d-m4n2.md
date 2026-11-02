---
title: CUDA 算子学习（三十八）：FP8 大 D：Split-D 与并行布局
description: 工作量增加与寄存器增长是不同瓶颈。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-fp8-split-d-m4n2/
date: 2026-11-02 17:00:37
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-fp8-persist-scale/) · 下一篇：[（三十九）FP4 Attention 的量化链](/llm/cuda-fp4-attention-preprocess/)

大 D 有两个不同瓶颈：shared tile 放不下，以及输出累加器寄存器太多。Split-D 主要处理前者；改变 M/N 排布和输出行块大小可以影响后者，但会引入新的通信。

本篇接续系列第 38 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 37 章（PDF 阅读器第 535–543 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** 工作量增加与寄存器增长是不同瓶颈。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| M/N warp 排布 | 计算组沿输出行/列方向的组织 |
| M4N2 | 本书某一 kernel 的 4×2 warp 排布名称，不是通用最优参数 |
| spill | 寄存器不足导致值放到 local memory |
| 跨 N 归约 | 同一行分布在多个组，需要合并 softmax 统计 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 先看总输出元素
若 Br、Dv 和线程数都不变，平均每线程输出元素 Br*Dv/T 不会仅因改名 M4N2 而下降。原书具体方案还改变 Br 等配置；必须把完整配置写出来才能解释寄存器变化。

### N 方向分工带来交换
把一行的列分到多个 warp 后，softmax 的 max/sum 可能需要跨 warp 合并，P 到 PV 的交接也可能需要 shared round-trip。减少每线程状态，可能增加通信和同步。

### 分别验证两种预算
shared 预算用 chunk 尺寸与 stage 数计算，寄存器预算用完整累加器和活跃临时值估计。然后看编译器报告与 local load/store，最后看总时间。occupancy 上升本身不等于性能上升。

## 公式：从定义到可核对的小例子

```text
O_regs_avg = Br*Dv/consumer_threads
Br=128,Dv=512,T=256 → 256
Br=64,Dv=512,T=256 → 128
减半来自完整配置中 Br 的变化，不是仅把 8×1 改成 4×2 的名字
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-fp8-split-d-m4n2.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-fp8-split-d-m4n2.svg" alt="FP8 大 D：Split-D 与并行布局的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-fp8-split-d-m4n2.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**分别估算 shared / registers → 沿 D chunk 降搬运容量 → 选择 M/N 排布与 Br → 合并行统计 / P 交接 → 检查 spill → 正确性 → 时间**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/37/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`Dv=128,consumer_threads=256`。到最后一步应满足：寄存器逻辑下界按真实 tile 计算，代价单独列出。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/37/cuda-fp8-split-d-m4n2.zip) · [独立 CPU 源码](/downloads/leetcuda/37/demo.py) · [运行说明](/downloads/leetcuda/37/README.txt) · [练习说明](/downloads/leetcuda/37/EXERCISES.txt)

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


def lab37(case):
    d = [128, 256, 512, 1024][case]
    a = 128 * d // 256
    b = 64 * d // 256
    assert a == 2 * b
    return result(
        f"Dv={d},consumer_threads=256",
        [
            step("Br=128 的累加器下界", {"values_per_thread": a}, "忽略全部其它临时寄存器。"),
            step("Br=64 的累加器下界", {"values_per_thread": b}, "下降来自实际输出块减小。"),
            step(
                "代价表",
                {"Q_CTA_count_ratio": 2, "new_cost": "跨组统计 / P shared 交接 / 更多任务"},
                "仅更改布局名称并不改变元素总数。",
            ),
        ],
        "寄存器逻辑下界按真实 tile 计算，代价单独列出。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab37(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** D=64/128/256/512 的分块 QK attention；M4N2 排布由专用 CUDA 路径提供。

```bash
python run_gpu.py --chapter 37 --backend triton
python run_gpu.py --chapter 37 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/vipshop/ffpa-attn/blob/861d75ee87824a14a9ebfe295d54c7dac3e3cabc/csrc/cuffpa/cute/fp8/sm_120/split_d_m4n2.cuh)。Linux NVIDIA SM120 环境；先按固定版本 README 准备 CUDA PyTorch、Python≥3.10、构建依赖和 CUDA 扩展。此入口运行整条量化 attention，前处理、主循环、误差报告一起观察。大 D 路径是否编入以构建的 headdim 配置为准。

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

**Triton 任务：** 大 D 的分块与任务划分对照。

**TileLang 任务：** M/N 方向的 warp 排布与流水调整。

Triton 对不同 Br 与 D chunk 做资源扫描；TileLang 对 M/N policy 变化画出 row ownership。至少包含 D=128/256/512 的语义参考和目标支持的大 D 路径，记录无法编译和 spill 配置。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 37 --backend triton --student student_triton
python run_gpu.py --chapter 37 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 37 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 并行度、spill 与耗时分别记录；不能仅看 occupancy。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

书中按 D 的切换阈值是该实现与设备的调参结果，不能作为所有 GPU 的固定阈值。低精度 causal 误差还需逐行检查。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（三十九）FP4 Attention 的量化链](/llm/cuda-fp4-attention-preprocess/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
