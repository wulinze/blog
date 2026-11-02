---
title: CUDA 算子学习（四十一）：基准与精度：把学习结果变成可复现实验
description: 环境、计时、缓存、误差统计和生态约束。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-benchmark-accuracy/
date: 2026-11-02 17:00:40
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-fp4-attention-probability/)

最后一章把前面的学习变成可复现实验。一个有用的结论必须同时告诉读者：算了同一件事吗、结果误差怎样、计时范围是什么、在什么硬件和软件条件下成立。

本篇接续系列第 41 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 40 章（PDF 阅读器第 561–570 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** 环境、计时、缓存、误差统计和生态约束。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| kernel-only | 仅主 kernel 的稳态设备时间 |
| 端到端 | 应用必须执行的初始化、预处理、kernel、后处理范围 |
| 误差分布 | 最大值、相对 L2、分位数与失败行，不能只报均值 |
| 参照后端 | 明确实际执行的库/算法，不能只写 auto |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 先正确，后计时
固定随机种子，覆盖边界、极值与业务形状。保存失败输入、配置和误差位置。原子输出要每次初始化；异步 GPU 运行要用正确设备事件和同步边界。JIT 编译、热身和实际计时分开。

### 后端与输入必须固定
基线库可能随 D、mask 或 dtype 选择不同算法。速度比突然上升时，先检查是不是基线变慢。记录实际后端、TF32/累加精度、是否 causal、是否包含量化，不把不同功能集合直接比较。

### 汇报分布和完整成本
重复测量并报告中位数与分位数，最小值可作辅助但不能独占结论。量化前处理占比为 f，主 kernel 加速 s 时，整体上限取决于 Amdahl 模型。记录温度、频率、功耗状态和 A/B 交错顺序，有助于解释漂移。

## 公式：从定义到可核对的小例子

```text
speedup = T_baseline/T_candidate
T_e2e = T_prepare+T_kernel+T_finalize
S_e2e = 1/(f+(1-f)/s)，f 为未被加速部分的基线时间占比
f=0.4,s=2 → S_e2e=1/0.7≈1.429
relative_L2 = norm(y-y_ref)/max(norm(y_ref),epsilon)
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-benchmark-accuracy.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-benchmark-accuracy.svg" alt="基准与精度：把学习结果变成可复现实验的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-benchmark-accuracy.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**冻结契约 / 版本 / 后端 → 边界与精度校验 → 热身与设备同步 → 重复 A/B 计时完整范围 → 保存分布、失败例与复现命令**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/40/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`未加速时间占比 f=0`。到最后一步应满足：端到端加速符合 Amdahl 推导；示例时间明确标为构造。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/40/cuda-benchmark-accuracy.zip) · [独立 CPU 源码](/downloads/leetcuda/40/demo.py) · [运行说明](/downloads/leetcuda/40/README.txt) · [练习说明](/downloads/leetcuda/40/EXERCISES.txt)

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
import statistics


def close(a, b, tol=1e-9):
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), (a, b)


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab40(case):
    prep = [0, 0.2, 0.4, 0.8][case]
    kernel_speedup = 2
    total = prep + (1 - prep) / kernel_speedup
    samples = [total * x for x in [1.02, 0.99, 1, 1.05, 1.01]]
    close(1 / total, 1 / (prep + (1 - prep) / 2))
    return result(
        f"未加速时间占比 f={prep}",
        [
            step("基线总时间=1", {"prepare": prep, "kernel": 1 - prep}, "所有时间是构造数据，不是设备 benchmark。"),
            step(
                "主 kernel 加速 2 倍后",
                {"prepare": prep, "kernel": (1 - prep) / 2, "total": total, "e2e_speedup": 1 / total},
                "不能把 kernel-only 的 2 倍写成端到端 2 倍。",
            ),
            step(
                "构造的重复测量",
                {"samples": samples, "median": statistics.median(samples), "minimum": min(samples)},
                "真实实验保存完整分布与环境。",
            ),
        ],
        "端到端加速符合 Amdahl 推导；示例时间明确标为构造。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab40(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** 相同输入契约的 attention + 设备事件计时，输出误差和完整语义基线时间。

```bash
python run_gpu.py --chapter 40 --backend triton
python run_gpu.py --chapter 40 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/vipshop/ffpa-attn/blob/861d75ee87824a14a9ebfe295d54c7dac3e3cabc/csrc/cuffpa/cute/fp8/fp8_pscale.cuh)。Linux NVIDIA SM120 环境；先按固定版本 README 准备 CUDA PyTorch、Python≥3.10、构建依赖和 CUDA 扩展。此入口运行整条量化 attention，前处理、主循环、误差报告一起观察。大 D 路径是否编入以构建的 headdim 配置为准。

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

**Triton 任务：** 统一测试 harness 与多形状报表。

**TileLang 任务：** 相同 harness 跑等价语义实现。

Triton 与 TileLang 接入同一个 harness，导出 JSON：设备、版本、shape、dtype、误差、kernel 与 e2e 时间。固定参照后端，保留失败配置；禁止把 CPU 时间或未同步 wall time 标成 CUDA kernel 延迟。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 40 --backend triton --student student_triton
python run_gpu.py --chapter 40 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 40 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 同时交付正确性、精度分布、端到端时间、条件和失败案例。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

较宽容差不是正确性的豁免。软件 fallback、SKIP、编译成功、设备正确性通过和性能改进必须分列，任何一个都不能冒充另一个。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
