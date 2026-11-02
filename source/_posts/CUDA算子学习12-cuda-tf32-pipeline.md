---
title: CUDA 算子学习（十二）：精度与异步搬运：TF32 和双缓冲
description: 区分精度改变、Tensor Core 通路与 cp.async 时序。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-tf32-pipeline/
date: 2026-11-02 17:00:11
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-gemm-tiling/) · 下一篇：[（十三）MMA 与 fragment：理解协作计算的数据归属](/llm/cuda-mma-fragments/)

精度改变和流水重叠是两条独立的优化轴。前者让计算进入不同吞吐的指令路径，后者减少等数据的时间。实验必须分别改变这两项，才能解释收益来源。

本篇接续系列第 12 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 11 章（PDF 阅读器第 118–128 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** 区分精度改变、Tensor Core 通路与 cp.async 时序。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| TF32 | 8 位指数、10 位显式尾数的矩阵乘输入精度，存储通常仍是 FP32 |
| cp.async | 异步 global→shared 搬运指令族 |
| 双缓冲 | 当前 tile 与下一 tile 使用不同存储槽 |
| prologue / drain | 流水填充与排空，短任务不可忽略 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### TF32 先做误差账
FP32 的尾数保留更少位后，乘法的输入发生舍入；FP32 累加不能恢复已经丢掉的输入信息。指数范围相近不表示数值完全相同。TF32x3 等软件组合路径也不能与一次 TF32 乘法混为一谈。

### 双缓冲是依赖图的重排
串行版每块都是 load→compute。双缓冲先加载 tile0，在计算 tile0 时提前发 tile1 的 load。使用 tile1 前必须等待其搬运完成；覆盖 tile0 的槽前必须确认全部读者结束。commit/wait 的 group 是发起线程的异步拷贝集合，跨线程可见性还要遵守完整同步协议。

### 增加 stage 的收益有上限
在理想独立搬运/计算模型里，稳态每块时间靠近 max(L,C)，首块仍需填充，末块仍需排空。stage 占 shared；缓冲增大可能减少 resident CTA，反而降低延迟掩盖。只看到 cp.async 指令不等于已证明 overlap。

## 公式：从定义到可核对的小例子

```text
T_serial = tiles*(L+C)
T_ideal = L+C+(tiles-1)*max(L,C)
例：4 tiles, L=3, C=5 → 32 与 23 时间单位
这些是理想模型值，不是 GPU 时钟或实测加速比
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-tf32-pipeline.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-tf32-pipeline.svg" alt="精度与异步搬运：TF32 和双缓冲的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-tf32-pipeline.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**先核对输入精度 → 填充 slot 0 → 提前加载 slot 1 → wait → 计算当前 slot → 读者完成 → 复用槽 / 排空**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/11/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`搬运 L=1，计算 C=4，4 tiles`。到最后一步应满足：理想时间线符合推导，且未将模型速度当实测。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/11/cuda-tf32-pipeline.zip) · [独立 CPU 源码](/downloads/leetcuda/11/demo.py) · [运行说明](/downloads/leetcuda/11/README.txt) · [练习说明](/downloads/leetcuda/11/EXERCISES.txt)

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


def lab11(case):
    count, load, compute = 4, [1, 3, 5, 8][case], 4
    serial = count * (load + compute)
    ideal = load + compute + (count - 1) * max(load, compute)
    timeline = []
    for t in range(count):
        start = load + t * max(load, compute)
        timeline.append([t, start, start + compute])
    assert ideal <= serial and timeline[-1][-1] == ideal
    v = 1.0003
    rounded = round(v * 1024) / 1024  # local binade [1,2), educational RNE
    return result(
        f"搬运 L={load}，计算 C={compute}，4 tiles",
        [
            step("串行时间", {"time": serial}, "单位是教学时间单位，不是 GPU 周期。"),
            step(
                "理想独立流水 compute 区间 [tile,start,end]", timeline, "假设可提前搬运、无资源争用；真实硬件需另测。"
            ),
            step(
                "精度与时间分别记录",
                {"ideal_time": ideal, "x": v, "10_fraction_bit_value": rounded, "input_error": rounded - v},
                "该舍入示例只覆盖 [1,2)；不是完整 TF32 编码器。",
            ),
        ],
        "理想时间线符合推导，且未将模型速度当实测。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab11(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** 相同 FP32 GEMM 正确性基线；TF32 与异步 stages 的改变是扩展题。

```bash
python run_gpu.py --chapter 11 --backend triton
python run_gpu.py --chapter 11 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/xlite-dev/LeetCUDA/blob/3b52c5cd3332c5e992c7aef63087d43f67f23b8e/kernels/interview/book/tests/ch10_sgemm_tf32.cu)。CUDA C++20、NVCC、对应目标 GPU 与 CUTLASS submodule。TMA 相关例子还需满足该版本头文件的 CUDA toolkit 要求。

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git LeetCUDA-lab
git -C LeetCUDA-lab checkout --detach 3b52c5cd3332c5e992c7aef63087d43f67f23b8e
git -C LeetCUDA-lab submodule update --init --recursive
python3 run_upstream.py --repo ./LeetCUDA-lab --arch sm_80
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** 切换 input_precision；分开报告误差与时间。

**TileLang 任务：** num_stages 对照；检查生成的 async 指令。

Triton 比较 input_precision="ieee" 与 "tf32"，固定形状分别报误差；TileLang 固定数学类型先扫 num_stages。记录生成代码、寄存器、shared 和时间，短 K 与长 K 各测一组。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 11 --backend triton --student student_triton
python run_gpu.py --chapter 11 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 11 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 必须先报告精度，再谈速度；只看到异步指令不算 overlap 已被证明。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

不能把“一条矩阵指令的 FLOPs”直接当成一个时钟的吞吐。指令延迟、发射率、硬件通路和资源限制都需要实际设备上的证据。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（十三）MMA 与 fragment：理解协作计算的数据归属](/llm/cuda-mma-fragments/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
