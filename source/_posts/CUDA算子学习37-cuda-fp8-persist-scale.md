---
title: CUDA 算子学习（三十七）：FP8 persist-D 与 scale 折叠
description: 保持 D 的注意力流水与合法的 scale 外提。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-fp8-persist-scale/
date: 2026-11-02 17:00:36
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-fp8-attention-preprocess/) · 下一篇：[（三十八）FP8 大 D：Split-D 与并行布局](/llm/cuda-fp8-split-d-m4n2/)

量化值参与 MMA 后得到的不是原始实数分数。scale 必须在 softmax 前作用到 logits；PV 的 scale 则要依据它是否随归约 key 改变，决定能否延后处理。

本篇接续系列第 37 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 36 章（PDF 阅读器第 524–534 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** 保持 D 的注意力流水与合法的 scale 外提。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| persist-D | 本实现把完整 head dimension 的有关状态保留在片上 |
| scale folding | 在条件允许时把比例因子合并进其它运算 |
| exp2 域 | 使用 2 为底的指数，需要乘 log2(e) |
| 输出修正 | PV 的 scale 与平滑均值在正确位置乘回或加回 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### QK scale 不能放到 softmax 外
如果同一 query-key 点积的 scale 乘积为 sQ*sK，则分数为 dot8*sQ*sK/sqrt(D)。softmax 非线性，不能先对 dot8 做 softmax，再乘 sQ*sK。使用 exp2 时系数还需乘 log2(e)。

### P/V scale 的消去要逐项证明
如果 V scale 只按输出通道变化，它可以从 key 归约外提；若随 key block 变化，则每个 block 的 PV partial 都要按自身 scale 处理。P 的公共缩放若同时进入未归一化分子和分母，才可能在最终除法中抵消。

### Lazy rescale 需要范围保证
跳过某次重缩放可能让中间指数大于 1。低精度 P 的可表示范围和累加类型会限制它是否安全。优化不能只证明最终实数式等价，还要检查中间范围、舍入与溢出。

## 公式：从定义到可核对的小例子

```text
score = dot(Q8,K8)*sQ*sK/sqrt(D)
exp(score-m)=exp2((score-m)*log2(e))
若 V[j,d]=sV[d]*V8[j,d]：O[d]=sV[d]*sum_j(p[j]*V8[j,d])
若 sV=sV[j,d]，不能直接把一个 sV[d] 外提
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-fp8-persist-scale.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-fp8-persist-scale.svg" alt="FP8 persist-D 与 scale 折叠的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-fp8-persist-scale.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**消费 Q8/K8/V8 与 scales → QK MMA → 恢复 score 尺度 → online softmax / P 量化 → PV 按合法尺度累加 → 最终归一化与输出修正**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/36/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`QK scale 乘积 / sqrt(D)=0.25`。到最后一步应满足：exp 与 exp2 路径等价；演示非法外提反例。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/36/cuda-fp8-persist-scale.zip) · [独立 CPU 源码](/downloads/leetcuda/36/demo.py) · [运行说明](/downloads/leetcuda/36/README.txt) · [练习说明](/downloads/leetcuda/36/EXERCISES.txt)

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


def lab36(case):
    scale = [0.25, 1, 2, 4][case]
    raw = [-1, 0, 2]
    v = [1, 4, 8]
    correct = dot(softmax([s * scale for s in raw]), v)
    wrong = dot(softmax(raw), v) * scale
    exp2 = [2 ** ((s - max(raw)) * scale * math.log2(math.e)) for s in raw]
    via_exp2 = dot([p / sum(exp2) for p in exp2], v)
    close(correct, via_exp2)
    return result(
        f"QK scale 乘积 / sqrt(D)={scale}",
        [
            step("正确 score 缩放", [s * scale for s in raw], "先恢复 logits 尺度，再 softmax。"),
            step("自然指数与 exp2", {"O_exp": correct, "O_exp2": via_exp2}, "log2(e) 只转换指数底，不改变数学。"),
            step(
                "错误把 scale 移出 softmax",
                {"wrong": wrong, "difference": wrong - correct},
                "scale=1 恰好不暴露错误，必须多 case。",
            ),
        ],
        "exp 与 exp2 路径等价；演示非法外提反例。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab36(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** 把 QK scale 正确放进 softmax 之前的 attention；FP8 专用折叠实现见 CUDA demo。

```bash
python run_gpu.py --chapter 36 --backend triton
python run_gpu.py --chapter 36 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/vipshop/ffpa-attn/blob/861d75ee87824a14a9ebfe295d54c7dac3e3cabc/csrc/cuffpa/cute/fp8/sm_120/persist_d.cuh)。Linux NVIDIA SM120 环境；先按固定版本 README 准备 CUDA PyTorch、Python≥3.10、构建依赖和 CUDA 扩展。此入口运行整条量化 attention，前处理、主循环、误差报告一起观察。大 D 路径是否编入以构建的 headdim 配置为准。

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

**Triton 任务：** 固定 D 的量化 attention 与反量化参考。

**TileLang 任务：** scale 融入 QK / PV 与输出修正。

Triton 和 TileLang 分别写显式反量化版与 scale 折叠版。先在无量化舍入的小例子中证明等价，再用 FP8 输入比较误差；加入跨 key 改变 scale 的反例，防止非法外提。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 36 --backend triton --student student_triton
python run_gpu.py --chapter 36 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 36 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 证明每个外提条件，动态 range 与不同 shape 下误差达标。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

代数抵消并不等于量化后 bitwise 抵消；在缩放前后分别舍入会产生不同结果。形状支持范围以具体 dispatch 与测试为准。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（三十八）FP8 大 D：Split-D 与并行布局](/llm/cuda-fp8-split-d-m4n2/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
