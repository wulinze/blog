---
title: CUDA 算子学习（二十七）：Attention 实现对照：统一流水、角色分工与双 consumer
description: 同一数学接口下比较不同数据流组织。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-cute-attention-variants/
date: 2026-11-02 17:00:26
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-cute-gemm/) · 下一篇：[（二十八）大 D 的布局代数与寄存器预算](/llm/cuda-cute-split-d-layout/)

同一个 attention 数学接口可以有不同执行结构。比较这些实现时，先把接口和数值约定统一，再解释流水、布局与分工的差别，否则测试结果无法归因。

本篇接续系列第 27 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 26 章（PDF 阅读器第 378–396 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** 同一数学接口下比较不同数据流组织。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| 统一流水 | 同一计算组负责搬运与计算 |
| WS | warp specialization，按角色分离生产与消费 |
| 双 consumer | 两路局部状态与一次最终 merge |
| smoke test | 仅验证某条链路的最小自检，不是完整算子验收 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 三条路径保留同一数学状态
统一 cp.async 方案、TMA+WS 方案和双 consumer 方案都必须产生同一逻辑输出。前两者可以维护一份行状态；第三者拆 key 集合后需要合并。Q/K/V 的形状、scale、mask 和输出类型必须一致。

### 逐层缩小错误范围
若完整 attention 不对，先做 TMA copy smoke；再只做 QK；再检查 max/l；再接 PV；最后检查 epilogue。每层用位置编码数据和小尺寸参考，比只看最终 allclose 更有信息。

### 公平对照需要完整路径
同样的 BM/BK、相同累加类型和相同 prefill/decode 形状下比较。双 consumer 的终态合并、所有预处理、必要初始化都计入范围。编译缓存与首次 JIT 不计入 steady-state kernel 时间，但另行记录首次调用成本。

## 公式：从定义到可核对的小例子

```text
O_ref = softmax(Q@K.T/sqrt(D)+mask)@V
路径 A/B/C 的输出形状均为 [Nq,Dv]
比较量：max_abs、相对 L2、完整延迟、寄存器/shared
copy smoke PASS ≠ attention PASS ≠ 性能改进
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-cute-attention-variants.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-cute-attention-variants.svg" alt="Attention 实现对照：统一流水、角色分工与双 consumer的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-cute-attention-variants.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**冻结统一输入契约 → copy smoke / QK 单独校验 → 三种流水执行相同数学 → 合并与 epilogue 完成 → 统一误差与总时间比较**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/26/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`N=2，同一输入三种数学组织`。到最后一步应满足：三种组织与独立完整参考对齐。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/26/cuda-cute-attention-variants.zip) · [独立 CPU 源码](/downloads/leetcuda/26/demo.py) · [运行说明](/downloads/leetcuda/26/README.txt) · [练习说明](/downloads/leetcuda/26/EXERCISES.txt)

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


def online(scores, values, block):
    m, total, accum = -math.inf, 0.0, 0.0
    snapshots = []
    for start in range(0, len(scores), block):
        s, v = scores[start : start + block], values[start : start + block]
        new_m = max(m, max(s))
        alpha = math.exp(m - new_m) if total else 0.0
        p = [math.exp(x - new_m) for x in s]
        total, accum = alpha * total + sum(p), alpha * accum + dot(p, v)
        m = new_m
        partial_ref = dot(softmax(scores[: start + block]), values[: start + block])
        close(accum / total, partial_ref)
        snapshots.append(
            step(
                f"处理 key [{start},{min(start + block, len(scores))})",
                {
                    "scores": s,
                    "m": m,
                    "alpha": alpha,
                    "denominator": total,
                    "u": accum,
                    "O": accum / total,
                },
                "旧 denominator 和旧 u 同乘 alpha；本块 p 以新 m 为指数基准。",
            )
        )
    return (accum / total if total else 0.0), snapshots


def state(scores, values):
    if not scores:
        return 0.0, -math.inf
    m = max(scores)
    return dot(softmax(scores), values), m + math.log(sum(math.exp(s - m) for s in scores))


def merge(a, b):
    oa, la = a
    ob, lb = b
    if la == -math.inf:
        return b
    if lb == -math.inf:
        return a
    c = max(la, lb)
    wa, wb = math.exp(la - c), math.exp(lb - c)
    return (wa * oa + wb * ob) / (wa + wb), c + math.log(wa + wb)


def lab26(case):
    scores = [-2, 1, 0, 4, 3][: case + 2]
    values = list(range(2, 2 + len(scores)))
    direct = dot(softmax(scores), values)
    one, _ = online(scores, values, 2)
    two = merge(state(scores[::2], values[::2]), state(scores[1::2], values[1::2]))[0]
    close(direct, one)
    close(direct, two)
    return result(
        f"N={len(scores)}，同一输入三种数学组织",
        [
            step("完整参考", {"O": direct}, "materialize 的数学参考。"),
            step("单状态在线", {"O": one}, "对应统一/WS 流水共同的数学状态。"),
            step("双状态 + merge", {"O": two}, "这里只验证组织等价，不模拟 CUDA 性能。"),
        ],
        "三种组织与独立完整参考对齐。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab26(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** 小尺寸 attention 语义基线，包含明确的 top-left causal mask；各章专用流水、布局与分工见下面的固定版本 CUDA demo。

```bash
python run_gpu.py --chapter 26 --backend triton
python run_gpu.py --chapter 26 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/xlite-dev/LeetCUDA/blob/3b52c5cd3332c5e992c7aef63087d43f67f23b8e/kernels/interview/book/tests/ch25_fa_cute.cu)。CUDA C++20、NVCC、对应目标 GPU 与 CUTLASS submodule。TMA 相关例子还需满足该版本头文件的 CUDA toolkit 要求。

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git LeetCUDA-lab
git -C LeetCUDA-lab checkout --detach 3b52c5cd3332c5e992c7aef63087d43f67f23b8e
git -C LeetCUDA-lab submodule update --init --recursive
python3 run_upstream.py --repo ./LeetCUDA-lab --arch sm_80
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** 对同一输入公平比较普通与 specialized 路径。

**TileLang 任务：** 对同一输入比较 cp.async / TMA 方案。

两语言均先跑同一个测试矩阵：N=1/65/257、D=64/128、dense/causal。只改变一项流水开关，记录实际选择的内核，不允许静默 fallback 冒充指定路径。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 26 --backend triton --student student_triton
python run_gpu.py --chapter 26 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 26 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 统一 mask、精度、计时范围；不把框架差异都归因于某一条指令。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

书中三种命名属于该仓库的实现组织，不等于所有同名论文的完整实现。框架与编译器变化的收益不能都归到 TMA 指令。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（二十八）大 D 的布局代数与寄存器预算](/llm/cuda-cute-split-d-layout/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
