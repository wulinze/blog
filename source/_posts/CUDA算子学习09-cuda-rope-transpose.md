---
title: CUDA 算子学习（九）：RoPE 与转置：数据重排为何影响性能
description: 旋转配对、相对位置性质、global 合并与 shared bank。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-rope-transpose/
date: 2026-11-02 17:00:08
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-normalization/) · 下一篇：[（十）GEMV：如何把一行交给 warp](/llm/cuda-gemv/)

RoPE 改变数值，转置改变位置。两者放在一起学习，是为了训练“逻辑坐标 → 物理地址”的思维：数学公式正确还不够，配对、stride、bank 和尾部都属于正确性的一部分。

本篇接续系列第 9 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 8 章（PDF 阅读器第 88–98 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** 旋转配对、相对位置性质、global 合并与 shared bank。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| RoPE | 对一对特征按位置角度做二维旋转 |
| 配对格式 | 相邻 (2i,2i+1) 或半区 (i,i+D/2)，不可混用 |
| 转置 | 逻辑坐标 (r,c) 变为 (c,r) |
| bank conflict | 同一 shared 请求访问同 bank 不同字的竞争，区别于同地址广播 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 旋转的两个不变量
一对 (x,y) 经过角度 theta 后，x′=x*cos(theta)-y*sin(theta)，y′=x*sin(theta)+y*cos(theta)。旋转保持平方范数；两个位置 p、q 的旋转内积只取决于角度差。每一对特征的频率不同，实际角度通常是 position*frequency，不能让所有通道共用一个频率。

### global 两端都要连续
按行读矩阵并直接按转置位置写，通常一端连续、一端跨距。shared tile 中转把“读入时的线程分工”和“写出时的线程分工”解开：合并载入、block 屏障、换坐标读取 shared、合并写出。矩形尾块对读与写分别检查边界。

### padding 改变 bank 不是改变数学
在简化 32-bank、每 bank 4 字节的 float 模型中，列读取 tile[lane][c]：行跨度 32 时各 lane 落到同 bank；跨度 33 时 bank=(33*lane+c)%32，遍历不同 bank。真实宽向量与 ldmatrix 请求需按其事务分组重新分析，不能套用这个标量模型。

## 公式：从定义到可核对的小例子

```text
x′²+y′² = x²+y²
bank = floor(byte_address/4) mod 32
float tile[32][32]: bank(lane,c)=c
float tile[32][33]: bank(lane,c)=(lane+c)%32
例：(1,0) 旋转 pi/2 得到 (0,1)
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-rope-transpose.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-rope-transpose.svg" alt="RoPE 与转置：数据重排为何影响性能的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-rope-transpose.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**特征配对 / 矩阵坐标 → RoPE 读一对并旋转 → 转置合并读 → shared → 屏障 → 换线程读取 → 合并写出并检查尾部**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/08/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`theta=0.000，shared stride=32`。到最后一步应满足：旋转范数保持；stride32/33 的 bank 覆盖符合手算。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/08/cuda-rope-transpose.zip) · [独立 CPU 源码](/downloads/leetcuda/08/demo.py) · [运行说明](/downloads/leetcuda/08/README.txt) · [练习说明](/downloads/leetcuda/08/EXERCISES.txt)

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


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def step(title, values, detail):
    return {"title": title, "values": values, "detail": detail}


def result(label, steps, check):
    return {"label": label, "steps": steps, "check": check}


def lab08(case):
    theta = [0, math.pi / 4, math.pi / 2, math.pi][case]
    x, y = 3.0, 4.0
    out = [x * math.cos(theta) - y * math.sin(theta), x * math.sin(theta) + y * math.cos(theta)]
    close(dot(out, out), 25)
    stride = 32 if case % 2 == 0 else 33
    banks = [(lane * stride) % 32 for lane in range(32)]
    assert len(set(banks)) == (1 if stride == 32 else 32)
    return result(
        f"theta={theta:.3f}，shared stride={stride}",
        [
            step(
                "旋转前后",
                {"before": [x, y], "after": out, "norm_squared": dot(out, out)},
                "RoPE 单个坐标对；真实各对有各自频率。",
            ),
            step("列读取 bank", banks, "教学 float32、32 bank 模型，列号为 0。"),
            step(
                "不同 bank 数",
                {"unique_banks": len(set(banks)), "stride": stride},
                "padding 改变地址，不改变转置的数学。",
            ),
        ],
        "旋转范数保持；stride32/33 的 bank 覆盖符合手算。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab08(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** 矩形转置的 gather 语义；RoPE 和 shared padding 是扩展题。

```bash
python run_gpu.py --chapter 8 --backend triton
python run_gpu.py --chapter 8 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/xlite-dev/LeetCUDA/blob/3b52c5cd3332c5e992c7aef63087d43f67f23b8e/kernels/interview/book/tests/ch07_rope_transpose.cu)。CUDA C++20、NVCC、对应目标 GPU 与 CUTLASS submodule。TMA 相关例子还需满足该版本头文件的 CUDA toolkit 要求。

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git LeetCUDA-lab
git -C LeetCUDA-lab checkout --detach 3b52c5cd3332c5e992c7aef63087d43f67f23b8e
git -C LeetCUDA-lab submodule update --init --recursive
python3 run_upstream.py --repo ./LeetCUDA-lab --arch sm_80
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** 两种 RoPE 配对格式；带尾部转置。

**TileLang 任务：** shared 中转转置；padding 布局实验。

Triton 实现两种 RoPE 配对和带 mask 转置；TileLang 实现 shared 中转，比较 stride 32/33。先检查范数，再检查相对位置点积；转置用 17×35，做两次必须回到原矩阵。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 8 --backend triton --student student_triton
python run_gpu.py --chapter 8 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 8 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 旋转范数近似保持，转置两次还原；手算真实 lane 地址后再判断冲突。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

padding 不总是最优；额外 shared 可能降低可驻留 CTA。旋转的相邻配对和半区配对都合理，但必须匹配模型权重的约定。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（十）GEMV：如何把一行交给 warp](/llm/cuda-gemv/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
