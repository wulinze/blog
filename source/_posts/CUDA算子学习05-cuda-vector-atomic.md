---
title: CUDA 算子学习（五）：访问粒度与原子冲突：向量化和直方图
description: 区分合并访存、单线程宽访存和原子串行化。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-vector-atomic/
date: 2026-11-02 17:00:04
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-reduction/) · 下一篇：[（六）稳定 Softmax：从整行到在线状态](/llm/cuda-softmax/)

逐元素加法很适合认识地址，直方图很适合认识竞争。它们都只做少量算术，但慢的原因不相同：跨距读扩大请求流量；标量循环可能增加指令；热点原子则让多个线程争同一个地址。先把三件事分开，才知道向量化能解决哪一件。

本篇接续系列第 5 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 4 章（PDF 阅读器第 52–59 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** 区分合并访存、单线程宽访存和原子串行化。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| 合并访问 | 同一条 warp 指令的地址合成 sector 请求；不是把多个循环合成一条指令 |
| 向量化 | 单线程一次读写多个相邻元素，例如 float4 的 16 字节 |
| 原子操作 | 对一个地址执行不可被其他更新拆开的读改写 |
| 热点 | 许多线程更新同一个 bin，使更新出现竞争 |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 从 lane 地址数 sector
给 32 个 lane 各读一个 float32。地址为 `base + 4*lane*stride`，先假设 base 按 32 字节对齐。stride=1 覆盖 4 个 sector；stride=2 覆盖 8 个；stride=8 覆盖 32 个。每次仍只要求 128 字节有效数据。这里统计的是一次请求的地址覆盖，不是断言这些字节全从 DRAM 搬来；缓存命中会改变下层流量。

### 宽访问还需要对齐和尾部
float4 把线程的四个相邻 float 作为一个 16 字节对象。起始地址必须满足该类型对齐，四个元素必须都有效，否则退回标量尾部。改变线程映射后，应重新枚举整个 warp 的地址。宽 load 不会自动把散乱地址变连续；编译器是否生成 128-bit 指令也要检查实际输出。

### 直方图先局部合并，再全局合并
输入整数 x，输出 `hist[b] = count(x==b)`。直接 atomicAdd 正确，但一个热点 bin 会收到大量同址请求。block 私有 shared histogram 先把更新聚合，最后把每个 bin 的局部计数原子加到全局。它增加清零、同步、shared 容量和最后合并的成本；当 bin 多、数据少时未必更快。每次运行前都要清零输出，计时必须包括所选方案必要的清零与合并。

## 公式：从定义到可核对的小例子

```text
S = count(unique(floor(address / 32)))
request_efficiency = useful_bytes / (32*S)
32 lanes × 4 bytes = 128 useful bytes
stride=1: 128/(32*4)=1; stride=8: 128/(32*32)=0.125
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-vector-atomic.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-vector-atomic.svg" alt="访问粒度与原子冲突：向量化和直方图的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-vector-atomic.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**lane → 元素地址 → 地址 → 32B sector → 连续 / 跨距请求对照 → 输入 → 局部 histogram → 全局合并 → 总数校验**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/04/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`stride=1，均匀 histogram`。到最后一步应满足：sector 覆盖与独立地址集合一致；hist 总数守恒。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/04/cuda-vector-atomic.zip) · [独立 CPU 源码](/downloads/leetcuda/04/demo.py) · [运行说明](/downloads/leetcuda/04/README.txt) · [练习说明](/downloads/leetcuda/04/EXERCISES.txt)

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


def lab04(case):
    stride = [1, 2, 4, 8][case]
    addr = [4 * lane * stride for lane in range(32)]
    sectors = [a // 32 for a in addr]
    expected = [4, 8, 16, 32][case]
    assert len(set(sectors)) == expected
    keys = [i % 4 if case < 2 else 0 for i in range(32)]
    hist = [keys.count(i) for i in range(4)]
    assert sum(hist) == len(keys)
    return result(
        f"stride={stride}，{'均匀' if case < 2 else '热点'} histogram",
        [
            step("32 lane 地址", addr, "每 lane 读一个 4 字节 float；base=0，按 32B 对齐。"),
            step("对应 sector 编号", sectors, "同编号合成请求；不是 DRAM 实测流量。"),
            step(
                "请求效率",
                {"sectors": expected, "useful_bytes": 128, "request_bytes": 32 * expected, "efficiency": 4 / expected},
                "向量宽度变化时必须重算分子分母。",
            ),
            step("直方图与竞争", hist, "计数总和必须为 32；热点使同址原子集中。"),
        ],
        "sector 覆盖与独立地址集合一致；hist 总数守恒。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab04(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** 整数 histogram：全局 atomic_add，覆盖均匀与单热点；shared 私有化是扩展题。

```bash
python run_gpu.py --chapter 4 --backend triton
python run_gpu.py --chapter 4 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/xlite-dev/LeetCUDA/blob/3b52c5cd3332c5e992c7aef63087d43f67f23b8e/kernels/interview/book/tests/ch03_elementwise.cu)。CUDA C++20、NVCC、对应目标 GPU 与 CUTLASS submodule。TMA 相关例子还需满足该版本头文件的 CUDA toolkit 要求。

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git LeetCUDA-lab
git -C LeetCUDA-lab checkout --detach 3b52c5cd3332c5e992c7aef63087d43f67f23b8e
git -C LeetCUDA-lab submodule update --init --recursive
python3 run_upstream.py --repo ./LeetCUDA-lab --arch sm_80
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** 连续 / 跨距访问；整数 histogram。

**TileLang 任务：** 宽 copy；每 block 局部 histogram 后合并。

Triton：先用 tl.arange 和 mask 写 stride copy，再用整数 atomic_add 做 histogram。TileLang：先写同样的全局原子基线，再引入 shared 私有计数。输入 [0,1,1,3,3,3] 的输出应为 [1,2,0,3]；均匀与全零两类输入都检查 sum(hist)==N。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 4 --backend triton --student student_triton
python run_gpu.py --chapter 4 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 4 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 同一有效数据量比较请求；均匀与热点分布均正确，直方图总数守恒。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

4 sector 的理想值只适用于这里的 32 个活跃 lane、每 lane 4 字节且对齐的访问。不要把 float4 的 16 sector 直接判为合并失败。原子吞吐高低必须结合热点分布；不能拿一组均匀随机输入代替全部应用。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（六）稳定 Softmax：从整行到在线状态](/llm/cuda-softmax/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
