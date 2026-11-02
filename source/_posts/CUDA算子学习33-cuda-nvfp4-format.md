---
title: CUDA 算子学习（三十三）：NVFP4：两级 scale 与打包
description: 4-bit 线格式、global/block scale、误差来源。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/cuda-nvfp4-format/
date: 2026-11-02 17:00:32
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇](/llm/cuda-fp8-gemm/) · 下一篇：[（三十四）Block-scaled MMA：scale 也是操作数](/llm/cuda-block-scaled-mma/)

4-bit 数据的格点很稀疏。NVFP4 用局部分组 scale 和更大范围的全局 scale 配合，编码与 scale 必须一起解释。只把两个 nibble 拼起来并不等于完成正确量化。

本篇接续系列第 33 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 32 章（PDF 阅读器第 470–481 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** 4-bit 线格式、global/block scale、误差来源。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

| 术语 | 本篇含义 |
| --- | --- |
| E2M1 | 4-bit 浮点数据格点，正值为 0、0.5、1、1.5、2、3、4、6 |
| 两级 scale | x_hat=global_scale*block_scale*q |
| packing | 两个 4-bit 编码装入一个 byte |
| block scale | 常见 NVFP4 路径中每 16 个值共享 FP8 scale |

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

### 先画 E2M1 数轴
正半轴有八个格点，负值对称并有符号零。最邻近舍入选择距离最近的格点，中点按约定处理。越界饱和到幅度 6；它没有普通 FP32 的 Inf/NaN 表示。本软件模型只接收有限值。

### Scale 自身也会舍入
理想局部 scale=amax/(6*global_scale)，但实际存储 scale 的 FP8 格点会产生第二次舍入。因此编码 q 时应使用实际存下来的 scale，而不是仍用未舍入的理想值。零块显式编码，不计算 0/0。

### 字节测试和数值测试互补
先逐 nibble 验证 pack/unpack 恢复编码，再解码并乘回两级 scale 比较原值。奇数长度最后一个 byte 的高 nibble 填 0，逻辑长度单独保存。硬件 scale layout 可能有 padding/swizzle，本章逻辑编码不冒充该物理布局。

## 公式：从定义到可核对的小例子

```text
x_hat = g*s_block*decode_e2m1(code)
packed_byte = code[2i] | (code[2i+1]<<4)
low=byte&15; high=(byte>>4)&15
16 个值：8 数据字节 + 1 个 FP8 block scale 字节（不计其它元数据）
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/cuda-nvfp4-format.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/cuda-nvfp4-format.svg" alt="NVFP4：两级 scale 与打包的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/cuda-nvfp4-format.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**选 global / block scale → 舍入并保存 block scale → 按实际 scale 量化 E2M1 → 两 nibble 打包一 byte → 解包 + 两级反量化验证**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/32/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`FP4 输入 [0, 0, 0]`。到最后一步应满足：所有 nibble pack/unpack 精确还原；实际硬件 SF layout 未模拟。 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包](/downloads/leetcuda/32/cuda-nvfp4-format.zip) · [独立 CPU 源码](/downloads/leetcuda/32/demo.py) · [运行说明](/downloads/leetcuda/32/README.txt) · [练习说明](/downloads/leetcuda/32/EXERCISES.txt)

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


def quant4(x, global_scale=1.0):
    """Logical one-block NVFP4-like quantization, not hardware SF layout."""
    assert global_scale > 0 and all(math.isfinite(v) for v in x)
    ideal = max(map(abs, x), default=0) / (6 * global_scale)
    sf = FP8[nearest(ideal, FP8)] if ideal else 1.0
    sf = max(sf, FP8[1])  # explicit finite tiny-block policy
    codes = [nearest(abs(v) / (global_scale * sf), FP4) | (8 if math.copysign(1, v) < 0 else 0) for v in x]
    packed = [codes[i] | ((codes[i + 1] if i + 1 < len(codes) else 0) << 4) for i in range(0, len(codes), 2)]
    unpacked = [n for byte in packed for n in (byte & 15, byte >> 4)][: len(codes)]
    assert unpacked == codes
    restored = [(-1 if c & 8 else 1) * FP4[c & 7] * sf * global_scale for c in codes]
    return restored, sf, codes, packed


def lab32(case):
    x = [[0, 0, 0], [0.5, 1, 1.5, 2, 3, 4, 6], [-6, -2, -0.5, 0, 1], [0.1, 0.3, 0.7, 2.7, 9]][case]
    restored, sf, codes, packed = quant4(x)
    return result(
        f"FP4 输入 {x}",
        [
            step(
                "block scale 与编码",
                {"scale": sf, "codes": codes},
                "global scale=1；本例逻辑组不足 16 时仍使用同组 scale。",
            ),
            step("打包 byte", packed, "低 nibble 是偶数元素；奇数尾部高 nibble=0。"),
            step(
                "重建",
                {"x_hat": restored, "error": [a - b for a, b in zip(restored, x)]},
                "数值误差与字节 round-trip 是两项独立验收。",
            ),
        ],
        "所有 nibble pack/unpack 精确还原；实际硬件 SF layout 未模拟。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab32(args.case), ensure_ascii=False, indent=2, allow_nan=False))
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** E2M1 数据舍入基线，scale 暂用 FP32；CPU demo 另验证两级 SF 舍入与 packing，硬件 NVFP4 见下面专用 demo。

```bash
python run_gpu.py --chapter 32 --backend triton
python run_gpu.py --chapter 32 --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试](https://github.com/xlite-dev/LeetCUDA/blob/d32a58f53e4122dc68d660698ee025751a596fc5/kernels/interview/bench/fp4_gemm.cu)。这个 FP4 GEMM 在较新的固定提交中，不能用前面旧提交的测试编号寻找它。需要支持 SM120 FP4 的 CUDA toolkit、CUTLASS、cuDNN/cuBLAS 等该构建脚本依赖。运行 --fp4-gemm 会执行量化与主 GEMM 的组合验证。

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
git clone https://github.com/xlite-dev/LeetCUDA.git LeetCUDA-lab
git -C LeetCUDA-lab checkout --detach d32a58f53e4122dc68d660698ee025751a596fc5
git -C LeetCUDA-lab submodule update --init --recursive
python3 run_upstream.py --repo ./LeetCUDA-lab --arch sm_120a
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** 软件量化/打包参考；支持时 scaled dot。

**TileLang 任务：** FP4 packing 与 scale layout。

Triton 与 TileLang 先做软件码点选择、uint8 pack/unpack；再用受支持的 scaled dot。测试 0、正负中点、极值、奇数长度及零组。字节一致与允许的重建误差分别验收。

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter 32 --backend triton --student student_triton
python run_gpu.py --chapter 32 --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter 32 --backend triton --benchmark --output result.json
```

**进阶通过条件：** 逐 byte round-trip；零值、极值、奇数长度和饱和策略均有定义。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

实际 NVFP4 转换与 scale 布局要匹配目标指令约定。软件小模型验证数值和 packing，不声称已实现全部设备专用 SF 布局。

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

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/) · 下一篇：[（三十四）Block-scaled MMA：scale 也是操作数](/llm/cuda-block-scaled-mma/)

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
