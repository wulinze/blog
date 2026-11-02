---
title: CUDA 学习补充：实验工具箱、构建入口与性能结果复现
description: 对应 LeetCUDA 附录 A–E，整理公共工具、性能表口径、构建测试分工、固定版本源码索引和延伸阅读。
permalink: llm/cuda-lab-toolbox/
date: 2026-11-02 17:01:01
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - 性能分析
---

## 附录不是另一组算子，而是复现实验的工具

本篇对应 PDF 阅读器第 571–591 页的附录 A–E。原书有 40 个编号章节，另有 CuTe 白皮书导读和这些附录；它们在学习目录中独立保留，避免编号和主题丢失。

正文一篇解决一个机制，附录帮助你找到公共宏、选择构建目标、理解性能表，并把自己的结果保存成可复现记录。这里不复制原书全部性能数字，因为那些数字属于作者的设备、版本与测试条件。

## 公共工具怎样对应回正文

| 工具族 | 职责 | 先读哪篇 |
| --- | --- | --- |
| 向量访问、warp reduce | 宽 load/store 与组内交换 | [向量化](/llm/cuda-vector-atomic/)、[归约](/llm/cuda-reduction/) |
| cp.async / commit / wait | global→shared 异步搬运 | [TF32 与双缓冲](/llm/cuda-tf32-pipeline/) |
| ldmatrix / MMA | shared→fragment 与协作矩阵乘 | [fragment](/llm/cuda-mma-fragments/) |
| XOR/swizzle | 地址位变换及其逆 | [三种布局与调度](/llm/cuda-swizzle/) |
| tensor map / mbarrier | 多维搬运与事务完成 | [Hopper TMA/WGMMA](/llm/cuda-hopper-tma-wgmma/) |
| setmaxnreg 等资源控制 | 受架构和工具链限制的资源调整 | [SM120 分工](/llm/cuda-sm120-warp-specialization/) |
| CuTe Layout/Copy/MMA | 用类型表达布局和线程分工 | [CuTe 导读](/llm/cute-layout-primer/) |

宏减少了重复代码，不会免除指令的参与线程、对齐、同步和架构限制。不要只复制宏定义后就改变 tile 或线程数；先找它所依赖的契约。

## 两类构建入口不要混用

`book/tests/build_tests.sh` 面向小规模正确性测试，通常用 CPU 高精度参考，不要求把大 benchmark 的全部对照库接起来。`kernels/interview/build.sh` 构建集成 benchmark，较新版本采用多个翻译单元，可能还依赖 cuBLAS、cuDNN 与子模块。

源码版本会改变文件名和入口，因此每篇实验包都附 `upstream.json`，记录完整提交号、测试 ID 或 benchmark 模式。博客第十五篇、PDF 第十四章、原仓库 `ch13` 可以指向同一主题；不要按数字想当然拼文件名。

本系列锁定三个源码快照：

- [LeetCUDA 基础与多数 CuTe 最小测试](https://github.com/xlite-dev/LeetCUDA/tree/3b52c5cd3332c5e992c7aef63087d43f67f23b8e)。
- [含后补 FP4 GEMM 与多文件 benchmark 的 LeetCUDA](https://github.com/xlite-dev/LeetCUDA/tree/d32a58f53e4122dc68d660698ee025751a596fc5)。该版本用于原书第 32–33 章实验入口。
- [FFPA 量化 Attention](https://github.com/vipshop/ffpa-attn/tree/861d75ee87824a14a9ebfe295d54c7dac3e3cabc)。用于量化 Attention 各篇的真实设备代码。

Hopper WGMMA 需要匹配家族目标，不能只判断 compute capability 数值大于 9。各篇 `run_upstream.py` 会检查源码提交；Hopper 入口显式使用 `arch=compute_90a,code=sm_90a`。SM120 FP4 也使用单独能力条件，不借用 SM100 的指令路径。

## 从失败结果往回定位

<figure class="cuda-diagram">
<img src="/images/leetcuda/cuda-benchmark-accuracy.svg" alt="冻结契约到正确性再到重复计时和保存证据的实验流程" loading="lazy">
<figcaption>先证明算的是同一件事，再比较时间。<a href="/images/leetcuda/cuda-benchmark-accuracy.excalidraw" download>编辑流程图</a></figcaption>
</figure>

编译失败先查 toolkit、目标架构、头文件与模板配置；输出不对先从 copy round-trip 查到 QK、统计量、PV、epilogue；只有正确性已通过，再采集资源与性能。`SKIP` 不是失败的数值结果，也不是通过的数值结果，需要单独记录。

原书性能表适合提出假设：例如有效带宽高于标称显存带宽时，先检查数据是否主要从 L2 服务；主 kernel 大幅加速但端到端收益很小时，先检查量化前处理占比。不要把“工作集小于缓存”当成保证每次访问命中的充分条件。

## 交互：为什么主 kernel 快一倍，整体未必快一倍

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/40/scenarios.json">
<label>选择未被加速部分的占比<select aria-label="端到端摊销实验"></select></label>
<p data-progress aria-live="polite"></p><div data-output></div><p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div><p data-check></p>
</div>

所有时间都是构造的教学数据。它演示 `S_e2e=1/(f+(1-f)/s)`：f 是未加速部分原来所占的比例，s 是可优化部分的加速比。

## 完整 demo 与保存结果

[下载基准实验包](/downloads/leetcuda/40/cuda-benchmark-accuracy.zip)，先运行不需要 GPU 的摊销模型，再运行所选 DSL：

```bash
python3 demo.py --case 2
python run_gpu.py --chapter 40 --backend triton --benchmark --output triton-result.json
python run_gpu.py --chapter 40 --backend tilelang --benchmark --output tilelang-result.json
```

GPU harness 保存设备、版本、形状、最大绝对误差、相对 L2 和重复时间样本。计时范围为整个语义基线组合，不冒充优化后的主 kernel-only。当前本地仅验证 CPU 与静态代码，GPU 后端编译、正确性和性能未实测。

下面这个独立小脚本可直接检查一组实验记录的口径，防止把缺失值当零或把 SKIP 当 PASS：

```python
import json
from pathlib import Path


def read_result(path):
    data = json.loads(Path(path).read_text())
    assert data["device"] and data["versions"] and data["scope"]
    assert len(data["cases"]) == 4
    for case in data["cases"]:
        assert case["status"] == "PASS"
        assert case["max_abs"] >= 0
        if "composition_ms_samples" in case:
            samples = case["composition_ms_samples"]
            assert len(samples) >= 10 and all(t > 0 for t in samples)
    return data


if __name__ == "__main__":
    import sys
    result = read_result(sys.argv[1])
    print(result["device"], result["scope"], "metadata checks passed")
```

Triton 练习：改一个 tile 参数，输出 A/B 两份结果并检查实际执行的配置。TileLang 练习：保持语义和形状相同，只改 stages，再解释资源和时间变化。两者都需要保留失败输入；元数据校验通过不能替代数值正确性或应用质量验收。

## 原始资料怎样继续查

指令语义查 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)，编程与同步模型查 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)，布局实现查 [CUTLASS CuTe](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cpp/cute)。Attention 算法依据回到 [FlashAttention](https://arxiv.org/abs/2205.14135)、[FlashAttention-2](https://arxiv.org/abs/2307.08691) 和 [FlashAttention-3](https://arxiv.org/abs/2407.08608)，量化实现从固定 FFPA 源码的引用继续追踪。

一篇学习笔记中的“为什么这样做”，最好同时能找到数学推导、源码实现和一个会失败的反例。只看性能表无法替代这三项。[回到完整系列](/systems/gpu-learning-path/)。

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
