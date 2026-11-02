---
title: LLM 推理
date: 2026-11-01 12:00:00
comments: false
---

近期主要关注 **LLM 的并行方式与 Cache 池化**，同时记录算子学习中的理解、实现和实验。

## 并行方式

整理模型与请求的并行组织方式，记录通信、调度与计算的关系，以及不同配置下的实践观察。

- [Qwen3.5 Context Parallel：系统设计与计算流程](/llm/qwen35-context-parallel/)：梳理 GDN、Full Attention 与 MoE 的布局衔接，以及分片缓存和 PD 传输。

## Cache 池化

关注缓存池的组织、共享复用与生命周期管理，记录存储层次、数据搬运和调度中的问题与尝试。

## 算子学习

从算子原理和实现入手，保留代码、测试条件、性能分析与学习过程中的疑问。

CUDA／LeetCUDA 系列按原文章节顺序创作，**一章一篇博客**，编号接续现有笔记。每篇包含原理与公式、流程图、交互演示、完整代码 demo，以及 Triton／TileLang 练习与验收。

- [完整学习目录与后续章节](/systems/gpu-learning-path/)
- [（一）CUDA 基础：执行模型与内存](/systems/cuda-basics/)
- [（二）性能分析：SM 调度、warp stall 与流水线](/systems/cuda-overlap/)
- [（三）GPU 架构与 Roofline：估算并验证性能上限](/systems/gpu-roofline/)
- [（四）归约与点积：从 warp 到全局结果](/llm/cuda-reduction/)


---

整理后的笔记会按这三个方向归档，尽量区分已有结论、实验观察和仍待验证的想法。

[查看 LLM 文章归档](/categories/llm-inference/) · [查看其他栏目](/categories/) · [GitHub](https://github.com/wulinze)
