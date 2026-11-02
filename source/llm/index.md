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

CUDA／LeetCUDA 系列按原文章节顺序整理，**一章一篇博客**，接续（一）至（四十一），覆盖 PDF 的 40 个编号章节。CuTe 独立导读与附录另列为补充资料。全部归入「LLM 推理」。

每篇有原理与术语、流程图、交互演示、具体代码与 Triton／TileLang 练习。新增篇章的 CPU 案例已经执行；GPU 基线与原书专用代码提供运行入口，尚未在当前环境做 GPU 编译、正确性及性能实测。

- [完整学习目录、来源页码与实验入口](/systems/gpu-learning-path/)
- [（一）CUDA 基础：执行模型与内存](/systems/cuda-basics/)
- [（二）性能分析：SM 调度、warp stall 与流水线](/systems/cuda-overlap/)
- [（三）GPU 架构与 Roofline：先预测，再测量](/systems/gpu-roofline/)
- [（四）归约与点积：从一组数到一个结果](/llm/cuda-reduction/)
- [（五）访问粒度与原子冲突：向量化和直方图](/llm/cuda-vector-atomic/)
- [（六）稳定 Softmax：从整行到在线状态](/llm/cuda-softmax/)
- [（七）LSE 与注意力状态合并](/llm/cuda-lse-merge/)
- [（八）RMSNorm 与 LayerNorm：统计量和广播](/llm/cuda-normalization/)
- [（九）RoPE 与转置：数据重排为何影响性能](/llm/cuda-rope-transpose/)
- [（十）GEMV：如何把一行交给 warp](/llm/cuda-gemv/)
- [（十一）GEMM：block tile 与寄存器 tile](/llm/cuda-gemm-tiling/)
- [（十二）精度与异步搬运：TF32 和双缓冲](/llm/cuda-tf32-pipeline/)
- [（十三）MMA 与 fragment：理解协作计算的数据归属](/llm/cuda-mma-fragments/)
- [（十四）布局与调度：三种不同的 swizzle](/llm/cuda-swizzle/)
- [（十五）TMA 与 WGMMA：Hopper 的完成协议](/llm/cuda-hopper-tma-wgmma/)
- [（十六）SM120 的生产消费流水](/llm/cuda-sm120-warp-specialization/)
- [（十七）FlashAttention：为什么可以不写出分数矩阵](/llm/cuda-flash-attention-math/)
- [（十八）Split-Q：让每组线程拥有自己的查询行](/llm/cuda-attention-split-q/)
- [（十九）注意力的双流水：K、V 与角色分工](/llm/cuda-attention-kv-pipeline/)
- [（二十）多 consumer 与分段合并](/llm/cuda-attention-multi-consumer/)
- [（二十一）大 head_dim：Split-D 的两次分块](/llm/cuda-attention-split-d/)
- [（二十二）Layout：坐标、地址与可逆映射](/llm/cuda-cute-layout/)
- [（二十三）Copy 的归属：谁搬什么，搬到哪里](/llm/cuda-cute-tiled-copy/)
- [（二十四）MMA 的归属：从原子到输出 fragment](/llm/cuda-cute-tiled-mma/)
- [（二十五）Swizzle 与 TMA 的地址契约](/llm/cuda-cute-tma-layout/)
- [（二十六）布局驱动的 GEMM 工程](/llm/cuda-cute-gemm/)
- [（二十七）Attention 实现对照：统一流水、角色分工与双 consumer](/llm/cuda-cute-attention-variants/)
- [（二十八）大 D 的布局代数与寄存器预算](/llm/cuda-cute-split-d-layout/)
- [（二十九）Persistent Attention：任务循环与尾波](/llm/cuda-persistent-attention/)
- [（三十）大 D 的 non-WS 流水：让所有线程参与 MMA](/llm/cuda-non-ws-split-d/)
- [（三十一）FP8 的格式与量化前处理](/llm/cuda-fp8-quantization/)
- [（三十二）FP8 GEMM：scale 如何进入累加](/llm/cuda-fp8-gemm/)
- [（三十三）NVFP4：两级 scale 与打包](/llm/cuda-nvfp4-format/)
- [（三十四）Block-scaled MMA：scale 也是操作数](/llm/cuda-block-scaled-mma/)
- [（三十五）量化 Attention：误差在哪里被放大](/llm/cuda-quantized-attention-math/)
- [（三十六）FP8 Attention 的前处理链](/llm/cuda-fp8-attention-preprocess/)
- [（三十七）FP8 persist-D 与 scale 折叠](/llm/cuda-fp8-persist-scale/)
- [（三十八）FP8 大 D：Split-D 与并行布局](/llm/cuda-fp8-split-d-m4n2/)
- [（三十九）FP4 Attention 的量化链](/llm/cuda-fp4-attention-preprocess/)
- [（四十）FP4 Attention：P 的两级量化](/llm/cuda-fp4-attention-probability/)
- [（四十一）基准与精度：把学习结果变成可复现实验](/llm/cuda-benchmark-accuracy/)

补充阅读：

- [CuTe 白皮书：坐标、函数与布局代数](/llm/cute-layout-primer/)
- [附录：实验工具箱、构建入口与性能结果复现](/llm/cuda-lab-toolbox/)


---

整理后的笔记会按这三个方向归档，尽量区分已有结论、实验观察和仍待验证的想法。

[查看 LLM 文章归档](/categories/llm-inference/) · [查看其他栏目](/categories/) · [GitHub](https://github.com/wulinze)
