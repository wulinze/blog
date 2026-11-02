---
title: CUDA 算子学习目录：按章阅读、逐篇实践
description: 按照 leetcuda-20261029 的章节顺序，一章对应一篇博客，接续既有编号，汇总交互演示、代码 demo 和 Triton／TileLang 练习。
permalink: systems/gpu-learning-path/
date: 2026-11-02 15:20:00
categories:
  - LLM 推理
tags:
  - CUDA
  - 学习路线
  - Triton
  - TileLang
---

## 从这里开始

本系列覆盖 `leetcuda-20261029.pdf` 的 **40 个编号章节**，按原文顺序一章对应一篇博客。额外的 CUDA 入门是（一），原文性能分析是（二），随后接续到（四十一）。分组仅帮助导航，不合并原文章节。

全部文章归入 [LLM 推理](/llm/)。本轮已将后续 37 章展开为独立文章，并补上原 PDF 的 CuTe 导读与附录。下面每个标题都能打开正文。

每篇包含术语、原理、公式、可编辑流程图、可切换案例的逐步交互、完整 CPU demo、GPU 语义基线或原书专用代码入口，以及 Triton／TileLang 练习。交互使用已运行校验的 CPU 计算快照。**当前没有 NVIDIA GPU：后端编译、设备正确性与性能仍为未实测，不把静态检查当设备验收。**

## 建议的学习动作

1. 先读术语和输入输出，手算文中的小例子。
2. 沿流程图看每一步的数据、结果归属与完成条件。
3. 切换交互案例，预测下一步，再观察中间状态。
4. 下载该篇实验包，先跑 CPU，再按硬件条件跑语义基线与原书专用代码。
5. 在独立学生入口中实现练习；正确后才用资源与 profiler 证据验证优化假设。

语义基线、架构专用路径与进阶优化有不同目标。比如一个普通 GEMM 基线验证结果，并不声称实现了 WGMMA；原书固定版本的 Hopper 实验才验证对应指令路径。每篇都列出两者的范围。

## 入门与性能分析

- [（一）CUDA 基础：GPU 结构、执行与内存](/systems/cuda-basics/)：额外入门篇，建立物理与逻辑映射。
- [（二）SM 调度、warp stall、访存优化与流水线](/systems/cuda-overlap/)：对应原文第 1 章，含性能分析流程与可交互实验。


## 机器、执行与测量

### [（三）GPU 架构与 Roofline：先预测，再测量](/systems/gpu-roofline/)

原文第 2 章 · PDF 第 34–42 页。读懂逻辑流量、分层流量和两种时间下界。

- **Triton：** 向量加法与 axpy；扫 tile / num_warps。
- **TileLang：** 同样语义；扫 T.Parallel 范围 / threads。
- **验收：** 先跑边界校验，再解释工作集与请求带宽；能说明为什么 roofline 不是测速结果。


## 基础并行原语与行算子

### [（四）归约与点积：从一组数到一个结果](/llm/cuda-reduction/)

原文第 3 章 · PDF 第 43–51 页。warp 内交换、block 两级归约、跨 block 合并与浮点误差。

- **Triton：** 块内 tl.sum；两阶段 dot，处理尾部。
- **TileLang：** fragment 归约；多 block partial 与最终合并。
- **验收：** 比较 FP64 参考，测非整除长度；解释为何不能在普通 grid 内随意全局同步。

### [（五）访问粒度与原子冲突：向量化和直方图](/llm/cuda-vector-atomic/)

原文第 4 章 · PDF 第 52–59 页。区分合并访存、单线程宽访存和原子串行化。

[直接打开实验包](/downloads/leetcuda/04/cuda-vector-atomic.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 连续 / 跨距访问；整数 histogram。
- **TileLang：** 宽 copy；每 block 局部 histogram 后合并。
- **验收：** 同一有效数据量比较请求；均匀与热点分布均正确，直方图总数守恒。

### [（六）稳定 Softmax：从整行到在线状态](/llm/cuda-softmax/)

原文第 5 章 · PDF 第 60–69 页。减最大值，推导分块 max 与 sum 的重缩放。

[直接打开实验包](/downloads/leetcuda/05/cuda-softmax.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 单行 softmax；长行分块状态。
- **TileLang：** reduce_max / reduce_sum；online 两状态。
- **验收：** 极大输入不溢出，行和接近 1；空行与全 mask 行有明确约定。

### [（七）LSE 与注意力状态合并](/llm/cuda-lse-merge/)

原文第 6 章 · PDF 第 70–77 页。把归一化输出与 log-sum-exp 组合成可合并状态。

[直接打开实验包](/downloads/leetcuda/06/cuda-lse-merge.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 实现两段与多段 LSE merge。
- **TileLang：** 并行分段 merge；避免额外中间张量。
- **验收：** 分段合并对齐完整参考，覆盖空段与两边都空，明确对数底。

### [（八）RMSNorm 与 LayerNorm：统计量和广播](/llm/cuda-normalization/)

原文第 7 章 · PDF 第 78–87 页。理解平方均值、中心化方差、eps 与稳定计算。

[直接打开实验包](/downloads/leetcuda/07/cuda-normalization.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** fused RMSNorm；LayerNorm 两遍统计。
- **TileLang：** fragment 归约；比较两遍方差与 Welford。
- **验收：** 小方差、大偏置、非整除 hidden size；匹配 eps 和方差定义。

### [（九）RoPE 与转置：数据重排为何影响性能](/llm/cuda-rope-transpose/)

原文第 8 章 · PDF 第 88–98 页。旋转配对、相对位置性质、global 合并与 shared bank。

[直接打开实验包](/downloads/leetcuda/08/cuda-rope-transpose.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 两种 RoPE 配对格式；带尾部转置。
- **TileLang：** shared 中转转置；padding 布局实验。
- **验收：** 旋转范数近似保持，转置两次还原；手算真实 lane 地址后再判断冲突。


## 矩阵乘与多层复用

### [（十）GEMV：如何把一行交给 warp](/llm/cuda-gemv/)

原文第 9 章 · PDF 第 99–106 页。长短 K 的并行分工，矩阵流量与向量复用。

[直接打开实验包](/downloads/leetcuda/09/cuda-gemv.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 每 program 一行与多行 GEMV。
- **TileLang：** 线程划分变化；短 K 多行打包。
- **验收：** M/K 尾部正确，区分任务太少与 DRAM 吞吐受限。

### [（十一）GEMM：block tile 与寄存器 tile](/llm/cuda-gemm-tiling/)

原文第 10 章 · PDF 第 107–117 页。从 2MNK 推导复用收益，区分 block / thread tile。

[直接打开实验包](/downloads/leetcuda/10/cuda-gemm-tiling.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 带 mask 的 tl.dot matmul；扫 BM/BN/BK。
- **TileLang：** shared tile + fragment accumulator + T.gemm。
- **验收：** 非方阵和尾块正确；解释 BK 与输出 tile 改变的不同资源。

### [（十二）精度与异步搬运：TF32 和双缓冲](/llm/cuda-tf32-pipeline/)

原文第 11 章 · PDF 第 118–128 页。区分精度改变、Tensor Core 通路与 cp.async 时序。

[直接打开实验包](/downloads/leetcuda/11/cuda-tf32-pipeline.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 切换 input_precision；分开报告误差与时间。
- **TileLang：** num_stages 对照；检查生成的 async 指令。
- **验收：** 必须先报告精度，再谈速度；只看到异步指令不算 overlap 已被证明。

### [（十三）MMA 与 fragment：理解协作计算的数据归属](/llm/cuda-mma-fragments/)

原文第 12 章 · PDF 第 129–141 页。lane / fragment / ldmatrix 与多级流水。

[直接打开实验包](/downloads/leetcuda/12/cuda-mma-fragments.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 检查 dot 的 lowering；扩展题使用显式布局。
- **TileLang：** 用布局可视化追踪 A/B/C 的 fragment。
- **验收：** 独立核对每个输出坐标与覆盖；不能把高层 tile 当每线程存储量。

### [（十四）布局与调度：三种不同的 swizzle](/llm/cuda-swizzle/)

原文第 13 章 · PDF 第 142–159 页。shared 地址 XOR、寄存器双缓冲、CTA 顺序互不等价。

[直接打开实验包](/downloads/leetcuda/13/cuda-swizzle.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** grouped program ordering；比较工作集复用。
- **TileLang：** shared swizzle 与分块调度分别做 A/B。
- **验收：** 验证映射一一对应，再用实际访问组验证 bank；不混淆 L2 与 shared。

### [（十五）TMA 与 WGMMA：Hopper 的完成协议](/llm/cuda-hopper-tma-wgmma/)

原文第 14 章 · PDF 第 160–175 页。descriptor、transaction barrier、异步 MMA 与 slot 释放。

[直接打开实验包](/downloads/leetcuda/14/cuda-hopper-tma-wgmma.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** Hopper descriptor / persistent matmul 练习，先检查支持。
- **TileLang：** TMA 搬运与 gemm 流水；核对生成屏障。
- **验收：** 列出 arrive、expected bytes、完成和复用事件；目标硬件不支持时仅做协议模拟。

### [（十六）SM120 的生产消费流水](/llm/cuda-sm120-warp-specialization/)

原文第 15 章 · PDF 第 176–190 页。架构能力边界、mma.sync 与资源分配协议。

[直接打开实验包](/downloads/leetcuda/15/cuda-sm120-warp-specialization.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 在目标设备检查生成指令和支持路径。
- **TileLang：** producer/consumer 方案与常规方案对照。
- **验收：** 能力核验在前；不能照搬 Hopper WGMMA，资源池预算与死锁条件须可解释。


## 注意力：从数学状态到分工

### [（十七）FlashAttention：为什么可以不写出分数矩阵](/llm/cuda-flash-attention-math/)

原文第 16 章 · PDF 第 191–204 页。QK、online softmax、PV 的状态不变量与 IO。

[直接打开实验包](/downloads/leetcuda/16/cuda-flash-attention-math.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 小尺寸 fused attention 与逐块 CPU 参考。
- **TileLang：** 分块 QK / PV；维护 m、l、O。
- **验收：** 与完整 attention 对齐，覆盖 causal、尾块；解释旧输出为何也必须缩放。

### [（十八）Split-Q：让每组线程拥有自己的查询行](/llm/cuda-attention-split-q/)

原文第 17 章 · PDF 第 205–226 页。减少跨 warp 归约；S/P 布局如何交给下一次 MMA。

[直接打开实验包](/downloads/leetcuda/17/cuda-attention-split-q.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 查询分块 attention；改变 BQ/BK。
- **TileLang：** QK fragment 到 PV fragment 的交接。
- **验收：** 每行状态有明确 owner；区分减少同步与增加重复加载。

### [（十九）注意力的双流水：K、V 与角色分工](/llm/cuda-attention-kv-pipeline/)

原文第 18 章 · PDF 第 227–244 页。不同数据的就绪时机，stage 占用和 full/empty 协议。

[直接打开实验包](/downloads/leetcuda/18/cuda-attention-kv-pipeline.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 比较编译器流水配置与支持的 warp specialization。
- **TileLang：** 分别调 K/V stages，追踪 consumer 释放。
- **验收：** 短序列、单 tile、尾 tile 先通过；生产/消费不平衡需要时间线证据。

### [（二十）多 consumer 与分段合并](/llm/cuda-attention-multi-consumer/)

原文第 19 章 · PDF 第 245–258 页。区分书中双 consumer 方案与泛称 FA3；合并局部状态。

[直接打开实验包](/downloads/leetcuda/19/cuda-attention-multi-consumer.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** split-KV attention + LSE merge。
- **TileLang：** 多 consumer 结构或分 kernel 等价基线。
- **验收：** 单段退化、奇数 KV tile、空段均正确；合并成本计入总时间。

### [（二十一）大 head_dim：Split-D 的两次分块](/llm/cuda-attention-split-d/)

原文第 20 章 · PDF 第 259–276 页。QK 的 D 归约与 PV 的输出 D 分块不同。

[直接打开实验包](/downloads/leetcuda/20/cuda-attention-split-d.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** D 分块点积后再 softmax；多输出 D tile。
- **TileLang：** QK 与 PV 使用不同 fragment 形状。
- **验收：** 先完成完整点积再做 softmax；比较 D=64/128/256/非整除。


## 布局代数与 CuTe 思维迁移

### [（二十二）Layout：坐标、地址与可逆映射](/llm/cuda-cute-layout/)

原文第 21 章 · PDF 第 310–324 页。从 shape/stride 到 composition、divide 与 product。

[直接打开实验包](/downloads/leetcuda/21/cuda-cute-layout.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 构造同逻辑张量的连续与转置 view。
- **TileLang：** 可视化 layout；核对 element 到 thread/value 映射。
- **验收：** Python 小模型穷举地址；分清逻辑 reshape 与真实搬运。

### [（二十三）Copy 的归属：谁搬什么，搬到哪里](/llm/cuda-cute-tiled-copy/)

原文第 22 章 · PDF 第 325–337 页。Tensor 视图与 TiledCopy 的 thread/value 划分。

[直接打开实验包](/downloads/leetcuda/22/cuda-cute-tiled-copy.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 不同 tile 载入映射，跟踪 mask 与布局。
- **TileLang：** T.copy 的 global/shared/fragment 路径。
- **验收：** 覆盖无遗漏、无非法重叠；对齐、向量宽度和尾部都可解释。

### [（二十四）MMA 的归属：从原子到输出 fragment](/llm/cuda-cute-tiled-mma/)

原文第 23 章 · PDF 第 338–349 页。TiledMMA 分层与 A/B/C 的不同坐标映射。

[直接打开实验包](/downloads/leetcuda/23/cuda-cute-tiled-mma.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** dot 两端布局变化及编译器中间表示。
- **TileLang：** T.gemm 片段分配和 accumulator 可视化。
- **验收：** 元素计数守恒；点积输入布局与输出布局不能直接混用。

### [（二十五）Swizzle 与 TMA 的地址契约](/llm/cuda-cute-tma-layout/)

原文第 24 章 · PDF 第 350–361 页。软件访问地址必须匹配 descriptor 生成的布局。

[直接打开实验包](/downloads/leetcuda/24/cuda-cute-tma-layout.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** descriptor 路径与普通 load 的等价验证。
- **TileLang：** 显式 swizzle 与 TMA copy 配对实验。
- **验收：** 先小 tile round-trip，再接 MMA；分别核验对齐、base 与 phase。

### [（二十六）布局驱动的 GEMM 工程](/llm/cuda-cute-gemm/)

原文第 25 章 · PDF 第 362–377 页。把数据布局、搬运划分、计算划分组合起来。

[直接打开实验包](/downloads/leetcuda/25/cuda-cute-gemm.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 有 epilogue 的矩阵乘，完整基线对照。
- **TileLang：** 共享内存布局 + 流水 + 向量写回。
- **验收：** 核心、epilogue 与尾块逐层校验，源码/配置/版本一起保存。

### [（二十七）Attention 实现对照：统一流水、角色分工与双 consumer](/llm/cuda-cute-attention-variants/)

原文第 26 章 · PDF 第 378–396 页。同一数学接口下比较不同数据流组织。

[直接打开实验包](/downloads/leetcuda/26/cuda-cute-attention-variants.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 对同一输入公平比较普通与 specialized 路径。
- **TileLang：** 对同一输入比较 cp.async / TMA 方案。
- **验收：** 统一 mask、精度、计时范围；不把框架差异都归因于某一条指令。

### [（二十八）大 D 的布局代数与寄存器预算](/llm/cuda-cute-split-d-layout/)

原文第 27 章 · PDF 第 397–413 页。QK/PV 两套 tile 与中间 P 的布局衔接。

[直接打开实验包](/downloads/leetcuda/27/cuda-cute-split-d-layout.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 不同 D 分块布局的 equivalent kernel。
- **TileLang：** 双 GEMM 片段与寄存器生命周期实验。
- **验收：** 手算元素归属，检查 spill 与 shared；禁用不能证明的 reinterpret。


## 架构专用调度与完整流水

### [（二十九）Persistent Attention：任务循环与尾波](/llm/cuda-persistent-attention/)

原文第 28 章 · PDF 第 414–428 页。CTA 常驻循环、linear task assignment、条件重缩放。

[直接打开实验包](/downloads/leetcuda/28/cuda-persistent-attention.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** persistent work queue 与普通 grid 对照。
- **TileLang：** persistent attention 及 epilogue 路径。
- **验收：** 任务无漏算/重算，短长序列都测；比较整段时间，不能预设胜过厂商库。

### [（三十）大 D 的 non-WS 流水：让所有线程参与 MMA](/llm/cuda-non-ws-split-d/)

原文第 29 章 · PDF 第 429–445 页。chunk 解耦、跨 tile 预取、独立 barrier 集。

[直接打开实验包](/downloads/leetcuda/29/cuda-non-ws-split-d.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 全员计算路径与分工路径的调参对照。
- **TileLang：** 独立 K/V stages 与 non-WS 编排。
- **验收：** 先画 buffer 生命周期，再测寄存器/等待/时间；更多 stage 不保证更快。


## 低精度矩阵乘：格式、scale 与误差

### [（三十一）FP8 的格式与量化前处理](/llm/cuda-fp8-quantization/)

原文第 30 章 · PDF 第 446–454 页。e4m3、动态范围、per-row/per-block scale。

[直接打开实验包](/downloads/leetcuda/30/cuda-fp8-quantization.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** amax 与缩放量化；零块和离群点。
- **TileLang：** 归约 + 转换融合；scale 排布。
- **验收：** 重建误差与饱和计数；明确舍入和 scale 方向，先不要求 Tensor Core。

### [（三十二）FP8 GEMM：scale 如何进入累加](/llm/cuda-fp8-gemm/)

原文第 31 章 · PDF 第 455–469 页。在线反量化、缩放粒度和矩阵乘等价式。

[直接打开实验包](/downloads/leetcuda/31/cuda-fp8-gemm.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** FP8 dot 与 scale 应用。
- **TileLang：** FP8 GEMM + scale epilogue。
- **验收：** 与反量化参考对齐，前处理成本计入端到端；不同 K scale 不能直接外提。

### [（三十三）NVFP4：两级 scale 与打包](/llm/cuda-nvfp4-format/)

原文第 32 章 · PDF 第 470–481 页。4-bit 线格式、global/block scale、误差来源。

[直接打开实验包](/downloads/leetcuda/32/cuda-nvfp4-format.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 软件量化/打包参考；支持时 scaled dot。
- **TileLang：** FP4 packing 与 scale layout。
- **验收：** 逐 byte round-trip；零值、极值、奇数长度和饱和策略均有定义。

### [（三十四）Block-scaled MMA：scale 也是操作数](/llm/cuda-block-scaled-mma/)

原文第 33 章 · PDF 第 482–497 页。scale fragment、数据布局与流水的协同。

[直接打开实验包](/downloads/leetcuda/33/cuda-block-scaled-mma.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 目标架构支持的 block-scaled matmul。
- **TileLang：** 数据和 scale 的协作加载。
- **验收：** 每个 scale 与正确数据块对应；对齐厂商库数值约定后再测。


## 量化注意力与最终验收

### [（三十五）量化 Attention：误差在哪里被放大](/llm/cuda-quantized-attention-math/)

原文第 34 章 · PDF 第 498–510 页。Q/K/V/P 的误差传播、平滑与不变性。

[直接打开实验包](/downloads/leetcuda/34/cuda-quantized-attention-math.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 高精度参考对照分步量化误差。
- **TileLang：** 按阶段插入量化并保留误差日志。
- **验收：** 区分 logits、概率、输出误差；极端分布/长序列都要覆盖。

### [（三十六）FP8 Attention 的前处理链](/llm/cuda-fp8-attention-preprocess/)

原文第 35 章 · PDF 第 511–523 页。融合、平滑、量化及 scale 生成。

[直接打开实验包](/downloads/leetcuda/35/cuda-fp8-attention-preprocess.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 融合前处理与多 kernel 基线。
- **TileLang：** 归约、转置与量化的 tile 融合。
- **验收：** 各中间量均可核验，先测端到端占比再优化主 kernel。

### [（三十七）FP8 persist-D 与 scale 折叠](/llm/cuda-fp8-persist-scale/)

原文第 36 章 · PDF 第 524–534 页。保持 D 的注意力流水与合法的 scale 外提。

[直接打开实验包](/downloads/leetcuda/36/cuda-fp8-persist-scale.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 固定 D 的量化 attention 与反量化参考。
- **TileLang：** scale 融入 QK / PV 与输出修正。
- **验收：** 证明每个外提条件，动态 range 与不同 shape 下误差达标。

### [（三十八）FP8 大 D：Split-D 与并行布局](/llm/cuda-fp8-split-d-m4n2/)

原文第 37 章 · PDF 第 535–543 页。工作量增加与寄存器增长是不同瓶颈。

[直接打开实验包](/downloads/leetcuda/37/cuda-fp8-split-d-m4n2.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 大 D 的分块与任务划分对照。
- **TileLang：** M/N 方向的 warp 排布与流水调整。
- **验收：** 并行度、spill 与耗时分别记录；不能仅看 occupancy。

### [（三十九）FP4 Attention 的量化链](/llm/cuda-fp4-attention-preprocess/)

原文第 38 章 · PDF 第 544–553 页。NVFP4 格式在 Q/K/V 预处理中的应用。

[直接打开实验包](/downloads/leetcuda/38/cuda-fp4-attention-preprocess.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 软件 packed 参考与完整前处理。
- **TileLang：** 两级 scale 生成和打包融合。
- **验收：** 数据/scale 同时对齐，精度与转换成本单独报告。

### [（四十）FP4 Attention：P 的两级量化](/llm/cuda-fp4-attention-probability/)

原文第 39 章 · PDF 第 554–560 页。online 状态、P 量化与输出缩放的交互。

[直接打开实验包](/downloads/leetcuda/39/cuda-fp4-attention-probability.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 量化 P 的逐阶段误差定位。
- **TileLang：** block-scaled PV 与 rescale 的接口。
- **验收：** 各 KV tile 后验证状态；短序列正确不能代替长序列稳定性。

### [（四十一）基准与精度：把学习结果变成可复现实验](/llm/cuda-benchmark-accuracy/)

原文第 40 章 · PDF 第 561–570 页。环境、计时、缓存、误差统计和生态约束。

[直接打开实验包](/downloads/leetcuda/40/cuda-benchmark-accuracy.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。

- **Triton：** 统一测试 harness 与多形状报表。
- **TileLang：** 相同 harness 跑等价语义实现。
- **验收：** 同时交付正确性、精度分布、端到端时间、条件和失败案例。

## 独立导读与附录

- [CuTe 白皮书学习导读](/llm/cute-layout-primer/)：PDF 第 277–309 页；层级 shape、composition、coalesce、complement、product、divide 与 inverse，配布局交互和代数 demo。
- [实验工具箱、构建入口与复现](/llm/cuda-lab-toolbox/)：PDF 第 571–591 页，覆盖附录 A–E；公共工具、性能表口径、两类构建入口、固定源码索引和参考资料，配端到端摊销交互与记录校验 demo。

附录中的原作者实测数据只作为原资料的实验记录；本系列没有把它们改写成自己的测量结果。封面、目录、术语前言中的公共约定分散到各篇开头，不另制造空内容文章。

## 下载与维护

实验包按篇独立下载，解压后可直接运行 `demo.py`，再阅读 `README.txt` 选择 GPU 后端与固定版本上游实验。每篇的输入契约、基线范围、扩展题与验证状态均保留在正文中。

源码中的 `docs/learning/leetcuda-source-map.json` 维护 PDF 章节、页码与博客编号的映射；`docs/learning/chapter-notes.json` 保存本轮正文的结构化原稿。网页中的图可下载 `.excalidraw` 编辑，Python、CUDA 源码入口与案例都随文章版本保存。
