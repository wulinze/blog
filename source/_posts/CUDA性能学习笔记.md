---
title: CUDA 性能学习（二）：从 SM 调度、warp stall 到访存优化与 TMA 流水线
description: 先分清性能术语与 SM 内部管线，再用手绘图、交互模型和完整 CUDA demo 理解 occupancy、延迟隐藏、合并访存、tiling、向量化及 TMA。
categories:
  - 系统知识
tags:
  - CUDA
  - 基础学习
  - GPU 性能
permalink: systems/cuda-overlap/
date: 2026-11-02 13:35:00
---

## 阅读路线与术语入口

[第一篇：CUDA 基础与 GPU 物理／逻辑结构](/systems/cuda-basics/) 解释了“任务在哪里执行、数据在哪里”。第二篇接着问：**SM 里面如何调度？线程为什么等？怎样让等待期间仍然有工作可做？**

先认识下表，再沿 **SM 结构 → 一条指令的流转 → 瓶颈判断 → resident warp 与 occupancy → 分层优化 → 异步流水线与 TMA** 阅读。最后附 Nsight Compute 的观察方法。不要第一次就记住所有 API。

| 术语 | 本篇使用的含义 | 首先排除的混淆 |
| --- | --- | --- |
| latency，延迟 | 从发起一个操作到所需结果可用的时间 | 不等于单位时间处理多少数据 |
| throughput，吞吐 | 单位时间完成的工作量，如 FLOP/s、指令/s | 高吞吐不保证每次操作延迟都短 |
| bandwidth，带宽 | 单位时间传输的字节数，常用 GB/s | 要说明是 HBM、L2、shared 还是 PCIe |
| compute-bound，计算受限 | 在本篇的吞吐模型中，目标计算管线能力限制速度 | 代码有乘法，不代表已算力饱和 |
| memory-bound，访存受限 | 广义指访存限制性能；Roofline 中通常特指带宽限制一侧 | 要继续区分带宽不足还是等待数据 |
| bandwidth-bound，带宽受限 | 某层数据通路已接近其可持续吞吐，搬运量成为限制 | 增加等待中的线程不增加硬件带宽 |
| latency-bound，延迟受限 | 独立工作不足，关键依赖的等待暴露出来 | 算术依赖也会造成延迟受限，不限于内存 |
| resident / active warp，驻留 warp | 状态和资源已在 SM 中占有位置的 warp | 驻留不等于此刻能发指令 |
| eligible warp，可发出 warp | 下一条指令已具备发出条件的 warp | 可发出也可能未被选中 |
| issued，已发出 | 调度器选择并发出了某条指令 | 发出不等于结果已经完成 |
| occupancy，占用率 | 驻留 warp 数相对该 SM 的 warp 容量上限的比例 | 不等于计算单元利用率、有效 lane 比例或 GPU 忙碌率 |
| overlap，重叠 | 两件独立工作在时间上有交叠 | async 返回早，不自动产生重叠 |
| tile / stage | tile 是数据分块；stage 是流水线中的缓冲与同步阶段 | 数据块大小与流水级数是两种参数 |

这些标签不是一套永远互斥的分类。例如一个访存受限 kernel 可以因为 HBM 带宽饱和而慢，也可以因为依赖链长、带宽尚未用满而慢。文献中的 memory-bound 有时采用狭义含义，阅读时要看它是否采用 Roofline 的吞吐假设。[NVIDIA GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/)

本文配有 **8 张可编辑手绘图、5 个浏览器概念实验、4 份完整 CUDA 程序**。图中的队列是功能示意，不是特定 GPU 的精确电路图；模拟时间单位不对应真实 GPU 周期。资料核对日期：2026-11-02。本机与当前可访问的计算环境均无可用 CUDA GPU／`nvcc`，CUDA 源码尚未编译和 GPU 实跑；不提供虚构测速数据。

[下载全部源码与 Makefile](/downloads/cuda-overlap/cuda-overlap-demos.zip) · [编译、校验和计时说明](/downloads/cuda-overlap/README.txt)

## 1. 先放大一个 SM：谁保存状态，谁调度，谁计算

### 1.1 看懂图里的部件

<strong>SM（Streaming Multiprocessor，流式多处理器）</strong> 承接 block 的执行。block 获得驻留资源之后，其 warp 由 SM 内部调度。理解调度时，需要把三类东西分开：**线程的状态、指令的控制、真正执行工作的通路**。

下图上方是 warp 状态与指令选择；中间是寄存器和指令流向；下方是计算与访存两条通路。蓝箭头表示指令／请求，绿箭头表示结果返回。带框的“队列”只表示有限的缓冲与接收能力，不表示所有架构都有同名、同深度的独立 FIFO。

<figure class="cuda-diagram">
<a href="/images/cuda-overlap/01-sm-pipelines.svg" target="_blank" rel="noopener"><img src="/images/cuda-overlap/01-sm-pipelines.svg" alt="SM 中的驻留 warp、依赖检查、调度、寄存器与计算和访存通路" loading="lazy"></a>
<figcaption>图 1：先顺蓝箭头看指令去向，再沿绿箭头看结果使依赖就绪。 <a href="/images/cuda-overlap/01-sm-pipelines.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-overlap/01-sm-pipelines.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

- **warp 状态与指令获取**：保留执行位置、活跃线程等状态，取得并解码要执行的指令。寄存器文件保存线程值。硬件切换候选 warp 时不需要像操作系统切换线程那样把整套寄存器上下文搬进搬出内存。
- **依赖跟踪（scoreboard）**：跟踪尚未完成的操作和相关依赖。比如 `x` 的 load 还没有完成，使用 `x` 的 FMA 就不能提前消费它。这里用 scoreboard 概括依赖控制，不假设其具体电路实现。
- **warp scheduler / issue / dispatch**：从本调度域的候选 warp 中选可发出的指令，送往对应功能单元。某个 warp 正在等待，不等于整个 SM 停止。
- **计算通路**：FP32、整数、特殊函数、Tensor Core 等处理不同指令，能力和接收速率各异。空闲的整数单元不能自动代替一个已经饱和的 Tensor Core 或特殊函数通路。
- **访存通路**：load/store 单元处理地址与访存请求。普通 global/local 访问进入缓存／内存系统；shared 访问走对应的片上存储通路。TMA 是后文讨论的异步搬运机制，不能把它等同所有普通 load。

现代 NVIDIA SM 常按子分区（SMSP）组织调度器、寄存器和执行资源，多个子分区共享 SM 级 L1/shared 等资源。Nsight Compute 的通用硬件图以四个子分区示意；本篇只放大一个调度域的工作方式，不把它画成“一个 SM 只有一个调度器”，也不假设一个调度器可任意挑选所有 SM 中的 warp。具体数量与执行通路应核对目标架构。[Nsight Compute：Hardware Model](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#hardware-model)

### 1.2 一条 load → FMA → store 怎样走完

用每线程的一小段工作跟图走：

```cpp
float x = input[i];
float y = fmaf(x, scale, bias); // FMA：一次乘法和一次加法，融合舍入
output[i] = y;
```

这里假定编译器保留这些操作，`scale`、`bias` 已可用；真实机器指令还包括地址计算、边界判断等。下图是**允许发生的一种执行顺序**，不是固定的轮转调度策略，也没有承诺每行耗时一样。

<figure class="cuda-diagram">
<a href="/images/cuda-overlap/02-issue-flow.svg" target="_blank" rel="noopener"><img src="/images/cuda-overlap/02-issue-flow.svg" alt="warp A 等待 load 和 FMA 结果时，warp B 发出独立指令" loading="lazy"></a>
<figcaption>图 2：从上往下跟踪 A，每一步都问 B 有没有独立且可发出的工作。 <a href="/images/cuda-overlap/02-issue-flow.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-overlap/02-issue-flow.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

1. warp A 的 load 具备发出条件且访存入口可接收，于是提交请求；相关目标寄存器尚未就绪。
2. A 接下来的 FMA 要用 `x`，因此等待数据依赖。A 仍然驻留，寄存器与 block 的 shared 分配不会因为这次等待就全部释放。
3. warp B 的独立算术指令已经就绪，且对应管线可以接收；调度器可发出 B 的指令。此时 A 的请求可能仍在缓存／显存系统中进行。
4. A 的数据返回，目标寄存器可用，相关依赖得到解除。A 能否立即发出 FMA，还取决于指令与目标执行资源等条件；“数据好了”不是“必定本周期选中”。
5. A 的 FMA 发出后还需等待计算结果可用。依赖 `y` 的 store 不能提前把未完成的结果写出；同时其他独立指令可能继续推进。
6. store 被接收以后还要经过写入路径。它不向普通目标寄存器返回一个计算值；不能把“store 已发出”当作其他线程已经按程序要求看见它，跨线程共享结果仍需正确的同步和内存顺序。

### 1.3 计算管线和访存管线如何排队

**管线有延迟，也有接收间隔。** 假设一个教学计算管线从进入到完成需要 4 个时间单位，但每单位可以接收一条独立指令：第一条在 `t=0` 进入、`t=4` 完成，第二条可在 `t=1` 进入。它们能重叠，因为并非等第一条完全离开才能接收下一条。但若第二条依赖第一条的结果，它仍要等到结果就绪。这是吞吐和延迟的区别，不是实际 FP32 指令的固定参数。

| 等待发生在哪里 | 等什么 | 增加 warp 可能怎样影响它 |
| --- | --- | --- |
| 指令获取／依赖检查 | 指令可用或输入操作数就绪 | 独立 warp 可能填补空档 |
| 计算通路入口 | 目标管线／发出能力可接收 | 可以换发另一类独立指令；同类指令已经饱和时继续堆积无益 |
| 访存入口／队列 | 请求缓冲或下游接收能力 | 尚未充分并发时可能改善；队列满时更多请求也会等待 |
| 缓存缺失／内存返回 | 所需数据到达 | 其他独立工作可隐藏等待，但不缩短这次物理传输 |
| shared 服务 | bank／通路服务当前请求 | bank conflict 可能让服务分批，增加 warp 不替代布局修正 |
| barrier／buffer 复用 | 其他参与者或异步操作完成 | 同步范围内的 warp 可能一起等，要找范围外的独立工作或改善工作分配 |

访存请求不是全 GPU 排成一条简单队列：warp 地址会合并或拆分，缓存命中与未命中走不同服务路径，下游也有有限的在途容量。可把“排队”先理解为**这一层暂时无法继续接收或完成工作**。后文的 `long scoreboard` 侧重结果依赖；`LG throttle`、`MIO throttle`、`math pipe throttle` 则帮助观察不同入口或通路压力，具体指标取决于芯片和工具版本。[Nsight Compute：Warp Stall Reasons](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#warp-stall-reasons)

## 2. memory-bound、compute-bound、latency-bound 怎么判断

### 2.1 先问“满了”，还是“没活可发”

下图把两种问题分开：左边有足够的独立工作，速度受服务吞吐限制；右边仍有空闲时间，但下一步工作卡在依赖上。图形占满表示教学情景的饱和，不是采集到的利用率。

<figure class="cuda-diagram">
<a href="/images/cuda-overlap/03-bottlenecks.svg" target="_blank" rel="noopener"><img src="/images/cuda-overlap/03-bottlenecks.svg" alt="计算吞吐、内存带宽限制与依赖延迟暴露的区别" loading="lazy"></a>
<figcaption>图 3：服务能力饱和与缺乏可发工作需要不同的优化。 <a href="/images/cuda-overlap/03-bottlenecks.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-overlap/03-bottlenecks.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

三个小例子帮助建立直觉：

- 大数组拷贝：每个字节都要读取再写入。如果相关内存层已接近可持续带宽，降低搬运字节数比继续加 warp 更直接。
- 指针追逐 `p = next[p]`：下一次地址依赖上次结果。即使需要的数据很多，也可能没有足够的独立请求同时在途，造成访存延迟受限。
- 大量可并行 FMA：如果目标计算管线持续繁忙，则可能计算吞吐受限；若只有一条串行累加链、没有其他独立工作，即使全是 FMA 也可能主要受依赖延迟限制。

### 2.2 用一个有边界的吞吐公式估算

设 `F` 是完成任务的浮点运算量，单位 FLOP；本篇把一个 FMA 计为 2 FLOP。`D` 是跨**所选内存层**的读写字节量，单位 byte。`P` 是匹配数据精度与指令类型的计算吞吐，单位 FLOP/s；`B` 是该内存层的带宽，单位 byte/s。

为什么先做除法？因为“工作量 / 每秒处理量”得到秒。为什么取最大值？如果两类工作可以理想重叠，总时间至少不能小于其中任何一项：

```text
T_compute = F / P
T_memory  = D / B
T_lower   = max(T_compute, T_memory)
I         = F / D                    // 算术强度，FLOP/byte
P_roof    = min(P, B * I)            // 理想吞吐上界，FLOP/s
I_balance = P / B                    // 两个吞吐上界的交点
```

`max` 不是说真实运行总能完全重叠；`T_lower` 只是该简化模型的下界。`P_roof` 取 `min`，因为实际吞吐不能越过计算和搬运两条约束中较低的一条。把 `F/P = D/B` 两边整理，即得 `F/D = P/B`，所以交点的单位也是 FLOP/byte。

代入一组**虚构教学参数**：`P=10 TFLOP/s`、`B=1 TB/s`，交点为 `10 FLOP/byte`。若 `F=20 GFLOP`、`D=4 GB`，则 `I=5`，计算需至少 `2 ms`、搬运需至少 `4 ms`，带宽侧上界更低。若实测却耗时 `20 ms`，不能仅凭这条式子说“已把带宽跑满”；可能还有依赖、并发不足、同步或别的通路限制。

这些单位采用十进制：`1 GB = 10^9 byte`。不要把 FP32 运算量代入某款 GPU 的稀疏 FP8 Tensor Core 峰值；也不要用 L2 的字节量配 HBM 带宽。[NVIDIA：Understanding Performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/) · [Nsight Compute：Roofline Charts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#roofline-charts)

<div class="cuda-demo" data-overlap-demo="bounds"><p>启用 JavaScript 调整工作量，比较两条吞吐约束。</p></div>

**小实验读法：** 保持 `D=4 GB`，把 `F` 从 20 改为 80 GFLOP，观察限制侧变化。计算结果只说明理想上界，不根据这些输入自动诊断真实 kernel。

## 3. warp stall 与更多 resident warp：怎样隐藏等待

### 3.1 resident、eligible、issued 是三个层次

一个调度域有 8 个 resident warp，其中 6 个等数据、2 个能发出，而本时刻只选择了 1 个：占位置的是 8 个，可发出的是 2 个，真正发出的只有 1 个。另一个 eligible warp 没被选中，并不说明数据依赖出了问题。

warp A 等待自己的 load 时，发出 B 的独立计算，不是提前得到 A 的数据，而是利用等待窗口。真正需要的是**足够多的可发出独立指令**；增加 resident warp 只是获得它们的一条途径。同一 warp 中也可能存在不依赖当前结果的工作，这对应指令级并行（ILP）；多个独立内存请求在途则提供内存级并行（MLP）。

### 3.2 一个能手算的调度模型

我们刻意只模拟一个发出槽：每时间单位最多发一条指令；每个 warp 都有一条依赖链，发出后需等 `L` 单位才能发下一条；有 `W` 个同样的 warp；没有其他争用与同步。

```text
一个 warp：最多每 L 单位发一次 → 平均发出率 <= 1 / L
W 个 warp：平均发出率 <= W / L
一个发出槽：平均发出率 <= 1
理想稳态发出率 U <= min(1, W / L)
```

`U` 的单位是“指令 / 模型时间单位”；因为容量恰好设为 1，其数值也等于槽位利用比例。`L=6, W=1` 时每 6 单位最多发一次；`W=3` 理想可填一半槽；`W>=6` 才可能填满。**这不是实际 SM 应该驻留几个 warp 的公式**：真实设备有多个调度域、不同指令通路、缓存延迟变化和资源限制。

<div class="cuda-demo" data-overlap-demo="warps"><p>启用 JavaScript 查看多个 warp 如何填补等待空档。</p></div>

格子从左到右是时间；`W0` 等表示该时刻发出哪个 warp 的指令，`—` 表示无候选可发出。先用 `W=1,L=6`，再增加到 3 和 6。模型用轮转选择来方便复现，并不声称硬件采用相同策略。

### 3.3 为什么加 warp 也可能没用

如果更多 warp 都在等同一阶段的 barrier，或都发请求把已经饱和的通路挤得更满，eligible 数量不一定增加。寄存器与 shared 的占用还可能减少每个 block 能保留的数据复用。内存并发的直觉可以写成：

```text
需要的在途数据量 Q ≈ 目标带宽 B_target × 平均往返延迟 L_mem
```

单位检查：`byte/s × s = byte`。这是稳态流量的近似关系，提示“高带宽需要足够在途工作”，不是分配某个 buffer 就保证达到带宽。举例：假定某条路径目标 `100 GB/s`、平均延迟 `200 ns`，需要约 `20,000 byte` 在途；真实结果还取决于请求粒度、队列、地址分布与资源竞争，不能直接转换成一个通用 warp 数。

## 4. SM occupancy：资源装得下多少 warp

### 4.1 从 block 数推到 warp 数

先区分 **theoretical occupancy**（按资源和启动配置推算的容量上限）与 **achieved occupancy**（运行中观测到的驻留情况，通常有时间和 SM 聚合）。grid 太小、尾部 block 逐渐结束、工作不均衡都可能让后者低于前者。要看具体指标的统计口径。

设一块有 `T` 个线程，每线程使用 `R` 个 32 位寄存器，每块 shared 使用量为 `S` byte。一个 SM 的容量分别记为 `R_SM` 个寄存器、`S_SM` byte shared、`T_SM` 个线程、`W_SM` 个 warp、`B_SM` 个 block。忽略资源分配粒度与其他架构限制的教学近似：

```text
W_block = ceil(T / 32)
B_resident <= min(
    floor(R_SM / (T * R)),
    floor(S_SM / S),           // S=0 时忽略这一项
    floor(T_SM / T),
    floor(W_SM / W_block),
    B_SM
)
W_resident = B_resident * W_block
occupancy = W_resident / W_SM
```

每个除法回答“这项资源能装几块”，不能装半块所以向下取整；需要同时满足所有限制，所以取最小值。真实寄存器分配按粒度取整，还有每 block 上限、分区等条件，不能用这张手算表替代 occupancy API。

<figure class="cuda-diagram">
<a href="/images/cuda-overlap/04-occupancy.svg" target="_blank" rel="noopener"><img src="/images/cuda-overlap/04-occupancy.svg" alt="寄存器、shared、线程、warp、block 多种容量取最小值" loading="lazy"></a>
<figcaption>图 4：每一种资源都允许容纳整个 block，最紧的约束决定上限。 <a href="/images/cuda-overlap/04-occupancy.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-overlap/04-occupancy.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

图中每个小格表示该项资源允许容纳一个完整 block，红色标出最紧的限制。图中使用一个**虚构 SM**：`R_SM=65536`、`S_SM=96 KiB`、`T_SM=2048`、`W_SM=64`、`B_SM=16`。取 `T=256,R=64,S=16 KiB`，各项能容纳的 block 数是 `4、6、8、8、16`，因此最多 4 块、32 warp，占用率 50%。这里 `1 KiB=1024 byte`。

把每块寄存器需求减小，可能让更多 block 驻留；但如果为压低寄存器数产生 spill，local memory 访问增加，耗时反而可能上升。增大 grid 只增加待调度 block，不会突破一个 SM 的驻留资源上限。[CUDA Best Practices：Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)

<div class="cuda-demo" data-overlap-demo="occupancy"><p>启用 JavaScript 调整 block、寄存器和流水级的资源占用。</p></div>

**实验：** 先保持 `T=256,R=64`。一层 buffer 占 16 KiB，stage 从 1 增到 2 时 shared 需求翻倍，模型最多变为 3 块、24 warp，即 37.5%。第三个 buffer 可能让预取更早，却继续消耗驻留资源。请同时看等待减少了多少和资源损失了多少。

### 4.2 查询当前 GPU 的真实容量

[occupancy.cu](/downloads/cuda-overlap/occupancy.cu) 使用 `cudaFuncGetAttributes` 读取编译后的寄存器需求，并用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 查询若干 block/shared 配置。它只查询容量，**不启动 kernel，也不测量 achieved occupancy 或性能**。

```cpp
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, kernel, threads_per_block, dynamic_shared_bytes);
```

最后一个参数是动态 shared 的字节数；静态 shared 与编译后的寄存器需求来自 kernel。cluster kernel 应使用相应 cluster occupancy 接口，不能直接套普通 block 估算。[CUDA Runtime：Occupancy API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__OCCUPANCY.html)

## 5. 对照存储层级：该减少搬运，还是把搬运提前

这一节的图沿用第一篇的物理层级，但新加“可以做什么”。箭头表示数据移动方向，旁注是改善该路径的办法；不存在自动的 `register → shared → L1 → L2 → HBM` 查找链。

<figure class="cuda-diagram">
<a href="/images/cuda-overlap/05-memory-overlap.svg" target="_blank" rel="noopener"><img src="/images/cuda-overlap/05-memory-overlap.svg" alt="按 host、global、shared、register、计算路径选择优化" loading="lazy"></a>
<figcaption>图 5：shared 的显式复用与缓存命中不同，数据也可直接 global load 到寄存器。 <a href="/images/cuda-overlap/05-memory-overlap.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-overlap/05-memory-overlap.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

| 路径或工作 | 可尝试的办法 | 成本与限制 |
| --- | --- | --- |
| 寄存器 → 计算通路 | 拆出独立累加器、展开合适的工作，增加 ILP | 活跃值增多，寄存器压力可能上升 |
| shared → 寄存器 → 计算 | register tiling、预取下一小块、复用当前片段 | 要避免 shared bank conflict，不能覆盖仍在使用的数据 |
| global（可命中 L1/L2）→ 寄存器 | 合并访存、对齐、适当向量化与独立请求 | 缓存不是程序可直接命名的多级 stage；更多请求可能造成争用 |
| global → shared | 用 tiling 提供复用；用异步拷贝提前填下一块 | 两者目的不同：复用减少重复访问，提前搬运隐藏等待 |
| CPU 内存 ↔ GPU | pinned host memory、支持并发的拷贝／计算引擎、不同 stream 和独立分块 | 默认流语义、数据依赖与硬件能力影响重叠；用 Nsight Systems 看时间线 |
| GPU ↔ GPU | 按通信依赖分块，使用对应互连和通信机制 | TMA 不是通用多 GPU 网络通信引擎，不能把 kernel 内搬运直接等同 NCCL overlap |

“计算通信掩盖”在这里首先指单 GPU 内数据搬运与计算的重叠；host/device 和多 GPU 通信是另外两个层级，必须分别确认传输机制、资源与完成信号。[CUDA Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html) · [CUDA Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)

## 6. 三种基础优化：合并访存、tiling、向量化

### 6.1 合并访存：先看同一条指令中相邻线程的地址

对一个满 warp 的一次标量 `float` load，假设起始地址按 32 byte 对齐。32 个线程连续读 32 个 float，共 128 byte，覆盖 4 个 32-byte sector；如果每个线程的地址相隔 32 个 float，就可能触及 32 个不同 sector。这里讨论的是典型 CC 6.0+ 下请求覆盖的 sector，**不是保证 HBM 必然搬运多少字节**：缓存命中、其他 warp 的复用与具体硬件会改变外部流量。

所以合并访存关注的是**跨线程地址关系**，不是把多个 CUDA API 合成一个，也不是让每个线程循环次数更少。[CUDA Best Practices：Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

<figure class="cuda-diagram">
<a href="/images/cuda-overlap/06-access-and-tiling.svg" target="_blank" rel="noopener"><img src="/images/cuda-overlap/06-access-and-tiling.svg" alt="合并访存对应跨 lane 地址、tiling 对应分块复用、向量化对应单线程宽度" loading="lazy"></a>
<figcaption>图 6：三个优化维度可以组合，但不能把它们当作同一件事。 <a href="/images/cuda-overlap/06-access-and-tiling.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-overlap/06-access-and-tiling.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

图的上半部只画 8 个 lane，帮助观察相邻 lane 如何对上内存格子；32-lane 的 sector 计算另外写在图中。下半部让一个输出 tile 对上被复用的输入 tile。两者分别回答“怎么取”和“取来后用几次”。

[coalescing.cu](/downloads/cuda-overlap/coalescing.cu) 的两个版本都完成 `out[j]=in[j]`，只改变线程到地址的映射。基线把每个 1024 元素小块看成 32×32，做一个行列交换的下标排列；优化版直接用连续下标。每个元素仍恰好读写一次，所以校验目标相同。

```cpp
// i 是原来的线性线程任务编号；N 必须为 1024 的倍数。
int j_strided = (i / 1024) * 1024
              + (i % 32) * 32
              + (i % 1024) / 32;
int j_coalesced = i;
```

固定同一个 warp 时，`i%32` 随 lane 增长，因此基线每 lane 跨 32 个元素；连续版每 lane 只跨一个元素。排列会增加少量地址计算，所以时间差也不能全部归因于带宽。完整程序拒绝不符合长度条件的输入，不拿越界访问作“慢版本”。

### 6.2 tiling：分块之后，要说明数据被谁复用

考虑 `C = A @ B`，`@` 表示矩阵乘法：`A` 为 `M×K`，`B` 为 `K×N`，`C` 为 `M×N`。每个输出 `C[row,col]` 是 K 对元素乘积之和。一个 `T×T` 输出 tile 由一个 block 处理，沿 K 方向每次推进 T。

本轮从 global 搬入 `T×T` 个 A 元素和 `T×T` 个 B 元素到 shared，然后让输出 tile 的线程重复使用它们。A 的一个值被同一输出行的多个线程使用，B 的一个值被同一输出列的多个线程使用。

推导一个**只计本轮输入、忽略输出和边界的源码级模型**：本轮计算 `T^2` 个输出，每个进行 T 次 FMA，因此运算量为 `2*T^3 FLOP`；输入为 `2*T^2` 个 float，即 `8*T^2 byte`。

```text
I_tile_input = (2 * T^3) / (8 * T^2) = T / 4 FLOP/byte
T = 16 时：8192 FLOP / 2048 byte = 4 FLOP/byte
```

这说明复用如何随 tile 增大，但它不等于完整 kernel 的 HBM 算术强度：还没计输出、尾块、不同 block 的重复读取和缓存。naive 写法中的重复地址也可能被广播／缓存服务，所以不能声称实际 DRAM 流量必然恰好减少 T 倍。

[tiling.cu](/downloads/cuda-overlap/tiling.cu) 对比 naive 与 16×16 shared tiled FP32 GEMM，默认 `M=129,N=193,K=257`，故意让尺寸不整除 tile。越界输入填零，所有 block 内线程都到达两次屏障，最后只让有效线程写输出：

```cpp
// 第一屏障：其他线程读 tile 之前，所有输入已填好。
__syncthreads();
for (int p = 0; p < TILE; ++p)
    sum += at[threadIdx.y][p] * bt[p][threadIdx.x];
// 第二屏障：覆盖旧 tile 之前，所有旧读取已结束。
__syncthreads();
```

增大 tile 不总更快：shared、寄存器和线程数都可能改变 occupancy，还要考虑访问布局。本 demo 没有异步搬运；因此 **tiling 本身并不等于 overlap**。[CUDA Best Practices：Shared Memory in Matrix Multiplication](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab)

### 6.3 向量化：改变单个线程的一次访问宽度

`float4` 表示连续四个 float，尺寸 16 byte。在编译器生成相应宽访存指令时，可以用更少的 load/store 指令完成同样字节量。它不会自动减少需要读取的数据，也不会自动把标量 FP32 算术变成 Tensor Core 运算。

[vectorized.cu](/downloads/cuda-overlap/vectorized.cu) 对比 scalar copy 和 `float4` copy；两者本来都可合并访问。完整代码使用 `cudaMalloc` 返回的对齐基址，每线程循环处理一个 float4，并用标量处理末尾不足四个的元素。

```cpp
const float4* in4 = reinterpret_cast<const float4*>(in);
float4* out4 = reinterpret_cast<float4*>(out);
for (int i = tid; i < n / 4; i += stride) out4[i] = in4[i];
if (tid == 0)
    for (int i = (n / 4) * 4; i < n; ++i) out[i] = in[i];
```

`in+1` 是偏移一个 float，不满足 float4 的 16-byte 对齐条件，不能强行转换后宽读。尾部也不能为了凑整而越界加载。源码使用向量类型后还应检查 SASS，确认实际指令宽度；吞吐已饱和或寄存器压力增加时，收益可能很小甚至变慢。[NVIDIA：Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)

### 6.4 完整源码与运行方式

下载 ZIP 后解压，确保这些文件在同一目录。需要 NVIDIA GPU、匹配驱动、支持目标架构的 CUDA Toolkit、C++17 编译支持和 `make`。下面 `sm_80` 是示例编译目标；Hopper 可选 `sm_90`，其他硬件请使用其 Toolkit 支持的正确目标。

```bash
make ARCH=sm_80
make check ARCH=sm_80
./coalescing 16777216
./vectorized 16777219
./tiling 256 256 256
./occupancy
```

`make check` 包含 N 为 1、3、4、1027 的向量化尾部，以及 1×1 和不规则矩阵。三个计算 demo 都有 CUDA 错误检查和 CPU 参考校验，通过时打印 `PASS`；失败返回非零。计时采用 5 次 warmup、20 次重复的 CUDA event 平均值，不包含分配、CPU 参考和 host/device 拷贝。反复访问同一工作集可能命中缓存，因此打印的 effective GB/s **不是 HBM 实测带宽**。

下面可以直接展开完整 `.cu` 文件。共用的错误检查、校验与计时在 `common.cuh`，下载包包含所有依赖。程序逻辑为本文编写；当前没有 CUDA 环境，尚不能声称编译或 GPU 结果已通过。

<details>
<summary>展开完整源码：common.cuh</summary>

```cpp
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <vector>
inline void check_cuda(cudaError_t code, const char* expression, int line) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "CUDA line %d: %s: %s\n", line, expression, cudaGetErrorString(code));
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(expr) check_cuda((expr), #expr, __LINE__)
inline int argument(char* value, int maximum) {
    char* end = nullptr; errno = 0;
    long n = std::strtol(value, &end, 10);
    if (errno || *end || end == value || n < 1 || n > maximum) {
        std::fprintf(stderr, "Expected integer in [1, %d]\n", maximum); std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(n);
}
inline cudaDeviceProp device() {
    int ordinal = 0; CHECK(cudaGetDevice(&ordinal));
    cudaDeviceProp prop{}; CHECK(cudaGetDeviceProperties(&prop, ordinal));
    std::printf("GPU: %s; CC %d.%d; SMs %d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    return prop;
}
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) { CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T))); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
template<class Launch> float measure(Launch launch, int repetitions = 20) {
    for (int i = 0; i < 5; ++i) { launch(); CHECK(cudaGetLastError()); }
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < repetitions; ++i) { launch(); CHECK(cudaGetLastError()); }
    CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    return ms / repetitions;
}
inline void validate(const std::vector<float>& actual, const std::vector<float>& expected) {
    for (size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 1e-4f + 1e-4f * std::fabs(expected[i]);
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance) {
            std::fprintf(stderr, "FAIL index %zu: got %.9g expected %.9g\n", i, actual[i], expected[i]);
            std::exit(EXIT_FAILURE);
        }
    }
    std::printf("PASS: %zu elements\n", actual.size());
}
```

</details>

<details>
<summary>展开完整源码：coalescing.cu</summary>

```cpp
#include "common.cuh"
// Both variants copy exactly the same N elements. Only thread-to-address mapping changes.
template<bool Strided> __global__ void copy_layout(const float* in, float* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int j = i;
        if constexpr (Strided) j = (i / 1024) * 1024 + (i % 32) * 32 + (i % 1024) / 32;
        out[j] = in[j];
    }
}
int main(int argc, char** argv) {
    if (argc > 2) { std::fprintf(stderr, "Usage: coalescing [N, multiple of 1024]\n"); return 1; }
    const int n = argc == 2 ? argument(argv[1], 1 << 28) : 1 << 24;
    if (n % 1024) { std::fprintf(stderr, "N must be a multiple of 1024 for the permutation.\n"); return 1; }
    auto prop = device();
    std::vector<float> in(n), out(n);
    for (int i = 0; i < n; ++i) in[i] = static_cast<float>((i % 251) - 125);
    DeviceBuffer<float> d_in(n), d_out(n);
    CHECK(cudaMemcpy(d_in.data, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    const int blocks = std::min((n + 255) / 256, prop.multiProcessorCount * 8);
    for (int variant = 0; variant < 2; ++variant) {
        CHECK(cudaMemset(d_out.data, 0xff, n * sizeof(float)));
        auto launch = [&] {
            if (variant == 0) copy_layout<true><<<blocks, 256>>>(d_in.data, d_out.data, n);
            else copy_layout<false><<<blocks, 256>>>(d_in.data, d_out.data, n);
        };
        launch(); CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(out.data(), d_out.data, n * sizeof(float), cudaMemcpyDeviceToHost));
        validate(out, in);
        const float ms = measure(launch);
        std::printf("%s: %.6f ms, effective %.3f GB/s\n", variant ? "coalesced" : "strided", ms, 2.0 * n * sizeof(float) / (ms * 1e6));
    }
    std::puts("Effective bytes = input + output. Repeated working set may be cached; this is not a DRAM bandwidth measurement.");
}
```

</details>

<details>
<summary>展开完整源码：tiling.cu</summary>

```cpp
#include "common.cuh"
constexpr int TILE = 16;
__global__ void matmul_naive(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0;
        for (int p = 0; p < k; ++p) sum += a[row * k + p] * b[p * n + col];
        c[row * n + col] = sum;
    }
}
__global__ void matmul_tiled(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float at[TILE][TILE], bt[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0;
    for (int base = 0; base < k; base += TILE) {
        int ak = base + threadIdx.x, bk = base + threadIdx.y;
        at[threadIdx.y][threadIdx.x] = row < m && ak < k ? a[row * k + ak] : 0.0f;
        bt[threadIdx.y][threadIdx.x] = bk < k && col < n ? b[bk * n + col] : 0.0f;
        __syncthreads(); // All tile values ready before cross-thread reads.
        for (int p = 0; p < TILE; ++p) sum += at[threadIdx.y][p] * bt[p][threadIdx.x];
        __syncthreads(); // All readers done before next iteration overwrites the tile.
    }
    if (row < m && col < n) c[row * n + col] = sum;
}
int main(int argc, char** argv) {
    if (argc != 1 && argc != 4) { std::fprintf(stderr, "Usage: tiling [M N K]\n"); return 1; }
    int m = argc == 4 ? argument(argv[1], 1024) : 129;
    int n = argc == 4 ? argument(argv[2], 1024) : 193;
    int k = argc == 4 ? argument(argv[3], 1024) : 257;
    device();
    std::vector<float> a(m * k), b(k * n), out(m * n), expected(m * n);
    for (int i = 0; i < m * k; ++i) a[i] = ((i % 13) - 6) / 8.0f;
    for (int i = 0; i < k * n; ++i) b[i] = ((i % 11) - 5) / 16.0f;
    for (int row = 0; row < m; ++row) for (int col = 0; col < n; ++col) {
        double sum = 0;
        for (int p = 0; p < k; ++p) sum += static_cast<double>(a[row * k + p]) * b[p * n + col];
        expected[row * n + col] = static_cast<float>(sum);
    }
    DeviceBuffer<float> da(a.size()), db(b.size()), dc(out.size());
    CHECK(cudaMemcpy(da.data, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db.data, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(TILE, TILE), grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    for (int variant = 0; variant < 2; ++variant) {
        CHECK(cudaMemset(dc.data, 0xff, out.size() * sizeof(float)));
        auto launch = [&] {
            if (variant == 0) matmul_naive<<<grid, block>>>(da.data, db.data, dc.data, m, n, k);
            else matmul_tiled<<<grid, block>>>(da.data, db.data, dc.data, m, n, k);
        };
        launch(); CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(out.data(), dc.data, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        validate(out, expected);
        const float ms = measure(launch);
        std::printf("%s: %.6f ms, effective %.3f GFLOP/s\n", variant ? "tiled" : "naive", ms, 2.0 * m * n * k / (ms * 1e6));
    }
    std::puts("FP32 teaching kernels, not Tensor Core GEMM. CPU reference and copies excluded from event timing.");
}
```

</details>

<details>
<summary>展开完整源码：vectorized.cu</summary>

```cpp
#include "common.cuh"
__global__ void copy_scalar(const float* in, float* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) out[i] = in[i];
}
__global__ void copy_float4(const float* in, float* out, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);
    for (int i = tid; i < n / 4; i += blockDim.x * gridDim.x) out4[i] = in4[i];
    if (tid == 0) for (int i = (n / 4) * 4; i < n; ++i) out[i] = in[i];
}
int main(int argc, char** argv) {
    if (argc > 2) { std::fprintf(stderr, "Usage: vectorized [N]\n"); return 1; }
    const int n = argc == 2 ? argument(argv[1], 1 << 28) : (1 << 24) + 3;
    auto prop = device();
    std::vector<float> in(n), out(n);
    for (int i = 0; i < n; ++i) in[i] = static_cast<float>((i % 127) - 63);
    DeviceBuffer<float> d_in(n), d_out(n); // cudaMalloc base pointers satisfy float4 alignment.
    CHECK(cudaMemcpy(d_in.data, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    const int blocks = std::min((n + 255) / 256, prop.multiProcessorCount * 8);
    for (int variant = 0; variant < 2; ++variant) {
        CHECK(cudaMemset(d_out.data, 0xff, n * sizeof(float)));
        auto launch = [&] {
            if (variant == 0) copy_scalar<<<blocks, 256>>>(d_in.data, d_out.data, n);
            else copy_float4<<<blocks, 256>>>(d_in.data, d_out.data, n);
        };
        launch(); CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(out.data(), d_out.data, n * sizeof(float), cudaMemcpyDeviceToHost));
        validate(out, in);
        const float ms = measure(launch);
        std::printf("%s: %.6f ms, effective %.3f GB/s\n", variant ? "float4" : "scalar", ms, 2.0 * n * sizeof(float) / (ms * 1e6));
    }
    std::puts("Both variants are coalesced. float4 changes per-thread width, not total useful bytes. Inspect SASS for actual instruction width.");
}
```

</details>

<details>
<summary>展开完整源码：occupancy.cu</summary>

```cpp
#include "common.cuh"
__global__ void probe(float* out) {
    extern __shared__ float tile[];
    tile[threadIdx.x] = static_cast<float>(threadIdx.x);
    __syncthreads();
    out[blockIdx.x * blockDim.x + threadIdx.x] = tile[(threadIdx.x + 1) % blockDim.x];
}
int main() {
    auto prop = device();
    cudaFuncAttributes attributes{}; CHECK(cudaFuncGetAttributes(&attributes, probe));
    std::printf("Registers/thread: %d; static shared/block: %zu bytes\n", attributes.numRegs, attributes.sharedSizeBytes);
    for (int threads : {128, 256, 512}) for (int bytes : {8192, 16384, 32768}) {
        if (threads > attributes.maxThreadsPerBlock || bytes + attributes.sharedSizeBytes > prop.sharedMemPerBlock) continue;
        int blocks = 0; CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, probe, threads, bytes));
        int warps = blocks * ((threads + prop.warpSize - 1) / prop.warpSize);
        int max_warps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
        std::printf("threads=%d dynamic_shared=%d: theoretical blocks/SM=%d warps/SM=%d occupancy=%.2f%%\n", threads, bytes, blocks, warps, 100.0 * warps / max_warps);
    }
    std::puts("This queries capacity; it does not launch the kernel or measure achieved occupancy/performance.");
}
```

</details>

## 7. 从同步 tiling 走到双缓冲：计算当前块，搬下一块

### 7.1 buffer 生命周期比“用了 async”更重要

同步版依次“搬 tile 0、算 tile 0、搬 tile 1、算 tile 1”。双缓冲提供两个 shared slot：一个给当前计算，另一个填下一块。图中横向是真实先后关系的示意，条长度按教学参数绘制；蓝色搬运、绿色计算，每块 tile 始终先搬完再计算。

<figure class="cuda-diagram">
<a href="/images/cuda-overlap/07-double-buffer.svg" target="_blank" rel="noopener"><img src="/images/cuda-overlap/07-double-buffer.svg" alt="四个 tile 的串行与双缓冲时间线，并标明 slot 生命周期" loading="lazy"></a>
<figcaption>图 7：先检查同一 tile 的搬算依赖，再检查同一 slot 何时允许覆盖。 <a href="/images/cuda-overlap/07-double-buffer.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-overlap/07-double-buffer.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

沿图看 tile 1：tile 0 开始计算时，tile 1 已可以搬到另一 slot；但 tile 2 不能覆盖 tile 0 的 slot，直到 tile 0 的所有使用者完成读取。每个 slot 的状态必须按以下顺序循环：

```text
FREE → FILLING → READY → READING → FREE
```

**READY** 需要证明异步写入完成且可见，**FREE** 需要证明所有消费者已不再使用它。两个证明不能合成一个含糊的“同步一下”。

### 7.2 为什么理想时间从相加变成取 max

设 N 是 tile 数，L 是每块搬运耗时，C 是每块计算耗时，均使用同一时间单位。假设有独立搬运与计算服务资源、一次各处理一块、至少双缓冲、无其他竞争或额外同步开销：

```text
T_serial   = N * (L + C)
T_pipeline = L + C + (N - 1) * max(L, C)
```

串行时每块两件事必须相加。流水线第一块必须先搬再算，付出 `L+C`；之后每完成一块的稳态间隔受较慢的一侧限制，因此为 `max(L,C)`。这也包含初始填充和末尾排空，不能只写 `N*max(L,C)` 就忽略首尾。

代入 `N=4,L=3,C=5`：串行 `32`，理想双缓冲 `3+5+3*5=23`。若 `L=9,C=2`，计算太短，搬运仍主导稳态；overlap 不创造额外带宽。实际硬件上的复制、计算与存储资源未必独立，此式只是理想对照。

<div class="cuda-demo" data-overlap-demo="pipeline"><p>启用 JavaScript 对照串行与双缓冲时间线。</p></div>

增加 stage 可以把请求提前更多轮。若每轮计算 C 时间，粗略希望提前窗口 `(n_stages - 1) * C` 不短于搬运延迟 L；其中 `n_stages` 是含当前消费 slot 的总 stage 数。它只提示延迟覆盖需求，不是普适最优 stage 公式：若搬运吞吐本身跟不上，多放 buffer 仍不能解决稳态缺口，还会降低 occupancy。

### 7.3 `cp.async` 与 pipeline API 的角色

Ampere（CC 8.0）引入硬件加速 global→shared 异步拷贝路径，可避免普通“global load 到临时寄存器、再 shared store”的中转。高级 `cuda::memcpy_async` 的具体路径还取决于架构、重载、类型及对齐；名字含 async 不保证所有情形都生成相同指令。

下面是**协议伪代码，不是可独立编译的 CUDA 程序**：

```text
预先提交 tile 0 到 slot 0
for k in [0, N):
    如果有下一块：
        等待下一 slot 已释放
        提交 tile k+1 的异步拷贝，并 commit
    等待 tile k 的拷贝完成
    用 slot k%2 计算
    保证所有消费者已用完后，release 当前 slot
```

`producer_acquire/commit` 管理提交端，`consumer_wait/release` 管理消费端。线程范围 pipeline 的 wait 不能自动替其他线程证明完成；跨线程读写需要匹配的协作和同步协议。warp 分歧还可能影响 commit/wait 的批次行为，应按所选 API 的规则保持相关操作的收敛。[CUDA Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html) · [Advanced Kernel Programming：Asynchronous Execution Features](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)

## 8. TMA：把一块 tensor 的搬运交给专门机制

### 8.1 它接在哪一层，替你做什么

<strong>TMA（Tensor Memory Accelerator，张量内存加速器）</strong> 是 Hopper／CC 9.0 引入的异步搬运能力。支持 bulk 与多维 tensor 搬运，典型路径为 global↔shared；一个线程可以发起一大块数据的搬运，使其他线程继续处理独立工作。它不是新的存储层，也不是 GEMM 计算单元。

与逐元素计算地址、发出很多普通 load/store 相比，tensor map 可描述数据布局，TMA 按描述和 tile 坐标搬运。它减少搬运中的指令与寄存器中转负担，但仍使用内存通路并受带宽、依赖和 buffer 容量约束。[Hopper Tuning Guide：Tensor Memory Accelerator](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator)

下图左边是描述与提交，中间是 buffer 生命周期，右边是消费者。箭头不表示一个函数调用就已完成数据可见性；“到达预期字节数”和“消费者用完”是不同事件。

<figure class="cuda-diagram">
<a href="/images/cuda-overlap/08-tma.svg" target="_blank" rel="noopener"><img src="/images/cuda-overlap/08-tma.svg" alt="TMA 描述、生产提交、transaction barrier、shared 就绪与消费者释放" loading="lazy"></a>
<figcaption>图 8：READY 与 FREE 由不同事件证明，不能在数据刚搬完时就覆盖正在消费的 slot。 <a href="/images/cuda-overlap/08-tma.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-overlap/08-tma.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

### 8.2 用一个二维 tile 理解 descriptor 和完成信号

设 global 中有行主序 `A[M,K]` 的 float 矩阵，要搬一个 `16×16` tile 到 shared。tensor map 需要描述基地址、维度、步长、tile 形状及相关布局设置；发起操作再指定 tile 起点。完整 tile 的有效数据为 `16*16*4=1024 byte`，这给出了本教学例子的事务字节计数。真实 descriptor 的维度顺序、stride、对齐和边界处理必须按 API 定义，不能只把 C++ 的维度数组原样照搬。

协议示意：

```text
初始化 shared slot 和关联的 transaction barrier
按所用指令协议发布 barrier 初始化，建立正确的 proxy 顺序
生产者约定本 phase 的参与到达数与待完成传输字节数
发起 global → shared 的 TMA 搬运
消费者等待匹配 phase 的完成信号
    既满足参与者到达条件，也满足关联传输完成条件
消费者读取 shared，执行计算
消费者完成后释放 slot；生产者才能在下一 phase 复用
```

这是协议伪代码，省略具体 API、初始化细节及 descriptor 构造，不能拿它替代可编译 TMA 实现。这里只学 global→shared 的生产／消费关系；前面的三份优化 demo 不依赖 Hopper，也没有伪装成 TMA demo。

<div class="cuda-demo" data-overlap-demo="tma"><p>启用 JavaScript 逐步查看一个 slot 从提交到释放。</p></div>

### 8.3 四个需要单独检查的正确性条件

1. **对齐与描述合法**：一维 bulk 和多维 tensor 操作的要求不完全相同。官方表格中一维 bulk 要求源／目标 16-byte 对齐、传输大小为 16 的倍数；多维 tensor 的 shared 目标对齐等约束更严格，不能把前者规则到处套用。
2. **phase 与完成计数正确**：transaction barrier 需要正确跟踪当前阶段的参与者和关联传输。等错 phase、字节数不匹配都可能导致永久等待或提前消费。
3. **proxy 可见性正确**：普通线程访存与 TMA 的 async proxy 访问需要符合指定的顺序协议。尤其 shared→global 时，普通线程刚写的 shared 数据必须按协议对 TMA 可见；一个随意放置的 `__syncthreads()` 不能替代所有 proxy fence 要求。
4. **区分读取完成和写入完成**：global→shared 的完成常通过 transaction barrier 跟踪；shared→global 使用相应 bulk async-group 机制。确认 TMA 已读完 shared 源，只能用于安全复用该源，不能把它等同所有线程都已看见 global 目标结果。

具体代码应从当前 Toolkit 对应的 [Asynchronous Data Copies：TMA 示例与约束](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html) 开始，连同描述符、fence、barrier 和等待代码一起理解，不能只复制发起指令的一行。

### 8.4 warp specialization 为何可能降低 occupancy 却更快

可以让生产者 warp 管搬运、消费者 warp 管计算；TMA 发起动作本身可能只需一个线程，但生产者角色还承担阶段和 buffer 协调。这样减少消费者在数据搬运和地址计算上的工作，并让两个角色处理不同 tile。

代价是专门的 warp、更多 shared stage、屏障和控制状态都要资源；消费者的寄存器需求也可能高。最终应比较正确完成相同工作所需的时间，而不是要求 occupancy 数字一定上升。异步矩阵计算还可能在消费者发出之后继续使用 shared 数据，必须等相应计算完成，不能仅凭“指令已发出”就释放 slot。[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)

## 9. 怎样用 profiler 把这些判断连起来

先确认结果正确，再固定输入、编译选项和 GPU 条件看耗时。端到端重叠用 Nsight Systems 看时间线；单 kernel 的管线、访存和 warp 行为用 Nsight Compute。采集本身可能改变执行，尤其多次 replay 不能直接代表原本跨 stream 的并发时间线。

```bash
# 查看本机支持的 section；名称与可用指标依工具版本而变。
ncu --list-sections
# 下列为常见 section 名称，先与上一步对照。
ncu --section LaunchStats --section Occupancy --section SchedulerStats \
    --section WarpStateStats --section MemoryWorkloadAnalysis ./tiling 256 256 256
```

| 观察 | 下一步的问题 | 不应直接下的结论 |
| --- | --- | --- |
| theoretical occupancy 低 | 被寄存器、shared、线程还是 block 上限限制？ | 立刻强压寄存器一定更快 |
| active 多，eligible 少，issue 空档多 | 等依赖、同步，还是资源入口？ | 只要继续增大 grid 就能修复 |
| long scoreboard 多 | 哪个消费者等 L1TEX 相关结果？实际地址、命中和流量怎样？ | 每一个样本都表示 HBM miss |
| short scoreboard 多 | 哪类 MIO 结果依赖？shared 布局是否有关？ | 一定是 shared bank conflict |
| LG/MIO throttle 多 | 相应队列／入口是否承压，是否有过多零碎指令？ | 只需要更多 resident warp |
| math pipe throttle 多 | 某类计算通路是否已饱和、指令混合是否失衡？ | 所有计算单元都已跑满 |
| barrier 多 | 哪些参与者先到，谁迟到，是否有工作不均衡？ | barrier 语句本身消耗了全部等待时间 |
| not selected 多 | 可发候选是否充足，发出槽是否在忙？ | 应把它像错误一样清零 |

stall 样本可能记在等待结果的**消费者指令**上，而不是发起请求的 load 那一行；要联系源码与 SASS。总 stalled 样本比例也不等于整个 SM 空闲比例：采样到 warp A 在等的同时，warp B 仍可能工作。优先判断 issue 和相关吞吐是否存在可改善的空档，再解释原因。[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) · [Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)

## 10. 检查自己是否理解

<details><summary>occupancy 从 50% 提高到 75%，耗时却增加，是否矛盾？</summary>

不矛盾。更多驻留 warp 可能伴随 spill、复用减少、队列竞争或不同的指令工作量。occupancy 是容量比例，不是性能结果。

</details>

<details><summary>合并访存、tiling、float4 分别改变了什么？</summary>

合并访存改变同一 warp 的地址分布；tiling 组织数据分块并创造复用或流水机会；float4 改变单线程访存宽度。三者可以组合，但作用与约束不同。

</details>

<details><summary>TMA 已经提交，可以立刻读取 shared 吗？</summary>

不可以。要等与这次传输匹配的完成协议。读完后还要保证所有消费者都不再使用 slot，才能让下一块覆盖它。

</details>

<details><summary>一条管线延迟为 4，是否每 4 个周期才能发一条指令？</summary>

不是。延迟与接收间隔是两回事。独立操作可以在流水级间重叠；依赖操作还要等待结果。实际周期与吞吐取决于目标指令和架构。

</details>

## 官方资料与后续学习

| 资料 | 对应问题 |
| --- | --- |
| [GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/) | 计算、内存与延迟限制，算术强度 |
| [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) | SM 子分区、调度状态、stall 与指标口径 |
| [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | occupancy、合并访存、shared tiling |
| [Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/) | 宽访存、对齐和指令数 |
| [CUDA Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html) | 生产、提交、等待与释放 |
| [Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html) | cp.async、TMA、descriptor、对齐与同步 |
| [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) | TMA 与 warp specialization 的架构背景 |

<link rel="stylesheet" href="/css/cuda-learning.css">
<link rel="stylesheet" href="/css/cuda-overlap.css">
<script src="/js/cuda-overlap.js" defer></script>
