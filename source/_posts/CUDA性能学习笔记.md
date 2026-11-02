---
title: CUDA 性能学习（二）：从 SM 调度、warp stall 到访存优化与 TMA 流水线
description: 先分清性能术语与 SM 内部管线，再用手绘图、交互模型和完整 CUDA demo 理解 occupancy、延迟隐藏、合并访存、tiling、向量化及 TMA。
categories:
  - LLM 推理
tags:
  - CUDA
  - 基础学习
  - GPU 性能
permalink: systems/cuda-overlap/
date: 2026-11-02 13:35:00
---

## 阅读路线与术语入口

后续学习按主题整理在 [GPU 算子学习路线](/systems/gpu-learning-path/)。当前可以继续读 [Roofline 与 Triton／TileLang 实验](/systems/gpu-roofline/)，把这里的性能判断接到实际练习。

[第一篇：CUDA 基础与 GPU 物理／逻辑结构](/systems/cuda-basics/) 解释了“任务在哪里执行、数据在哪里”。第二篇接着问：**SM 里面如何调度？线程为什么等？怎样让等待期间仍然有工作可做？**

术语先分成三类：**bound 是瓶颈判断，warp 状态是某一时刻发生了什么，指标是如何统计这些现象**。先读下面三张表，再沿 **SM 结构 → 指令流转 → 瓶颈判断 → 驻留与调度 → tiling → overlap 实现** 阅读。不要用一个指标直接替代瓶颈结论。

### A. 瓶颈分类：bound 到底被什么限制

`bound` 的意思是“受某种限制约束”，讨论时必须说清对象：整个程序、某个 kernel，还是其中一个阶段。这里先区分**工作能否持续供应**与**服务吞吐是否已饱和**。

| 分类 | 限制是什么 | 如何建立判断 | 不能直接推出什么 |
| --- | --- | --- | --- |
| compute-bound，计算吞吐受限 | 某类算术执行管线的处理能力 | 独立工作充足，相关计算通路接近可持续吞吐，减少对应计算能改善耗时 | 有很多乘加就一定算力饱和；某条管线满就代表所有单元都满 |
| memory-bound，访存受限 | 广义指数据访问限制进度 | 继续定位内存层和机制：HBM/L2/shared，带宽、依赖或访问效率 | 不能只凭这个标签决定加 warp；Roofline 中该词常采用“带宽受限”的狭义含义 |
| bandwidth-bound，带宽受限 | 某条数据通路单位时间能服务的字节数 | 相关层接近有效可持续带宽，流量与运行时间对应 | HBM 没满不代表所有访存层都没满；加在途请求不会增加已饱和带宽 |
| latency-bound，延迟受限 | 关键依赖的等待没有被独立工作覆盖 | issue 存在空档，结合依赖链、eligible warp 和通路利用率寻找等待来源 | 低带宽本身不足以证明它；算术依赖也能造成延迟受限 |
| synchronization-limited，同步限制（补充） | 工作等待其他参与者或某一阶段完成 | 对照 barrier、工作不均衡及生产／消费时间线 | barrier 上采样多，不等于屏障指令本身执行很慢 |
| launch／host-overhead limited，提交或主机开销限制（补充） | 程序时间主要花在 kernel 外或大量短任务提交上 | 看端到端 CPU/GPU 时间线和空档，改变任务粒度验证 | 单个 kernel 内的 occupancy 通常解释不了整个程序的空闲 |

这不是互斥标签的考试题。例如“访存受限”还可以细分为“HBM 带宽受限”或“地址依赖造成访存延迟受限”；同步等待也可能表现为可发出指令不足。分析时给出**范围 + 机制 + 证据**，例如“该 kernel 的主要等待来自连续依赖的 global load，HBM 吞吐尚未饱和”。[NVIDIA GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/) · [Nsight Compute：Compute Triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/)

### B. warp 状态：驻留、就绪和发出不是一件事

以下描述的是调度域中 warp 的状态与调度事件；一个 warp 发出一条指令后仍可能继续驻留，不是沿表从上到下只走一次就退出。

| 状态或事件 | 准确含义 | 它与其他状态的关系 |
| --- | --- | --- |
| pending／尚未驻留（说明性用语） | block 还未获得执行所需资源，其 warp 尚未进入相应驻留池 | grid 很大不等于所有 warp 都同时 resident；这里不是特定 profiler 的状态枚举 |
| resident / active，驻留 | warp 的执行状态及所需资源已分配在 SM 的相应调度域中 | 包含能发出与暂时不能发出的 warp；active 不等于正在做有效计算 |
| eligible，可发出 | 下一条指令可用、依赖已满足，相关资源允许发出 | 是 resident 中此刻可供选择的候选；不保证马上被选中 |
| selected / issued，被选择／已发出 | 调度器选择该 warp，并向相应通路发出指令 | 是本次发出事件，不等于操作完成，也不释放该 warp 的全部资源 |
| not selected，未被选中 | 本来可发出，但本次选择了其他候选 | 通常说明还有可用候选；不应当作数据未就绪那种真 stall |
| stalled，暂不能发出 | 因指令、数据依赖、同步或资源条件等不能推进下一条指令 | 它仍可能 resident；要继续看等待原因与其他 warp 是否填补空档 |
| completed，执行完成（说明性用语） | 该 warp 已完成其工作 | warp 完成与整个 block 的全部资源可回收不是同一件事；block 生命周期仍要看其余参与者 |

例如一个调度域有 `8 resident`，此刻其中 `2 eligible`，本次发出了其中 1 个：另一个是未选中的候选，其余 6 个需继续定位不能发出的原因。不能把 8、2、1 都叫“正在执行的 warp”。[Nsight Compute：Hardware Model 与 Scheduler Statistics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#hardware-model)

### C. 相关指标：每个数字的单位和统计范围是什么

| 指标／量 | 定义与单位 | 阅读时要核对的口径 |
| --- | --- | --- |
| latency，延迟 | 发起到所需结果可用的间隔，常用 ns、cycle | 哪一种操作、哪一层，是否包含排队；cycle 换算需频率 |
| kernel duration，执行耗时 | 一次 kernel 的时间，常用 μs、ms | 是否含 host 开销；event 计时与 profiler 采集的条件是否一致 |
| throughput，吞吐率 | 工作量 / 时间，如 FLOP/s、指令/s | 数据精度、指令种类、有效工作与实际执行工作是否一致 |
| bandwidth，带宽 | 字节 / 时间，如 GB/s | HBM、L2、shared、PCIe 分别看；逻辑有效字节不等于真实事务字节 |
| arithmetic intensity，算术强度 I | FLOP / 跨所选层的 byte | 哪一层的流量，是否计写回、缓存复用、尾块和额外搬运 |
| theoretical occupancy，理论占用率 | 资源与启动配置允许的驻留 warp 数 / SM warp 上限 | 容量估计；看寄存器、shared、线程和 block 等限制 |
| achieved occupancy，实际占用率 | 运行中观测的 active warp 相对上限的统计 | 时间与 SM 聚合口径、尾部和不均衡；不等于计算单元利用率 |
| active / eligible warps，活跃／可发 warp 数 | 调度域中相应状态的数量，常见为每周期平均 | `sm` 与 `smsp` 范围不同，平均值可以是小数；不能与瞬时整数快照直接相减解释所有周期 |
| issue activity，发出活跃程度 | 有指令发出的周期比例或具体发出率 | 具体指标的分母和单位；不能把它等同所有计算管线利用率 |
| pipe utilization，管线利用率 | 相应通路活动或吞吐相对基准的比例 | 哪个通路、哪种峰值与统计窗口；不等于 occupancy |
| cache hit rate，缓存命中率 | 按该指标定义统计的命中请求／sector 等比例 | 请求还是 sector、哪层缓存；高命中率仍可能受该层吞吐或延迟限制 |
| stall samples，等待样本 | 采样到某类 warp 等待状态的数量或比例 | 是样本不是直接累计耗时；同周期其他 warp 可能在工作，不能简单相加当总停机时间 |

`long scoreboard`、`short scoreboard`、`barrier`、`LG/MIO throttle` 是帮助定位等待机制的原因名称；相关采样指标统计这些现象。它们不是另一组与 compute-bound 并列的最终结论。第 9 节再将指标和优化动作对应起来。[Nsight Compute：Metrics 与 Warp Sampling](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

后文出现的 **tile** 是数据分块，**slot/buffer** 是存放块的空间，**stage** 是流水线中的缓冲与同步阶段，**overlap** 是独立工作在时间上重叠。这些是实现机制，不放进 bound 或 warp 状态表。第 6、7 节会让同一组数据从原理图走到代码与实验。

本文配有 **13 张可编辑手绘图、6 个浏览器概念实验、5 份完整 CUDA 程序**。图中的队列是功能示意，不是特定 GPU 的精确电路图；模拟时间单位不对应真实 GPU 周期。资料核对日期：2026-11-02。本机与当前可访问的计算环境均无可用 CUDA GPU／`nvcc`，CUDA 源码尚未编译和 GPU 实跑；不提供虚构测速数据。

[下载全部源码与 Makefile](/downloads/cuda-overlap/cuda-overlap-demos.zip) · [编译、校验和计时说明](/downloads/cuda-overlap/README.txt)

## 1. 先放大一个 SM：谁保存状态，谁调度，谁计算

### 1.1 先看组成：哪些部件属于子分区，哪些由整个 SM 共享

下面先画**包含和共享关系**，下一张图再画指令流。外层绿框是一个 SM；内层是子分区，其中只把一个子分区展开。紫色是保存线程值的寄存器，橙色表示计算或依赖条件，蓝色表示调度／访存相关部件；连线只突出部分主要联系，不是完整端口布线。

<figure class="cuda-diagram">
<a href="/images/cuda-overlap/09-sm-components.svg" target="_blank" rel="noopener"><img src="/images/cuda-overlap/09-sm-components.svg" alt="一个 SM 内的多个子分区，每个子分区包含 warp 状态、调度、寄存器与执行资源，共享 SM 级 L1 和 shared" loading="lazy"></a>
<figcaption>SM 组成图：先读包含关系，再看部件如何相互提供数据、指令和就绪条件。 <a href="/images/cuda-overlap/09-sm-components.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-overlap/09-sm-components.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

| 组件或层级 | 保存什么／做什么 | 依赖哪些其他部件 |
| --- | --- | --- |
| SM 子分区（SMSP） | 一个局部调度与执行域，管理一组驻留 warp | block 已获准在这个 SM 驻留；warp 的调度受本域资源约束 |
| warp 执行状态与指令获取 | 记录执行位置、活跃线程等，提供下一条候选指令 | 需要取到指令并满足解码等条件；它不保存全部数组数据 |
| warp scheduler／dispatch | 选择满足条件的指令，发往适合的执行通路 | 候选指令可用、依赖解除、目标通路允许接收 |
| register file，寄存器文件 | 保存线程操作数、地址值、累加器等活跃值 | 算术和地址计算读取它；计算或 load 的结果使目标值就绪 |
| scoreboard 等依赖控制 | 跟踪未完成操作与相关依赖 | 根据操作完成情况更新；向调度控制提供是否可推进的条件 |
| FP／INT／SFU／Tensor 等计算资源 | 按指令类型执行浮点、整数、特殊函数或张量运算 | 需要输入操作数和通路容量；各类型不能随意互换 |
| load/store 单元与访存通路 | 形成地址、组织请求，执行 load/store | 地址与 store 数据就绪、请求入口和下游容量可用；load 返回又解除计算依赖 |
| SM 级 L1／shared 存储 | 缓存数据或存放程序管理的 shared 对象 | 接收各子分区的相关访问；物理共享不代表所有 block 的 shared 数组互相可见 |
| SM 外互连、L2、内存控制器 | 为需要下一级服务的访问提供数据路径 | 缓存策略、命中、请求竞争和内存容量；它们不属于某个线程私有资源 |

可以把依赖记成四种关系：**包含**（SM 有子分区）、**资源分配**（线程用寄存器、block 用 shared）、**数据流**（load 返回值供 FMA 使用）、**调度条件**（值就绪且目标通路可接收）。一个 block 的线程在同一个 SM 上，并不意味着它的所有 warp 必须归同一个子分区调度；不要把 block 框与 SMSP 框画成一一对应。

这里采用 [Nsight Compute 通用 SM 硬件模型](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#hardware-model) 的分组方式。具体功能单元数量、共享方式与通路随架构变化；图中的四个子分区不是要求程序员固定写四个 warp。

### 1.2 再看这些部件如何工作

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

### 1.3 一条 load → FMA → store 怎样走完

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

### 1.4 计算管线和访存管线如何排队

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

先沿图中的箭头看**同一条 load 指令里，各线程分别读哪里**。lane 是线程在 warp 内的编号；一个 sector 是这里用于统计请求覆盖范围的 32-byte 单元，可容纳 8 个 float。图上半部每个线程只用到一个 sector 中的一个 float，下半部 8 个相邻线程一起用满一个 sector；两部分为便于阅读使用不同的格子宽度，均不表示物理距离。

<figure class="cuda-diagram">
<a href="/images/cuda-overlap/14-coalescing-addresses.svg" target="_blank" rel="noopener"><img src="/images/cuda-overlap/14-coalescing-addresses.svg" alt="合并访存前后：lane 0到7跨步读取下标0、32到224，分散在8个sector；连续读取下标0到7，只覆盖一个32-byte sector" loading="lazy"></a>
<figcaption>图 6-1：箭头表示读取地址，橙色或绿色格子是需要的数据，空白格是本次未使用的数据。 <a href="/images/cuda-overlap/14-coalescing-addresses.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-overlap/14-coalescing-addresses.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

跟踪 lane 1：跨步版读 `in[32]`，落在 sector 4；连续版读 `in[1]`，与 lane 0 的 `in[0]` 落在同一个 sector 0。这里 sector 编号相对数组起点计算，起点按 32 byte 对齐。**优化的是线程到地址的映射，让同一次请求中需要的数据更集中。**

扩展到一个满 warp 的一次标量 `float` load：32 个线程连续读 32 个 float，共 128 byte，覆盖 4 个 32-byte sector；如果每个线程的地址相隔 32 个 float，就会触及 32 个不同 sector。这里讨论的是典型 CC 6.0+ 下请求覆盖的 sector，**不是保证 HBM 必然搬运多少字节**：缓存命中、其他 warp 的复用与具体硬件会改变外部流量。

所以合并访存关注的是**跨线程地址关系**，不是把多个 CUDA API 合成一个，也不是让每个线程循环次数更少。[CUDA Best Practices：Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

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

#### 先看整块矩阵：为什么 A 取这个 tile，B 取另一个 tile

先不代入元素的数值，取 `M=N=K=4,T=2`，把每个矩阵切成四块。图中的 `A00`、`A01` 等是**tile 的行列编号**，每一块仍是一个 `2×2` 小矩阵；例如 `A01` 是 A 的右上块，并非元素 `A[0,1]`。下面固定计算 C 的右上块 `C01`，先读第一行图，再读第二行图：蓝色 A 块向右移动，绿色 B 块向下移动，橙色 C 块一直留在原位累加。

<figure class="cuda-diagram">
<a href="/images/cuda-overlap/15-tiling-matrix-blocks.svg" target="_blank" rel="noopener"><img src="/images/cuda-overlap/15-tiling-matrix-blocks.svg" alt="计算C的右上tile：第一轮A左上块乘B右上块，第二轮A右上块乘B右下块，两轮结果累加到同一个C右上块" loading="lazy"></a>
<figcaption>图 6-2：每个大格包含 2×2 个元素；蓝色和绿色表示本轮选中的输入，橙色表示输出累加位置，灰色表示本轮不参与。 <a href="/images/cuda-overlap/15-tiling-matrix-blocks.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-overlap/15-tiling-matrix-blocks.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

选块规则来自单个输出的计算定义：

```text
C[row,col] = A[row,0]*B[0,col] + ... + A[row,K-1]*B[K-1,col]
```

因此，要计算 `C01` 中的全部四个输出，就有三个约束：

1. **C 的行决定 A 的行。** 输出行固定为 0、1，A 就只取这两行；取 A 下半部会算到 C 的下半部。
2. **C 的列决定 B 的列。** 输出列固定为 2、3，B 就只取这两列；取 B 左半部会算到 C 的左半部。
3. **A 的列与 B 的行必须取同一段 K。** 第一轮配对 `k=0,1`，第二轮配对 `k=2,3`。例如把 `A00` 配上 `B11`，虽然形状仍能相乘，却会把 A 的第 0、1 列与 B 的第 2、3 行相乘，已经不是原来的矩阵乘法。

用切片写得更直接。`a:b` 表示包含 a、不包含 b 的范围；两轮输入乘积都是 `2×2`，逐元素加到同一个 `2×2` 输出块：

```text
C[0:2,2:4] = A[0:2,0:2] @ B[0:2,2:4]
           + A[0:2,2:4] @ B[2:4,2:4]

分块记法：C01 = A00 @ B01 + A01 @ B11
```

第一轮只算完 K 的前半段，不能把它当成最终结果；第二轮继续加到原来的输出累加器。推广到任意完整方形 tile，令 `r0`、`c0` 为输出块的起始行列，`base` 为本轮 K 的起点：

```text
A_tile = A[r0:r0+T, base:base+T]     # T×T，固定输出行
B_tile = B[base:base+T, c0:c0+T]     # T×T，固定输出列
C_tile += A_tile @ B_tile           # T×T，跨轮保留累加值
base 每轮增加 T；遍历完整个 K 后才写回 C_tile。
```

#### 选好块之后，复用发生在哪里

本轮从 global 搬入 `T×T` 个 A 元素和 `T×T` 个 B 元素到 shared，然后让输出 tile 的线程重复使用它们。A 的一个值被同一输出行的多个线程使用，B 的一个值被同一输出列的多个线程使用。

对应图 6-2 的第一轮，`A[1,0]` 同时参与 `C[1,2]` 和 `C[1,3]`，`B[0,2]` 同时参与 `C[0,2]` 和 `C[1,2]`。把输入块合作搬进 shared 后，各输出线程可以读取同一份值；这就是分块带来的复用。这里一个线程负责一个输出元素，后面的数值和代码都沿用这一规则。

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

#### 从图上的 tile 走到 `matmul_tiled` 的线程索引

现在给图 6-2 的矩阵填上具体数字，手算同一套实现规则。仍取 `M=N=K=4,T=2`；完整 CUDA demo 使用 `TILE=16`，这里只缩小数字，不改变算法。输入为：

```text
A = [[ 1,  2,  3,  4],    B = [[1, 0, 1, 0],
     [ 5,  6,  7,  8],         [0, 1, 0, 1],
     [ 9, 10, 11, 12],         [1, 1, 1, 1],
     [13, 14, 15, 16]]         [2, 0, 0, 2]]
```

将这两个输入按图中的边界切块，可以先算完整个输出 tile：

```text
第 1 轮：A00 @ B01 = [[1,2], [5,6]] @ [[1,0], [0,1]]
                  = [[1,2], [5,6]]
第 2 轮：A01 @ B11 = [[3,4], [7,8]] @ [[1,1], [0,2]]
                  = [[3,11], [7,23]]
相加得到 C01      = [[4,13], [12,29]]
```

接着跟踪 C 右上角输出 tile 内的一个线程：`blockIdx=(x=1,y=0)`，`threadIdx=(x=0,y=1)`。它负责 `row=0*2+1=1`、`col=1*2+0=2`，即 `C[1,2]`，也就是上面结果块的左下格 12。注意二维编号明确按 x、y 解释，矩阵下标按 row、col 解释。

<figure class="cuda-diagram">
<a href="/images/cuda-overlap/10-tiling-walkthrough.svg" target="_blank" rel="noopener"><img src="/images/cuda-overlap/10-tiling-walkthrough.svg" alt="用 TILE 等于2的例子将 C[1,2] 的计算分成两轮输入 tile，第一轮增加5，第二轮增加7" loading="lazy"></a>
<figcaption>图 6A：同一线程的 sum 跨轮保留；shared 中的 A/B tile 在每轮结束后被安全替换。 <a href="/images/cuda-overlap/10-tiling-walkthrough.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-overlap/10-tiling-walkthrough.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

| 图中的一步 | `tiling.cu` 对应代码 | 跟踪的线程实际做什么 |
| --- | --- | --- |
| 找到输出格子 | `row = blockIdx.y*TILE + threadIdx.y`，`col = blockIdx.x*TILE + threadIdx.x` | 固定负责 `C[1,2]`，不是每轮换一个输出 |
| 选择本轮输入 | `for (base=0; base<k; base+=TILE)` | 先看 K 的 0、1 项，再看 2、3 项 |
| 合作装入 shared | `at[ty][tx]=a[row*k+base+tx]`；`bt[ty][tx]=b[(base+ty)*n+col]`（完整代码含边界检查） | 本线程第一轮搬 `A[1,0]=5` 和 `B[1,2]=0`；其他线程搬其余值 |
| 第一道屏障 | 填充后的 `__syncthreads()` | 等所有 tile 元素可用，才读其他线程装入的值 |
| 计算本轮贡献 | `sum += at[ty][p]*bt[p][tx]` | 第一轮 `5*1+6*0=5`；第二轮 `7*1+8*0=7` |
| 第二道屏障 | 内积后的 `__syncthreads()` | 旧数据无人再读，下一轮才能覆盖它 |
| 写回结果 | K 循环后 `c[row*n+col]=sum` | 最终 `C[1,2]=12`；只写一次最终累加值 |

<div class="cuda-demo" data-overlap-demo="tiling"><p>启用 JavaScript 按轮推进 tile，观察输入块和 C[1,2] 的累加值。</p></div>

**怎么观察：** 点“下一轮 K 分块”，应看到 `base: 0→2`、`sum: 5→12`。这个小实验解释数据复用与屏障的位置，不测速度。随后运行 `./tiling 17 19 23`，检查两种实现的 `PASS`，理解同样规则如何通过填零处理尾块；再用较大输入比较耗时。图中手算矩阵与程序默认生成矩阵不同，数值 12 只属于本手算例子。

### 6.3 向量化：改变单个线程的一次访问宽度

`float4` 表示连续四个 float，尺寸 16 byte。在编译器生成相应宽访存指令时，可以用更少的 load/store 指令完成同样字节量。它不会自动减少需要读取的数据，也不会自动把标量 FP32 算术变成 Tensor Core 运算。

先看同一批线程在一次循环中负责的范围：上半图每个线程拿一个元素，下半图每个线程拿四个相邻元素。颜色区分下半图中不同线程负责的连续范围；箭头表示读取关系，不表示执行时间。

<figure class="cuda-diagram">
<a href="/images/cuda-overlap/16-vectorized-access.svg" target="_blank" rel="noopener"><img src="/images/cuda-overlap/16-vectorized-access.svg" alt="标量与float4访问对比：标量线程t0到t3各读一个float，float4线程t0读0到3、t1读4到7、t2读8到11、t3读12到15" loading="lazy"></a>
<figcaption>图 6-3：单个线程的访问宽度变大；两种映射的线程间地址都连续，都可以合并访问。 <a href="/images/cuda-overlap/16-vectorized-access.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-overlap/16-vectorized-access.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

沿图跟踪线程 `t1`：标量版第一轮读取 `in[1]`；向量版第一轮读取 `in4[1]`，覆盖 `in[4]`、`in[5]`、`in[6]`、`in[7]`。因此同一批线程在一轮中处理的数据更多。对于固定的总数据量，向量版可减少循环和线程级访存操作；图中的“16 次与 4 次”是**线程级访问计数**，不能直接当作 warp 发出的指令条数或内存事务数。实际编译结果还需检查。

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

下载 ZIP 后解压，确保这些文件在同一目录。需要 NVIDIA GPU、匹配驱动、支持目标架构的 CUDA Toolkit、C++17 编译支持和 `make`。下面 `sm_80` 是示例编译目标；Hopper 可选 `sm_90`，其他硬件请使用其 Toolkit 支持的正确目标。下载包还包含第 7.4 节的 `overlap.cu`，因此整包编译需要 CC 8.0+ 目标；只运行本节与 occupancy 示例时可以单独编译对应文件。

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

### 7.4 对应到完整 `overlap.cu`：不只停在伪代码

[下载 overlap.cu](/downloads/cuda-overlap/overlap.cu)。这个程序比较同步搬运与 `cp.async` 双缓冲的完整流程；它使用 CUDA C primitives（`<cuda_pipeline.h>`），要求 **CC 8.0+ 与支持目标 GPU 的 CUDA Toolkit**。它没有调用 TMA，TMA 的 descriptor／transaction barrier 留在第 8 节单独解释。

为把搬运协议看清楚，计算任务采用“分块读取后进行若干轮 FMA”，不是再写一个复杂 GEMM。每块 128 个线程，每线程负责一个 `float4`，因此一个输入 tile 是 `128*16=2048 byte`。两个 shared slot 共占 4096 byte。线程读取本 tile 中下一个线程搬来的 float4，因此必须说明**跨线程完成可见性**；同步版与异步版计算、输出和两个 slot 的逻辑分配相同。

<figure class="cuda-diagram">
<a href="/images/cuda-overlap/11-overlap-slots.svg" target="_blank" rel="noopener"><img src="/images/cuda-overlap/11-overlap-slots.svg" alt="将 overlap.cu 的预提交、下一块提交、当前块等待计算及释放对应到两个 shared slot" loading="lazy"></a>
<figcaption>图 7A：按行看循环阶段，按列看同一块 shared 空间何时被覆盖。 <a href="/images/cuda-overlap/11-overlap-slots.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-overlap/11-overlap-slots.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

先跟踪 `k=0`：循环前提交 tile 0 到 slot 0；进入循环又提交 tile 1 到 slot 1；等 tile 0 完成后计算它，此时 tile 1 的搬运允许继续。第二道块内屏障保证 tile 0 已无人使用，下一轮才能让 tile 2 覆盖 slot 0。异步并没有移除数据依赖，只把**下一块**不依赖当前结果的工作提前。

| 协议目的 | 实际语句 | 对应的图与注意点 |
| --- | --- | --- |
| 预填第一块 | `__pipeline_memcpy_async(...)`，随后 `__pipeline_commit()` | 图第一行；提交 tile 0 后还不能直接读取 |
| 提前填下一块 | 拷贝到 `slot[(k+1)%2][t]` 并 commit | 图中 `k=0` 搬 tile 1、`k=1` 搬 tile 2；必须保证目标 slot 的旧消费者已读完 |
| 当前块完成，下一块可继续 | 有下一块时 `__pipeline_wait_prior(1)` | 等待除最新一批外的已提交批次；不是等待“编号为 1 的 tile”，也不是任意等一轮 |
| 排空末尾 | 没有下一块时 `__pipeline_wait_prior(0)` | 最后一批必须完成；`tiles=1` 时同样走此分支 |
| 允许跨线程读 | wait 后 `__syncthreads()` | 每线程先等自己的拷贝，再集合，才能读邻居搬来的值；这道屏障不取代异步 wait |
| 安全复用 | 计算之后再 `__syncthreads()` | 所有消费者已用完当前 slot，未来循环才能覆盖 |

这些 primitives 的 commit/wait 批次跟踪有特定的线程／warp 语义；代码中的循环和分支对整个 block 一致，不让不同线程自行跳过阶段。16-byte 搬运的两端都满足 float4 对齐，所有 tile 完整分配。本例没有“不足一个 tile 的尾部”；命令行指定每块的完整 tile 数，禁止零或负数。[CUDA：异步拷贝与预取协议](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)

```bash
make overlap ARCH=sm_80  # 根据 GPU 与 Toolkit 改为支持的目标
./overlap 1 16          # 只有一块：检查初始填充和最终排空
./overlap 2 16          # 两块：看到两个 slot 都参与
./overlap 8 1           # 计算工作少，未必能遮住搬运
./overlap 8 64          # 增加独立计算窗口；仍需实测判断收益
compute-sanitizer --tool racecheck ./overlap 3 16
```

参数依次是 `tiles_per_block` 和 `rounds`。CPU 按相同的块内邻居读取和 FMA 规则生成参考，两个版本必须先通过校验才计时。比较同一组参数的两个版本；不要把 `rounds=1` 与 `rounds=64` 的绝对耗时当同工作量加速比。计时差反映整个实现的差异，还包括指令、屏障等成本；应检查编译后的 SASS 和 profiler，不能只凭源码含 async 就宣称已成功重叠。

当前仍没有 `nvcc`／GPU，本文件是完整待编译验证源码，尚无 GPU 正确性或性能实测。网页时间线是独立的教学模型，不是这个程序的 trace。

<details>
<summary>展开完整源码：overlap.cu</summary>

```cpp
#include "common.cuh"
#include <cuda_pipeline.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "Build overlap.cu for compute capability 8.0 or newer."
#endif
constexpr int THREADS = 128;
__host__ __device__ float transform(float x, int rounds) {
    for (int r = 0; r < rounds; ++r) x = fmaf(x, 1.0001f, 0.0001f);
    return x;
}
__host__ __device__ float4 compute(float4 x, int rounds) {
    return make_float4(transform(x.x, rounds), transform(x.y, rounds),
                       transform(x.z, rounds), transform(x.w, rounds));
}
// The baseline uses the same number of shared slots, so resource allocation is comparable.
__global__ void serial_tiles(const float4* in, float4* out, int tiles, int rounds) {
    __shared__ float4 slot[2][THREADS];
    int t = threadIdx.x;
    for (int k = 0; k < tiles; ++k) {
        int base = (blockIdx.x * tiles + k) * THREADS;
        slot[k % 2][t] = in[base + t];
        __syncthreads();
        out[base + t] = compute(slot[k % 2][(t + 1) % THREADS], rounds);
        __syncthreads();
    }
}
__global__ void pipelined_tiles(const float4* in, float4* out, int tiles, int rounds) {
    __shared__ float4 slot[2][THREADS];
    int t = threadIdx.x;
    int first = blockIdx.x * tiles * THREADS;
    __pipeline_memcpy_async(&slot[0][t], &in[first + t], sizeof(float4));
    __pipeline_commit(); // Submit tile 0 before entering the loop.
    for (int k = 0; k < tiles; ++k) {
        int base = (blockIdx.x * tiles + k) * THREADS;
        if (k + 1 < tiles) {
            __pipeline_memcpy_async(&slot[(k + 1) % 2][t], &in[base + THREADS + t], sizeof(float4));
            __pipeline_commit();
            __pipeline_wait_prior(1); // Current tile complete; newest tile may still be in flight.
        } else {
            __pipeline_wait_prior(0); // Drain the final tile, including the tiles=1 case.
        }
        __syncthreads(); // Every thread has waited for its copies; allow cross-thread reads.
        out[base + t] = compute(slot[k % 2][(t + 1) % THREADS], rounds);
        __syncthreads(); // All consumers finished before a future iteration reuses this slot.
    }
}
int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) { std::fprintf(stderr, "Usage: overlap [tiles_per_block rounds]\n"); return 1; }
    const int tiles = argc == 3 ? argument(argv[1], 64) : 8;
    const int rounds = argc == 3 ? argument(argv[2], 256) : 16;
    auto prop = device();
    if (prop.major < 8) { std::fprintf(stderr, "Requires CC 8.0+ for this async-copy example.\n"); return 1; }
    const int blocks = prop.multiProcessorCount * 2;
    const size_t count = static_cast<size_t>(blocks) * tiles * THREADS;
    std::vector<float4> input(count), output(count);
    std::vector<float> actual(count * 4), expected(count * 4);
    for (size_t i = 0; i < count; ++i) {
        float x = static_cast<float>(i % 97) / 128.0f;
        input[i] = make_float4(x, x + .25f, x + .5f, x + .75f);
    }
    for (size_t i = 0; i < count; ++i) {
        size_t neighbor = (i / THREADS) * THREADS + (i + 1) % THREADS;
        float4 y = compute(input[neighbor], rounds);
        expected[4*i] = y.x; expected[4*i+1] = y.y; expected[4*i+2] = y.z; expected[4*i+3] = y.w;
    }
    DeviceBuffer<float4> din(count), dout(count);
    CHECK(cudaMemcpy(din.data, input.data(), count * sizeof(float4), cudaMemcpyHostToDevice));
    for (int variant = 0; variant < 2; ++variant) {
        CHECK(cudaMemset(dout.data, 0xff, count * sizeof(float4)));
        auto launch = [&] {
            if (variant == 0) serial_tiles<<<blocks, THREADS>>>(din.data, dout.data, tiles, rounds);
            else pipelined_tiles<<<blocks, THREADS>>>(din.data, dout.data, tiles, rounds);
        };
        launch(); CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(output.data(), dout.data, count * sizeof(float4), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) {
            actual[4*i] = output[i].x; actual[4*i+1] = output[i].y;
            actual[4*i+2] = output[i].z; actual[4*i+3] = output[i].w;
        }
        validate(actual, expected);
        std::printf("%s: %.6f ms; tiles/block=%d rounds=%d\n", variant ? "async_double_buffer" : "serial", measure(launch), tiles, rounds);
    }
    std::puts("Timing compares whole implementations, not isolated overlap. Check SASS/profiling; speedup is not guaranteed.");
}
```

</details>

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

<a id="performance-analysis"></a>

## 9. 性能分析流程：从症状走到证据，再决定优化

这一节根据学习截图中的决策树和指标字典重新整理，并对照 NVIDIA 官方指标定义补充适用范围。**先找真正影响目标耗时的工作，再把“怀疑什么、怎样验证、改什么”连起来。** 图中的分支是调查入口，不是只看两个百分比就能确定的结论。

### 9.1 先定位热点，确认采集可信且工作量足够

1. **先验正确性与基线。** 固定输入、GPU、编译选项、精度、时钟／功耗条件及其他负载，记录未插入重型 profiler 时的耗时。区分单 kernel 时间和端到端时间。
2. **Nsight Systems 找优先级。** 看 kernel、拷贝、CPU 提交和同步的时间线。总耗时高或频繁出现的 kernel 是候选，但并发 kernel 的耗时之和不等于端到端关键路径；不能仅凭汇总表的占比宣称优化它就能等比例加速应用。
3. **对少数目标启动做 Nsight Compute。** 使用优化编译和 `-lineinfo`，按需要采集 section。核对 replay、缓存控制、时钟控制与原始运行场景是否一致；指标矛盾时先修采集，不先改 kernel。
4. **先排除工作量不足。** 看 grid、block、SM 数、waves 和尾部。若独立 block 太少，GPU 天然填不满，优先考虑把独立工作展开或合理批处理，而不是直接增加 stage。若只是大量短 kernel 的提交开销，先处理任务粒度、图执行等端到端问题。

这些前置检查解释了为什么“SM% 和 MEM% 双低”不一定是 kernel 内的访存等待：也可能是任务太小、尾部太长、提交空档或采集条件不同。[Nsight Systems：Kernel Summary](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html) · [Nsight Compute：Triage Workflow](https://docs.nvidia.com/nsight-compute/ComputeTriage/)

### 9.2 读 SOL：它是入口，不是占用率或瓶颈判决

SOL 是 Speed Of Light 的缩写，工具用它表示相对某种峰值／可持续吞吐基准的表现。先在 **SpeedOfLight** 概览读 SM 和 memory 侧，再展开底层管线；不要把它与 warp occupancy、`nvidia-smi` 的 GPU utilization 混用。

| 观察对象 | 常见指标或界面入口 | 读数需要补上的含义 |
| --- | --- | --- |
| SM 侧概览 | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 不是所有计算单元“忙碌百分比”的简单平均，更不等于 occupancy；展开看由哪些通路主导 |
| memory 侧概览 | SpeedOfLight 的 Memory Throughput／breakdown；某些版本提供 `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 不直接等于 HBM/DRAM 带宽；继续拆 L1/TEX、L2、DRAM 和指令／请求通路 |
| 目标层吞吐 | 如 `dram__throughput.avg.pct_of_peak_sustained_elapsed`，以及对应 L1/TEX、L2 的 breakdown | 是哪个层的哪些计数、用 active 还是 elapsed 周期作分母；不能混用口径比较 |
| 指令是否发得出来 | Scheduler Statistics 的 active、eligible、issue activity | 用来区分候选不足、依赖等待和通路接收能力；与吞吐概览交叉看 |

Nsight Compute 的 throughput 指标会取其构成计数器的峰值百分比中的最大值来表达压力，因此“某个概览很高”不意味着其覆盖的全部资源同时饱和。`sm__...` 与 `smsp__...` 的聚合范围也不同。具体指标名称和可用性按目标 GPU、工具版本查询。[Nsight Compute：Metrics Structure](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#metrics-structure)

截图中的“相差 20”“双低小于 40”“双高大于 60”“DRAM 大于 70”适合作为特定实验的筛选经验，不能作为所有硬件和 kernel 的硬阈值。本篇流程图因此使用“哪侧更忙、是否接近合理可持续水平”，并要求下一步证据；未明显落入某一象限时，继续看 breakdown 和绝对耗时，不强行分类。

<figure class="cuda-diagram">
<a href="/images/cuda-overlap/12-analysis-flow.svg" target="_blank" rel="noopener"><img src="/images/cuda-overlap/12-analysis-flow.svg" alt="性能分析从热点定位与采集验证，经任务规模和SOL分流，到内存、计算、等待及高利用分支，最终统一复测" loading="lazy"></a>
<figcaption>图 9：调查入口与优化结论分开，每条分支最后都要验证。 <a href="/images/cuda-overlap/12-analysis-flow.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-overlap/12-analysis-flow.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

**如何沿图走：** 蓝色是前置检查；橙色调查内存服务压力，紫色调查计算／指令通路，红色调查未被隐藏的等待，绿色调查是否已接近当前实现的合理上限。四条分支可以同时存在，只是每轮先处理最重要的一个。左侧绿色回线表示复测后重新分流。所有分支最后都回到正确性与同条件耗时验证，不能把绿色分支直接写成“停止优化”。

### 9.3 四条分支各自要回答什么

| 概览现象 | 首先建立的假设 | 下一步证据与动作 |
| --- | --- | --- |
| memory 侧更忙 | 某层搬运或请求服务成为主要限制 | 按 9.6 看哪层吞吐高、流量为何这么大、地址是否浪费；若带宽未饱和且 issue 有空档，再转入等待分支 |
| SM／计算侧更忙 | 某种计算或指令发出通路压力较大 | 按 9.7 看具体 FP／INT／Tensor／特殊函数等通路，核对算法工作量与 SASS；不是一律换 MMA |
| 两侧都不忙 | 工作供应不足或关键依赖等待暴露 | 排除 9.1 的小任务／采集问题后，看 active、eligible、issue 与主要等待原因，再查 9.5 的驻留资源 |
| 两侧都忙 | 当前实现可能已较好利用资源，也可能有大量无效工作 | 检查有效字节／有效 FLOP、重复计算、端到端贡献；决定是否改算法、融合或减少物化，而不是凭百分比结束 |

“停止优化”的依据是：正确性成立、代表性输入下收益已小、端到端目标已满足，进一步复杂度不值得。吞吐比例下降但耗时更短也可能是好事，例如减少了多余字节或计算。[NVIDIA Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)

### 9.4 warp stall 字典：先定位等待的消费者，再回溯生产者

下面列的是**原因名称**，不是独立的瓶颈标签。常见采样指标族为 `smsp__pcsamp_warps_issue_stalled_*`；部分平台另有仅在调度器未发出周期采样的 `_not_issued` 变体。实际后缀、可用性和含义以目标设备的指标说明为准。不要把不同采样族的比例直接混算。

| 原因 | 它说明什么 | 先补哪项证据 | 证据成立后可尝试什么 |
| --- | --- | --- | --- |
| `wait` | 等固定延迟的执行依赖 | 对应 SASS 的生产者／消费者链，独立指令与 eligible 是否不足 | 拆独立累加链、增加 ILP、重排独立工作；有驻留余量时再考虑更多 warp |
| `long_scoreboard` | 等 L1TEX 相关操作的结果依赖，如 global/local/texture/surface | 找等待结果的指令与相关 load；看命中、流量、合并度、是否 spill | 先修地址与复用，再评估预取、`cp.async`／TMA 或 stage；不能直接把它解释成“在等 TMA” |
| `short_scoreboard` | 等非 L1TEX 的 MIO 操作结果，shared 是常见来源之一 | 是否真有 shared 依赖、bank conflict，也检查特殊函数或其他相关操作 | 减少 shared 往返、改善布局、增加独立工作；不能仅凭名字断定 bank conflict |
| `math_pipe_throttle` | 目标数学执行通路暂不能接收／不可用 | 哪类 pipe 已繁忙，发出、指令混合与依赖情况 | 若目标 pipe 饱和，减少相应工作或调整指令混合；若还有空档，查依赖与调度，不能一律加 warp |
| `lg_throttle` / `mio_throttle` | 相应 local/global 或 MIO 入口／队列压力，名称依设备而异 | 指令粒度、发出密度、下游服务和队列压力 | 适当宽访存、减少冗余操作、交错独立算术／搬运，并检查副作用 |
| `sleeping` | warp 的线程处于阻塞、让出或休眠等状态 | 是显式 sleep、同步实现的等待，还是生产／消费不均衡；看另一角色此时在做什么 | 调整等待策略或 producer/consumer 工作分配；必要的等待不是无条件的浪费 |
| `barrier` | 等同 CTA 的其他参与者到达相应屏障 | 谁早到、谁晚到，是否分工不均或数据迟到 | 先改善屏障前的工作与协作范围；只在证明依赖允许后减少屏障，不能直接删同步 |
| `not_selected` | 有可发出的候选，但本次选择了别的 warp | issue 是否持续有效，相关 pipe 是否有用地工作 | 通常无需清零；存在候选不等于整个 kernel 已无其他瓶颈 |
| `selected` | 被选择并发出指令的调度状态，不是真 stall | 发出的是什么指令，完成了多少有效工作 | 用耗时和有效吞吐评价；不是无条件追求样本数量更大 |

尤其注意：TMA／异步矩阵操作的等待可能体现在 transaction barrier、group wait 或其他架构相关同步状态中，必须看实际指令和协议，不能把所有异步数据等待都归成 `long_scoreboard`。也不能把 MIO“结果没好”的 scoreboard 依赖与 MIO“入口满”的 throttle 当成同一机制。[Nsight Compute：Warp Stall Reasons](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#warp-stall-reasons) · [CUDA：异步拷贝完成机制](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)

采样常记在**消费尚未就绪结果的指令**处。例如 `y=fmaf(x,...)` 等前面的 `x=input[i]`，应沿依赖回到 load 的地址和访问模式。只把采样热点那一行代码删减，可能没有触及原因。一个 warp 的等待样本也不是整个 SM 的停机时间，其他 warp 可能正在工作。

### 9.5 occupancy 分支：是装不下，还是没给够任务

先把两种情况分开：

- **理论容量就低**：读 occupancy 的限制项。寄存器过多、shared 太大、线程／block 数上限等都可能限制驻留。再评估缩小 tile、减少 stage、缩短活跃值生命周期、调整 block 等候选；这些动作也可能损失复用或延迟覆盖。
- **理论容量尚可、实际偏低**：回看 grid／waves、尾部、负载不均以及执行阶段。不能仅凭 achieved occupancy 低，就认定寄存器是瓶颈。

`__launch_bounds__` 是编译约束工具，不是“把指定数量 block 强行塞进 SM”的开关；压寄存器也可能引入 spill。增加 resident warp 之前，先确认有资源和足够独立工作；目标通路已经饱和时，更多驻留未必改善耗时。

local load/store 非零还要区分**编译器安排在线程私有 local memory 的对象**与**寄存器 spill**。结合编译器资源／spill 报告、SASS、local 字节量和源码定位，不能仅凭 `LDL/STL` 就断定超过某个固定寄存器阈值。每线程与每 SM 的限制依架构而定。[CUDA Best Practices：Occupancy 与 Registers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)

### 9.6 Memory Workload：分层看吞吐、命中与访问放大

这里有三个相互独立的问题：**哪条通路忙、数据由哪层提供、有没有为有效数据付出多余请求／传输**。只看命中率无法回答另外两问。

<figure class="cuda-diagram">
<a href="/images/cuda-overlap/13-memory-diagnosis.svg" target="_blank" rel="noopener"><img src="/images/cuda-overlap/13-memory-diagnosis.svg" alt="分别检查L1/TEX、L2、DRAM的吞吐命中及访问放大，并独立检查shared bank冲突和资源限制" loading="lazy"></a>
<figcaption>图 9A：按层归因，并将访问模式、缓存与驻留资源交叉检查。 <a href="/images/cuda-overlap/13-memory-diagnosis.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-overlap/13-memory-diagnosis.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

图从上往下调查 global 访问的服务层级；shared 是旁边的独立分支。每层的框都是“检查什么”，不是说每次访问必定走到最下面。L1 允许被某些访问策略绕过；TMA 等路径应按其实际指令语义分析。

| 层级或维度 | 观察入口／常见例子 | 怎样解释，下一步做什么 |
| --- | --- | --- |
| L1/TEX | `l1tex__t_sector_hit_rate.pct`、相关 throughput／请求表 | 先核对缓存策略和该访问是否本来应复用。一次性流式读低命中可以正常；应复用却反复 miss 时，再考虑布局、tile 或显式 shared 复用 |
| L2 | `lts__t_sector_hit_rate.pct`、L2 throughput、字节／sector | 与工作集、CTA 的实际访问顺序和重复读取对应。复用距离太长时可试调整分块／调度；高命中也可能使 L2 本身成为瓶颈 |
| DRAM | `dram__throughput.avg.pct_of_peak_sustained_elapsed`、读写字节和带宽 | 接近代表性访问模式下的可持续水平且字节量解释耗时时，优先减少流量；带宽低但 issue 空档多时回查依赖与在途并发 |
| 合并访存 | L1/TEX 表中 Requests、Sectors、Sectors/Req，以及对应访问类型 | 同时核对 active lane、每 lane 字节宽度、对齐和访问模式；比率高并不总意味着浪费 |
| shared | bank conflicts、wavefronts、shared 指令与服务吞吐 | 判断是否有冲突重放或服务压力，随后试 padding／swizzle 或减少往返；这不是 L1 缓存命中问题 |

这些是常见名称示例，运行前用 `ncu --query-metrics` 确认本机支持的完整名字；global/local、load/store、texture 等流量混在一个总指标里时，尽量拆出与当前假设对应的限定项。[Nsight Compute：Memory Workload Analysis 与 Memory Tables](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#memory-tables)

**一个容易误判的“4”。** 对 32 个活跃 lane、每 lane 读取一个 4-byte float、连续且按 sector 对齐的教学访问，128 byte 覆盖四个 32-byte sector。但若每 lane 一次合法读取 float4，整 warp 的有效数据就变成 `32*16=512 byte`，至少覆盖 16 个这样的 sector；不能继续拿“4”作为所有向量化访问的正确性或效率阈值。部分活跃 lane、广播／重叠地址和未对齐也会改变关系。实际 request 拆分需看对应架构与工具计数口径。

因此诊断访问放大时，需要把**相同有效数据量**对应的指令、请求、sector 和下游字节一起比较，而不是只追某个比率越小越好。向量化主要改变单线程的指令宽度；跨线程地址有跨距时，应先修映射，不能指望套 `float4` 自动解决合并访存。[NVIDIA：Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/) · [CUDA Best Practices：Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

**L1/shared 容量的代价。** 许多架构二者共用底层 SRAM，且每块 shared 需求还会限制驻留。增加 tile/stage 同时可能改变复用、缓存容量和 occupancy，应该一起测量。截图中的“某 GB202 kernel 每多 16 KB shared 慢若干微秒”缺少本篇可复现实验条件，不能推广为线性规律；本篇也未在该 GPU 上测得这个结果。

**不要把不同 swizzle 混起来。** CTA swizzle 是改变逻辑工作分块的调度／访问顺序以探索局部性；shared swizzle 是改变 shared 中的地址布局以影响 bank 映射。Stream-K 则涉及矩阵乘法的工作划分与负载平衡，并不是一个“L2 命中率低就打开”的通用缓存选项。初学时先用第 6 节的固定 tiling 例子把数据复用弄清楚。

### 9.7 compute 分支：查“忙的是谁”，再查“忙得是否有价值”

- **Tensor 通路高**：检查是否确实完成了有价值的矩阵工作，数据类型与计算量是否合适。降低 FLOP、融合消除中间工作或改变精度，需要算法和误差要求允许；不能只为让指标漂亮改变正确性。
- **FP／INT／特殊函数通路高，Tensor 低**：先确认任务是否有可映射的矩阵运算及合适形状／精度。如果只是地址计算、归约或不适合 MMA 的工作，Tensor 低很正常。适合 GEMM 的情形才进一步评估库实现或 MMA／WGMMA 等路径。
- **发射指令很多，有效算术少**：查看分歧、地址与索引开销、零碎访存、额外转换和编译产生的控制指令。不同类型的 pipe 压力不能一概解释为“算力不够”。

`math_pipe_throttle` 可以提示某通路的接收能力问题，但它与 `wait` 的结果依赖不同。目标 pipe 已持续饱和时，增加同类指令的候选 warp 不会凭空提高该 pipe 的吞吐。[Nsight Compute：Compute Workload Analysis](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

### 9.8 把流程真正用到本篇 demo

下面是可在安装工具的 NVIDIA GPU 环境执行的采集方式；本机无此环境，以下是操作示例，不是本文已取得的测量报告。先执行程序的校验，再采集，报告文件保存在运行命令的当前目录。

```bash
# 先看时间线：哪些 kernel / 拷贝 / 主机间隔影响整体。
nsys profile --trace=cuda,nvtx --output=overlap_trace ./overlap 8 16
nsys stats --report cuda_gpu_kern_sum overlap_trace.nsys-rep

# 再按本机实际可用的 section 和 metric 采集。
ncu --list-sections
ncu --query-metrics

# 本文 tiling 程序每个版本先校验一次，再 warmup 五次。
# 因而跳过目标 kernel 的前六次匹配启动，采一轮计时阶段的启动。
ncu --kernel-name regex:matmul_tiled --launch-skip 6 --launch-count 1 \
    --section SpeedOfLight --section LaunchStats --section Occupancy \
    --section SchedulerStats --section WarpStateStats \
    --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
    --output=tiled_report ./tiling 256 256 256
```

先检查筛选结果确实匹配目标 kernel；若修改了 demo 的启动顺序，应相应修改 skip。只采某一个 kernel 是为了定位它内部的机制；Nsight Compute 可能通过 replay 等方式改变原本并发条件，不能把该报告当作端到端 overlap 时间线。[Nsight Systems 使用指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) · [Nsight Compute 采集说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

| 用哪个 demo | 本轮只检验什么假设 | 对照哪些证据 | 结果如何解释 |
| --- | --- | --- | --- |
| `coalescing.cu` | 连续映射减少每份有效数据的访问放大 | 相同 N 下的 sectors/requests、下游字节和耗时 | 请求改善但时间不变时，继续查缓存复用、地址指令和其他瓶颈，不自动宣称带宽加速 |
| `tiling.cu` | 显式块内复用减少重复 global 访问 | naive/tiled 的 global 与 shared 流量、寄存器／shared、occupancy 和时间 | global 访问减少可能伴随 shared 与屏障成本；最终看总耗时 |
| `vectorized.cu` | 宽访存减少完成相同字节所需的指令开销 | SASS 访问宽度、指令数、有效字节与时间 | 每个请求覆盖更多 sector 可能是宽指令的正常结果，不按标量的“4”判坏 |
| `overlap.cu` | 下一 tile 的搬运提前，减少暴露的等待 | 固定 tiles/rounds 下的两个版本、等待的消费者、issue 与时间 | async 源码和 stall 占比下降都不独立证明加速；确认正确性和绝对耗时 |
| `occupancy.cu` | 启动配置和 shared 如何限制理论驻留 | API 给出的容量，结合实际 kernel 的资源与 profiler | 它只查询示例 kernel 的理论容量，不是其他 kernel 的 achieved occupancy |

举一个**假设分析过程**：`tiling` 的 tiled 版本比 naive 快，但 achieved occupancy 较低。先核对全量输出正确，再看到 global 请求减少而 shared 使用增加，这就支持“复用收益超过了驻留损失”的解释。若随后加深 stage 反而变慢，应回查 shared 占用、可发 warp 与新增同步，而不是坚持“更深流水一定更好”。这些是如何推理的例子，不是本篇实测结论。

### 9.9 每轮保留一条可以被否证的记录

```text
对象：GPU / Toolkit / 编译参数 / 输入 / 目标 kernel 或阶段
基线：正确性；独立计时；端到端时间；采集设置
症状：例如 issue 有空档，活跃 warp 多、可发 warp 少
假设：哪条具体依赖或通路造成空档
证据：源代码与 SASS 位置、相关层级指标、反例排查
改动：一次主要改变一个因素，注明附带资源变化
结果：相同工作量再次校验、计时、采集
结论：保留或回退；若瓶颈迁移，重新进入流程
```

这条闭环把前文的概念连起来：**SM 结构说明资源在哪里，warp 状态说明为何不能发，分层指标说明等谁或挤在哪里，tiling／向量化／overlap 是带着证据选择的实现手段。**

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
