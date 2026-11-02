---
title: CUDA 基础学习（一）：从一次向量加法理解执行与内存
description: 用十张 Excalidraw 手绘图、八个交互实验与 CUDA demo，对应 GPU 物理结构、线程组织和多层访存路径。
categories:
  - LLM 推理
tags:
  - CUDA
  - 基础学习
permalink: systems/cuda-basics/
date: 2026-11-02 12:03:33
---

## 这篇笔记想弄懂什么

第一次读 CUDA 代码，容易同时遇到 kernel、block、warp、shared memory、stream 等术语。我的第一步是把它们放回一个具体问题里：**CPU 想让 GPU 完成一次向量加法，任务如何分配，数据放在哪里，什么时候能读取结果？**

本文面向能读懂 C/C++ 数组、指针和循环的读者，采用 CUDA C++ 的线程编程方式。重点是原理和结构；Tensor Core、异步流水线、复杂算子优化留到后续。文中的小规模输入用于理解，不用于证明 GPU 比 CPU 快。

资料查阅日期为 **2026-11-02**，核心依据是 NVIDIA 官方编程指南和入门教程。完整示例是为这篇笔记编写的；当前本机没有 `nvcc`，尚未在 CUDA 环境编译或在 GPU 上运行，文中输出标为预期结果。

这版配有 **10 张可放大的 Excalidraw 手绘图、8 个浏览器概念演示和 4 份 CUDA 源码 demo**。浏览器演示只计算概念模型，不调用 GPU；下载的 CUDA 程序需要 NVIDIA GPU 环境。建议每节先看图，再操作小实验，最后按需展开代码。

[下载全部 CUDA demo（ZIP）](/downloads/cuda-basics/cuda-basics-demos.zip) · [运行说明](/downloads/cuda-basics/README.txt)

## 1. 从一个普通循环开始

设 `A`、`B`、`C` 都是长度为 `N` 的一维数组，元素类型为 `float`。目标是：

```text
C[i] = A[i] + B[i]
其中 0 <= i < N
```

CPU 上最直接的写法如下：

```cpp
for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
}
```

观察这个循环：计算 `C[0]` 不需要等待 `C[1]`，不同下标也不会写同一个位置。因此，可以把不同下标分给不同执行者。CUDA 版本中，我们先采用“一个线程负责一个元素”的分工。

这里的关键是先找到彼此独立的工作，再把工作映射给线程。GPU 能否带来收益，还取决于数据规模、数据传输和启动开销；只有十个元素的教学例子没有性能代表性。[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#what-runs-on-a-cuda-enabled-device)

### 小 demo：改变输入，看各元素独立计算

把下面的倍数从 2 改成 3，观察每个输出如何变化。试着只看第 2 个元素：它需要知道其他位置的结果吗？

<div class="cuda-demo" data-cuda-demo="vector"><p>启用 JavaScript 可操作此概念演示；正文仍可独立阅读。</p></div>

## 2. host、device 和 kernel：谁在执行什么

在 CUDA 的语境中，host 指 CPU 侧，device 指 GPU 侧。在本文的程序里，`main()` 在 CPU 上运行，负责准备数组、申请设备内存和提交计算；`add_arrays` 是运行在 GPU 上的 kernel。一次 kernel 启动会组织许多线程执行同一份函数代码。[Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)

```text
CPU / host                         GPU / device
准备 A、B
分配 device_a、device_b、device_c
复制输入 ------------------------> 保存 A、B
启动 add_arrays -----------------> 多个线程计算 C
等待完成 <------------------------ 计算结束
接收 C <-------------------------- 复制输出
检查结果、释放资源
```

<figure class="cuda-diagram">
<a href="/images/cuda-basics/01-host-device.svg" target="_blank" rel="noopener"><img src="/images/cuda-basics/01-host-device.svg" alt="CPU 准备和拷贝输入，GPU 执行 kernel，再把结果拷回 CPU" loading="lazy"></a>
<figcaption>图 1：先沿三个箭头看一次完整计算。注意“提交 kernel”与“得到结果”之间还有等待。 <a href="/images/cuda-basics/01-host-device.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-basics/01-host-device.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

这张图描述的是本例采用的显式分配与拷贝流程，不代表所有 CUDA 系统都必须显式搬运每份数据。

还要把三个常见名字分开：CUDA Toolkit 是开发工具、头文件和库的集合；`nvcc` 是其中用于编译 CUDA C++ 的工具；CUDA Runtime API 提供 `cudaMalloc`、`cudaMemcpy` 等接口，底层再使用驱动提供的能力。Toolkit 和 NVIDIA 驱动是不同的软件组件。[The CUDA Platform](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html)

### 小 demo：一步步推进程序

在“提交 kernel”处停一下：此时只知道工作已提交，还不能声称计算完成。再推进到等待完成和拷回结果，观察 GPU 存储与 CPU 数组的区别。

<div class="cuda-demo" data-cuda-demo="flow"><p>启用 JavaScript 可操作此概念演示；正文仍可独立阅读。</p></div>

## 3. thread、block、grid 与 warp、SM

### 3.1 先看程序如何组织任务

- **thread（线程）**：执行 kernel 的一个实例，拥有自己的索引和局部状态。
- **block（线程块）**：一组线程，块内可以协作，例如交换 shared memory 中的数据并进行块内同步。
- **grid（网格）**：一次普通 kernel 启动所组织的全部线程块。

把它们画在一起：

```text
一次 kernel 启动
└─ grid
   ├─ block 0：thread 0、1、2、3
   ├─ block 1：thread 0、1、2、3
   └─ block 2：thread 0、1、2、3
```

每个 block 内都有一个 `thread 0`，所以只看块内编号还不够定位全局任务。需要同时知道“第几个 block”和“block 里的第几个 thread”。[Writing SIMT Kernels：Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#thread-hierarchy)

### 3.2 再看硬件怎样承接这些任务

<strong>SM（Streaming Multiprocessor，流式多处理器）</strong> 是 GPU 上执行线程的硬件单元。一个普通 block 的线程由同一个 SM 执行；一个 SM 在资源允许时可以同时驻留多个 block。block 的编号不表示固定的 SM 编号，也不保证执行先后。

**warp** 是同一个 block 内组织线程执行的一组，当前 CUDA 的 warp 大小为 32 个线程。它采用 SIMT（Single Instruction, Multiple Threads，单指令多线程）模型：线程执行同一个 kernel，但拥有各自的数据和状态，也可能走不同分支。[Programming Model：GPU Hardware Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#gpu-hardware-model)

<figure class="cuda-diagram">
<a href="/images/cuda-basics/02-grid-sm.svg" target="_blank" rel="noopener"><img src="/images/cuda-basics/02-grid-sm.svg" alt="每个含64个线程的block拆为两个warp，再由SM承接线程块" loading="lazy"></a>
<figcaption>图 2：左边是程序划分的任务，右边只画一种可能的硬件分配。颜色箭头不表示固定调度策略。 <a href="/images/cuda-basics/02-grid-sm.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-basics/02-grid-sm.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

因此，读代码时可以分成两层：`grid/block/thread` 说明任务怎样组织；warp 是线程的执行分组，由 SM 内的硬件承接。**不要把一个 CUDA 线程想成永久独占一个物理计算核心。**

同一个 warp 内的线程如果走不同分支，执行相应路径时会有部分线程不参与，这就是理解分支分歧的起点。但“同属一个 warp”不能替代明确的线程协作与同步规则。线程间交换数据时，仍应使用适合其范围的同步机制。[Writing SIMT Kernels：Basics of SIMT](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#basics-of-simt)

### 小 demo：block 和 warp 怎么数

先试 64 个线程，再改成 40：每块仍要分成两个 warp，但第二个只有 8 个有效线程位置。再改成 4：一个 block 对应一个只有 4 个有效线程位置的 warp；多个小 block 也不会拼成同一个 warp。

<div class="cuda-demo" data-cuda-demo="hierarchy"><p>启用 JavaScript 可操作此概念演示；正文仍可独立阅读。</p></div>

[下载 thread-layout.cu](/downloads/cuda-basics/thread-layout.cu) 可在 GPU 上记录每个线程的 block、块内 warp 与 lane。默认启动 2 个 block，每块 40 个线程。输出由 CPU 按数组下标整理，因此打印顺序不代表 GPU 的执行顺序。

```bash
nvcc -std=c++17 thread-layout.cu -o thread-layout
./thread-layout
```

预期记录之一为 `i=39 block=0 thread=39 warp_in_block=1 lane=7`；第 40 条全局记录进入 block 1，块内 thread、warp、lane 都重新从 0 开始。这些是按索引定义推导的值。

### 3.3 把整个 GPU 的物理结构和逻辑结构对起来

可以先把 GPU 想成由许多 SM 组成的计算设备，再放大其中一个 SM。下面画的是**面向 CUDA 计算的结构全景**：涵盖计算单元、片上存储、显存与 host 连接；省略图形、显示、视频等专用模块。它不是某一型号的精确芯片平面图，实际数量、分区和多芯粒组织依架构而变。

<figure class="cuda-diagram">
<a href="/images/cuda-basics/08-physical-logical.svg" target="_blank" rel="noopener"><img src="/images/cuda-basics/08-physical-logical.svg" alt="左侧 grid、block、warp、thread 对应右侧 GPU、SM、warp 调度器及寄存器；下方连接 L2、内存控制器与设备显存" loading="lazy"></a>
<figcaption>图 2A：先沿右侧看硬件包含关系，再看左侧的任务怎样使用它。箭头表示承接关系，不是固定的一对一绑定。 <a href="/images/cuda-basics/08-physical-logical.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-basics/08-physical-logical.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

从外往里读右侧：

1. **CPU 与主机内存**在 host 一侧。CPU 提交工作，并通过平台支持的连接与 GPU 传输数据；独立显卡常见 PCIe，特定平台也有其他 CPU–GPU 互连。它与 SM 内部访存不是同一层连接。
2. **GPU 的计算部分**包含多组 SM。NVIDIA 把 SM 组织在 GPC（Graphics Processing Cluster）等硬件分组中；有些架构还会出现 TPC 等中间分组。初学写普通 kernel 时，不需要直接选择这些分组。
3. **SM 内部**有 warp 调度器、寄存器文件、不同类型的执行单元，以及 load/store 单元。调度器从可执行的 warp 中发出指令；算术指令交给相应计算单元，访存指令交给访存通路。常说的 CUDA Core 是其中的算术执行资源，不能把它理解为一个固定运行某个 CUDA thread 的 CPU 核。
4. **SM 附近的片上存储**包括 register、shared memory 和 L1。许多现代 NVIDIA GPU 的 L1 与 shared 使用同一块物理 SRAM 的不同容量分配，但两者的使用方式不同。SRAM 指静态随机访问存储器，常用于片上高速存储。
5. **L2 位于 SM 之外**，服务多个 SM，物理上可以分成多个分区。再向外是内存控制器和设备 DRAM（动态随机访问存储器），也就是通常说的显存，例如 HBM 或 GDDR。HBM 可以与计算芯片同封装；同封装不等于位于 SM 内，也不等于寄存器或缓存。

这些组件的角色和内存通路可以对照 [NVIDIA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) 与 [Nsight Compute：GPU 和 Memory Chart 模型](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 阅读。

| CUDA 代码里的逻辑对象 | 硬件如何承接 | 不能据此推断什么 |
| --- | --- | --- |
| 一次 kernel 的 grid | block 被分配到可用 SM，可能分多批执行 | grid 不要求全部 block 同时驻留 |
| 一个普通 block | 其线程在同一个 SM 上执行，使用该 SM 的寄存器与 shared 等资源 | `blockIdx.x` 不是 SM 编号；一个 SM 可驻留多个 block |
| 一个 warp | 同一 block 内最多 32 个线程位置，由 SM 的调度器发出指令 | 不保证一条 warp 指令只用一个时钟周期完成 |
| 一个 thread | 有自己的索引、寄存器状态和执行状态；其指令使用 SM 内的功能单元 | 不永久独占某个 CUDA Core |
| block 的 shared 数组 | 驻留 SM 上为该 block 分配的 shared 空间 | 不同 block 的同名数组不是同一份数据 |
| global 数组 | 本例的分配由设备 DRAM 支撑，访问可能命中 L1/L2 | `A[i]` 不意味着每次都去显存芯片读取 |

硬件层级中有 GPC，并不意味着普通 grid 下面必须添加一个对应它的逻辑对象。支持的设备上还可以显式使用 **thread block cluster**：计算能力 9.0 起提供这一可选层级，让一组 block 在同一个 GPC 内协同调度，并支持集群内同步与 distributed shared memory。本篇普通 kernel 没有启用 cluster，后文的 shared 都指普通块内用法。[Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters)

用已有 demo 对照：`thread-layout.cu` 的 `<<<2, 40>>>` 表示一个 grid、两个 block、每块两个 warp，共 80 个线程。GPU 有多少个 SM 不会改变这些逻辑数量；实际驻留在哪个 SM，以及同一时刻驻留几个 block，由硬件和资源约束决定。

## 4. 线程怎样找到自己负责的数组元素

这里使用一维 grid 和一维 block，四个内置变量的含义如下：

| 变量 | 本例中的含义 |
| --- | --- |
| `threadIdx.x` | 当前线程在 block 内的编号 |
| `blockIdx.x` | 当前 block 在 grid 内的编号 |
| `blockDim.x` | 每个 block 的线程数 |
| `gridDim.x` | grid 中的 block 数 |

这是 CUDA 提供的位置与尺寸信息；“每个线程具体处理什么数据”由程序计算决定。[Writing SIMT Kernels：Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#thread-hierarchy)

设每个 block 有 `T` 个线程，当前 block 编号为 `b`，块内线程编号为 `t`。我们给这个线程分配的数组下标为：

```text
i = b * T + t
  = blockIdx.x * blockDim.x + threadIdx.x
```

为什么要乘 `T`？因为在当前 block 前面，已经按编号划出了 `b` 组任务，每组 `T` 个。再加上组内位置 `t`，就得到全局下标。这里的“前面”指编号关系，不指时间先后。

用 `N = 10`、`T = 4` 手算：

| block 编号 | 块内线程编号 | 算出的全局下标 | 本次工作 |
| --- | --- | --- | --- |
| 0 | 0、1、2、3 | 0、1、2、3 | 处理前四个元素 |
| 1 | 0、1、2、3 | 4、5、6、7 | 处理中间四个元素 |
| 2 | 0、1、2、3 | 8、9、10、11 | 仅下标 8、9 有效 |

需要的 block 数是向上取整的 `N / T`。对本例的正整数与小规模索引，可以写成：

```text
blocks = (N + T - 1) / T  // 整数除法
       = (10 + 4 - 1) / 4
       = 3
```

我们启动 12 个线程，但只有 10 个元素，所以 kernel 必须检查 `i < N`。后两个线程仍然存在，只是不访问数组。

<figure class="cuda-diagram">
<a href="/images/cuda-basics/03-thread-index.svg" target="_blank" rel="noopener"><img src="/images/cuda-basics/03-thread-index.svg" alt="三个block每块四个线程覆盖十个元素，下标10和11被边界条件排除" loading="lazy"></a>
<figcaption>图 3：绿色格子会访问数组，红色格子只是不做本次有效工作，并不是没有启动线程。 <a href="/images/cuda-basics/03-thread-index.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-basics/03-thread-index.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

“一个线程一个元素”只是本例的分工选择。以后遇到一个线程循环处理多个元素的写法，也应回到下标公式逐步分析。

### 小 demo：自己调整 N 和 T

试试 `N = 9, T = 4`：应有 3 个越界线程。再试 `N = 12, T = 4`：最后一块刚好填满。下面会把每个线程对应的数组下标列出来。

<div class="cuda-demo" data-cuda-demo="index"><p>启用 JavaScript 可操作此概念演示；正文仍可独立阅读。</p></div>

## 5. 数据放在哪里：区分访问范围和物理位置

本例用 `std::vector` 保存 CPU 侧数组，用 `cudaMalloc` 申请 GPU 侧存储，再通过 `cudaMemcpy` 传输。`device_a` 这个指针变量本身在 host 代码中，但它指向 device 分配；不能因为拿到了指针数值，就在 CPU 上直接执行 `device_a[0]` 来访问那份设备数据。[Intro to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)

读 kernel 时，常见的几种存储需要分清：

| 名称 | 基本访问范围 | 理解重点 |
| --- | --- | --- |
| register | 线程私有 | 编译器管理，位于 SM 上 |
| shared memory | 普通用法下由同一个 block 的线程共享 | 用于块内协作，程序显式组织访问 |
| global memory | 持有合法地址的设备线程可以访问 | 常用于保存输入、输出与跨 kernel 保留的数据 |
| local memory | 线程私有 | “local”描述作用范围，存储位于设备内存空间 |

例如，局部变量并不保证一定放进 register；较大的线程私有数组或寄存器溢出可能使用 local memory。另一方面，shared memory 的“shared”也不表示整个 grid 自动共享同一份空间。[Writing SIMT Kernels：Memory Spaces](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)

<figure class="cuda-diagram">
<a href="/images/cuda-basics/04-memory.svg" target="_blank" rel="noopener"><img src="/images/cuda-basics/04-memory.svg" alt="GPU中的register和block级shared位于SM，global和线程私有local位于设备内存空间" loading="lazy"></a>
<figcaption>图 4：先问谁能访问，再问放在哪里。A、B 两个 block 的 shared memory 是不同的对象。 <a href="/images/cuda-basics/04-memory.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-basics/04-memory.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

对向量加法来说，各线程只读自己的 `A[i]`、`B[i]`，写自己的 `C[i]`，没有块内交换数据的需求，因此不需要为了“用上 GPU 特性”而添加 shared memory。

### 5.1 存储空间和缓存层级，为什么不能混成一个列表

`global / shared / local` 主要回答程序访问哪种存储空间、谁可以访问；`L1 / L2 / DRAM` 则回答数据由哪个物理层级提供。**同一个 global 地址可以在显存中有数据，也在缓存中有副本。** L1、L2 不是需要分别 `cudaMalloc` 的两种数组。

| 物理位置 | 对应的逻辑用途 | 谁负责管理内容 |
| --- | --- | --- |
| SM 的寄存器文件 | 保存线程的活跃值，例如加载结果、下标和中间结果 | 编译器分配；硬件执行寄存器读写 |
| SM 的 shared 存储 | `__shared__` 数组，供一个普通 block 协作 | 程序显式写入、读取，并安排同步 |
| SM 的 L1 数据缓存 | 缓存部分 global/local 等访问的数据，具体策略依架构与指令而定 | 硬件管理命中、填充和替换 |
| 多个 SM 共用的 L2 | 缓存流向设备内存的访问数据 | 硬件管理，物理上可分区 |
| 设备 DRAM（HBM/GDDR） | 本例 global 分配，以及需要使用 local 的线程私有存储 | 程序/运行时管理分配，硬件完成实际访问 |

shared 和 L1 即便共享底层 SRAM，也不是同一种地址空间：普通 `A[i]` 的缓存命中不会自动变成 `tile[i]`；声明 shared 数组也不会自动把 global 数据搬进去。[Nsight Compute：Memory Chart](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-chart)

这样的分层让容量与访问成本有所取舍：靠近执行单元的片上存储容量有限，设备 DRAM 则提供更大的存储空间。实际访问耗时还受命中率、带宽、竞争和指令影响，不能给每一层贴一个适用于所有 GPU 的固定延迟。

### 5.2 跟踪一次 `float x = A[i]`：请求向外走，数据送回寄存器

假设 `A` 指向当前 GPU 的普通设备分配、编译器确实生成了这次普通标量 global load，并且该指令允许使用 L1 缓存。先忽略地址翻译、只读专用通路和异步拷贝等细节：

1. 线程计算自己的地址。一个 warp 执行这条 load 时，活跃线程的地址交给访存单元；硬件按地址分布组合、拆分为相应访问请求。因此，“32 个线程”不等于“向显存发送 32 次独立读取”。
2. 请求检查本 SM 的 **L1**。需要的数据已经在这里时，可以由 L1 提供；不需要为这次命中的部分继续读取 L2 或 DRAM。
3. L1 未命中的部分交给 **L2**。L2 命中则从 L2 返回；只有仍未命中的部分才继续访问设备内存。
4. L2 也未命中时，通过**内存控制器**读取 HBM/GDDR。返回的数据按具体缓存策略处理，最终把线程需要的值交给目标寄存器。
5. 后面的 `x + 1` 依赖加载结果。等待这个结果的指令不能提前使用它；SM 可以发出其他就绪 warp 的指令，但不保证总能隐藏等待时间。

<figure class="cuda-diagram">
<a href="/images/cuda-basics/09-load-path.svg" target="_blank" rel="noopener"><img src="/images/cuda-basics/09-load-path.svg" alt="global load 请求按条件查询 L1、L2 与 DRAM；任一命中层向目标寄存器返回数据，shared 不在这条必经路径上" loading="lazy"></a>
<figcaption>图 4A：橙色是向外查找的请求，绿色是数据返回。图中假定允许 L1 缓存；命中哪一层就由哪一层提供数据。 <a href="/images/cuda-basics/09-load-path.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-basics/09-load-path.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

寄存器在这里是**加载的目的地**，不是执行新 load 之前先查找的一级地址缓存。若 `x` 已经保存在寄存器，后续复用 `x` 直接使用寄存器值即可；这是复用已取得的值，而不是又执行了一次对 `A[i]` 的 load。编译器也可能消除重复加载，不能仅凭 C++ 代码行数计算真实访存次数。

还要保留两个边界：有些加载会绕过 L1，直接使用 L2 通路；一个 warp 的请求也可能有一部分命中、一部分未命中。图中为了入门只跟踪一个所需数据片段，不假设所有线程总是命中同一层。缓存命中也不能替代线程间同步。[Nsight Compute：L1/TEX Cache](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#l1-tex-cache) · [CUDA Best Practices：Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)

### 小 demo：切换访问对象，看路径是否还经过显存

先选择 global load，依次切换 L1 命中、L2 命中、全部未命中；再切换到 shared 和寄存器。这里的命中状态由你指定，用于解释路径，不预测真实缓存命中率，也不模拟延迟。local 一项假定编译器确实产生了 local memory 访问。

<div class="cuda-demo" data-cuda-demo="memory"><p>启用 JavaScript 可切换 global、shared、register、local 的读取路径。</p></div>

### 5.3 shared memory 怎样接到这条路径上

shared memory 是程序显式使用的另一条分支，不是 `register → shared → L1 → L2 → DRAM` 这种自动查找链中的一级。普通 C++ 写法 `tile[t] = A[i]` 通常先执行 global load 得到寄存器临时值，再执行 shared store。支持的异步拷贝指令可以直接把 global 数据送入 shared、绕过中间寄存器；这是后续进阶路径，不能用来推断所有普通赋值的实现。

<figure class="cuda-diagram">
<a href="/images/cuda-basics/10-shared-reuse.svg" target="_blank" rel="noopener"><img src="/images/cuda-basics/10-shared-reuse.svg" alt="global 数据通过普通加载和 shared store 填入 shared；块内同步后，线程通过 shared load 读入寄存器并计算" loading="lazy"></a>
<figcaption>图 4B：上半部是显式搬运，下半部是块内复用。shared load 本身不需要再次查询原 global 地址的 L1/L2/DRAM。 <a href="/images/cuda-basics/10-shared-reuse.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-basics/10-shared-reuse.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

将后面的 `shared-rotate.cu` 核心语句拆开，就能逐行对上硬件。下面是用于理解的代码片段，假设每个 block 恰好 4 个线程、输入长度是 4 的倍数：

```cpp
__shared__ float tile[4];
int t = threadIdx.x;
int i = blockIdx.x * blockDim.x + t;
float x = input[i];          // global load：缓存或 DRAM → 寄存器
tile[t] = x;               // shared store：寄存器 → 本块 shared
__syncthreads();             // 本块参与线程写完之后，才能互读
float y = tile[(t + 1) % 4]; // shared load：本块 shared → 寄存器
output[i] = y;              // global store：走写入通路
```

例如 block 0 的 thread 0 先把 `input[0]` 搬进 `tile[0]`；屏障后读取的是 thread 1 已写入的 `tile[1]`。这次 shared load 直接使用 shared 中的数据，不需要重新沿 global 路径读取 `input[1]`。该例主要展示协作；是否加速需要实测。

这里的变量名只是源码概念，具体是否分配独立寄存器以及使用什么机器指令由编译器决定。最后一行是 **store**，其缓存策略与可见性规则不能简单套用前面的 load 命中流程。普通缓存也不是 block 间通信的同步机制。

线程私有的 **local memory** 则是另一种容易误会的情况：它的“私有”是逻辑范围，实际存储由设备内存支撑，读取可能经过缓存。寄存器压力导致 spill 时，某些原本希望留在寄存器的值需要保存到 local、随后重新加载；因此“多用局部变量”并不自动等于“只访问 SM 寄存器”。[Writing SIMT Kernels：Memory Spaces](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html) · [Nsight Compute：Load Global Store Shared](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-chart)

### 5.4 Unified Memory 为什么看起来少了拷贝步骤

一些入门教程使用 `cudaMallocManaged`，使同一份 managed allocation 能被 CPU 和 GPU 代码访问。它简化了内存管理，但不意味着数据访问没有代价，也不取消读写之间的同步要求；具体机制与平台能力有关。本篇选择显式拷贝，方便先看清数据位置。[Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)

### 小 demo：每块各自交换数据

[shared-rotate.cu](/downloads/cuda-basics/shared-rotate.cu) 把 8 个整数分给两个 block，每块 4 个线程。每线程先把自己的值写进本块的 shared memory；块内屏障之后，再读取本块的下一个位置。

<div class="cuda-demo" data-cuda-demo="shared"><p>启用 JavaScript 可操作此概念演示；正文仍可独立阅读。</p></div>

```cpp
// 固定以 <<<2, 4>>> 启动，输入输出各含 8 个整数。
__shared__ int tile[4];
int t = threadIdx.x;
int i = blockIdx.x * 4 + t;
tile[t] = input[i];
__syncthreads();
output[i] = tile[(t + 1) % 4];
```

预期输出为 `[1, 2, 3, 0, 5, 6, 7, 4]`。为什么不是把整个数组左移一格？因为每个 block 只在自己的四元素 shared 数组里循环，第二个 block 的 `tile[0]` 对应输入 4。这个 kernel 的数组与启动尺寸配套，不能只改变线程数。

## 6. 同步要回答：谁在等待谁

### 6.1 kernel 启动返回，不等于计算完成

CPU 发出 kernel 启动后，可以继续向下执行；GPU 可能仍在计算。因此，“执行到了下一行 host 代码”和“已经可以读取 GPU 结果”是两件事。[CUDA Runtime API：Synchronization Behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)

把常见同步范围并列看，会更容易理解：

| 操作 | 使用位置 | 等待或约束的范围 |
| --- | --- | --- |
| `__syncthreads()` | kernel 内 | 同一 block 内参与该屏障的线程，不能当作全 grid 屏障 |
| `cudaStreamSynchronize(stream)` | host 侧 | 指定 stream 中此前提交的工作 |
| `cudaDeviceSynchronize()` | host 侧 | 当前设备上此前提交的工作；范围比单一 stream 更大 |

`__syncthreads()` 用来协调块内协作。例如，一个线程要读取另一个线程刚写入的 shared memory，就必须建立必要的先后关系。初学时把块级屏障放在整个 block 都能一致到达的位置，不要让部分仍在执行的线程绕过屏障、另一部分却等待它。[Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)

<figure class="cuda-diagram">
<a href="/images/cuda-basics/05-block-sync.svg" target="_blank" rel="noopener"><img src="/images/cuda-basics/05-block-sync.svg" alt="同一block内两个线程写入shared后在syncthreads会合，再读取对方的值" loading="lazy"></a>
<figcaption>图 5：屏障把“所有参与者已写完”放在“读取别人的值”之前。图中线程快慢只是一次可能的执行。 <a href="/images/cuda-basics/05-block-sync.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-basics/05-block-sync.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

### 小 demo：屏障到底挡住了什么

<div class="cuda-demo" data-cuda-demo="sync"><p>启用 JavaScript 可操作此概念演示；正文仍可独立阅读。</p></div>

后两种同步在 host 侧等待设备工作，不能代替 kernel 内线程间的协作屏障。[Asynchronous Execution：Explicit Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#explicit-synchronization)

### 6.2 stream：表达工作顺序

可以先把 stream 理解为有顺序的工作队列。在本文使用的常规执行方式下，同一 stream 中提交的拷贝与 kernel 按其队列顺序建立依赖关系；不同 stream 的工作是否重叠，取决于依赖与可用资源。

```text
同一 stream：输入拷贝 -> kernel -> 输出拷贝
host：提交这些工作后，在需要结果的位置等待
```

“异步”描述提交者是否需要等到工作完成；“并发”描述不同工作是否实际重叠。使用异步接口并不保证发生并发。本例没有显式创建 stream，而是使用默认 stream；默认 stream 还有 legacy 与 per-thread 等语义，学习多流协作时再展开。[Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)

<figure class="cuda-diagram">
<a href="/images/cuda-basics/06-streams.svg" target="_blank" rel="noopener"><img src="/images/cuda-basics/06-streams.svg" alt="CPU提交工作后可做其他任务，stream A按顺序执行，stream B可能重叠，CPU只等待A" loading="lazy"></a>
<figcaption>图 6：横向位置表示示意时间。A、B 独立时可以有重叠机会；如果 B 依赖 A 的输出，就需要额外建立依赖。 <a href="/images/cuda-basics/06-streams.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-basics/06-streams.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

另外，不能只凭 API 名字判断所有细节：`cudaMemcpyAsync` 在某些内存条件下仍可能阻塞 host，而普通 device-to-host `cudaMemcpy` 返回时，拷贝已经完成。具体行为要看方向、内存类型和 API 文档。[CUDA Runtime API：Synchronization Behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)

### CUDA demo：两个 stream 之间建立先后关系

[下载 stream-order.cu](/downloads/cuda-basics/stream-order.cu)。生产者在 stream A 写入 5，消费者在 stream B 将它乘以 3。这里 B 需要 A 的结果，不能只凭 host 先提交了 A 就假定顺序正确。

```cpp
produce<<<1, 1, 0, producer_stream>>>(device_value);
cudaEventRecord(produced, producer_stream);
cudaStreamWaitEvent(consumer_stream, produced, 0);
consume<<<1, 1, 0, consumer_stream>>>(device_value);
```

上面是从完整 demo 摘取的核心步骤，完整文件对 API 和 kernel 启动进行错误检查。event 标记 A 流中的生产者完成位置，B 流等待这个 event 后才运行消费者；最终在 B 流中拷回结果并等待 B。预期得到 `15`，不是并发加速的证明。[官方 stream/event 同步说明](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#explicit-synchronization)

完整 demo 的异步拷回使用 `cudaMallocHost` 分配的 pinned host memory。不要删除事件依赖后，用某次偶然得到的 15 来判断程序正确。

## 7. 用完整例子把概念连起来

令 `A[i] = i`，`B[i] = 2 * i`，所以预期 `C[i] = 3 * i`。继续使用十个元素、每块四个线程，方便对照前面的表格。

下面是完整的 `vector-add.cu`。可直接[下载 vector-add.cu](/downloads/cuda-basics/vector-add.cu)，也可以展开阅读完整代码。

<details>
<summary>展开完整向量加法代码（包含错误检查与结果校验）</summary>

```cpp
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void add_arrays(const float* a, const float* b, float* c, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    // Small launch dimensions make the thread-to-element mapping easy to trace.
    constexpr int n = 10;
    constexpr int threads_per_block = 4;
    constexpr int blocks = (n + threads_per_block - 1) / threads_per_block;
    const size_t bytes = n * sizeof(float);

    std::vector<float> a(n), b(n), c(n);
    for (int index = 0; index < n; ++index) {
        a[index] = static_cast<float>(index);
        b[index] = static_cast<float>(2 * index);
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    check_cuda(cudaMalloc(&device_a, bytes), "allocate a");
    check_cuda(cudaMalloc(&device_b, bytes), "allocate b");
    check_cuda(cudaMalloc(&device_c, bytes), "allocate c");

    check_cuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice), "copy a to GPU");
    check_cuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice), "copy b to GPU");

    add_arrays<<<blocks, threads_per_block>>>(device_a, device_b, device_c, n);
    check_cuda(cudaGetLastError(), "launch add_arrays");
    check_cuda(cudaDeviceSynchronize(), "wait for add_arrays");
    check_cuda(cudaMemcpy(c.data(), device_c, bytes, cudaMemcpyDeviceToHost), "copy c to CPU");

    bool correct = true;
    for (int index = 0; index < n; ++index) {
        const float expected = a[index] + b[index];
        if (!std::isfinite(c[index]) || std::fabs(c[index] - expected) > 1e-6f) {
            std::fprintf(stderr, "Mismatch at %d: got %g, expected %g\n", index, c[index], expected);
            correct = false;
            break;
        }
    }

    check_cuda(cudaFree(device_a), "free a");
    check_cuda(cudaFree(device_b), "free b");
    check_cuda(cudaFree(device_c), "free c");
    if (!correct) return EXIT_FAILURE;
    std::printf("PASS: %d elements; C[9] = %.0f\n", n, c[9]);
    return EXIT_SUCCESS;
}
```

</details>

### 沿执行顺序读代码

1. `main()` 准备 CPU 数组；`device_a` 等分配对应的 GPU 存储。
2. 两次 host-to-device 拷贝提供输入。
3. `add_arrays<<<blocks, threads_per_block>>>` 启动 grid；每个线程用自己的下标决定访问位置。
4. `cudaGetLastError()` 检查启动后的错误状态；它不会等待 kernel 完成。接着显式同步，以便在这里检查执行期错误。
5. 将结果拷回 CPU，逐元素比较，再释放设备分配。

启动后的错误检查与执行完成后的错误检查覆盖不同阶段。例子中额外写出 `cudaDeviceSynchronize()` 是为了让等待位置清楚；在这条默认 stream 上，后续的阻塞 device-to-host 拷贝本来也会等待相关工作后返回，因此不能把这次显式全设备等待理解为所有程序的必需写法。[Intro to CUDA C++：Error Checking](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)；[API Synchronization Behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)

### 如何运行与如何解读结果

在具有受支持的 NVIDIA GPU、匹配的驱动、CUDA Toolkit 和 host C++ 编译器的环境中，可在示例目录执行：

```bash
nvcc -std=c++17 vector-add.cu -o vector-add
./vector-add
```

实际目标架构选项需要与你的 GPU 和 Toolkit 支持范围匹配。本机没有 `nvcc`，以下是由输入与计算规则推导的**预期输出，不是本次实测日志**：

```text
PASS: 10 elements; C[9] = 27
```

`__global__` 标识这个函数是 kernel；`<<<...>>>` 是 CUDA C++ 的启动配置，括号中的参数才是传给函数的数组与长度。NVIDIA 的 [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) 也从向量加法切入，可作为读完本例后的补充材料。

例子采用遇错退出的简单教学处理，没有加入复杂的资源管理封装。四个线程一块是为了便于手算，不是性能建议；本例也没有进行任何计时。

## 8. 从“能读懂”走向“知道为什么可能慢”

第一版只建立三个性能问题的入口：

**数据访问是否集中？** 同一个 warp 中相邻线程访问相邻数据，通常更容易形成有效的合并访存。但“地址连续”并不保证任何条件下都只有一次内存事务，数据宽度与对齐等也有影响。本例按连续下标访问三个数组，便于从这里继续学习 coalescing。[CUDA Best Practices Guide：Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

<figure class="cuda-diagram">
<a href="/images/cuda-basics/07-coalescing.svg" target="_blank" rel="noopener"><img src="/images/cuda-basics/07-coalescing.svg" alt="同一warp部分线程的连续访问与跨步访问地址对比" loading="lazy"></a>
<figcaption>图 7：看一组线程合起来访问了哪些地址。下排没有画出全部间隙，格子不是实际内存事务。 <a href="/images/cuda-basics/07-coalescing.svg" target="_blank" rel="noopener">放大看图</a> · <a href="/images/cuda-basics/07-coalescing.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

**是否有值得复用的数据？** 如果多个线程反复读取同一批数据，就可以进一步研究是否适合协作加载到 shared memory；但额外的读写与同步也有成本，不能只凭“shared memory 较快”就断言整体更快。[Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)

**衡量的是哪段时间？** 应区分 kernel 执行、数据传输以及整个程序的时间。优化 GPU 计算本身，不一定能抵消频繁的 CPU/GPU 往返。本文没有性能数据，后续实践笔记再用测量判断具体瓶颈。[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

### 小 demo：改变步长，观察地址分布

把步长从 1 改成 4。有效数据仍是 8 个 float，但这些地址从连续分布变成更大的跨步范围。这能说明访存模式变化，不能单独给出 GPU 耗时或内存事务数。

<div class="cuda-demo" data-cuda-demo="address"><p>启用 JavaScript 可操作此概念演示；正文仍可独立阅读。</p></div>

## 9. 检查自己是否真正理解

先尝试回答，再展开答案：

<details>
<summary>如果 N = 1000，每块 256 个线程，需要多少个 block？</summary>

需要 4 个 block，逻辑上启动 1024 个线程。只有全局下标 0 到 999 的线程访问数组，其余 24 个线程被边界条件挡住。这里没有假设全部线程在同一时刻执行。

</details>

<details>
<summary>为什么向量加法里没有 __syncthreads()？</summary>

每个有效线程读取自己的两个输入元素，并写入独立的输出元素；没有线程要读取其他线程在本次 kernel 中产生的中间结果。CPU 获取结果前的等待属于另一个层次。

</details>

<details>
<summary>如果一个 block 把结果写入 global memory，另一个 block 就一定能立即使用吗？</summary>

不能仅凭存储位置得出这个结论。还必须解决执行先后与内存可见性；普通 block 没有按编号执行的保证。初学时可把跨块有依赖的两个阶段拆成同一 stream 中先后提交的两个 kernel。更高级的跨块协作需要明确的机制和条件。

</details>

<details>
<summary>某个变量对一个线程私有，是否说明它一定在寄存器里？</summary>

不是。访问范围与物理存放位置是两个问题；线程私有数据也可能使用 local memory。

</details>

这版笔记建立的主线是：**先划分工作，再确定每个线程访问什么，最后说明数据依赖如何得到保证。** 后续可以沿着这条线分别整理二维线程索引、shared memory 协作和 stream 三篇基础笔记。

## 继续学习

[第二篇：SM 调度、warp stall、occupancy 与 TMA 流水线](/systems/cuda-overlap/) 接着解释 SM 内部如何发出指令、为什么等待，以及合并访存、tiling、向量化和双缓冲；附完整 CUDA demo。

## 参考资料与阅读顺序

以下都来自 NVIDIA 官方渠道。顺序按本篇的学习目标安排，不要求从头读完整份手册。

| 阶段 | 资料 | 本次主要用途 |
| --- | --- | --- |
| 先有整体印象 | [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) | 用小程序认识 kernel、启动与等待；文章使用 managed memory |
| 理清概念 | [Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) | host/device、线程组织、SM 与 warp |
| 对照 C++ 代码 | [Intro to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html) | 设备存储、启动语法与错误检查 |
| 查线程与内存 | [Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html) | 内置索引与不同存储空间 |
| 理清先后关系 | [Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) | stream、异步与同步范围 |
| 精确核对 API | [API Synchronization Behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html) | 拷贝方向和内存类型对应的阻塞行为 |
| 补充内存理解 | [Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html) | managed memory 与平台差异 |
| 进阶阅读 | [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/) | 块内数据交换与同步；2013 年文章中的旧架构参数不当作当前硬件规格 |
| 对照物理访存通路 | [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-chart) | 逻辑存储空间与 L1/shared/L2/设备内存的对应 |
| 开始性能实践时 | [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | 合并访存、传输成本与测量方法 |

官方在线文档会更新。学习具体 API 或硬件行为时，应再核对自己使用的 Toolkit 版本与 GPU 架构；本篇的目的，是先建立足够清楚的概念关系，能够带着问题继续查资料。

<link rel="stylesheet" href="/css/cuda-learning.css">
<script src="/js/cuda-learning.js" defer></script>
