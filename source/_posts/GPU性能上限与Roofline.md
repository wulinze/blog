---
title: CUDA 算子学习（三）：GPU 架构与 Roofline——估算 GPU 性能上限，再用 Triton 和 TileLang 验证
description: 从字节账、运算量和时间下界理解 Roofline，结合手绘图、交互模型、完整向量算子 demo 和两种语言的练习建立性能分析习惯。
permalink: systems/gpu-roofline/
date: 2026-11-02 15:25:00
categories:
  - LLM 推理
tags:
  - CUDA
  - 基础学习
  - Roofline
  - Triton
  - TileLang
---

## 学完以后，要能独立完成什么

你已经知道 [GPU 的结构与内存层级](/systems/cuda-basics/)，也看过 [SM 调度、warp stall 和优化流程](/systems/cuda-overlap/)。现在接着解决一个具体问题：**写 kernel 之前，怎样估计这份工作至少要多长时间；测得很慢之后，怎样判断下一步该查什么？**

本文根据 `leetcuda-20261029.pdf` 的“GPU 架构、执行模型与 Roofline”主题重新组织（PDF 阅读器第 34–42 页）。保留其从机器资源走向算术强度的主线，补充流量口径、缓存条件与实测验证，使用新的例子和图。完整后续主题见 [GPU 算子学习路线](/systems/gpu-learning-path/)，博客编号接续已有的（一）基础、（二）性能分析；本篇为（三），对应原文第二章。

完成本篇后，你应能：

- 对向量加法、GEMV、GEMM 写出运算量和**声明假设后的**字节账。
- 手推 `AI`、`ridge` 和时间下界，解释为什么屋顶线是上界而不是预测实测值。
- 分清 `BLOCK` 的逻辑元素数、线程数和 occupancy。
- 跑通边界校验，分别用 Triton 与 TileLang 实现向量加法和融合 axpy。
- 交出一份“假设 → 数据 → 解释”的记录，包括没有加速时的原因。

**交付与验证范围：** 本篇有 4 张可编辑图、2 个网页交互模型、纯 Python 计算 demo、Triton／TileLang 完整参考实现、独立练习入口与统一校验/计时程序。CPU 解析模型和网页演示已验证；当前编辑环境没有 NVIDIA CUDA GPU，GPU 程序尚未完成后端编译、GPU 正确性与性能实测。参考实现是待在目标环境验证的完整源码，不附虚构测速数据。

## 先把三类术语分开

### 性能限制：模型假设与实际瓶颈

| 术语 | 在本篇指什么 | 需要什么证据 |
| --- | --- | --- |
| bandwidth-bound，带宽受限 | 某层数据通路的字节服务能力限制进度 | 该层流量、持续吞吐、时间，以及减少流量后的变化 |
| compute-bound，计算吞吐受限 | 指定精度和指令通路的计算能力限制进度 | 对应 pipe、指令混合、有效运算量和耗时 |
| latency-bound，延迟受限 | 关键依赖的等待未被其他独立工作覆盖 | eligible/issue、依赖链、任务规模；不能只看低带宽 |
| memory-bound，广义访存受限 | 数据访问影响进度的统称 | 继续区分带宽、请求效率、延迟和内存层级 |
| Roofline 下的分类 | 在所选计算峰值与带宽之间，哪条屋顶更低 | 静态模型只筛选调查方向；不能证明实现已达到相应极限 |

### 执行对象与状态

| 术语 | 含义 | 本篇容易混淆的点 |
| --- | --- | --- |
| thread / warp | CUDA 线程与由 32 个线程组成的 warp | warp 是发出相关工作的基本组织，不表示任意时刻 32 个 lane 都有效 |
| block / program | CUDA block；本文普通 Triton program 的一次 tile 工作 | program 个数不是 SM 个数；多个 program 可能排队 |
| resident warp | 执行状态和资源已分配的 warp | 包括当前因依赖等原因不可发出的 warp |
| eligible warp | 当前满足发出条件的候选 | resident 多不保证 eligible 多 |
| tile | 逻辑上划出的数据块 | 一个 tile 可由多个线程协作，一个线程可处理多个元素 |

### 量与指标：每个数字都要带单位

| 记号 | 定义与单位 | 使用边界 |
| --- | --- | --- |
| `F` | 有效浮点运算量，FLOP | 乘加按 2 FLOP；不是每秒速率 |
| `Q` | 指定口径下的字节量，byte | 必须说明逻辑读写量、请求流量或某层实测流量 |
| `AI = F/Q` | 算术强度，FLOP/byte | 随 dtype、融合方式、缓存层级和实现改变 |
| `P` | 匹配计算类型的可用吞吐上界，FLOP/s | FP32 / Tensor Core、dense / sparse 不可混用 |
| `BW` | 指定内存层与访问场景的带宽，byte/s | 官方理论峰值和实测持续带宽分别标明 |
| `ridge = P/BW` | 两条屋顶相交处的 AI，FLOP/byte | 分子分母必须属于同一设备和一致口径 |
| occupancy | active warp / 对应硬件 warp 容量 | 容量比例，不等于计算单元忙碌程度 |

下面统一使用十进制单位：`1 GB = 10^9 byte`，`1 TFLOP = 10^12 FLOP`。另说 MiB 时才使用 `2^20 byte`。先别将这些单位混进同一个除法。

## 先记字节账：同一个 out = x + y，为什么会出现不同带宽

设 x、y、out 都是长度 N 的连续 float32 向量。每个有效输出做一次加法、读取两个 4-byte 数、写一个 4-byte 数，因此：

```text
F = N
Q_logical = 4N + 4N + 4N = 12N
AI_logical = N / (12N) = 1/12 FLOP/byte
```

这张账记录的是**有用工作与逻辑读写**。它还没告诉你本次 launch 究竟从 DRAM 搬了多少。

<figure class="cuda-diagram">
<a href="/images/gpu-roofline/traffic-ledger.svg" target="_blank" rel="noopener"><img src="/images/gpu-roofline/traffic-ledger.svg" alt="逻辑读写、请求访问与L1 L2 DRAM分层流量的区别" loading="lazy"></a>
<figcaption>图：同一个计算要保留三本字节账。<a href="/images/gpu-roofline/traffic-ledger.excalidraw" download>下载可编辑原图</a></figcaption>
</figure>

沿图看一次 N=1003 的加法：逻辑账固定是 12036 byte。若地址跨距或尾部导致 sector 利用不足，请求侧可能付出更多；若输入已经在 L2，global load 仍发生，但部分读请求可以由 L2 服务。这就是右支“驻留状态”影响分层流量的含义。global 地址空间不是“每次都直达显存”的保证。

所以用 `12N / 时间` 算出的应叫**有效请求带宽**。它超过显存规格并不自动表示测速造假，也不证明 DRAM 超频；先查是否大量由缓存服务。反过来，小数据集的低带宽也可能主要受 launch 与依赖延迟影响。[NVIDIA：有效带宽与实际吞吐](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#bandwidth) 对这两种口径作了区分。

原文“每个输入读一次、每个输出写一次”的算法字节模型很适合做第一张账，但本篇不把它当作**无论缓存如何、每次 kernel 都必须付出的 DRAM 下界**。讨论 DRAM roofline 时，要加上初始数据驻留、缓存控制和传输计数范围；也不能把所有中间量一律忽略——一旦某个版本把中间张量写出，它就应记入那个版本的流量账。

## 用三个算子把 AI 算明白

### 向量加法与 axpy：多做一次乘法，不增加完整向量流量

本篇 demo 的两种语义：

```text
add:  out[i] = x[i] + y[i]
axpy: out[i] = 1.25 * x[i] + y[i]
```

axpy 每个元素是一次乘法加一次加法，按常用 FLOP 计数为 2；即使硬件用一条 FMA 指令实现，仍按 2 FLOP 计。标量 1.25 为编译期常量，不引入第三条输入向量。

```text
add:  F = N;  Q_logical = 12N;  AI = 1/12
axpy: F = 2N; Q_logical = 12N;  AI = 1/6
```

两者 AI 都很低。在同一带宽条件下，多一次算术不一定令大向量的耗时翻倍。但“可能主要看带宽”必须由足够大的任务和实际分层指标验证。

若先生成 `tmp = 1.25*x` 再运行 `out = tmp+y`，逻辑流量是第一步 8N 加第二步 12N，总共 20N。融合版本变成 12N，省下的是临时结果的一次写和一次读。`20/12≈1.67` 只是等带宽、搬运主导等假设下的比值，不是承诺的实测加速。

### GEMV：矩阵读一遍，输出每行一个数

`A` 为 M×K，`x` 为 K，输出 `y` 为 M；`y = A @ x`，其中 `@` 表示矩阵乘法。按一次乘加 2 FLOP 的常用近似计数，假定 A/x 各读一次、y 写一次，均为 float32：

```text
F = 2MK
Q_logical = 4(MK + K + M)
AI = 2MK / [4(MK + K + M)]
```

M、K 足够大时，MK 项主导，AI 接近 0.5。这个极限说明矩阵元素的复用机会少；但小 M 导致独立任务不足时，真实表现还可能落在延迟/并行度限制上。向量 x 在不同 block 间如何复用，也会影响请求及缓存流量。

### GEMM：算术强度来自复用，需要实现兑现

`A[M,K] @ B[K,N] = C[M,N]`，不读取旧 C，数据全为 float32，采用输入各读一次的理想复用模型：

```text
F = 2MNK
Q_ideal = 4(MK + KN + MN)
AI_ideal = 2MNK / [4(MK + KN + MN)]

若 M=N=K=D：
AI_ideal = 2D^3 / (12D^2) = D/6
```

D=4096 时，`AI_ideal=682.6667 FLOP/byte`。这并不意味着任何写法的 4096 方阵乘都自动 compute-bound：每个输出各自反复读取 A/B 的实现，在 global 请求层的复用差得多；缓存、分块、数据搬运和计算通路共同决定真正的位置。

手算一个小例子：`A[2,4] @ B[4,3]` 有 6 个输出，每个输出 4 次乘加，所以 F=48；读 8+12 个输入、写 6 个输出，Q=104 byte，AI=48/104≈0.4615。这有助于避免把“大 GEMM 的结论”搬到很小的矩阵上。

换成 FP16 输入但保留 FP32 输出时，字节账改成 `2(MK+KN)+4MN`，不能把三个矩阵全乘 2。若还要读取旧 C、加 bias 或写额外统计量，应继续加上对应项。计算屋顶也必须按实际生成的指令和累加路径选择。

### 为什么这里没有给 Softmax 一个通用的“固定 FLOP 数”

Softmax 还涉及 max、exp、sum、除法与同步。给 exp 规定某个“等效 FLOP 数”可以形成特定模型，但不能直接据此套 FP32 FMA 峰值，忽略特殊函数通路。这里先用加法、乘加作可核对的例子；Softmax 专题会把访存与特殊函数、归约成本分别分析。

## Roofline：从时间预算推导，而不是背一条公式

设工作有 F FLOP，需要在选定层级搬 Q byte；相应计算上界是 P FLOP/s，带宽是 BW byte/s。两个不可绕过的时间预算为：

```text
T_compute >= F / P
T_memory  >= Q / BW
```

即使计算和搬运能理想重叠，整个工作也不可能早于两者中更慢的那个结束。因此：

```text
T >= max(F/P, Q/BW)
F/T <= min(P, F*BW/Q)
     = min(P, AI*BW)

交点满足：P = AI*BW
所以 ridge = P/BW
```

取 max 的理由是**最乐观的重叠**。若某阶段必须先搬完再算，相关时间可能更接近相加；依赖、同步、资源不足和提交开销会把实测时间继续抬高。Roofline 保留的是最简单的两资源上界，不是所有等待的完整模拟。

<figure class="cuda-diagram">
<a href="/images/gpu-roofline/roofline-derivation.svg" target="_blank" rel="noopener"><img src="/images/gpu-roofline/roofline-derivation.svg" alt="从计算与搬运时间下界推导Roofline及ridge point" loading="lazy"></a>
<figcaption>图：先看两个时间预算，再推性能上界。<a href="/images/gpu-roofline/roofline-derivation.excalidraw" download>下载可编辑原图</a></figcaption>
</figure>

用一台**假想机器**来代入：匹配 FP32 计算的 P=60 TFLOP/s，选定层级 BW=1500 GB/s。这不是某款 GPU 的规格表。换成同单位后，ridge 是 `60*10^12 / (1500*10^9) = 40 FLOP/byte`。

N=1,048,576 的向量加法需要 F=1,048,576、Q=12,582,912 byte：计算预算约 0.01748 μs，搬运预算约 8.38861 μs。模型因此给出时间下界约 8.39 μs。如果实测是 20 μs，不能简单宣称剩下的 11.61 μs 都是 DRAM 延迟；需要回到采集证据。

同一设备可以画多个精度、指令类型和内存层级的屋顶。切换精度时通常同时改变 P、Q、舍入误差和指令路径，不能只换一个峰值数字；稀疏 Tensor Core 峰值也不能放进稠密算子的模型。[Nsight Compute：Roofline Charts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#roofline-charts) 提供了对应的工具入口。

## 网页实验：改变工作，不先猜结论

下面的黄点表示**模型上界**，并非测量点。图用对数轴，便于同时查看小 AI 和大 AI。修改元素宽度以后，要自己提供匹配的计算吞吐与带宽假设；网页不会自动把“FP16”理解成一定使用 Tensor Core。

<div class="gpu-lab" data-roofline-lab>
<h3>算术强度与 Roofline 计算器</h3>
<div class="gpu-lab-controls">
<label>工作类型<select name="operation"><option value="add">向量加法</option><option value="axpy">融合 axpy</option><option value="gemv">GEMV</option><option value="gemm" selected>GEMM</option></select></label>
<label>M<input name="m" type="number" min="1" step="1" value="4096"></label>
<label>N（add 的长度）<input name="n" type="number" min="1" step="1" value="4096"></label>
<label>K<input name="k" type="number" min="1" step="1" value="4096"></label>
<label>输入 byte/元素<input name="ib" type="number" min="1" step="1" value="4"></label>
<label>输出 byte/元素<input name="ob" type="number" min="1" step="1" value="4"></label>
<label>P，TFLOP/s<input name="peak" type="number" min="0.001" step="any" value="60"></label>
<label>BW，GB/s<input name="bw" type="number" min="0.001" step="any" value="1500"></label>
</div>
<output data-roofline-result aria-live="polite"></output>
<div class="gpu-roofplot" tabindex="0" role="region" aria-label="Roofline图表，窄屏可横向滚动"><svg viewBox="0 0 760 350" role="img" aria-label="模型Roofline对数坐标曲线与预测上界"></svg></div>
<p>窄屏可横向滚动图表，计算结果也完整显示在上方。</p>
<noscript>启用 JavaScript 可操作计算器；相同模型也可下载 roofline.py 在 CPU 运行。</noscript>
</div>

建议按顺序做三个实验：先切 add，观察 AI 不随 N 改变；再切 GEMM 同时增大 M/N/K，观察理想复用下 AI 上升；最后只提高 P，观察 add 的较低屋顶为什么基本不变。每次都先写预测，再动输入框。GEMV 使用 M/K，add/axpy 只使用 N。

无 GPU 也能运行同一份计算逻辑：

```bash
python roofline.py --self-check
python roofline.py --operation add --n 1048576 --peak-tflops 60 --bandwidth-gbs 1500
python roofline.py --operation gemm --m 4096 --n 4096 --k 4096
python roofline.py --operation gemm --m 2 --n 3 --k 4 --input-bytes 2 --output-bytes 4
```

预期解析值分别包含：加法 AI=1/12、方阵 AI=4096/6、最后一个混合位宽例子的 F=48/Q=64。这些都是手算/模型值，不是设备性能数据。

## 把逻辑 tile 放到执行模型上

本篇两个语言的基线都实现“一个 program/block 处理连续 BLOCK 个逻辑元素”。这里的 BLOCK 是数据分块，不一定是 CUDA block 的线程数。

<figure class="cuda-diagram">
<a href="/images/gpu-roofline/tile-mapping.svg" target="_blank" rel="noopener"><img src="/images/gpu-roofline/tile-mapping.svg" alt="N1003与BLOCK256形成四个tile，最后235个有效位置，区别逻辑元素数与线程数" loading="lazy"></a>
<figcaption>图：先算逻辑索引，再决定协作线程数。<a href="/images/gpu-roofline/tile-mapping.excalidraw" download>下载可编辑原图</a></figcaption>
</figure>

以图中 N=1003/BLOCK=256 为例：`grid=ceil(1003/256)=4`。pid=3 的起点是 768，逻辑范围本来覆盖到 1023，但只允许 768…1002，共 235 个位置。余下 21 个位置必须在 load 与 store 两端都被保护，不能只防写而让读越界。

Triton 使用 `tl.program_id(0)` 选 tile，`tl.arange(0,BLOCK)` 产生逻辑位置，`num_warps` 指定协作 warp 数。TileLang 使用 `T.Kernel(...,threads=...)` 与 `T.Parallel(BLOCK)` 分别表达 block 线程配置和 tile 内工作。编译器将逻辑位置映射到线程/寄存器，不应从 BLOCK=256 直接推出“256 个 CUDA 线程”。[Triton 向量加法教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) · [TileLang 语言基础](https://tilelang.com/programming_guides/language_basics.html)

<div class="gpu-lab" data-mapping-lab>
<h3>tile 尾部与线程配置</h3>
<div class="gpu-lab-controls">
<label>N<input name="length" type="number" min="0" step="1" value="1003"></label>
<label>BLOCK<select name="tile"><option>128</option><option selected>256</option><option>512</option><option>1024</option></select></label>
<label>Triton num_warps<select name="warps"><option selected>4</option><option>8</option></select></label>
</div>
<output aria-live="polite"></output>
</div>

### 为什么不能从配置直接算出 achieved occupancy

给出一个只用于手算的假想 SM：最多 2048 线程、64 warp、32 block、65536 个 32-bit 寄存器、64 KiB shared。一个 block 用 256 线程、每线程 64 寄存器、32 KiB shared：

```text
每 block warp 数 = ceil(256/32) = 8
线程约束：floor(2048/256) = 8 block
warp 约束：floor(64/8) = 8 block
寄存器约束：floor(65536/(256*64)) = 4 block
shared 约束：floor(65536/32768) = 2 block
硬件 block 上限：32 block

可驻留 block 的简化上界 = min(8,8,4,2,32) = 2
可驻留 warp 上界 = 2*8 = 16
理论 occupancy 上界 = 16/64 = 25%
```

实际还存在分配粒度、分区、每 block 上限等约束，应以目标设备与编译结果以及 occupancy 工具为准。即使理论容量较高，grid 太小、尾波或不均衡也会让 achieved occupancy 较低。增加 BLOCK/num_warps 可能增加每 block 消耗，也可能减少 grid 任务数；“调大”没有单调收益。完整状态区别见 [SM 调度与 occupancy](/systems/cuda-overlap/)。

## 动手前：下载、环境与文件职责

[下载完整实验包（ZIP）](/downloads/gpu-roofline/gpu-roofline-lab.zip)。解压后进入目录运行，不需要把文件放进某个大工程。

| 文件 | 用途 |
| --- | --- |
| [roofline.py](/downloads/gpu-roofline/roofline.py) | 纯 Python 模型与解析检查；无第三方依赖 |
| [reference_triton.py](/downloads/gpu-roofline/reference_triton.py) | Triton 完整基线，支持 add/axpy 和尾部 mask |
| [reference_tilelang.py](/downloads/gpu-roofline/reference_tilelang.py) | TileLang 完整基线，同样的输入输出语义 |
| [student_triton.py](/downloads/gpu-roofline/student_triton.py) | 自己实现的入口；未完成会明确抛出异常 |
| [student_tilelang.py](/downloads/gpu-roofline/student_tilelang.py) | 自己实现的入口；不会偷偷回退到参考答案 |
| [run_gpu.py](/downloads/gpu-roofline/run_gpu.py) | 全量输出校验、输入保护检查、计时、参数扫描与 JSON 记录 |
| [README.txt](/downloads/gpu-roofline/README.txt) | 环境、命令、验收与限制 |

GPU 路线面向 Linux、NVIDIA GPU 和对应 CUDA 版 PyTorch。Python 建议 3.10 或更高，GPU 与工具链需满足所选 Triton/TileLang 版本的支持范围；两种框架可以使用独立虚拟环境，分别测试，避免依赖解析时破坏已有 PyTorch 环境。不要直接在 macOS 上照抄 GPU 命令并把 import 成功当作 CUDA 已可用。

先按 [PyTorch 官方安装入口](https://pytorch.org/get-started/locally/)、[Triton 安装说明](https://triton-lang.org/main/getting-started/installation.html) 与 [TileLang 安装说明](https://tilelang.com/get_started/Installation.html) 配好目标环境，再确认：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())"
python -m pip show torch triton tilelang
nvidia-smi
```

本篇 TileLang 写法对照其官方 `@tilelang.jit`、`T.prim_func`、显式输出 buffer 接口；没有给出未经 GPU 验证的依赖锁定组合。实际运行的版本会写入 JSON，首次运行需先完成 JIT。若编译失败，先记录完整版本与报错，不能将编译失败记成性能差。

## 两种语言做同一件事：先跑基线，再独立写

### Triton：索引、mask、计算与写回

核心路径与图中四步一致：pid 选 tile → 生成 offset → 带 mask 读 → 带 mask 写。add/axpy 使用编译期选择，同一次测量内不切换语义。

```python
"""Masked float32 elementwise baselines. GPU execution has not been verified here."""

import triton
import triton.language as tl


@triton.jit
def elementwise(X, Y, Out, N: tl.constexpr, BLOCK: tl.constexpr, AXPY: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    x = tl.load(X + offsets, mask=valid, other=0.0)
    y = tl.load(Y + offsets, mask=valid, other=0.0)
    if AXPY:
        value = 1.25 * x + y
    else:
        value = x + y
    tl.store(Out + offsets, value, mask=valid)


def prepare(n: int, block: int, workers: int, operation: str):
    """Create a launcher; workers means warps per Triton program."""

    def launch(x, y, out):
        if n:
            elementwise[(triton.cdiv(n, block),)](x, y, out, n, block, operation == "axpy", num_warps=workers)

    return launch
```

### TileLang：tile 范围与物理线程数分别给出

`T.Parallel(block)` 描述逻辑工作范围，`threads` 控制参与的线程数。显式 `if offset < n` 使边界意图清楚；不依赖读者猜测编译器是否自动插入保护。输出由运行器预先分配，不把分配开销混入 kernel 对比。

```python
"""Explicit boundary guards; workers means CUDA threads per TileLang block."""

import tilelang
import tilelang.language as T


@tilelang.jit
def make_elementwise(n: int, block: int, threads: int, axpy: bool):
    @T.prim_func
    def elementwise(X: T.Tensor((n,), "float32"), Y: T.Tensor((n,), "float32"), Out: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, block), threads=threads) as pid:
            for lane in T.Parallel(block):
                offset = pid * block + lane
                if offset < n:
                    if axpy:
                        Out[offset] = 1.25 * X[offset] + Y[offset]
                    else:
                        Out[offset] = X[offset] + Y[offset]

    return elementwise


def prepare(n: int, block: int, workers: int, operation: str):
    """Create an explicit-output callable, avoiding output allocation in timing."""
    if n == 0:
        return lambda x, y, out: None
    return make_elementwise(n, block, workers, operation == "axpy")
```

两者都从 host 单独处理 n=0，非空输入只支持本练习约定的一维、连续、float32 数组。不同 stride、dtype、重叠输入输出是之后的扩展题，不在本次验收接口内。

先分别运行参考版：

```bash
python run_gpu.py --backend triton --mode check --operation add --n 1003
python run_gpu.py --backend triton --mode check --operation axpy --n 1003
python run_gpu.py --backend tilelang --mode check --operation add --n 1003
python run_gpu.py --backend tilelang --mode check --operation axpy --n 1003
```

校验器不仅测试你传入的 N，还测试 0、1、31/32/33、127/128/129、BLOCK−1/BLOCK/BLOCK+1。先把输出填成 NaN，再比较**全部有效元素**，并确认输入没有被修改。容差为 `rtol=1e-5, atol=1e-6`；允许融合 FMA 与参考算式的舍入差异，不接受漏写尾部。

## 练习：有接口、有提示，也有可以执行的验收

### Triton 练习：从完整覆盖到资源实验

**基础任务 T1。** 在 `student_triton.py` 内实现 `prepare(n, block, workers, operation)`，返回 `launch(x,y,out)`。先支持 add；workers 作为 num_warps。不得调用 PyTorch 完成输出，也不要导入参考 kernel 代替自己的实现。

提示依次是：`pid*BLOCK` 选起点；`tl.arange` 产生逻辑偏移；load/store 都使用 `offset<N`。不要先针对 N=1024 写一个永远整除的版本再忘记尾部。

**融合任务 T2。** 增加 axpy，输出 `1.25*x+y`，保持相同输入输出 buffer；不生成 global 临时张量。先推导 12N 与拆分版本 20N 的字节账，再看实际差异。runner 的 torch axpy 对照也已融合，因此不能把它误称为“拆分版本”。

**验证任务 T3。** 扫 BLOCK=128/256/512/1024，workers=4/8。固定其他条件，对比小 N 与较大 N。提交每种配置的正确性、耗时、有效请求带宽和解释；不要求最大 BLOCK 获胜，也不要求测出固定倍数。

```bash
python run_gpu.py --backend triton --impl student --mode check --operation add --sweep --n 1003
python run_gpu.py --backend triton --impl student --mode check --operation axpy --sweep --n 1003
python run_gpu.py --backend triton --impl student --mode bench --operation add --sweep --n 1048576 --output triton-add.json
```

### TileLang 练习：把逻辑 tile 与线程配置分开

**基础任务 L1。** 在 `student_tilelang.py` 实现同一个接口。workers 表示 threads，block 表示 T.Parallel 的范围。先使用 block=256/threads=128，再处理所有尾部；n=0 返回不做 device launch 的 callable。

**融合任务 L2。** 加入 axpy，输出由外部传入。用生成代码检查是否存在额外全局临时存储，并观察线程覆盖；不要求某条向量指令一定出现，源码形式不能替代编译结果。

**验证任务 L3。** 独立扫描 block=128/256/512/1024 与 threads=128/256。把“多逻辑元素/线程”和“更多协作线程”分别作为假设，比较 kernel 时间及编译资源；不要同时改变 dtype 和问题规模。

```bash
python run_gpu.py --backend tilelang --impl student --mode check --operation add --sweep --n 1003
python run_gpu.py --backend tilelang --impl student --mode check --operation axpy --sweep --n 1003
python run_gpu.py --backend tilelang --impl student --mode bench --operation add --sweep --n 1048576 --output tilelang-add.json
python run_gpu.py --backend tilelang --mode profile --n 1003 --dump-source generated.cu
```

`--dump-source` 用于返回对象具有 `get_kernel_source()` 的 TileLang 实现，默认参考版符合该接口；学生版若包了一层普通 Python 函数，需要保留/暴露相同接口才可导出。获取源码的 API 对照 [TileLang JIT](https://tilelang.com/autoapi/tilelang/jit/index.html)。

### 两种语言共同的最终验收

你需要提交三件东西：通过边界校验的自己的实现；一张带环境信息的实验表；一段能被证据支持或推翻的解释。两种实现应使用相同 shape/dtype/数学表达式比较，workers 的数值则按各自语义记录。

| 必交项目 | 通过条件 |
| --- | --- |
| 数学与字节账 | 能手推 add/axpy 的 F、Q、AI；声明逻辑与 DRAM 的区别 |
| 索引与边界 | 输出全覆盖，空输入不发起 kernel，尾部读写均合法 |
| 正确性 | 所有校验尺寸通过；输入保持不变；异常不能被吞掉 |
| 计时 | JIT/分配在外，预热和重复数明确；分别报告小/大工作集 |
| 性能解释 | 说明可能的限制与反证；不以高 occupancy 或单项 stall 直接定性 |

## 测量流程：先能复现，再讨论快慢

<figure class="cuda-diagram">
<a href="/images/gpu-roofline/experiment-loop.svg" target="_blank" rel="noopener"><img src="/images/gpu-roofline/experiment-loop.svg" alt="固定条件、手算、边界验证、计时、分层采集与单因素修改的实验闭环" loading="lazy"></a>
<figcaption>图：沿着同一个向量算子走完验证闭环。<a href="/images/gpu-roofline/experiment-loop.excalidraw" download>下载可编辑原图</a></figcaption>
</figure>

图里的绿色校验是进入计时的门槛。运行器先编译、检查每份 buffer，再 warmup 10 次；计时取 7 组，每组 50 次 launch 的 CUDA event 总时间除以 50，报告组均值的中位数与最小/最大值。不是 350 个独立 launch 时间的中位数。

计时区间排除了 JIT、随机数、分配和正确性比较，但 CPU 提交跟不上时，event 区间仍可能包含 GPU 上的提交空档。因此非常小的 kernel 要结合 nsys 看时间线；不能把这套批量 event 方法叫作纯硬件指令延迟。缓存不会被显式清空，默认为同一份工作集重复使用。

```bash
# 同语义 PyTorch 基线，输出也预先分配。
python run_gpu.py --backend torch --mode bench --operation add --n 1048576 --output torch-add.json

# 一个重复使用的工作集。
python run_gpu.py --backend triton --mode bench --n 1048576 --output add-hot.json

# 更大工作集 + 轮换 buffer；按显存容量调整 N/buffers。
python run_gpu.py --backend triton --mode bench --n 16777216 --buffers 4 --output add-rotating.json
```

第二个配置的三份数组逻辑 footprint 是 12 MiB；最后一个配置四套数组合计 768 MiB，还需为校验临时张量保留空间。轮换工作集可能降低缓存复用，但**容量超过 L2 并不等于完全禁用缓存**，仍应测 L2/DRAM 流量。别把所有 N 都统一要求达到理论显存带宽。

需要细查时，先用 nsys 判断提交与 kernel 时间占比，再选目标 kernel 做 ncu：

```bash
nsys profile --trace=cuda,nvtx --output=vector-timeline \
  python run_gpu.py --backend triton --mode bench --n 1048576

ncu --list-sections
ncu --query-metrics
ncu --kernel-name regex:elementwise --launch-skip 1 --launch-count 1 \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section SchedulerStats --section MemoryWorkloadAnalysis \
  --output=vector-report \
  python run_gpu.py --backend triton --mode profile --n 1048576
```

最后一条针对默认 `--buffers 1`：目标 kernel 在正确性校验中启动一次，profile 模式随后再启动一次；skip=1 是跳过第一轮**匹配的目标启动**。修改后端、buffer 数或启动流程以后，先核对实际 kernel 名与筛选结果，再调整 skip。ncu replay/缓存策略可能改变工作条件，其计时不与独立 benchmark 直接混为一组。[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/) 与 [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) 说明了这些采集机制。

观察结果时走这几条分支：

- **有效带宽很高甚至超过 DRAM 规格**：先核对 byte 与时间单位，再查缓存和真实 DRAM 字节，不能立刻宣称“突破硬件上限”。
- **带宽低，grid 很小**：先增加独立工作量看趋势，检查提交与尾波；不要先加 shared stage。
- **更多 warp 反而慢**：检查每 block 资源、grid 数、指令与调度；逻辑工作量没变不代表执行开销没变。
- **融合后时间变化小**：核对被比较的基线是否本来已融合，再看当前瓶颈。减少理论流量不保证消除主限制。

## 学习记录与接下来的方向

```text
对象：GPU、驱动、框架版本、dtype、N、BLOCK、workers、buffers
语义：add 或 axpy；输出是否预分配；允许误差
预测：F / Q 的口径，选择的 P / BW，预计哪条屋顶更低
正确性：测试尺寸、最大误差、输入是否改变
测量：计时设置、耗时统计、工作集、profiler 设置
解释：哪些证据支持预测，哪些证据与预测不符
下一轮：只修改一个主要因素，写出预期与反证条件
```

下一篇：[（四）归约与点积：从 warp 到全局结果](/llm/cuda-reduction/)。它对应原文下一章，继续学习 warp 内交换、block 内合并、跨 block 归约与浮点求和误差。

<link rel="stylesheet" href="/css/cuda-learning.css">
<link rel="stylesheet" href="/css/gpu-roofline.css">
<script defer src="/js/gpu-roofline.js"></script>
