---
title: CUDA 算子学习（四）：归约与点积——从 warp 到全局结果
description: 对应 LeetCUDA 归约原语一章，用交互逐轮解释 shuffle、block 汇合、跨 block 归约与浮点误差，并提供 CUDA、Triton、TileLang 完整 demo 和练习。
permalink: llm/cuda-reduction/
date: 2026-11-02 16:20:00
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - 归约
  - Triton
  - TileLang
---

## 本篇在系列中的位置

[LLM 算子学习](/llm/) · [完整编号目录](/systems/gpu-learning-path/) · 上一篇：[（三）GPU 架构与 Roofline](/systems/gpu-roofline/)

这是系列的第四篇，对应 `leetcuda-20261029.pdf` 的“归约原语：Warp/Block Reduce 与 Dot”一章（PDF 阅读器第 43–51 页）。保持一章一篇、按原文顺序继续；本篇补充可独立操作的实验、完整实现，以及部分原文表述的适用条件。

向量加法中，线程只写自己负责的输出。归约则是**很多线程共同产生更少的结果**：sum/max、点积、Softmax 的分母、RMSNorm 的平方和都需要它。难点不只有加法，而是：结果由谁持有、谁可以读谁的数据、何时可以读取、浮点运算顺序改变后怎样验证。

本篇学习目标：能从输入手算每层 partial；解释 shuffle 与 shared 同步；实现不会重复写全局结果的 block reduce；跑通多层 sum/max/dot；比较原子与分层方案的完整耗时。

**材料状态：** 4 张可编辑图、3 个网页交互、CPU 模型、完整 CUDA 和 Triton／TileLang demo、独立练习入口。CPU 模型与网页演示已执行验证；当前编辑环境没有 NVIDIA CUDA GPU，GPU 代码尚未经过后端编译、设备正确性及性能实测。下面的运行结果示例是可手算的预期值，没有伪造 GPU 速度。

## 先定义计算、参与者和结果归属

### 数学契约与单位元

本练习输入 x/y 为连续一维 float32，长度 N≥0，非空输入元素为有限数。输出是一个 float32 标量。NaN/±Inf 输入策略和非连续 stride 留到扩展题，不能在没有定义的情况下宣称都支持。

| 算子 | 数学定义 | 无效位置填什么 | 空输入约定 |
| --- | --- | --- | --- |
| sum | `s = Σ x[i]` | 0；与任何数相加不改变它 | 0 |
| max | `m = max(x[i])` | `-inf`；不会压过有限有效值 | `-inf`，这是本 demo 的约定 |
| dot | `d = Σ x[i]*y[i]` | 先把无效乘积设为 0 | 0 |

单位元的作用是让尾部仍可参加同一个固定形状的归约。比如全负数 `[-7,-3,-9]` 的 max 应为 -3，尾部若填 0 就变成了错误的 0。`-FLT_MAX` 对“只允许有限 float”可作某些实现的哨兵；若接口允许真正的 `-inf`，则必须重新审视它是否仍符合约定。本篇直接用 `-inf`。

### 参与者、通信范围与归约结果

| 名称 | 范围与职责 | 容易混淆的点 |
| --- | --- | --- |
| lane | warp 内的位置，CUDA warp 通常有 lane 0…31 | lane 0 在每个 warp 中都有，不是整个 block 唯一线程 |
| `tid` | block 内线程编号；本 CUDA demo 是 threadIdx.x | `tid==0` 才是整个一维 block 唯一写出者 |
| warp reduce | warp 内多值汇合 | 可以只让一个 lane 持有最终结果 |
| warp all-reduce | warp 内所有参与 lane 都取得同一结果 | 本篇指单 GPU 的 warp 集合操作，不是 NCCL 跨 GPU 通信 |
| block reduce | 多个 warp 的结果进一步汇合 | shuffle 本身不能跨 warp |
| partial | 某一分块或阶段的中间结果 | dot 的 partial 已是乘积之和，后续不能再乘 y |
| collective | 一组线程按约定协作的操作 | 输入无效与线程不参与是两件事 |

### 性能与数值观测分开记录

本篇关注三类等待：读取数据及结果依赖、block 内汇合、同地址原子争用。低 AI 只是访存方向的初步假设，不能忽略小任务的启动开销和最后一层只剩很少任务的情况。更多分类见 [（二）性能分析](/systems/cuda-overlap/)。

指标分别记录：完整归约耗时（包括所有层）、各层 kernel 数量与时间、global 字节/atomic 请求、资源与 issue 情况，以及数值绝对误差和按贡献量缩放的误差。不能拿第一层的带宽代替整个标量结果的完成时间。

## 原理：把长依赖链变成浅树

单个结果的求和最少需要 N−1 次有用加法。串行累加把它们接成一条链；平衡二叉树将同一层的独立加法并行执行，理想依赖深度约为 `ceil(log2 N)`。

```text
输入：1 2 3 4 5 6 7 8
第 1 层：(1+2) (3+4) (5+6) (7+8) = 3 7 11 15
第 2 层：(3+7) (11+15)             = 10 26
第 3 层：(10+26)                  = 36

有用加法数：4+2+1 = 7
树深度：3
```

这里的“层”不是 GPU 时钟周期。真实执行还要付出数据交换、访存、同步、指令发出等成本；每线程先累加多个元素，也会形成局部依赖链。另外，若目标是所有 lane 都得到结果，复制结果的过程可能引入更多运算，不应把单结果树的 N−1 计数直接套到 XOR all-reduce。

<figure class="cuda-diagram">
<a href="/images/gpu-reduction/reduction-tree.svg" target="_blank" rel="noopener"><img src="/images/gpu-reduction/reduction-tree.svg" alt="8个数串行7次依赖与三层归约树的对照" loading="lazy"></a>
<figcaption>图：减少的是依赖深度；实际成本还要看数据如何交换。<a href="/images/gpu-reduction/reduction-tree.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

图中第一层四组加法彼此独立，第二层要等自己的两个输入，而不需要等待不相关的分支。将这张依赖图映射到 warp、shared 与 kernel 边界，就是本篇接下来的三层实现。

## Warp 内：down 与 XOR 分别保证什么

shuffle 让线程取得同 warp 内另一个 lane 的寄存器值。一次 shuffle 只完成一次数据交换，不会自动把 32 个值全部归约。完整 32-lane 的常见树形序列有 16、8、4、2、1 五轮。

**down 的目标可以是 reduce-to-one。** lane i 从 i+offset 取得值并累加，最后只消费 lane 0 的全和。其他 lane 的最终数不需要是全和，也不应该被当作独立输出。它是有效的归约方案，不能因为别的 lane 没有全和就称整个算法错误。

**XOR 可构造 all-reduce。** 每轮 lane i 与 `i XOR mask` 交换旧值；两者拥有的不同子集合并。W=8 时 mask 从 4 到 2 到 1，三轮后所有 lane 都持全和。第一次是同低两位的一对，第二次扩展为同最低位的四个元素，第三次合并全部八个。

```text
以 lane 0 为例，初值 x0 = 1：
mask 4：从旧 lane 4 取 x4，拥有 {0,4}，值 1+5=6
mask 2：从旧 lane 2 取 {2,6}，拥有 {0,2,4,6}，值 16
mask 1：从旧 lane 1 取 {1,3,5,7}，拥有全部，值 36
```

每轮必须读取**上一轮快照**。在 CPU 上从 lane 0 到 lane 7 依次原地更新同一个数组，会让后面的 lane 读到本轮的新值，已经不是同一个数学过程。

<figure class="cuda-diagram">
<a href="/images/gpu-reduction/result-ownership.svg" target="_blank" rel="noopener"><img src="/images/gpu-reduction/result-ownership.svg" alt="down只消费lane0全和与xor所有lane都取得全和的归属对照" loading="lazy"></a>
<figcaption>图：先定义谁需要最终结果，再选择交换模式。<a href="/images/gpu-reduction/result-ownership.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

沿图看 `[1…8]`：down 终态 lane 0 是 36；XOR 终态八个 lane 都是 36。两条路径满足不同输出契约，性能要结合上下文测量，不能把“XOR 每个 lane 都有用”推成“XOR 必然比 down 快”。[NVIDIA warp 归约示例](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/) 同时介绍了这两种用途。

### 交互：逐轮看谁拿到了哪些值

下面用 8 lane 便于手算；真实 CUDA demo 使用完整的 32-lane warp。数值面板演示算术关系，不模拟硬件时间，也不保证与任意 GPU 指令序列逐 bit 一致。

<div class="gpu-lab" data-reduction="warp">
<h3>warp 交换与结果归属</h3>
<div class="gpu-lab-controls">
<label>8 个输入（英文逗号）<input value="1,2,3,4,5,6,7,8" aria-label="八个lane初值"></label>
<label>模式<select><option value="xor">XOR：所有 lane 得到结果</option><option value="down">down：只消费 lane 0</option></select></label>
</div>
<div class="reduce-lanes" data-lanes></div>
<div class="reduce-actions"><button type="button" data-next>下一轮交换</button><button type="button" data-reset>重置</button></div>
<output aria-live="polite"></output>
</div>

先预测 XOR 的 lane 0 三轮应是 6、16、36，再点击。切到 down 并比较终态，判断哪些位置属于该函数保证的输出。这里使用的“越过逻辑 width 后保留本 lane 值”的 down 行为，不应混同“读取一个未参加的源 lane”。

### mask、width 与尾部的正确性条件

shuffle 的参与 mask 用来声明协作线程，width 约束逻辑子组；它们不是将缺失线程自动补零的开关。被读取的源线程必须符合该指令的参与语义。不要让部分线程先 return，再在剩余线程上照抄 full mask。

本 CUDA demo 固定 256 个线程，每个 warp 都完整。输入尾部只影响线程加载的值：越界位置填单位元，线程仍到达 shuffle 和 block 屏障。随后只有 warp 0 做第二次归约时，该 warp 的 32 个线程仍全部参加。这是一种容易检查的安全前提，不是任意稀疏 mask 的通用算法。shuffle 也不提供 shared memory 所需要的内存同步语义。[NVIDIA：Warp-level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)

## Block 内：shared 交换的是 warp 的部分结果

256 个线程形成 8 个 warp。每个 warp 先归约，lane 0 把结果写进 `warp_totals[warp_id]`。然后必须让整个 block 到达 `__syncthreads()`，保证 warp 0 读取时，其他 warp 的结果已写好。

下一步，warp 0 的前 8 个 lane 各读一个 partial，另外 24 个 lane 填单位元，再做一次完整 warp reduce。只有 `tid==0` 写出该 block 的最终结果。

```text
每个 warp 的结果 → shared[0…7]
                         ↓ 全 block 屏障
warp 0 的输入：[p0 p1 p2 p3 p4 p5 p6 p7 0 … 0]
                         ↓ warp reduce
tid 0 写 partial[block_id]
```

这个 demo 只需一个线程写出，不要求全 block 都持有结果。若后续是 Softmax 除法，每个线程都需要分母，就应增加**明确的 block 广播过程**：结果写 shared，再经过全 block 同步后读取；或采用经过证明的每 warp 冗余归约方案。`__shfl_sync(...,0)` 只能在每个 warp 内广播，不能单独把 warp 0 的结果直接传给其他 warp。

本 CUDA 版本第二次仍用 32-lane reduce，因此是“5 轮 warp 交换 + shared/屏障 + 5 轮 warp 交换”的结构，而不是固定的 9 个时钟周期。原文采用子组和冗余归约时的步骤统计也不能直接当作硬件周期。

<figure class="cuda-diagram">
<a href="/images/gpu-reduction/reduction-hierarchy.svg" target="_blank" rel="noopener"><img src="/images/gpu-reduction/reduction-hierarchy.svg" alt="线程局部值经warp归约shared汇合再到global partial和后续kernel" loading="lazy"></a>
<figcaption>图：每层都说明结果在哪里，以及下一层什么时候能读。<a href="/images/gpu-reduction/reduction-hierarchy.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

图里的绿色框是 block 内同步，橙色框之间是 kernel 顺序。二者范围不同：普通 `__syncthreads()` 不会等待整个 grid。也不能让 block 0 忙等其他 block 的标志，假定所有 block 已经同时驻留；这种写法可能阻塞尚未调度的生产者。

## 跨 block：先做 partial，再合并

设每个 program 处理 B 个元素，第一层输出个数 `G=ceil(N/B)`。给每个 block 一个独立的 partial 元素，避免多个 block 同写一个普通地址。下一次 kernel 在同一 stream 中读取这份 partial，再次归约；直到只剩一个元素。

```text
N = 10，B = 4，x = [1,2,3,4,5,6,7,8,9,10]
第一层：sum(1…4)=10，sum(5…8)=26，sum(9,10,0,0)=19
partial = [10,26,19]
第二层：sum(10,26,19,0)=55
```

“两个阶段”是最常见的结构名称，但 G 仍很大时会继续分层。本练习代码支持多层，不会假定第二个 block 永远装得下所有 partial。比如 B=256、N=65537：第一层 257 个值，第二层 2 个值，第三层才得到 1 个值。

<div class="gpu-lab" data-reduction="hierarchy">
<h3>逐层提交归约</h3>
<div class="gpu-lab-controls">
<label>N（输入为 1…N）<input type="number" min="0" max="64" step="1" value="10"></label>
<label>教学 tile 大小<select><option>2</option><option selected>4</option><option>8</option></select></label>
</div>
<div data-levels></div>
<div class="reduce-actions"><button type="button" data-next>提交下一层</button><button type="button" data-reset>重置</button></div>
<output aria-live="polite"></output>
</div>

先用 N=10/B=4 核对图中的 `[10,26,19]→55`，再用 N=17/B=4 看为什么需要三层。交互为展示各层输入输出而缩小了 tile；真实 demo 的 block 选择为 128/256/512/1024。

### Dot：只在第一层做乘法

点积先生成每个位置的贡献 `x[i]*y[i]`，后面的归约结构与 sum 相同。例如 x=[1,2,3]、y=[4,5,6]，先得到 [4,10,18]，归约结果 32。

最容易写错的是复用同一个 dot kernel 处理 partial：partial 已经是某段的乘积和，第二层必须做 sum，不能再乘一次 y。两个语言参考版都把第一层的 OP 与后续层明确区分。

### 与 Roofline 连起来：不要漏掉 partial 的读写

按一次理想读取/写出计，sum 的输入约 4N byte，dot 约 8N byte。如果第一层产生 G 个 float、第二层全部读回并写出一个 float，那么中间流量还有 `4G+4G+4`：

```text
sum 两层的逻辑字节：4N + 8G + 4
dot 两层的逻辑字节：8N + 8G + 4
```

有更多层就继续加入该层 partial 的写与读；缓存会改变真实 DRAM 流量。sum 的单结果最少有 N−1 次有用加法，dot 另加 N 次乘法，通常简记约 2N FLOP；padding 和具体归约模式可能执行额外操作。因此 dot 的大 N 算术强度约为 0.25 FLOP/byte，但实际耗时还受归约依赖、启动、同步或原子争用影响。

## 原子累加：简单，但要比较完整流程

另一种做法是先将 out 清零，每个 block 得到一个 partial 后，仅让 `tid==0` 执行一次 `atomicAdd(out,partial)`。原子操作使并发更新不会相互覆盖；不意味着浮点结果严格等于实数求和，也不保证更新顺序固定。

<figure class="cuda-diagram">
<a href="/images/gpu-reduction/global-merge.svg" target="_blank" rel="noopener"><img src="/images/gpu-reduction/global-merge.svg" alt="每轮清零后原子累加与多层partial归约的成本和顺序对照" loading="lazy"></a>
<figcaption>图：原子方案的清零与分层方案的后续 kernel 都属于工作成本。<a href="/images/gpu-reduction/global-merge.excalidraw" download>下载 Excalidraw 原图</a></figcaption>
</figure>

图左边必须**每次调用都清零**，不能只在 benchmark 开始前清一次，否则第 2、3 次运行会把上一次结果继续加进去。图右边每个 partial 都由唯一生产者覆盖写，不需要为了求和先清空，但要保证所有有效 partial 都被写到。

原子方案省去了独立合并 kernel，却增加同地址争用与初始化；分层方案增加 partial 流量与启动。N、block 数和设备不同，优劣可能变化。本 CUDA demo 支持两条路径；Triton/TileLang 基线先使用分层方案，原子版作为完成基础后再增加的实验，避免把两种机制一次混在一起。

## 浮点误差：树更浅不等于任何输入都更准确

浮点加法不满足实数的结合律。以 `A=2^25` 的 binary32 输入 `[A,1,-A,1]` 为例，每次加法都舍入到 FP32：

```text
数学精确和：2
从左往右累加：1
相邻成对树：(A+1) + (-A+1) → A + (-A) → 0
先抵消大数：(A-A) + (1+1) → 2
```

这个例子说明顺序影响误差；不能把某一个顺序的正确结果推广成所有输入的定理。固定分层树减少了原子更新完成顺序带来的变化，但编译器、设备、归约布局或配置变化仍可能改变求和顺序，不能承诺跨环境 bitwise 一致。

<div class="gpu-lab" data-reduction="rounding">
<h3>用 FP32 舍入观察相消</h3>
<label>大数的指数（24…30）<input type="range" min="24" max="30" step="1" value="25" aria-label="大数指数"></label>
<output aria-live="polite"></output>
</div>

校验使用 FP64 CPU 参考，同时报告绝对误差与 `abs(error)/sum(abs(contribution))`。当正确结果接近 0，单独使用相对结果误差会失去意义。练习包采用 `1e-6 + 2e-6*sum_abs` 的保守教学容差，并打印实际误差；它不是通用的数值稳定性证明，生产算法应按精度要求设计更全面的输入分布和误差标准。max 对本篇有限输入则要求精确匹配。

## 下载与环境：先运行现成 demo

[下载完整归约实验包](/downloads/gpu-reduction/gpu-reduction-lab.zip)。解压后进入目录。纯 CPU 模型只依赖 Python 标准库；GPU 代码需要 NVIDIA CUDA 环境，Triton/TileLang 需安装对应框架与 CUDA PyTorch，准备方式见 [上一篇环境说明](/systems/gpu-roofline/)。

| 文件 | 用途 |
| --- | --- |
| [reduction_model.py](/downloads/gpu-reduction/reduction_model.py) | CPU 上打印 XOR 轮次、partial 层级及 FP32 相消例子 |
| [reduction.cu](/downloads/gpu-reduction/reduction.cu) | CUDA warp/block/global、sum/max/dot、受保护 float4、原子对照与校验 |
| [reference_triton.py](/downloads/gpu-reduction/reference_triton.py) | `tl.sum` / `tl.max` 与递归 partial 合并 |
| [reference_tilelang.py](/downloads/gpu-reduction/reference_tilelang.py) | fragment 归约与相同的多层接口 |
| [run_gpu.py](/downloads/gpu-reduction/run_gpu.py) | 多尺寸、多分布校验，完整流程计时与 JSON 记录 |
| [student_triton.py](/downloads/gpu-reduction/student_triton.py)、[student_tilelang.py](/downloads/gpu-reduction/student_tilelang.py) | 独立练习入口；未实现时明确报错 |
| [README.txt](/downloads/gpu-reduction/README.txt) | 命令、契约、验收与验证状态 |

```bash
# 无 GPU 也可先观察完整中间结果。
python reduction_model.py
python reduction_model.py --self-check

# CUDA：在装有匹配 Toolkit 的 NVIDIA GPU 环境编译。
# -arch=native 需要支持该选项的 nvcc；否则改成目标设备匹配的 sm_XX。
nvcc -O3 -std=c++17 -lineinfo -arch=native reduction.cu -o reduction
./reduction 1003 sum two-pass
./reduction 1003 max two-pass
./reduction 1048576 dot two-pass
./reduction 1048576 dot atomic
./reduction 520 dot-vec4 two-pass
compute-sanitizer --tool memcheck ./reduction 520 dot-vec4 two-pass
```

CUDA 程序先检查 0、1、31/32/33、127/128/129、255/256/257、520、65537，再运行指定 N。max 用全负数数据验证单位元，sum/dot 对照 FP64。原子版本每次调用都重新初始化；max 未提供原子版，会明确拒绝该组合。例子 `dot-vec4` 的向量读只在 `index+3<N` 时发生，尾部逐元素受保护，避免“先越界 float4 读取，再用条件把乘积清零”的错误。

### CUDA 核心：图中的三层如何落到代码

```cpp
// 完整源码中的关键顺序，THREADS 固定为 256。
value = warp_reduce<op>(value);
if (lane == 0) warp_totals[warp] = value;
__syncthreads();
if (warp == 0) {
    value = lane < THREADS / 32 ? warp_totals[lane] : neutral<op>();
    value = warp_reduce<op>(value);
    if (tid == 0) output[blockIdx.x] = value;
}
```

第一行是寄存器交换；shared 写与同步对应图中的绿色汇合；最后一次写入对应橙色 partial。完整可编译程序包含输入保护、host 分配、后续层调度、错误检查与计时，不要求读者自行补这些外壳。

<details class="gpu-source"><summary>展开完整 CUDA 源码：reduction.cu</summary>

```cpp
// Complete CUDA teaching baseline. Backend compilation/GPU execution not performed by the author.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA(call) do { cudaError_t err = (call); if (err != cudaSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(err)); } while (0)
constexpr int THREADS = 256;  // Multiple of 32; every warp participates in its collective.
enum Op { SUM, MAXIMUM, DOT };

template<Op op> __device__ float combine(float a, float b) {
    if constexpr (op == MAXIMUM) return fmaxf(a, b);
    return a + b;
}
template<Op op> __device__ float neutral() {
    if constexpr (op == MAXIMUM) return -CUDART_INF_F;
    return 0.0f;
}
template<Op op> __device__ float warp_reduce(float value) {
    for (int offset = 16; offset; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffffu, value, offset);
        value = combine<op>(value, other);
    }
    return value;  // Only lane 0 is consumed as the warp's complete reduction.
}

template<Op op, bool vec4 = false, bool atomic = false>
__global__ void reduce_stage(const float* x, const float* y, float* output, int n) {
    __shared__ float warp_totals[THREADS / 32];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int index = (blockIdx.x * THREADS + tid) * (vec4 ? 4 : 1);
    float value = neutral<op>();
    if constexpr (vec4) {
        // Valid only for DOT. Base pointers from cudaMalloc; index is a multiple of 4.
        if (index + 3 < n) {
            float4 a = *reinterpret_cast<const float4*>(x + index);
            float4 b = *reinterpret_cast<const float4*>(y + index);
            value = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
        } else {
            for (int j = 0; j < 4; ++j)
                if (index + j < n) value += x[index+j] * y[index+j];
        }
    } else if (index < n) {
        if constexpr (op == DOT) value = x[index] * y[index];
        else value = x[index];
    }
    // Do not return early for an input tail: every launched warp is full.
    value = warp_reduce<op>(value);
    if (lane == 0) warp_totals[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < THREADS / 32 ? warp_totals[lane] : neutral<op>();
        value = warp_reduce<op>(value);
        if (tid == 0) {
            if constexpr (atomic) atomicAdd(output, value);
            else output[blockIdx.x] = value;
        }
    }
}

struct Buffers {
    float *x=nullptr, *y=nullptr, *a=nullptr, *b=nullptr, *out=nullptr;
    explicit Buffers(int n) {
        auto alloc=[](float** p, size_t count) { CUDA(cudaMalloc(p, std::max(size_t(1), count)*sizeof(float))); };
        alloc(&x,n); alloc(&y,n); alloc(&a,(n+THREADS-1)/THREADS); alloc(&b,(n+THREADS-1)/THREADS); alloc(&out,1);
    }
    ~Buffers() { cudaFree(x); cudaFree(y); cudaFree(a); cudaFree(b); cudaFree(out); }
};

void launch(Buffers& b, int n, Op op, bool vec4, bool atomic) {
    if (n == 0) {
        float identity = op == MAXIMUM ? -std::numeric_limits<float>::infinity() : 0.0f;
        CUDA(cudaMemcpy(b.out,&identity,sizeof(float),cudaMemcpyHostToDevice));
        return;
    }
    int groups = (n + THREADS*(vec4?4:1)-1)/(THREADS*(vec4?4:1));
    float* dst = atomic || groups==1 ? b.out : b.a;
    if (atomic) CUDA(cudaMemsetAsync(b.out,0,sizeof(float))); // Included in timed workflow.
    if (op == MAXIMUM) reduce_stage<MAXIMUM><<<groups,THREADS>>>(b.x,b.y,dst,n);
    else if (op == SUM) {
        if (atomic) reduce_stage<SUM,false,true><<<groups,THREADS>>>(b.x,b.y,dst,n);
        else reduce_stage<SUM><<<groups,THREADS>>>(b.x,b.y,dst,n);
    } else if (vec4) {
        if (atomic) reduce_stage<DOT,true,true><<<groups,THREADS>>>(b.x,b.y,dst,n);
        else reduce_stage<DOT,true><<<groups,THREADS>>>(b.x,b.y,dst,n);
    } else {
        if (atomic) reduce_stage<DOT,false,true><<<groups,THREADS>>>(b.x,b.y,dst,n);
        else reduce_stage<DOT><<<groups,THREADS>>>(b.x,b.y,dst,n);
    }
    CUDA(cudaGetLastError());
    if (atomic) return;
    while (groups > 1) {
        const float* src=dst;
        int length=groups; groups=(length+THREADS-1)/THREADS;
        dst=groups==1 ? b.out : (src==b.a ? b.b : b.a);
        if (op==MAXIMUM) reduce_stage<MAXIMUM><<<groups,THREADS>>>(src,src,dst,length);
        else reduce_stage<SUM><<<groups,THREADS>>>(src,src,dst,length); // DOT partials must be summed.
        CUDA(cudaGetLastError());
    }
}

void run_case(int n, Op op, bool vec4, bool atomic, bool timing) {
    std::vector<float> x(n), y(n);
    double ref=op==MAXIMUM ? -std::numeric_limits<double>::infinity() : 0.0, magnitude=0.0;
    for(int i=0;i<n;++i) {
        x[i]=op==MAXIMUM ? -1.0f-float(i%29)/29 : float((i%23)-11)/11;
        y[i]=float((i%19)-9)/9;
        double v=op==DOT ? double(x[i])*double(y[i]) : x[i];
        if(op==MAXIMUM) ref=std::max(ref,v); else ref+=v;
        magnitude+=std::abs(v);
    }
    Buffers b(n);
    if(n) { CUDA(cudaMemcpy(b.x,x.data(),n*sizeof(float),cudaMemcpyHostToDevice)); CUDA(cudaMemcpy(b.y,y.data(),n*sizeof(float),cudaMemcpyHostToDevice)); }
    float got;
    for(int repeat=0;repeat<2;++repeat) {
        launch(b,n,op,vec4,atomic);
        CUDA(cudaMemcpy(&got,b.out,sizeof(float),cudaMemcpyDeviceToHost));
        bool ok=op==MAXIMUM ? double(got)==ref : std::isfinite(got) && std::abs(double(got)-ref)<=1e-6+2e-6*magnitude;
        if(!ok) throw std::runtime_error("numerical validation failed for n="+std::to_string(n));
    }
    std::cout<<"PASS n="<<n<<" result="<<got<<" fp64_ref="<<ref;
    if(timing && n) {
        for(int i=0;i<5;++i) launch(b,n,op,vec4,atomic);
        CUDA(cudaDeviceSynchronize());
        cudaEvent_t start,end; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end));
        CUDA(cudaEventRecord(start));
        for(int i=0;i<50;++i) launch(b,n,op,vec4,atomic);
        CUDA(cudaEventRecord(end)); CUDA(cudaEventSynchronize(end));
        float ms; CUDA(cudaEventElapsedTime(&ms,start,end));
        CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end));
        std::cout<<" complete_workflow_ms="<<ms/50;
    }
    std::cout<<'\n';
}

int main(int argc,char** argv) {
    try {
        int n=argc>1 ? std::stoi(argv[1]) : 1003;
        std::string name=argc>2 ? argv[2] : "sum", method=argc>3 ? argv[3] : "two-pass";
        if(n<0 || n>(1<<26)) throw std::invalid_argument("n must be in [0,2^26]");
        if(name!="sum" && name!="max" && name!="dot" && name!="dot-vec4") throw std::invalid_argument("operation: sum|max|dot|dot-vec4");
        if(method!="two-pass" && method!="atomic") throw std::invalid_argument("method: two-pass|atomic");
        Op op=name=="max" ? MAXIMUM : name=="sum" ? SUM : DOT;
        bool atomic=method=="atomic", vec4=name=="dot-vec4";
        if(atomic && op==MAXIMUM) throw std::invalid_argument("atomic demo supports sum/dot, not max");
        for(int size : {0,1,31,32,33,127,128,129,255,256,257,520,65537}) run_case(size,op,vec4,atomic,false);
        run_case(n,op,vec4,atomic,true);
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
```

</details>

### Triton：一轮产生一个 partial

`tl.sum`/`tl.max` 表达逻辑轴上的归约；具体 warp/线程布局由编译器决定，不保证严格对应上面的手写 CUDA 指令数。使用 `tl.arange` 的 BLOCK 是逻辑元素数，workers 才用于 num_warps。[Triton sum API](https://triton-lang.org/main/python-api/generated/triton.language.sum.html)

```python
"""Hierarchical float32 sum/max/dot; no global atomic accumulator."""

import torch
import triton
import triton.language as tl


@triton.jit
def reduce_tiles(X, Y, Out, N: tl.constexpr, BLOCK: tl.constexpr, OP: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if OP == 1:
        values = tl.load(X + offsets, offsets < N, other=-float("inf"))
        total = tl.max(values, axis=0)
    else:
        values = tl.load(X + offsets, offsets < N, other=0.0)
        if OP == 2:
            values *= tl.load(Y + offsets, offsets < N, other=0.0)
        total = tl.sum(values, axis=0)
    tl.store(Out + tl.program_id(0), total)


def prepare(n: int, block: int, workers: int, operation: str):
    """Allocate partial buffers once and return launch(x, y, out[1])."""
    op = {"sum": 0, "max": 1, "dot": 2}[operation]
    lengths, partials = [], []
    length = n
    while length > 0:
        lengths.append(length)
        count = triton.cdiv(length, block)
        if count == 1:
            break
        partials.append(torch.empty(count, device="cuda", dtype=torch.float32))
        length = count

    def launch(x, y, out):
        if n == 0:
            out.fill_(-float("inf") if op == 1 else 0.0)
            return
        source = x
        for level, count in enumerate(lengths):
            target = partials[level] if level < len(partials) else out
            current_op = op if level == 0 else (1 if op == 1 else 0)
            reduce_tiles[(triton.cdiv(count, block),)](source, y, target, count, block, current_op, num_warps=workers)
            source = target

    launch.partials = partials
    return launch
```

### TileLang：先把 fragment 全部初始化，再沿轴归约

这里使用 `[1,BLOCK]` 的 fragment，沿 dim=1 归约成 `[1]`。尾部的每个位置都赋单位元，没有残留的未初始化值。每个 tile 是独立归约，故使用 `clear=True`；如果之后改成在循环里跨 tile 累加，必须重新安排初始化和累积语义，不能照抄清零策略。[TileLang reduction API](https://tilelang.com/autoapi/tilelang/language/reduce_op/index.html)

```python
"""Fragment-based hierarchical reduction with explicit neutral values."""

import tilelang
import tilelang.language as T
import torch


@tilelang.jit
def make_reduce(n: int, block: int, threads: int, op: int):
    groups = (n + block - 1) // block

    @T.prim_func
    def reduce_tiles(X: T.Tensor((n,), "float32"), Y: T.Tensor((n,), "float32"), Out: T.Tensor((groups,), "float32")):
        with T.Kernel(groups, threads=threads) as pid:
            values = T.alloc_fragment((1, block), "float32")
            total = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(block):
                offset = pid * block + j
                if offset < n:
                    if op == 2:
                        values[0, j] = X[offset] * Y[offset]
                    else:
                        values[0, j] = X[offset]
                else:
                    if op == 1:
                        values[0, j] = -T.infinity("float32")
                    else:
                        values[0, j] = 0.0
            if op == 1:
                T.reduce_max(values, total, dim=1, clear=True)
            else:
                T.reduce_sum(values, total, dim=1, clear=True)
            for i in T.Parallel(1):
                Out[pid] = total[i]

    return reduce_tiles


def prepare(n: int, block: int, workers: int, operation: str):
    """Create all levels before timing; y is only used for the first dot stage."""
    op = {"sum": 0, "max": 1, "dot": 2}[operation]
    kernels, partials = [], []
    length = n
    while length > 0:
        current_op = op if not kernels else (1 if op == 1 else 0)
        kernels.append(make_reduce(length, block, workers, current_op))
        count = (length + block - 1) // block
        if count == 1:
            break
        partials.append(torch.empty(count, device="cuda", dtype=torch.float32))
        length = count

    def launch(x, y, out):
        if n == 0:
            out.fill_(-float("inf") if op == 1 else 0.0)
            return
        source = x
        for level, kernel in enumerate(kernels):
            target = partials[level] if level < len(partials) else out
            kernel(source, y if level == 0 else source, target)
            source = target

    launch.partials = partials
    return launch
```

两个版本都预分配 partial，运行时按同一 stream 顺序提交各层；dot 只有第一层乘 x/y。n=0 在 host 路径将输出设为单位元，不发起空的归约 grid。输出初始化本身仍可能触发设备操作，所以不把空输入当作 GPU kernel 性能用例。

## Triton 与 TileLang 练习：完成后用同一套验收

### Triton T1：sum/max 的一层与多层

在 `student_triton.py` 实现 `prepare(n,block,workers,operation)`，返回 `launch(x,y,out)`，out 的形状固定为 `(1,)`。先让每个 program 写一个 partial，再用设备上的后续 kernel 汇合；partial 缓冲在 prepare 中分配。禁止用 CPU 求和、`.item()` 或 PyTorch reduction 代替学生实现。

提示：load 的默认值随操作变化；每个 program 只向 `partial[pid]` 写一次；即使 N=BLOCK²+1，也要继续归约到一个标量。N=0 单独按接口定义处理。

### Triton T2：把 dot 的乘法限制在第一层

第一层计算 x*y 再求和，第二层开始只对 partial 求和。用 x=[1,2,3]、y=[4,5,6] 验证 32；再用全 1 输入，任意 N 的 sum/dot 应为 N，可快速发现重复累加或遗漏。

进阶实验：保持数学语义不变，只改变 BLOCK/num_warps 或每线程数据量，记录完整归约时间。若加入原子版，清零必须进入每次调用和计时范围，再对照同一个两阶段基线。

### TileLang L1：保证 fragment 尾部与归约轴正确

在 `student_tilelang.py` 实现相同接口。先使用 `(1,BLOCK)` 与 `dim=1`，明确所有有效/无效元素的来源。workers 表示 threads，不是 BLOCK；用 threads=128/BLOCK=256 开始，再调整。

提示：全负数 max 最容易抓出错用 0 的问题。改变输出 fragment 形状时，重新核对被消掉的维度，不能只改一个 `dim` 数字。每层 output 长度是 `ceil(input_length/BLOCK)`。

### TileLang L2：复用多层调度并解释 clear

实现 dot 首层与 sum 后续层；后续 kernel 的 tensor 形状要和当前 partial 长度一致。解释为何本代码每轮 `clear=True` 正确，而“在一个循环里想累加多个 tile 的结果”不能每轮把累加器清掉。

进阶实验：导出生成代码，观察归约布局与 shared 同步；分别调整 BLOCK 与 threads，再解释哪种改变影响并行工作、哪种改变影响线程协作。不能只用“看起来用了 shuffle”判定实现已经最快。

### 统一运行入口与通过条件

```bash
# 先跑完整参考；在各自安装好框架的环境中运行。
python run_gpu.py --backend triton --operation sum --mode check
python run_gpu.py --backend triton --operation max --mode check
python run_gpu.py --backend triton --operation dot --mode check
python run_gpu.py --backend tilelang --operation sum --mode check
python run_gpu.py --backend tilelang --operation max --mode check
python run_gpu.py --backend tilelang --operation dot --mode check

# 做完练习后，用相同测试检查自己的实现。
python run_gpu.py --backend triton --impl student --operation dot --mode check
python run_gpu.py --backend tilelang --impl student --operation max --mode check

# 每种后端都重复检查 sum / max / dot，再测完整流程。
python run_gpu.py --backend triton --operation dot --mode bench --n 1048576 --output triton-dot.json
python run_gpu.py --backend tilelang --operation dot --mode bench --n 1048576 --output tilelang-dot.json
```

检查模式覆盖空输入、warp 和 block 边界、520、BLOCK²+1 及指定长度；每个尺寸有随机、全 1、全负、交替正负四种分布。每个测试重复调用两次，检查结果和输入未被修改；部分和较小时会一起输出，方便定位错误层。

| 验收 | 需要证明什么 |
| --- | --- |
| 覆盖 | 每个输入恰好贡献一次；每个 partial 有唯一写出者 |
| 边界 | 无效输入填正确单位元；所有 collective 参与条件合法 |
| 多层 | BLOCK²+1 能继续合并，dot 后续不重复乘 y |
| 数值 | FP64 参考、绝对误差、贡献量缩放误差分别记录 |
| 重复调用 | 不遗留旧结果，不依赖未初始化 partial，不修改输入 |
| 性能 | 比较完整结果完成时间，包含原子清零或所有合并层 |

## 怎么测：第一层很快，不代表整个归约就快

Python 计时器在编译、分配、校验后预热 10 次；取 7 组，每组 50 次**完整 launch 函数**，报告各组均值的中位数/min/max。多层 kernel 全部在区间内；CUDA demo 则预热 5 次、取连续 50 次完整流程的平均值。两种计时统计不同，不直接拼成同一张速度排名。

工作集会重复使用，未显式清 cache；小 kernel 的 event 区间可能包含提交空档。输出的 input_GB/s 只按输入字节除以整体时间，用于参考，不包含所有 partial 流量，也不是 DRAM 实测带宽。

```bash
nsys profile --trace=cuda,nvtx --output=reduce-timeline \
  python run_gpu.py --backend triton --operation dot --mode bench --n 1048576

ncu --list-sections
ncu --kernel-name regex:reduce_tiles \
  --section LaunchStats --section SchedulerStats --section MemoryWorkloadAnalysis \
  --output=reduce-report \
  python run_gpu.py --backend triton --operation dot --mode profile --n 65537
```

profile 模式包含两次完整校验调用和一次额外调用，每次可能有多个归约层；上面刻意没有硬编码某一 launch 编号。先从 kernel grid/输入长度识别第一层与合并层，再按需要缩小采样范围。不要只比较某层的 ncu 耗时就宣称端到端加速。[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

按现象选择下一步：第一层流量大且吞吐已高，先减少无效访问；第一层很快但总时间偏高，检查后续小 kernel 和提交；原子方案随 block 数增大变慢，查同地址争用与清零成本；加大 tile 后变慢，核对寄存器、并行任务数与局部依赖。每次主要改一个因素，没有固定“某版本必须胜出”的结论。

## 从本篇进入下一篇

你现在应能解释：一次 shuffle 为什么不够；down 与 XOR 的结果契约为何不同；为什么 lane 0 不能代替 tid 0；shared 屏障为何不能同步 grid；dot 的第二层为何变成 sum；数值正确为何不能只用 bitwise 比较。

下一篇按原文顺序是 **（五）向量化访存与原子操作**，专门展开合并访问、单线程宽访存、地址对齐、尾部保护与直方图争用。本篇已经用安全的 dot-vec4 和原子对照建立入口，后续仍保持一章一篇。

[返回 LLM 算子学习栏目](/llm/) · [返回完整编号目录](/systems/gpu-learning-path/)

<link rel="stylesheet" href="/css/cuda-learning.css">
<link rel="stylesheet" href="/css/gpu-roofline.css">
<link rel="stylesheet" href="/css/gpu-reduction.css">
<script defer src="/js/gpu-reduction.js"></script>
