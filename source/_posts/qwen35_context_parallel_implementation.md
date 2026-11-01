---
title: "Qwen3.5 Context Parallel：系统设计与计算流程"
date: "2026-11-01 12:00:00"
categories: ["LLM 推理"]
tags: ["Qwen3.5", "Context Parallel", "GDN", "MoE", "KV Cache", "SGLang"]
description: "解析混合模型 Context Parallel 的系统设计与计算流程，涵盖 token 布局、GDN 状态递推、MoE 写回、缓存分片、PD 传输与调度依赖。"
permalink: "/llm/qwen35-context-parallel/"
---

Qwen3.5 的混合结构同时包含有序递推的 Gated DeltaNet（GDN）、需要完整历史 KV 的 Full Attention，以及按专家路由的 MoE。三者对 token 顺序和数据归属的要求不同，Context Parallel（CP）需要同时处理计算布局、缓存布局和跨层依赖。

系统以 **CP4、Attention TP1、MoE EP4（Mega）** 组织计算：各层使用适合自身算子的 token 布局，KV 与 GDN state 按固定 owner 持久化，通过统一 metadata 衔接层间转换、缓存访问和 PD 传输。

<!-- more -->

## 整体计算流程

图中 Encoder 对应多模态输入；前缀匹配后，CPU 规划 chunk 的布局与索引，CP 计算组执行各层计算，最终还原输出并传输 KV 与 state。

[![请求入口、CP 计算组与 PD 传输的整体链路](/images/qwen35-context-parallel/system-overview.png)](/images/qwen35-context-parallel/system-overview.png)

*请求入口、CP 计算组与 PD 传输的整体链路*

## 1. 计算布局与缓存归属分开维护

GDN 使用连续序列段，保证递推顺序；Full Attention 使用 round-robin query 布局，持久 KV 按绝对 token 位置交错分片。zigzag query 保留为兼容路径。

| 对象 | 布局与职责 |
| --- | --- |
| GDN 输入 | 每个请求按连续段切分，段间传递卷积历史与递推状态 |
| Full Attention query | round-robin；兼容 zigzag 两端块分配 |
| 持久 KV | 按绝对位置确定 owner，不随层间 hidden 排列变化 |
| hidden 与 residual | 由同一份 token 映射转换，进入下一层时身份一致 |
| MoE | 本地有效 token 路由到专家，再按原 token 身份归并 |

[![GDN 连续段、zigzag query 与交错 KV 的 token 映射](/images/qwen35-context-parallel/token-layouts.png)](/images/qwen35-context-parallel/token-layouts.png)

*GDN 连续段、zigzag query 与交错 KV 的 token 映射*

`prepare_attn` 根据当前层与 metadata 转换 hidden/residual，同布局时直接复用。`--prefill-cp-mode round-robin-split` 对应交错 query，`in-seq-split` 对应 zigzag；两者都保持 GDN 连续分片。该参数描述序列布局，不限定请求并发数。

设 CP 大小为 $C$，hidden size 为 $D$，Q/KV head 数为 $H_q/H_{kv}$，head dimension 为 $d_h$，逻辑页大小为 $P$。hidden 的本地形状为 $[N_{local},D]$，Q 为 $[N_q,H_q,d_h]$，保留完整 hidden/head 维度。在页大小可被 $C$ 整除的配置下，本地单页 K、V 各为 $[P/C,H_{kv},d_h]$。

GDN 每请求的物理段宽按下式对齐，其中 $N$ 是本次新增 token 数：

$$
L_{physical}=64\left\lceil\frac{N}{64C}\right\rceil.
$$

padding 只服务算子对齐，不进入有效 MoE 路由、KV 写入或真实状态更新。请求长度、prefix、有效 token 和还原映射均逐请求记录，尾部和空 shard 按实际长度处理。

## 2. Full Attention：Q 留在本地，KV 按 owner 轮转

新增 token 的 KV 必须写入持久 owner。round-robin query 与 KV owner 一致，可直接写入；zigzag 的两种 owner 不一致，需要先转换新 K/V 的归属。

[![zigzag 与 round-robin 的 query 和 KV owner 对照](/images/qwen35-context-parallel/round-robin-kv-owner.png)](/images/qwen35-context-parallel/round-robin-kv-owner.png)

*zigzag 与 round-robin 的 query 和 KV owner 对照*

计算过程如下：

1. 将新 K/V 写入本地缓存，读取包含 prefix 的本地 KV shard。
2. Q 保留在当前 rank，双缓冲轮转其他 owner 的 KV。
3. 每轮按全局位置确定 causal 可见范围，调用已有 attention kernel。
4. 用各轮 partial output 与 log-sum-exp（LSE）合并输出，最后恢复本地 Q 行序。

图中的 K/V 换 owner 对应 zigzag 路径；round-robin 路径跳过该步骤，后续 KV 轮转与输出合并流程相同。

[![Full Attention 的 KV 轮转、双缓冲与输出归并](/images/qwen35-context-parallel/attention-kv-ring.png)](/images/qwen35-context-parallel/attention-kv-ring.png)

*Full Attention 的 KV 轮转、双缓冲与输出归并*

轮转的是 KV，合并的是 attention 输出。若第 $r$ 个非空分片返回归一化输出 $O_r$ 和自然对数形式的 LSE $\ell_r$，则可用以下形式表达合并关系：

$$
m=\max_r\ell_r,\qquad
O=\frac{\sum_r e^{\ell_r-m}O_r}{\sum_r e^{\ell_r-m}}.
$$

这是对分片 softmax 合并的数学表达；具体 backend 的 LSE 对数底数需先统一，全 masked 的行需单独处理，不能直接计算未定义的差值。

### Query owner 与本地行序

zigzag 先按请求两端块分配 Q，再在本 rank 内按位置余数整理为供 varlen 接口使用的虚拟序列。它不会把 Q 再次分发给 `position % C` 对应的 rank。

例如，CP4、64 个 token 时，rank0 可持有 `[0…7]` 与 `[56…63]`。本地分组改变调用时的排列，不改变这些 Q 的 owner。各组根据全局位置确定当前 KV shard 的可见截止范围，继续复用底右对齐 causal mask；多个组在一次 varlen 调用中批量处理。

[![zigzag query 在本 rank 内分组并调用 varlen attention](/images/qwen35-context-parallel/zigzag-query-groups.png)](/images/qwen35-context-parallel/zigzag-query-groups.png)

*zigzag query 在本 rank 内分组并调用 varlen attention*

### Prefix 与物理地址

新增 token 的 owner 从完整逻辑位置计算：

```python
kv_owner = (prefix_len + new_token_index) % cp_size
page_row = local_row * cp_size + rank
```

不能在每个新 chunk 从 rank0 重新计数。物理 slot 继续由现有页表、分配器与 CP 映射定位，逻辑位置不能直接充当物理地址。

KV ring 通过 `--attn-cp-transport` 选择 `nccl` 或 `symm_rma`，通信流与计算流以事件衔接。KV ring、GDN 状态通信和 Mega dispatch 是不同通路，应分别管理依赖。Attention 继续复用已有 FA3/TRTLLM prefill backend，CP 不改变 backend 的 dtype 约束。

## 3. GDN：分别处理卷积 halo 与递推状态

连续切分后，本段短卷积缺少前驱尾部，递推缺少前面各段作用后的初始 state。这两种依赖需要分别处理。

### 卷积 halo

满足条件的 CP4 路径发布投影后 QKV 的最后三行，由卷积 kernel 通过对称内存读取前驱历史。这里传递的是卷积所需历史，不是把前驱 hidden 拼到 Mega combine 后重复投影。尾部不整齐、空 shard 等情况保留通用交换路径。

[![GDN 短卷积读取前驱 halo 的数据流](/images/qwen35-context-parallel/gdn-convolution-halo.png)](/images/qwen35-context-parallel/gdn-convolution-halo.png)

*GDN 短卷积读取前驱 halo 的数据流*

### 有序状态摘要

每段产生局部贡献 $H$ 与状态变换 $M$。为解释组合顺序，这里将状态向量化，并约定以左乘形式表示一段的作用：

$$
S_{out}=MS_{in}+H.
$$

先执行段 $a$，再执行段 $b$，组合关系为：

$$
(M_b,H_b)\circ(M_a,H_a)
=(M_bM_a,\ M_bH_a+H_b).
$$

这一表达用于说明摘要的有序组合；实际 kernel 的张量布局和乘法方向应以实现约定为准。组合顺序与原序列一致，不能将状态摘要当作普通数值求和，也不能用一次广播替代前缀计算。

peer-prefix kernel 从对称内存读取前面各段摘要，按序得到本段初始 state，再生成本段输出与快照。实现也保留 ring 算法和未启用对称内存时的兼容路径。

[![GDN 局部状态摘要、有序前缀合成与快照发布](/images/qwen35-context-parallel/gdn-state-prefix.png)](/images/qwen35-context-parallel/gdn-state-prefix.png)

*GDN 局部状态摘要、有序前缀合成与快照发布*

数学推导参考：[GDN / Context Parallel 数学推导参考（知乎）](https://zhuanlan.zhihu.com/p/2007937984738129405)。

### 状态发布与算子边界

final state 与 checkpoint 按 owner 发布，可使用 multicast。工作区暂时存在相同结果，不意味着每卡长期保存同一份 cache。

公共分配、peer 地址和同步原语复用 `TorchSymmMemCommunicator`；递推与状态语义留在 GDN/FLA 算子内。CP 路径通过已有 kernel dispatcher 的 extend 入口接入，decode 与 target verify 保留各自支持的入口。

## 4. MoE EP：保持 token 身份，调整最终写回位置

本地有效 token 执行 top-k，dispatch 到持有完整专家的 rank，由 Mega 完成通信与专家计算，再按路由权重归并回原 token。该 EP 路径不再依赖旧 MoE TP 的全量 hidden All-Gather/Reduce-Scatter。

[![Mega EP 按 token 路由到专家并按原身份归并](/images/qwen35-context-parallel/moe-expert-routing.png)](/images/qwen35-context-parallel/moe-expert-routing.png)

*Mega EP 按 token 路由到专家并按原身份归并*

| 并行配置 | Attention | MoE |
| --- | --- | --- |
| `tp=4, cp=4, ep=1` | CP4、TP1 | TP4 |
| `tp=4, cp=4, ep=4`，启用 Mega | CP4、TP1 | EP4、TP1 |

Mega 经已有 `FusedMoE.forward` 分派，各层复用 workspace。Hopper W4AFP8 路径通过 `--enable-mxfp4-mega` 转换专家权重表示，并由 FlashInfer stage 量化输入。这一过程同时涉及权重表示与输入量化，需要与通信路径分别处理。

### Combine 直接写入下一层布局

常规路径先把专家结果 combine 回本层输入行，下一层再把 hidden 从连续段转换为 round-robin，或反向转换。直接写回的设计将目标位置映射放进最终 reducer：

1. dispatch、专家计算、top-k 权重和 token 归并身份保持原语义。
2. reducer 按既有 BF16 舍入顺序归并，并加入同一 token 的 shared expert 输出。
3. 根据下一层 CP metadata 的目标 rank/row，将最终 hidden 写入目标对称内存。
4. residual 使用相同映射转换，hidden 与 residual 都就绪后更新当前布局。

[![专家归并保持 token 身份，最终写入下一层目标 rank 和 row](/images/qwen35-context-parallel/combine-target-layout.png)](/images/qwen35-context-parallel/combine-target-layout.png)

*专家归并保持 token 身份，最终写入下一层目标 rank 和 row*

shared expert 只相加一次。改动发生在最终写回阶段，不是提前把 FC2 专家结果改投到另一 rank。跨 rank 数据仍需搬运，只是 hidden 的布局转换并入最终 store；residual 转换仍然存在。

目标映射由 batch 阶段的 CPU plan 提供，经 `prepare_mlp`、`postprocess_layer`、`prepare_attn` 衔接。同布局继续本地消费；末层没有下一层目标，保留模型出口的 token 顺序还原。

round-robin 转回连续段时，可预建源行排列，让相邻 CTA 的工作交错面向不同目标 rank，避免连续工作集中写一个 peer。排列只改变工作分配，不改变输出地址和数值，也不规定 GPU 的实际执行顺序。

[![按目标 rank 交错安排 CTA 的写回工作](/images/qwen35-context-parallel/combine-write-scheduling.png)](/images/qwen35-context-parallel/combine-write-scheduling.png)

*按目标 rank 交错安排 CTA 的写回工作*

对称 workspace 在 warmup 准备。生产者写完后消费者才能读取，上一轮消费结束后才能覆盖；取消中间 clone 后，这些依赖必须显式成立。

### Residual 的两种搬运位置

独立 stream 方案在 post-attention Add/Norm 生成更新后的 residual 后，提前提交布局交换。计算流继续执行 shared expert、router/top-k 和 Mega，到下一层 Norm 消费前再等待完成事件。

Mega 内搬运方案让 kernel 接收 residual、目标 rank/row 和 peer 指针，由 dispatch 中的搬运工作将 residual 一次写到目标位置。residual 不参与 top-k 复制，也不执行专家 combine；完成同步复用 Mega 的尾部机制。目标 buffer 交替使用，避免快 rank 覆盖慢 rank 尚未消费的数据。

两种搬运方式都必须保证 residual 只写一次，并与下一层 hidden 的 token 身份一致。融合搬运与专家计算共享执行资源，独立 stream 则通过完成事件衔接消费者；选择时需要同时考虑资源占用、buffer 生命周期与依赖关系。

## 5. Cache：完整前缀匹配，KV 与 state 分片持久化

混合模型恢复前缀时，既需要对应 KV，也需要同一位置的 GDN state。只在大 chunk 末尾保存 state，会让中间公共前缀缺少恢复点。

[![公共前缀根据可用 state checkpoint 选择恢复点](/images/qwen35-context-parallel/prefix-state-checkpoints.png)](/images/qwen35-context-parallel/prefix-state-checkpoints.png)

*公共前缀根据可用 state checkpoint 选择恢复点*

radix 仍按完整 prompt token 序列匹配和插入，节点保存逻辑 KV 索引。层执行时才将这些索引解析到各 rank 的物理 KV。Mamba state 则按逻辑 slot 分散持久化，slot 0 留作空状态：

```python
def state_location(slot, cp_size):
    assert slot > 0  # slot 0 is reserved
    owner = (slot - 1) % cp_size
    local_slot = (slot - 1) // cp_size + 1
    return owner, local_slot
```

读取 prefix 时从实际 owner 取 state，写回也只由 owner 持久化。一个 checkpoint 同时包含 temporal state 与对应的短卷积历史，不保存逐 token 的 GDN KV。

[![逻辑 state slot 与物理 owner 的映射及读写路径](/images/qwen35-context-parallel/state-slot-ownership.png)](/images/qwen35-context-parallel/state-slot-ownership.png)

*逻辑 state slot 与物理 owner 的映射及读写路径*

在原有分叉点与末尾 checkpoint 基础上，可按预算补充对齐的 shard 端点快照。新增快照复用已有 KV 前缀链；预算不足时减少可选快照，不为短尾快照驱逐完整前缀。分叉恢复点也不应截断整个 CP chunk 的缓存发布。

缓存可恢复性需要联合检查请求所在实例、radix 匹配位置、state 是否存在以及是否已被驱逐，不能仅由 KV 容量推断。

## 6. PD bulk：按 entry 身份传输并恢复目标布局

Prefill（P）与 Decode（D）的缓存层数可能不同，例如 D 多出 draft/NEXTN 层。两端不能仅依靠 buffer 数量或顺序对齐。

bulk 路径按 `(target/draft, K/V, layer)` 匹配 entry，再协商每行字节数和偏移。P 将本 rank 的 KV pack 成连续数据，各 writer 写入各自 staging 区域；D 等相关 writer 和 scatter 完成后消费目标 KV。

[![PD bulk 的 entry 匹配、分片搬运与目标页还原](/images/qwen35-context-parallel/pd-bulk-transfer.png)](/images/qwen35-context-parallel/pd-bulk-transfer.png)

*PD bulk 的 entry 匹配、分片搬运与目标页还原*

需要保持以下约束：

- D 端额外 draft/NEXTN entry 不被 target 数据覆盖，也不应仅因多一层就让 target bulk 回退。
- 页内行恢复遵循 `local_row * C + rank`；BF16/FP8 按字节布局搬运，不隐式转换数值。
- fallback writer 用完成 mask 区分有效区域，未写入的 staging 数据不能 scatter 到目标页。
- Mamba state 独立传输，由逻辑 slot 的实际 owner 发送，不固定由 CP0 发送。
- 保持已有 `SGLANG_DISAGG_STAGING_BUFFER` 入口兼容；显式要求 bulk 时，失败应报告原因。

EPD 场景仍按 encoder → prefill → decode 分工。encoder 生成 embedding，P 在切分前按有效 token 位置接入 embedding/MRoPE；这里的 CP 作用于 prefill，不把 encoder 当作 CP rank。

## 7. Metadata 与 overlap：提前规划，按依赖消费

scheduler 在进入 forward stream 前，以 CPU/NumPy 构建 batch 只读 metadata，包含层间重排、KV 路由、attention segments 和模型出口还原计划。各字段按对齐偏移打包到 pinned buffer，一次异步 H2D 后恢复 view；权威 KV slot 表继续在 GPU 查询。

[![CPU metadata 统一规划、打包 H2D 与 GPU view](/images/qwen35-context-parallel/metadata-packed-h2d.png)](/images/qwen35-context-parallel/metadata-packed-h2d.png)

*CPU metadata 统一规划、打包 H2D 与 GPU view*

同形状计划可以复用，但每批 request slot、prefix 和在途工作必须独立持有。下一批不能覆盖上一批仍在使用的 workspace。

安全的同请求 continuation 可以提前准备和提交，真实 KV/state 递推依赖仍在设备侧保留。DP=1 且完整 CP 组持有相同 batch 信息时，可复用本地信息；实际数据通信不会因此消失。

[![相邻 chunk 的 CPU 规划、异步 H2D 与 GPU 执行依赖](/images/qwen35-context-parallel/chunk-overlap.png)](/images/qwen35-context-parallel/chunk-overlap.png)

*相邻 chunk 的 CPU 规划、异步 H2D 与 GPU 执行依赖*

overlap 是否成立取决于提交时机和依赖图。CPU 尚未提交完 eager 算子时，后续工作无法仅靠一个 overlap 开关自动提前执行。

## 8. Warmup 与实现验证

`ModelRunner` 按模型几何与启用路径统一准备重排、norm、卷积、状态计算、attention merge、FlashQLA 上下文与通信 workspace，再准备 Mega 权重、buffer 和 tactic。参数相关的编译与调优放在启动期，数据相关的 batch plan 留在调度阶段。

[![ModelRunner 在启动阶段准备算子、workspace 与通信资源](/images/qwen35-context-parallel/warmup-resources.png)](/images/qwen35-context-parallel/warmup-resources.png)

*ModelRunner 在启动阶段准备算子、workspace 与通信资源*

执行继续进入已有模型层、GDN/Full Attention backend 和 MoE layer。计算算法、token 映射和公共对称内存资源分别维护，避免另建一套包含完整 forward 的 CP backend。

实现验证应覆盖：

- round-robin/zigzag 与连续段之间的 token 身份、hidden/residual 一致性。
- prefix 续接、ragged、padding、空 shard 和多请求 batch。
- 分片 attention 合并、GDN 前缀状态与 checkpoint 的对应关系。
- shared expert 只加一次、既有舍入顺序及模型出口行序还原。
- 多轮 workspace 复用，以及快慢 rank 间的读写完成依赖。
- PD 两端 entry 不同、fallback 完成 mask 和 state owner 传输。

验证按算子、层间组合和完整服务逐层展开，分别检查数值语义、布局衔接和端到端依赖。

## 参考资料

- [数学推导参考（知乎）](https://zhuanlan.zhihu.com/p/2007937984738129405)
- [返回 LLM 专题](/llm/)
