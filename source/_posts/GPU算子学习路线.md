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

本系列归入 **LLM 推理 → 算子学习**，依据 `leetcuda-20261029.pdf` 持续创作。**原文每一章对应一篇独立博客，按原文章节顺序继续，编号接着已有文章往下排。** 下方的大组只是导航分组，不会合并成一篇博客。

编号关系固定为：已有 CUDA 基础是（一），性能分析是（二）；原文第二章的 GPU 架构与 Roofline 为（三），原文第三章的归约与点积为（四），依此继续到（四十一）。博客编号与原文编号分开记录，正文保留原文主题与 PDF 阅读器页码便于核对。

当前可直接学习：

- [（一）CUDA 基础：GPU 结构、执行与内存](/systems/cuda-basics/)——先建立物理结构与编程模型。
- [（二）SM 调度、warp stall、访存优化与流水线](/systems/cuda-overlap/)——理解等待、驻留、tiling、overlap，并学会采集证据。
- [（三）Roofline：估算性能上限，再用 Triton 和 TileLang 验证](/systems/gpu-roofline/)——接续当前学习，附图、交互计算器、CPU demo、两套 GPU 基线和练习入口。

- [（四）归约与点积：从 warp 到全局结果](/llm/cuda-reduction/)——逐轮交互、CUDA/Triton/TileLang 完整 demo、分层归约与原子对照。

**下面覆盖后续全部学习主题，但“学习任务已梳理”与“完整教程已发布”分开标注。** 标为“待展开”的条目已有学习目标和练习设计，尚没有配套成品；完成讲解、流程图、demo、练习与检查后，逐篇替换成文章链接。不会将空模板列为已完成教程。

<figure class="cuda-diagram">
<a href="/images/gpu-roofline/learning-route.svg" target="_blank" rel="noopener"><img src="/images/gpu-roofline/learning-route.svg" alt="机器测量到基础并行，再到矩阵计算、注意力布局和低精度的知识依赖路线" loading="lazy"></a>
<figcaption>这张图概括知识依赖；实际文章保持下方的连续编号与原文章节顺序。<a href="/images/gpu-roofline/learning-route.excalidraw" download>下载可编辑原图</a></figcaption>
</figure>

路线图把基础能力汇合到 Attention，再由布局、调度和量化各自深化。图中的箭头表示理解上的依赖：例如不了解 online 状态就直接改 FlashAttention 流水，数值问题与同步问题会混在一起；先做稳定 Softmax 和 LSE merge，可以把两者分开验证。

## 每篇发布前必须齐备的内容

每个学习主题都需要**可操作的交互演示和具体代码 demo**，并与原理、流程图使用同一个例子。下方“待展开”主题中的演示与 demo 是明确的制作任务，目前尚未提供可运行成品。

| 组成 | 要达到的效果 |
| --- | --- |
| 交互演示 | 有输入/配置变化、单步或重置等合适操作，展示中间状态与解释；不是只有一张静态图片 |
| 具体代码 demo | 提供完整源码、依赖、运行命令、最小输入与预期结果；可直接运行已有版本，再开始改写 |
| 图与公式 | 标明数据、控制、时间或布局的关系，公式变量和假设可手算核对 |
| Triton 与 TileLang 练习 | 独立学生入口、提示、参考实现、统一接口和验收用例，不能只有题目名称 |
| 验证说明 | CPU 模型、后端编译、GPU 正确性、性能实测分别写实际状态；交互模拟不冒充硬件测速 |

架构专用主题先提供任何电脑可运行的索引/协议/数值小模型，再给支持该能力的 GPU 代码与明确运行条件。它们互相补充，CPU 演示不会替代 GPU 实现的交付。完成这些内容后才把主题标为“完整教程已发布”。

## 每个主题怎样学

一轮学习按五步走：**说清输入输出 → 用小例子手算 → 沿图追踪数据与控制 → 跑完整 demo → 独立完成练习并验收**。CUDA 用于解释硬件机制，Triton 练习强调块级表达，TileLang 练习强调 tile、布局与搬运配合。高级硬件机制先检查 GPU 和编译器支持，不能为了凑语言对照把不支持的指令伪装成可用。

- 练习分为“实现正确结果”和“验证一个性能假设”。没有固定加速倍数要求，变慢也可以形成正确结论。
- 每份验收记录保留 shape、dtype、布局、GPU、软件版本、误差、独立计时及必要 profiler 证据。
- CPU 模型验证索引或代数；GPU correctness 验证输出；实际性能需要真实设备计时。三种状态分别记录。
- 全书实验性能不等于本机性能。所有微架构数量、API 和编译目标按当时官方资料及目标设备复核。

## 机器、执行与测量

先读上方两篇已发布基础资料；会用 Python 和张量 shape。

### [（三）GPU 架构与 Roofline：先预测，再测量](/systems/gpu-roofline/)

**状态：已发布，可开始学习。** 读懂逻辑流量、分层流量和两种时间下界。

- **交互演示**：改变尺寸、位宽和 P/BW 观察屋顶；改变 tile/warp 查看尾部 mask。
- **完整代码 demo**：roofline.py 与 add/axpy 两语言参考实现：手算、边界校验、计时。
- **Triton 练习**：向量加法与 axpy；扫 tile / num_warps。
- **TileLang 练习**：同样语义；扫 T.Parallel 范围 / threads。
- **通过条件**：先跑边界校验，再解释工作集与请求带宽；能说明为什么 roofline 不是测速结果。

## 基础并行原语与行算子

前置：线程/warp、掩码与同步。归约是 Softmax 和 Norm 的公共前置，LSE 合并在 Softmax 之后。

### [（四）归约与点积：从一组数到一个结果](/llm/cuda-reduction/)

**状态：已发布，可开始学习。** warp 内交换、block 两级归约、跨 block 合并与浮点误差。

- **交互演示**：点击下一轮展开归约树，显示每个 lane 的值与跨 warp 合并。
- **完整代码 demo**：两阶段 sum/dot：输出 partial 和最终值，对照 FP64，并覆盖尾块。
- **Triton 练习**：块内 tl.sum；两阶段 dot，处理尾部。
- **TileLang 练习**：fragment 归约；多 block partial 与最终合并。
- **通过条件**：比较 FP64 参考，测非整除长度；解释为何不能在普通 grid 内随意全局同步。

### （五）访问粒度与原子冲突：向量化和直方图

**状态：待展开；学习任务已梳理。** 区分合并访存、单线程宽访存和原子串行化。

- **交互演示**：切换 stride、访问宽度、热点分布，显示 sector 覆盖与原子冲突。
- **完整代码 demo**：连续/跨距 copy 与 histogram：相同字节量对照，直方图总数校验。
- **Triton 练习**：连续 / 跨距访问；整数 histogram。
- **TileLang 练习**：宽 copy；每 block 局部 histogram 后合并。
- **通过条件**：同一有效数据量比较请求；均匀与热点分布均正确，直方图总数守恒。

### （六）稳定 Softmax：从整行到在线状态

**状态：待展开；学习任务已梳理。** 减最大值，推导分块 max 与 sum 的重缩放。

- **交互演示**：逐块推进 max/sum 状态，修改极端输入观察溢出与重缩放。
- **完整代码 demo**：naive、stable、online softmax：中间状态输出、行和与参考误差检查。
- **Triton 练习**：单行 softmax；长行分块状态。
- **TileLang 练习**：reduce_max / reduce_sum；online 两状态。
- **通过条件**：极大输入不溢出，行和接近 1；空行与全 mask 行有明确约定。

### （七）LSE 与注意力状态合并

**状态：待展开；学习任务已梳理。** 把归一化输出与 log-sum-exp 组合成可合并状态。

- **交互演示**：合并两段 LSE/output，切换空段，显示权重与最终状态。
- **完整代码 demo**：merge_states：两段/多段/空段对照一次完整计算。
- **Triton 练习**：实现两段与多段 LSE merge。
- **TileLang 练习**：并行分段 merge；避免额外中间张量。
- **通过条件**：分段合并对齐完整参考，覆盖空段与两边都空，明确对数底。

### （八）RMSNorm 与 LayerNorm：统计量和广播

**状态：待展开；学习任务已梳理。** 理解平方均值、中心化方差、eps 与稳定计算。

- **交互演示**：逐步显示均值、方差或平方均值；调 eps 和输入偏置。
- **完整代码 demo**：RMSNorm/LayerNorm：两套数学参考与 GPU 基线，覆盖小方差和尾部。
- **Triton 练习**：fused RMSNorm；LayerNorm 两遍统计。
- **TileLang 练习**：fragment 归约；比较两遍方差与 Welford。
- **通过条件**：小方差、大偏置、非整除 hidden size；匹配 eps 和方差定义。

### （九）RoPE 与转置：数据重排为何影响性能

**状态：待展开；学习任务已梳理。** 旋转配对、相对位置性质、global 合并与 shared bank。

- **交互演示**：拖动旋转角度观察坐标对；切换 padding 查看 lane→bank 映射。
- **完整代码 demo**：RoPE 与矩阵转置：范数/相对位置验证、转置往返与尾块检查。
- **Triton 练习**：两种 RoPE 配对格式；带尾部转置。
- **TileLang 练习**：shared 中转转置；padding 布局实验。
- **通过条件**：旋转范数近似保持，转置两次还原；手算真实 lane 地址后再判断冲突。

## 矩阵乘与多层复用

前置：归约、内存访问与 Roofline；先 GEMV/GEMM，再看 Tensor Core、流水和架构专用协议。

### （十）GEMV：如何把一行交给 warp

**状态：待展开；学习任务已梳理。** 长短 K 的并行分工，矩阵流量与向量复用。

- **交互演示**：切换 K 与行分工，显示每组 lane 读取的矩阵元素及归约。
- **完整代码 demo**：一行/多行 GEMV：短 K、长 K 和非整除形状的校验与计时。
- **Triton 练习**：每 program 一行与多行 GEMV。
- **TileLang 练习**：线程划分变化；短 K 多行打包。
- **通过条件**：M/K 尾部正确，区分任务太少与 DRAM 吞吐受限。

### （十一）GEMM：block tile 与寄存器 tile

**状态：待展开；学习任务已梳理。** 从 2MNK 推导复用收益，区分 block / thread tile。

- **交互演示**：逐个 K tile 累加；显示 A/B 复用次数和 C 的部分和。
- **完整代码 demo**：naive/tiled GEMM：非方阵、尾块与 CPU/框架参考对照。
- **Triton 练习**：带 mask 的 tl.dot matmul；扫 BM/BN/BK。
- **TileLang 练习**：shared tile + fragment accumulator + T.gemm。
- **通过条件**：非方阵和尾块正确；解释 BK 与输出 tile 改变的不同资源。

### （十二）精度与异步搬运：TF32 和双缓冲

**状态：待展开；学习任务已梳理。** 区分精度改变、Tensor Core 通路与 cp.async 时序。

- **交互演示**：切换串行/双缓冲时间线；量化输入观察舍入误差。
- **完整代码 demo**：TF32 精度对照与 staged matmul：误差、编译指令和耗时分别记录。
- **Triton 练习**：切换 input_precision；分开报告误差与时间。
- **TileLang 练习**：num_stages 对照；检查生成的 async 指令。
- **通过条件**：必须先报告精度，再谈速度；只看到异步指令不算 overlap 已被证明。

### （十三）MMA 与 fragment：理解协作计算的数据归属

**状态：待展开；学习任务已梳理。** lane / fragment / ldmatrix 与多级流水。

- **交互演示**：选择 lane 查看 A/B/C fragment；逐步推进 load/MMA/store。
- **完整代码 demo**：fragment 地址枚举小模型与受支持 GPU 上的 MMA 矩阵乘。
- **Triton 练习**：检查 dot 的 lowering；扩展题使用显式布局。
- **TileLang 练习**：用布局可视化追踪 A/B/C 的 fragment。
- **通过条件**：独立核对每个输出坐标与覆盖；不能把高层 tile 当每线程存储量。

### （十四）布局与调度：三种不同的 swizzle

**状态：待展开；学习任务已梳理。** shared 地址 XOR、寄存器双缓冲、CTA 顺序互不等价。

- **交互演示**：切换 padding/XOR/CTA 顺序，分别显示 bank 与 tile 访问顺序。
- **完整代码 demo**：swizzle 地址 round-trip 与布局 A/B GEMM，先验映射再采集性能。
- **Triton 练习**：grouped program ordering；比较工作集复用。
- **TileLang 练习**：shared swizzle 与分块调度分别做 A/B。
- **通过条件**：验证映射一一对应，再用实际访问组验证 bank；不混淆 L2 与 shared。

### （十五）TMA 与 WGMMA：Hopper 的完成协议

**状态：待展开；学习任务已梳理。** descriptor、transaction barrier、异步 MMA 与 slot 释放。

- **交互演示**：逐步推进 expected bytes、phase、full/empty、MMA completion。
- **完整代码 demo**：CPU slot 协议模拟；Hopper 上最小 TMA copy 与 WGMMA GEMM。
- **Triton 练习**：Hopper descriptor / persistent matmul 练习，先检查支持。
- **TileLang 练习**：TMA 搬运与 gemm 流水；核对生成屏障。
- **通过条件**：列出 arrive、expected bytes、完成和复用事件；目标硬件不支持时仅做协议模拟。

### （十六）SM120 的生产消费流水

**状态：待展开；学习任务已梳理。** 架构能力边界、mma.sync 与资源分配协议。

- **交互演示**：调整 producer/consumer 数与资源预算，显示等待环和资源不足。
- **完整代码 demo**：资源池与屏障计数模拟；SM120 专用 TMA+mma.sync 对照基线。
- **Triton 练习**：在目标设备检查生成指令和支持路径。
- **TileLang 练习**：producer/consumer 方案与常规方案对照。
- **通过条件**：能力核验在前；不能照搬 Hopper WGMMA，资源池预算与死锁条件须可解释。

## 注意力：从数学状态到分工

前置：Softmax、LSE merge、GEMM 和流水；先证明数学状态，再改变线程角色。

### （十七）FlashAttention：为什么可以不写出分数矩阵

**状态：待展开；学习任务已梳理。** QK、online softmax、PV 的状态不变量与 IO。

- **交互演示**：逐 KV tile 更新 m/l/O，与完整 softmax attention 并排对照。
- **完整代码 demo**：完整 attention 参考与 blocked online 实现，逐 tile 核对状态。
- **Triton 练习**：小尺寸 fused attention 与逐块 CPU 参考。
- **TileLang 练习**：分块 QK / PV；维护 m、l、O。
- **通过条件**：与完整 attention 对齐，覆盖 causal、尾块；解释旧输出为何也必须缩放。

### （十八）Split-Q：让每组线程拥有自己的查询行

**状态：待展开；学习任务已梳理。** 减少跨 warp 归约；S/P 布局如何交给下一次 MMA。

- **交互演示**：切换查询行归属，显示共享状态、归约范围与重复加载。
- **完整代码 demo**：split-Q attention：固定语义比较行划分，校验 causal 与尾块。
- **Triton 练习**：查询分块 attention；改变 BQ/BK。
- **TileLang 练习**：QK fragment 到 PV fragment 的交接。
- **通过条件**：每行状态有明确 owner；区分减少同步与增加重复加载。

### （十九）注意力的双流水：K、V 与角色分工

**状态：待展开；学习任务已梳理。** 不同数据的就绪时机，stage 占用和 full/empty 协议。

- **交互演示**：分别调 K/V 搬运延迟与 stages，观察 slot 占用和等待。
- **完整代码 demo**：K/V 双流水协议模拟与 supported GPU attention A/B。
- **Triton 练习**：比较编译器流水配置与支持的 warp specialization。
- **TileLang 练习**：分别调 K/V stages，追踪 consumer 释放。
- **通过条件**：短序列、单 tile、尾 tile 先通过；生产/消费不平衡需要时间线证据。

### （二十）多 consumer 与分段合并

**状态：待展开；学习任务已梳理。** 区分书中双 consumer 方案与泛称 FA3；合并局部状态。

- **交互演示**：奇偶 KV tile 分流到两个 consumer，最后可视化 LSE 合并。
- **完整代码 demo**：split-KV + merge：单段、奇数 tile 和空段均与完整 attention 对照。
- **Triton 练习**：split-KV attention + LSE merge。
- **TileLang 练习**：多 consumer 结构或分 kernel 等价基线。
- **通过条件**：单段退化、奇数 KV tile、空段均正确；合并成本计入总时间。

### （二十一）大 head_dim：Split-D 的两次分块

**状态：待展开；学习任务已梳理。** QK 的 D 归约与 PV 的输出 D 分块不同。

- **交互演示**：逐 D chunk 累加 QK，再逐输出 D tile 做 PV，标出不可提前 softmax。
- **完整代码 demo**：Split-D attention：完整点积和输出分块的逐阶段结果校验。
- **Triton 练习**：D 分块点积后再 softmax；多输出 D tile。
- **TileLang 练习**：QK 与 PV 使用不同 fragment 形状。
- **通过条件**：先完成完整点积再做 softmax；比较 D=64/128/256/非整除。

## 布局代数与 CuTe 思维迁移

前置：地址/stride、tile、MMA 基础；先用坐标枚举验证映射，再组合 Copy 与 MMA。

### （二十二）Layout：坐标、地址与可逆映射

**状态：待展开；学习任务已梳理。** 从 shape/stride 到 composition、divide 与 product。

- **交互演示**：编辑 shape/stride 并点击坐标，显示 offset、重复地址和覆盖。
- **完整代码 demo**：纯 Python layout 枚举与 GPU view/copy 对照，区分逻辑视图与搬运。
- **Triton 练习**：构造同逻辑张量的连续与转置 view。
- **TileLang 练习**：可视化 layout；核对 element 到 thread/value 映射。
- **通过条件**：Python 小模型穷举地址；分清逻辑 reshape 与真实搬运。

### （二十三）Copy 的归属：谁搬什么，搬到哪里

**状态：待展开；学习任务已梳理。** Tensor 视图与 TiledCopy 的 thread/value 划分。

- **交互演示**：选 thread/value 显示其源和目标坐标，切换向量宽度与尾部。
- **完整代码 demo**：Copy 映射枚举与 global→shared→global round-trip。
- **Triton 练习**：不同 tile 载入映射，跟踪 mask 与布局。
- **TileLang 练习**：T.copy 的 global/shared/fragment 路径。
- **通过条件**：覆盖无遗漏、无非法重叠；对齐、向量宽度和尾部都可解释。

### （二十四）MMA 的归属：从原子到输出 fragment

**状态：待展开；学习任务已梳理。** TiledMMA 分层与 A/B/C 的不同坐标映射。

- **交互演示**：从 MMA atom 展开到 tile，点击输出位置追踪拥有者。
- **完整代码 demo**：A/B/C fragment 元素计数与小矩阵乘，校验布局归属。
- **Triton 练习**：dot 两端布局变化及编译器中间表示。
- **TileLang 练习**：T.gemm 片段分配和 accumulator 可视化。
- **通过条件**：元素计数守恒；点积输入布局与输出布局不能直接混用。

### （二十五）Swizzle 与 TMA 的地址契约

**状态：待展开；学习任务已梳理。** 软件访问地址必须匹配 descriptor 生成的布局。

- **交互演示**：分别显示 descriptor 和软件生成的地址，故意错配定位错误。
- **完整代码 demo**：swizzle round-trip 与目标设备 TMA copy 自检，先不接 MMA。
- **Triton 练习**：descriptor 路径与普通 load 的等价验证。
- **TileLang 练习**：显式 swizzle 与 TMA copy 配对实验。
- **通过条件**：先小 tile round-trip，再接 MMA；分别核验对齐、base 与 phase。

### （二十六）布局驱动的 GEMM 工程

**状态：待展开；学习任务已梳理。** 把数据布局、搬运划分、计算划分组合起来。

- **交互演示**：联动显示数据布局、copy 分工、MMA tile 和 epilogue。
- **完整代码 demo**：含 epilogue 的 GEMM：逐层启用搬运/计算/写回优化并校验。
- **Triton 练习**：有 epilogue 的矩阵乘，完整基线对照。
- **TileLang 练习**：共享内存布局 + 流水 + 向量写回。
- **通过条件**：核心、epilogue 与尾块逐层校验，源码/配置/版本一起保存。

### （二十七）Attention 实现对照：统一流水、角色分工与双 consumer

**状态：待展开；学习任务已梳理。** 同一数学接口下比较不同数据流组织。

- **交互演示**：同一输入切换统一流水、角色分工和双 consumer 的协议时间线。
- **完整代码 demo**：同一 attention 接口的多个实现与统一正确性/计时入口。
- **Triton 练习**：对同一输入公平比较普通与 specialized 路径。
- **TileLang 练习**：对同一输入比较 cp.async / TMA 方案。
- **通过条件**：统一 mask、精度、计时范围；不把框架差异都归因于某一条指令。

### （二十八）大 D 的布局代数与寄存器预算

**状态：待展开；学习任务已梳理。** QK/PV 两套 tile 与中间 P 的布局衔接。

- **交互演示**：选择 D chunk 和输出 tile，显示片段计数及同时存活的状态。
- **完整代码 demo**：Split-D 布局小模型与 GPU attention，比较寄存器、spill 和误差。
- **Triton 练习**：不同 D 分块布局的 equivalent kernel。
- **TileLang 练习**：双 GEMM 片段与寄存器生命周期实验。
- **通过条件**：手算元素归属，检查 spill 与 shared；禁用不能证明的 reinterpret。

## 架构专用调度与完整流水

前置：Attention、布局代数、异步完成协议；需明确目标 GPU 能力。

### （二十九）Persistent Attention：任务循环与尾波

**状态：待展开；学习任务已梳理。** CTA 常驻循环、linear task assignment、条件重缩放。

- **交互演示**：切换普通 grid 与 persistent 分配，显示任务覆盖和尾波。
- **完整代码 demo**：persistent task-mapping 检查与 attention，对照普通 grid。
- **Triton 练习**：persistent work queue 与普通 grid 对照。
- **TileLang 练习**：persistent attention 及 epilogue 路径。
- **通过条件**：任务无漏算/重算，短长序列都测；比较整段时间，不能预设胜过厂商库。

### （三十）大 D 的 non-WS 流水：让所有线程参与 MMA

**状态：待展开；学习任务已梳理。** chunk 解耦、跨 tile 预取、独立 barrier 集。

- **交互演示**：调独立 K/V stages 与 chunk，观察全员计算和预取依赖。
- **完整代码 demo**：non-WS Split-D 协议模拟与目标 GPU 实现，独立调参并计时。
- **Triton 练习**：全员计算路径与分工路径的调参对照。
- **TileLang 练习**：独立 K/V stages 与 non-WS 编排。
- **通过条件**：先画 buffer 生命周期，再测寄存器/等待/时间；更多 stage 不保证更快。

## 低精度矩阵乘：格式、scale 与误差

前置：GEMM、数值误差和布局；先软件量化参考，再硬件低精度乘法。

### （三十一）FP8 的格式与量化前处理

**状态：待展开；学习任务已梳理。** e4m3、动态范围、per-row/per-block scale。

- **交互演示**：调整输入、scale 与量化粒度，显示可表示值、饱和和重建误差。
- **完整代码 demo**：FP8 软件量化参考与前处理 GPU kernel，覆盖零块和离群点。
- **Triton 练习**：amax 与缩放量化；零块和离群点。
- **TileLang 练习**：归约 + 转换融合；scale 排布。
- **通过条件**：重建误差与饱和计数；明确舍入和 scale 方向，先不要求 Tensor Core。

### （三十二）FP8 GEMM：scale 如何进入累加

**状态：待展开；学习任务已梳理。** 在线反量化、缩放粒度和矩阵乘等价式。

- **交互演示**：展示 per-row/per-K-block scale 在乘法与求和中的位置。
- **完整代码 demo**：FP8 GEMM：反量化参考与受支持硬件路径，计入前处理成本。
- **Triton 练习**：FP8 dot 与 scale 应用。
- **TileLang 练习**：FP8 GEMM + scale epilogue。
- **通过条件**：与反量化参考对齐，前处理成本计入端到端；不同 K scale 不能直接外提。

### （三十三）NVFP4：两级 scale 与打包

**状态：待展开；学习任务已梳理。** 4-bit 线格式、global/block scale、误差来源。

- **交互演示**：修改 nibble、block scale 和全局 scale，观察打包字节与重建值。
- **完整代码 demo**：NVFP4 软件编码/解码与 GPU 打包，逐字节和数值双校验。
- **Triton 练习**：软件量化/打包参考；支持时 scaled dot。
- **TileLang 练习**：FP4 packing 与 scale layout。
- **通过条件**：逐 byte round-trip；零值、极值、奇数长度和饱和策略均有定义。

### （三十四）Block-scaled MMA：scale 也是操作数

**状态：待展开；学习任务已梳理。** scale fragment、数据布局与流水的协同。

- **交互演示**：点击数据 tile 查看对应 scale fragment，切换 layout 检查错配。
- **完整代码 demo**：block-scaled matmul：先验证数据/scale 映射，再与高精度参考比较。
- **Triton 练习**：目标架构支持的 block-scaled matmul。
- **TileLang 练习**：数据和 scale 的协作加载。
- **通过条件**：每个 scale 与正确数据块对应；对齐厂商库数值约定后再测。

## 量化注意力与最终验收

前置：稳定 Attention、FP8/FP4 格式与 scale；逐阶段定位误差，再看端到端收益。

### （三十五）量化 Attention：误差在哪里被放大

**状态：待展开；学习任务已梳理。** Q/K/V/P 的误差传播、平滑与不变性。

- **交互演示**：分别扰动 Q/K/V/P，显示 logits、概率、输出误差如何变化。
- **完整代码 demo**：逐阶段量化 attention 参考，记录多形状和长序列误差。
- **Triton 练习**：高精度参考对照分步量化误差。
- **TileLang 练习**：按阶段插入量化并保留误差日志。
- **通过条件**：区分 logits、概率、输出误差；极端分布/长序列都要覆盖。

### （三十六）FP8 Attention 的前处理链

**状态：待展开；学习任务已梳理。** 融合、平滑、量化及 scale 生成。

- **交互演示**：切换分离/融合前处理图，显示中间张量与逻辑流量。
- **完整代码 demo**：FP8 前处理链：逐中间量正确性和端到端时间对照。
- **Triton 练习**：融合前处理与多 kernel 基线。
- **TileLang 练习**：归约、转置与量化的 tile 融合。
- **通过条件**：各中间量均可核验，先测端到端占比再优化主 kernel。

### （三十七）FP8 persist-D 与 scale 折叠

**状态：待展开；学习任务已梳理。** 保持 D 的注意力流水与合法的 scale 外提。

- **交互演示**：移动 scale 的位置并检查等价条件，观察在线状态的变化。
- **完整代码 demo**：persist-D 量化 attention：scale 折叠前后数值等价验证。
- **Triton 练习**：固定 D 的量化 attention 与反量化参考。
- **TileLang 练习**：scale 融入 QK / PV 与输出修正。
- **通过条件**：证明每个外提条件，动态 range 与不同 shape 下误差达标。

### （三十八）FP8 大 D：Split-D 与并行布局

**状态：待展开；学习任务已梳理。** 工作量增加与寄存器增长是不同瓶颈。

- **交互演示**：调 D 与 warp 排布，分别显示任务数、累加器和资源预算。
- **完整代码 demo**：FP8 大 D attention：布局/分块 A/B，记录误差与编译资源。
- **Triton 练习**：大 D 的分块与任务划分对照。
- **TileLang 练习**：M/N 方向的 warp 排布与流水调整。
- **通过条件**：并行度、spill 与耗时分别记录；不能仅看 occupancy。

### （三十九）FP4 Attention 的量化链

**状态：待展开；学习任务已梳理。** NVFP4 格式在 Q/K/V 预处理中的应用。

- **交互演示**：逐步走过 Q/K/V 的两级量化和打包，显示 scale 与数据绑定。
- **完整代码 demo**：NVFP4 attention 前处理：软件参考、GPU 转换和完整校验。
- **Triton 练习**：软件 packed 参考与完整前处理。
- **TileLang 练习**：两级 scale 生成和打包融合。
- **通过条件**：数据/scale 同时对齐，精度与转换成本单独报告。

### （四十）FP4 Attention：P 的两级量化

**状态：待展开；学习任务已梳理。** online 状态、P 量化与输出缩放的交互。

- **交互演示**：逐 KV tile 显示 P 两级量化与 O 重缩放，观察误差累积。
- **完整代码 demo**：FP4 PV/attention：逐阶段参考、极端输入与长序列验证。
- **Triton 练习**：量化 P 的逐阶段误差定位。
- **TileLang 练习**：block-scaled PV 与 rescale 的接口。
- **通过条件**：各 KV tile 后验证状态；短序列正确不能代替长序列稳定性。

### （四十一）基准与精度：把学习结果变成可复现实验

**状态：待展开；学习任务已梳理。** 环境、计时、缓存、误差统计和生态约束。

- **交互演示**：切换统计口径、工作集和误差阈值，观察比较结论如何改变。
- **完整代码 demo**：统一 benchmark/accuracy runner：导出环境、误差分布、时间与失败案例。
- **Triton 练习**：统一测试 harness 与多形状报表。
- **TileLang 练习**：相同 harness 跑等价语义实现。
- **通过条件**：同时交付正确性、精度分布、端到端时间、条件和失败案例。

## 材料范围与持续维护

依据本地提供的 `leetcuda-20261029.pdf`（591 页）梳理：基础并行、GEMM、Attention、CuTe、低精度矩阵乘、量化注意力均已映射到上述主题。CuTe 布局白皮书译注作为布局主题的补充；工具箱、构建说明与实验汇总并入环境准备和验收，不另造一套编号。

教程使用重新组织的讲解和原创图。参考资料中的特定设备实测、经验阈值和专用实现会明确标注来源与适用条件；本站没有运行过的 GPU 实验不写成实测结论。每篇完整资料会保留原文主题或 PDF 阅读器页码，便于返回原文核对。

<link rel="stylesheet" href="/css/cuda-learning.css">
