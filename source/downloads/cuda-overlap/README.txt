CUDA 性能学习（二）配套程序
https://wulinze.github.io/systems/cuda-overlap/

文件
  common.cuh       共享错误检查、CUDA event 计时和 CPU 参考校验
  coalescing.cu    相同 copy 工作的跨 lane 地址排列 vs 连续地址
  vectorized.cu    scalar copy vs float4 copy，处理非 4 倍数尾部
  tiling.cu        naive FP32 GEMM vs 16x16 shared tiled GEMM，处理尾块
  occupancy.cu    查询编译后 kernel 资源与理论驻留容量，不启动 kernel
  Makefile        编译与边界案例校验

验证状态（2026-11-02）
  当前作者环境无 NVIDIA GPU / nvcc，尚未编译或在 GPU 实跑。
  这是完整 CUDA 源码，不是已测得性能的报告。
  网页交互是概念模型，不调用 CUDA。

环境与编译
  NVIDIA GPU + 匹配驱动 + CUDA Toolkit + C++17 host 编译器 + make。
  这些程序不使用 cp.async / TMA，不要求 Hopper 专属硬件。
  将 ARCH 改为目标 GPU 且当前 Toolkit 支持的架构：
    make ARCH=sm_80
    make check ARCH=sm_80
  Hopper 示例：make ARCH=sm_90
  单独编译：nvcc -O3 -std=c++17 -lineinfo -arch=sm_80 vectorized.cu -o vectorized
  common.cuh 必须与 .cu 位于同一目录。

运行
  ./coalescing 16777216
  ./vectorized 16777219
  ./tiling 256 256 256
  ./occupancy
  coalescing 的 N 必须是 1024 的正整数倍；代码会拒绝不支持的长度。
  vectorized 支持任意正 N，cudaMalloc 的基址满足 float4 对齐；不要随意偏移指针。
  tiling 参数为 M N K，均为 1..1024，缺省 129 193 257；CPU 参考在大矩阵上会耗时。
  copy 示例 N 最大为 2^28；自行按显存和主机内存容量选择较小值。

校验和诊断
  每个基线/优化版本计时前先运行并校验，PASS 表示与 CPU 参考匹配。
  CUDA 出错或数值不匹配会返回非零；occupancy 只报告理论容量。
  make check 包含极小 N、向量尾部、非方阵和 tile 尾块。
  可进一步运行：
    compute-sanitizer --tool memcheck ./vectorized 1027
    compute-sanitizer --tool memcheck ./tiling 17 19 23
    compute-sanitizer --tool racecheck ./tiling 17 19 23
  nvcc -Xptxas=-v 可查看寄存器等信息；cuobjdump --dump-sass vectorized 可检查访存指令。

计时口径
  5 次 warmup，20 次重复，CUDA event 计时取每次平均；固定 default stream。
  不包含分配、CPU 参考、H2D/D2H。前后版本使用相同输入与逻辑输出。
  反复访问同一工作集可能命中 L2 等缓存；打印 effective GB/s 不是 HBM 带宽。
  effective GB/s = 读写逻辑字节 / 秒 / 10^9。
  effective GFLOP/s = 2*M*N*K / 秒 / 10^9；尾块的额外执行不计入有效工作。
  小尺寸可能被 launch 开销主导；测试顺序、缓存、时钟、其他进程都可能影响结果。
  没有保证优化版总更快，更不能用理论 sector 或指令数比推断耗时倍数。

TMA 说明
  正文 TMA 部分是描述符、phase、transaction barrier、proxy 与 slot 生命周期的协议解释。
  它不是可直接编译的 TMA demo；需要按目标 Toolkit 的 NVIDIA 官方完整示例实现。
