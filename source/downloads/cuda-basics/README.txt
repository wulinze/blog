CUDA 基础学习 demo

配套笔记：https://wulinze.github.io/systems/cuda-basics/

运行条件：受支持的 NVIDIA GPU、匹配的驱动、CUDA Toolkit（含 nvcc）及 host C++ 编译器。
这些 CUDA 示例尚未在本次工作中实际编译或 GPU 运行；下列值是按程序语义推导的预期结果。
示例用于理解概念，不是 benchmark；出错时立即报告并退出。

在解压目录执行：
nvcc -std=c++17 vector-add.cu -o vector-add
./vector-add
预期：PASS: 10 elements; C[9] = 27

nvcc -std=c++17 thread-layout.cu -o thread-layout
./thread-layout
预期：按全局索引打印 80 条记录，最后 PASS。
block 0 的 thread 39 属于本块 warp 1、lane 7；block 1 的 thread 0 重新从本块 warp 0 开始。
结果按 CPU 数组下标打印，不表示 GPU 按这个顺序执行。

nvcc -std=c++17 shared-rotate.cu -o shared-rotate
./shared-rotate
预期：1 2 3 0 5 6 7 4，然后 PASS。
该 kernel 固定使用 <<<2, 4>>>；不可单独改变启动尺寸而不修改内存和边界处理。
不要把删除 __syncthreads() 后偶然得到正确输出当作正确性证明。

nvcc -std=c++17 stream-order.cu -o stream-order
./stream-order
预期：PASS: result = 15。
producer stream 写 5；event 建立跨 stream 依赖；consumer stream 乘 3 并拷回。
该例演示先后依赖，不演示加速比或并发收益。

若 nvcc 的默认目标架构与 GPU 不匹配，请按照当前 Toolkit 支持范围设置适当的 -arch 选项。
