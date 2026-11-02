CUDA 算子学习（二十八）：大 D 的布局代数与寄存器预算
来源：leetcuda-20261029.pdf，第 397–413 页。

CPU：Python 3.9+ 标准库，解压后运行：
python3 demo.py --case 0
python3 demo.py --case 3
python3 learning_lab.py --self-test

GPU 语义基线：Linux NVIDIA GPU；Python>=3.10；匹配 CUDA 的 PyTorch；选装 triton 或 tilelang。
PyTorch 安装 https://pytorch.org/get-started/locally/
Triton https://triton-lang.org/main/getting-started/installation.html
TileLang https://tilelang.com/get_started/installation.html
在隔离环境里安装所选 DSL（例如 python -m pip install triton 或 tilelang）。
具体版本由 run_gpu.py 写入 result.json；当前未做 GPU 后端编译/设备/性能验证。

python run_gpu.py --chapter 27 --backend triton
python run_gpu.py --chapter 27 --backend tilelang
python run_gpu.py --chapter 27 --backend triton --student student_triton
python run_gpu.py --chapter 27 --backend tilelang --student student_tilelang
python run_gpu.py --chapter 27 --backend triton --benchmark --output result.json

基线范围：小尺寸 attention 语义基线，包含明确的 top-left causal mask；各章专用流水、布局与分工见下面的固定版本 CUDA demo。
参考实现使用共享原语组合，包含框架编排和若干独立 kernel，不能当成已优化的 fused kernel。
--benchmark 报整个语义基线组合的设备事件时间，包含必要清零和中间操作，不是主 kernel-only。
输入必须为测试契约中的有限 float32/合法索引；具体整数 histogram 是 int32。
GPU 不可用退出 77，明确 SKIP；不静默改跑 CPU。学生未实现会失败。
FP4 DSL 基线是 E2M1+FP32 scale；CPU demo 提供 FP8 SF 舍入、pack/unpack；真实 NVFP4 见下方。

架构专用完整源码与测试：https://github.com/xlite-dev/LeetCUDA/blob/3b52c5cd3332c5e992c7aef63087d43f67f23b8e/kernels/interview/book/tests/ch26_ffpa_cute.cu
CUDA C++20、NVCC、对应目标 GPU 与 CUTLASS submodule。TMA 相关例子还需满足该版本头文件的 CUDA toolkit 要求。
以下命令在本解压目录执行，新建独立上游 checkout：
git clone https://github.com/xlite-dev/LeetCUDA.git LeetCUDA-lab
git -C LeetCUDA-lab checkout --detach 3b52c5cd3332c5e992c7aef63087d43f67f23b8e
git -C LeetCUDA-lab submodule update --init --recursive
python3 run_upstream.py --repo ./LeetCUDA-lab --arch sm_80

run_upstream.py 检查完整提交号；SKIP 不算 GPU PASS。
原作者 CUDA 代码未复制进此压缩包；通过固定版本 checkout 获得完整依赖与许可。
阅读上游 LICENSE；博客讲解和 CPU/DSL 教学代码为本学习资料新增。
