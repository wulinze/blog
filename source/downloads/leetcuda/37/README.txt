CUDA 算子学习（三十八）：FP8 大 D：Split-D 与并行布局
来源：leetcuda-20261029.pdf，第 535–543 页。

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

python run_gpu.py --chapter 37 --backend triton
python run_gpu.py --chapter 37 --backend tilelang
python run_gpu.py --chapter 37 --backend triton --student student_triton
python run_gpu.py --chapter 37 --backend tilelang --student student_tilelang
python run_gpu.py --chapter 37 --backend triton --benchmark --output result.json

基线范围：D=64/128/256/512 的分块 QK attention；M4N2 排布由专用 CUDA 路径提供。
参考实现使用共享原语组合，包含框架编排和若干独立 kernel，不能当成已优化的 fused kernel。
--benchmark 报整个语义基线组合的设备事件时间，包含必要清零和中间操作，不是主 kernel-only。
输入必须为测试契约中的有限 float32/合法索引；具体整数 histogram 是 int32。
GPU 不可用退出 77，明确 SKIP；不静默改跑 CPU。学生未实现会失败。
FP4 DSL 基线是 E2M1+FP32 scale；CPU demo 提供 FP8 SF 舍入、pack/unpack；真实 NVFP4 见下方。

架构专用完整源码与测试：https://github.com/vipshop/ffpa-attn/blob/861d75ee87824a14a9ebfe295d54c7dac3e3cabc/csrc/cuffpa/cute/fp8/sm_120/split_d_m4n2.cuh
Linux NVIDIA SM120 环境；先按固定版本 README 准备 CUDA PyTorch、Python≥3.10、构建依赖和 CUDA 扩展。此入口运行整条量化 attention，前处理、主循环、误差报告一起观察。大 D 路径是否编入以构建的 headdim 配置为准。
以下命令在本解压目录执行，新建独立上游 checkout：
git clone https://github.com/vipshop/ffpa-attn.git ffpa-attn-lab
git -C ffpa-attn-lab checkout --detach 861d75ee87824a14a9ebfe295d54c7dac3e3cabc
git -C ffpa-attn-lab submodule update --init --recursive
cd ffpa-attn-lab
# 在已装好匹配 CUDA PyTorch 的隔离环境中安装项目依赖
python -m pip install packaging ninja wheel setuptools setuptools-scm matplotlib
FFPA_SKIP_CUDA_EXT=1 python -m pip install -e ".[bench]" --no-build-isolation
bash build.sh --arch sm_120a --ext all --headdim 128,512 --jobs 4
cd ..
python3 run_upstream.py --repo ./ffpa-attn-lab --arch sm_120a

run_upstream.py 检查完整提交号；SKIP 不算 GPU PASS。
原作者 CUDA 代码未复制进此压缩包；通过固定版本 checkout 获得完整依赖与许可。
阅读上游 LICENSE；博客讲解和 CPU/DSL 教学代码为本学习资料新增。
