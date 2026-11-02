CUDA operator learning (IV): Reduction and Dot
Article: https://wulinze.github.io/llm/cuda-reduction/
Series: https://wulinze.github.io/systems/gpu-learning-path/

Status: CPU model executed; Python syntax/static checks performed.
CUDA/Triton/TileLang backend compilation, GPU correctness and timing NOT performed
in the authoring environment. No measured GPU speedup is claimed.

Contract: contiguous one-dimensional finite float32 x/y, n>=0, output float32[1].
sum(empty)=dot(empty)=0; max(empty)=-inf. NaN/Inf inputs, strides and aliasing
are outside this exercise contract. Max must match exactly for finite inputs.
Sum/dot compare to CPU FP64 with abs_error <= 1e-6 + 2e-6*sum_abs_contributions.
This is a teaching tolerance, not a numerical error theorem; actual errors are printed.

CPU (standard library only)
  python reduction_model.py
  python reduction_model.py --self-check
Expected: XOR [1..8] reaches [36]*8; hierarchical [1..10] -> [10,26,19] -> [55].
FP32 [2^25,1,-2^25,1]: serial=1; adjacent-pair tree=0; exact sum=2.

CUDA (C++17, compatible NVIDIA GPU/Toolkit)
  nvcc -O3 -std=c++17 -lineinfo -arch=native reduction.cu -o reduction
Use a matching sm_XX if your nvcc does not support -arch=native.
  ./reduction 1003 sum two-pass
  ./reduction 1003 max two-pass
  ./reduction 1048576 dot two-pass
  ./reduction 1048576 dot atomic
  ./reduction 520 dot-vec4 two-pass
  compute-sanitizer --tool memcheck ./reduction 520 dot-vec4 two-pass
Fixed 256-thread full warps. Invalid data positions retain participating threads
and contribute neutral values. Full float4 reads are guarded by index+3<n;
remaining scalar reads are also guarded. cudaMalloc bases guarantee alignment.
Atomic demo supports sum/dot; reset is performed on EVERY call and included in timing.
Two-pass may need more than two levels. All levels are timed.
CUDA timing: 5 warmups and mean of 50 complete workflow calls. Reused data, no cache flush.

Python GPU setup: Linux, supported NVIDIA GPU/driver, CUDA PyTorch and selected backend.
Use separate environments when needed. Follow official setup instructions:
https://pytorch.org/get-started/locally/
https://triton-lang.org/main/getting-started/installation.html
https://tilelang.com/get_started/Installation.html
No untested fixed version combination is claimed.

Complete references (run each operation on each backend)
  python run_gpu.py --backend triton --operation sum --mode check
  python run_gpu.py --backend triton --operation max --mode check
  python run_gpu.py --backend triton --operation dot --mode check
  python run_gpu.py --backend tilelang --operation sum --mode check
  python run_gpu.py --backend tilelang --operation max --mode check
  python run_gpu.py --backend tilelang --operation dot --mode check

Student implementation
Implement prepare(n,block,workers,operation) returning launch(x,y,out).
Allocate partials in prepare. Keep the computation on the GPU; no CPU/.item()/torch
reduction as a substitute. Dot multiplies x*y only at the FIRST level.
workers = num_warps for Triton (4/8), CUDA threads for TileLang (128/256).
block is logical tile size (128/256/512/1024).
  python run_gpu.py --backend triton --impl student --operation dot --mode check
  python run_gpu.py --backend tilelang --impl student --operation max --mode check
Repeat for sum/max/dot. Student files explicitly fail until implemented.
Checks: empty, warp/block boundaries, 520, block^2+1 and requested n;
random/ones/all-negative/cancellation inputs; repeated calls; unchanged inputs.
For small n, references expose first-stage partials for inspection.

Benchmark the COMPLETE reduction
  python run_gpu.py --backend triton --operation dot --mode bench --n 1048576 --output triton.json
  python run_gpu.py --backend tilelang --operation dot --mode bench --n 1048576 --output tilelang.json
10 warmups, 7 batches of 50 full calls, CUDA events; report median/min/max batch mean.
JIT/allocations/checks excluded; host submission gaps may remain. No cache flush.
Input GB/s excludes partial traffic and is not measured DRAM bandwidth.
JSON records environment and numerics; empty max uses the string "-inf" in JSON files.

Profiling
  nsys profile --trace=cuda,nvtx --output=reduce-timeline python run_gpu.py --backend triton --operation dot --mode bench --n 1048576
  ncu --kernel-name regex:reduce_tiles --section LaunchStats --section MemoryWorkloadAnalysis --output=reduce-report python run_gpu.py --backend triton --operation dot --mode profile --n 65537
Profile mode runs two correctness workflows, then one extra workflow. Each has
multiple stages; identify target grid/stage before narrowing launch filters.
