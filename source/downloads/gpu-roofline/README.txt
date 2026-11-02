GPU Roofline learning lab
========================
Article: https://wulinze.github.io/systems/gpu-roofline/
Route: https://wulinze.github.io/systems/gpu-learning-path/

Validation status
- CPU model: executed and checked against hand-derived values.
- Python GPU sources: syntax checked; NOT backend-compiled or GPU-executed here.
- No GPU benchmark numbers are claimed by this package.

Environment
CPU model: Python 3.9+; standard library only.
GPU work: Linux, compatible NVIDIA GPU/driver, CUDA-enabled PyTorch,
and a supported Triton or TileLang installation. Python 3.10+ recommended.
Use separate environments if the two compilers require different dependencies.
Follow official installers; no untested universal version pin is supplied:
https://pytorch.org/get-started/locally/
https://triton-lang.org/main/getting-started/installation.html
https://tilelang.com/get_started/Installation.html
Record your actual versions; run_gpu.py includes them in --output JSON.

Start on CPU
  python roofline.py --self-check
  python roofline.py --operation add --n 1048576
  python roofline.py --operation gemm --m 4096 --n 4096 --k 4096
Default peak=60 TFLOP/s and bandwidth=1500 GB/s are hypothetical values.
The byte count is logical/ideal traffic, NOT measured DRAM traffic.

Run a complete reference (inside the appropriate GPU environment)
  python run_gpu.py --backend triton --mode check --operation add --n 1003
  python run_gpu.py --backend triton --mode check --operation axpy --n 1003
  python run_gpu.py --backend tilelang --mode check --operation add --n 1003
  python run_gpu.py --backend tilelang --mode check --operation axpy --n 1003

Exercise contract
Implement prepare(n, block, workers, operation) in student_triton.py or
student_tilelang.py. Return launch(x, y, out), writing into preallocated out.
Inputs: separate contiguous one-dimensional float32 CUDA tensors, length n.
Output: add -> x+y; axpy -> 1.25*x+y. Do not modify x or y.
Empty input: no device launch. Tail reads AND writes must be guarded.
BLOCK means logical tile size; workers is num_warps for Triton, threads for TileLang.
Do not call the reference kernel or PyTorch to produce the student output.
Student files intentionally raise NotImplementedError until completed.

  python run_gpu.py --backend triton --impl student --mode check --sweep --n 1003
  python run_gpu.py --backend tilelang --impl student --mode check --sweep --n 1003
Repeat for --operation axpy. Correctness uses rtol=1e-5, atol=1e-6.
Checks include 0,1,31,32,33,127,128,129,block-1,block,block+1,1003 and requested n.
NaN-filled output detects missing writes; input clones detect modifications.
This is a numerical harness, not a substitute for compute-sanitizer memory checking.

Benchmark and record
  python run_gpu.py --backend triton --mode bench --sweep --n 1048576 --output triton.json
  python run_gpu.py --backend tilelang --mode bench --sweep --n 1048576 --output tilelang.json
  python run_gpu.py --backend torch --mode bench --n 1048576 --output torch.json
  python run_gpu.py --backend triton --mode bench --n 16777216 --buffers 4 --output rotating.json

Timing: JIT/allocation/checks excluded; 10 warmups, 7 batches of 50 launches,
CUDA events, median/min/max of batch averages. Host submission gaps may remain.
No cache flush. Default reuses one buffer set. Requested GB/s = 12*n / time,
NOT DRAM GB/s. buffers rotates separately allocated data; larger footprint
reduces potential cache reuse but does not guarantee cold data.
Large example uses 768 MiB for 4 sets plus validation temporaries/runtime.
Torch axpy baseline is already fused; it is not the 20N-byte split implementation.
No fixed speedup requirement: an evidence-backed explanation is the goal.

Profile
  ncu --kernel-name regex:elementwise --launch-skip 1 --launch-count 1 --section SpeedOfLight --section MemoryWorkloadAnalysis --output=vector-report python run_gpu.py --backend triton --mode profile --n 1048576
Default buffers=1: one matched correctness launch followed by one profile launch.
Verify actual kernel names, supported metrics and replay/cache settings first.
  python run_gpu.py --backend tilelang --mode profile --n 1003 --dump-source generated.cu
Source export needs a TileLang object exposing get_kernel_source().

Acceptance
1. Correct output for every case and unchanged inputs.
2. Explain F, logical Q, layer-specific Q, AI and matching roofline parameters.
3. Record environment, workload, configuration, timing distribution and cache policy.
4. Compare one changed factor; investigate non-improvement instead of inventing speedups.
