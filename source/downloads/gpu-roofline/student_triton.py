"""Exercise entry point. Keep this API; the runner supplies contiguous float32 vectors."""
# Import triton and triton.language, then implement your own masked kernel.
# T1: operation="add" -> out[i] = x[i] + y[i].
# T2: operation="axpy" -> out[i] = 1.25*x[i] + y[i], without an intermediate tensor.
# n=0: return a callable that performs no device launch.
# Use workers as num_warps. Do not interpret block as a CUDA thread count.


def prepare(n: int, block: int, workers: int, operation: str):
    """Return launch(x, y, out). Compile outside timing if your backend allows it."""
    raise NotImplementedError("Implement T1/T2; hints and reference are in the article")
