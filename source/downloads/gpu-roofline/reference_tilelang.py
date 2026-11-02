"""Explicit boundary guards; workers means CUDA threads per TileLang block."""

import tilelang
import tilelang.language as T


@tilelang.jit
def make_elementwise(n: int, block: int, threads: int, axpy: bool):
    @T.prim_func
    def elementwise(X: T.Tensor((n,), "float32"), Y: T.Tensor((n,), "float32"), Out: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, block), threads=threads) as pid:
            for lane in T.Parallel(block):
                offset = pid * block + lane
                if offset < n:
                    if axpy:
                        Out[offset] = 1.25 * X[offset] + Y[offset]
                    else:
                        Out[offset] = X[offset] + Y[offset]

    return elementwise


def prepare(n: int, block: int, workers: int, operation: str):
    """Create an explicit-output callable, avoiding output allocation in timing."""
    if n == 0:
        return lambda x, y, out: None
    return make_elementwise(n, block, workers, operation == "axpy")
