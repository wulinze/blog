"""Fragment-based hierarchical reduction with explicit neutral values."""

import tilelang
import tilelang.language as T
import torch


@tilelang.jit
def make_reduce(n: int, block: int, threads: int, op: int):
    groups = (n + block - 1) // block

    @T.prim_func
    def reduce_tiles(X: T.Tensor((n,), "float32"), Y: T.Tensor((n,), "float32"), Out: T.Tensor((groups,), "float32")):
        with T.Kernel(groups, threads=threads) as pid:
            values = T.alloc_fragment((1, block), "float32")
            total = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(block):
                offset = pid * block + j
                if offset < n:
                    if op == 2:
                        values[0, j] = X[offset] * Y[offset]
                    else:
                        values[0, j] = X[offset]
                else:
                    if op == 1:
                        values[0, j] = -T.infinity("float32")
                    else:
                        values[0, j] = 0.0
            if op == 1:
                T.reduce_max(values, total, dim=1, clear=True)
            else:
                T.reduce_sum(values, total, dim=1, clear=True)
            for i in T.Parallel(1):
                Out[pid] = total[i]

    return reduce_tiles


def prepare(n: int, block: int, workers: int, operation: str):
    """Create all levels before timing; y is only used for the first dot stage."""
    op = {"sum": 0, "max": 1, "dot": 2}[operation]
    kernels, partials = [], []
    length = n
    while length > 0:
        current_op = op if not kernels else (1 if op == 1 else 0)
        kernels.append(make_reduce(length, block, workers, current_op))
        count = (length + block - 1) // block
        if count == 1:
            break
        partials.append(torch.empty(count, device="cuda", dtype=torch.float32))
        length = count

    def launch(x, y, out):
        if n == 0:
            out.fill_(-float("inf") if op == 1 else 0.0)
            return
        source = x
        for level, kernel in enumerate(kernels):
            target = partials[level] if level < len(partials) else out
            kernel(source, y if level == 0 else source, target)
            source = target

    launch.partials = partials
    return launch
