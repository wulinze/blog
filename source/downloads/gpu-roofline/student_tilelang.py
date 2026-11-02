"""Exercise entry point; same numerical contract as student_triton.py."""
# Import tilelang and tilelang.language as T.
# L1: T.Kernel partitions n; T.Parallel(block) covers a tile; guard offset < n.
# L2: fuse 1.25*x + y; inspect generated source with get_kernel_source().
# workers means physical CUDA threads, independent of tile element count block.
# n=0 must return a no-op callable, not launch a zero-sized grid.


def prepare(n: int, block: int, workers: int, operation: str):
    """Return launch(x, y, out), with an explicit output buffer."""
    raise NotImplementedError("Implement L1/L2; hints and reference are in the article")
