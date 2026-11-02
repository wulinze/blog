"""Masked float32 elementwise baselines. GPU execution has not been verified here."""

import triton
import triton.language as tl


@triton.jit
def elementwise(X, Y, Out, N: tl.constexpr, BLOCK: tl.constexpr, AXPY: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    x = tl.load(X + offsets, mask=valid, other=0.0)
    y = tl.load(Y + offsets, mask=valid, other=0.0)
    if AXPY:
        value = 1.25 * x + y
    else:
        value = x + y
    tl.store(Out + offsets, value, mask=valid)


def prepare(n: int, block: int, workers: int, operation: str):
    """Create a launcher; workers means warps per Triton program."""

    def launch(x, y, out):
        if n:
            elementwise[(triton.cdiv(n, block),)](x, y, out, n, block, operation == "axpy", num_warps=workers)

    return launch
