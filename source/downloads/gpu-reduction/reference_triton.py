"""Hierarchical float32 sum/max/dot; no global atomic accumulator."""

import torch
import triton
import triton.language as tl


@triton.jit
def reduce_tiles(X, Y, Out, N: tl.constexpr, BLOCK: tl.constexpr, OP: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if OP == 1:
        values = tl.load(X + offsets, offsets < N, other=-float("inf"))
        total = tl.max(values, axis=0)
    else:
        values = tl.load(X + offsets, offsets < N, other=0.0)
        if OP == 2:
            values *= tl.load(Y + offsets, offsets < N, other=0.0)
        total = tl.sum(values, axis=0)
    tl.store(Out + tl.program_id(0), total)


def prepare(n: int, block: int, workers: int, operation: str):
    """Allocate partial buffers once and return launch(x, y, out[1])."""
    op = {"sum": 0, "max": 1, "dot": 2}[operation]
    lengths, partials = [], []
    length = n
    while length > 0:
        lengths.append(length)
        count = triton.cdiv(length, block)
        if count == 1:
            break
        partials.append(torch.empty(count, device="cuda", dtype=torch.float32))
        length = count

    def launch(x, y, out):
        if n == 0:
            out.fill_(-float("inf") if op == 1 else 0.0)
            return
        source = x
        for level, count in enumerate(lengths):
            target = partials[level] if level < len(partials) else out
            current_op = op if level == 0 else (1 if op == 1 else 0)
            reduce_tiles[(triton.cdiv(count, block),)](source, y, target, count, block, current_op, num_warps=workers)
            source = target

    launch.partials = partials
    return launch
