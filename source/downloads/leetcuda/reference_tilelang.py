"""Simple TileLang semantic baselines. mm uses scalar FMA, not TMA/MMA.

Advanced optimized CUDA implementations are provided as pinned upstream labs.
"""

import tilelang
import tilelang.language as T
import torch


@tilelang.jit
def _make_row(rows: int, n: int, block: int, kind: int):
    @T.prim_func
    def kernel(X: T.Tensor((rows, n), "float32"), Y: T.Tensor((rows, n), "float32")):
        with T.Kernel(rows, threads=128) as row:
            x = T.alloc_fragment((1, block), "float32")
            work = T.alloc_fragment((1, block), "float32")
            stat = T.alloc_fragment((1,), "float32")
            mean = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(block):
                x[0, j] = T.if_then_else(j < n, X[row, j], 0)
            if kind == 0:
                for j in T.Parallel(block):
                    work[0, j] = T.if_then_else(j < n, x[0, j], -T.infinity("float32"))
                T.reduce_max(work, stat, dim=1, clear=True)
                for j in T.Parallel(block):
                    work[0, j] = T.exp(work[0, j] - stat[0])
                T.reduce_sum(work, stat, dim=1, clear=True)
                for j in T.Parallel(block):
                    if j < n:
                        Y[row, j] = work[0, j] / stat[0]
            elif kind == 1:
                for j in T.Parallel(block):
                    work[0, j] = x[0, j] * x[0, j]
                T.reduce_sum(work, stat, dim=1, clear=True)
                for j in T.Parallel(block):
                    if j < n:
                        Y[row, j] = x[0, j] * T.rsqrt(stat[0] / n + 1e-5)
            elif kind == 2 or kind == 3:
                T.reduce_sum(x, mean, dim=1, clear=True)
                for j in T.Parallel(block):
                    work[0, j] = T.if_then_else(j < n, (x[0, j] - mean[0] / n) * (x[0, j] - mean[0] / n), 0)
                T.reduce_sum(work, stat, dim=1, clear=True)
                for j in T.Parallel(block):
                    if j < n:
                        if kind == 2:
                            Y[row, j] = (x[0, j] - mean[0] / n) * T.rsqrt(stat[0] / n + 1e-5)
                        else:
                            Y[row, j] = x[0, j] - mean[0] / n
            else:
                for j in T.Parallel(block):
                    work[0, j] = T.abs(x[0, j])
                T.reduce_max(work, stat, dim=1, clear=True)
                for j in T.Parallel(block):
                    if j < n:
                        if kind == 4:
                            scale = T.if_then_else(stat[0] > 0, stat[0] / 448, 1.0)
                            Y[row, j] = T.cast(T.cast(x[0, j] / scale, "float8_e4m3fn"), "float32") * scale
                        else:
                            scale = T.if_then_else(stat[0] > 0, stat[0] / 6, 1.0)
                            a = T.abs(x[0, j] / scale)
                            q = T.if_then_else(
                                a <= 0.25,
                                0.0,
                                T.if_then_else(
                                    a < 0.75,
                                    0.5,
                                    T.if_then_else(
                                        a <= 1.25,
                                        1.0,
                                        T.if_then_else(
                                            a < 1.75,
                                            1.5,
                                            T.if_then_else(
                                                a <= 2.5,
                                                2.0,
                                                T.if_then_else(a < 3.5, 3.0, T.if_then_else(a <= 5.0, 4.0, 6.0)),
                                            ),
                                        ),
                                    ),
                                ),
                            )
                            Y[row, j] = T.if_then_else(x[0, j] < 0, -q, q) * scale

    return kernel


def row(x, kind):
    x = x.contiguous()
    out = torch.empty_like(x)
    block = 1 << (x.shape[1] - 1).bit_length()
    _make_row(x.shape[0], x.shape[1], block, kind)(x, out)
    return out


@tilelang.jit
def _make_mm(m: int, n: int, k: int):
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float32"), B: T.Tensor((k, n), "float32"), C: T.Tensor((m, n), "float32")):
        with T.Kernel(T.ceildiv(m, 16), T.ceildiv(n, 16), threads=128) as (bm, bn):
            acc = T.alloc_fragment((16, 16), "float32")
            T.clear(acc)
            for kk in T.serial(k):
                for i, j in T.Parallel(16, 16):
                    if bm * 16 + i < m and bn * 16 + j < n:
                        acc[i, j] += A[bm * 16 + i, kk] * B[kk, bn * 16 + j]
            for i, j in T.Parallel(16, 16):
                if bm * 16 + i < m and bn * 16 + j < n:
                    C[bm * 16 + i, bn * 16 + j] = acc[i, j]

    return kernel


def mm(a, b):
    a, b = a.contiguous(), b.contiguous()
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
    _make_mm(a.shape[0], b.shape[1], a.shape[1])(a, b, out)
    return out


@tilelang.jit
def _make_gather(size: int, n: int):
    @T.prim_func
    def kernel(X: T.Tensor((size,), "float32"), Indices: T.Tensor((n,), "int64"), Y: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 128), threads=128) as pid:
            for j in T.Parallel(128):
                i = pid * 128 + j
                if i < n:
                    Y[i] = X[Indices[i]]

    return kernel


def gather(x, index):
    x, index = x.contiguous().flatten(), index.contiguous()
    out = torch.empty(index.shape, device=x.device, dtype=x.dtype)
    _make_gather(x.numel(), index.numel())(x, index.flatten(), out.flatten())
    return out


@tilelang.jit
def _make_hist(n: int, bins: int):
    @T.prim_func
    def kernel(X: T.Tensor((n,), "int32"), Y: T.Tensor((bins,), "int32")):
        with T.Kernel(T.ceildiv(n, 128), threads=128) as pid:
            for j in T.Parallel(128):
                i = pid * 128 + j
                if i < n:
                    T.atomic_add(Y[X[i]], 1)

    return kernel


def histogram(x, bins):
    out = torch.zeros(bins, device=x.device, dtype=torch.int32)
    _make_hist(x.numel(), bins)(x, out)
    return out
