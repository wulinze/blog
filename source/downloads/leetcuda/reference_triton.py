"""Readable GPU semantic baselines; advanced hardware kernels are linked separately."""

import torch
import triton
import triton.language as tl


@triton.jit
def _row(X, Y, N: tl.constexpr, B: tl.constexpr, KIND: tl.constexpr):
    r = tl.program_id(0)
    j = tl.arange(0, B)
    x = tl.load(X + r * N + j, j < N, 0)
    if KIND == 0:
        v = tl.where(j < N, x, -float("inf"))
        z = tl.exp(v - tl.max(v, 0))
        out = z / tl.sum(z, 0)
    elif KIND == 1:
        out = x * tl.rsqrt(tl.sum(x * x, 0) / N + 1e-5)
    elif KIND == 2:
        mean = tl.sum(x, 0) / N
        v = tl.where(j < N, x - mean, 0)
        out = v * tl.rsqrt(tl.sum(v * v, 0) / N + 1e-5)
    elif KIND == 3:
        out = x - tl.sum(x, 0) / N
    elif KIND == 4:
        amax = tl.max(tl.abs(x), 0)
        scale = tl.where(amax > 0, amax / 448, 1.0)
        out = (x / scale).to(tl.float8e4nv).to(tl.float32) * scale
    else:
        # E2M1 data quantizer with ideal FP32 scale, deliberately NOT NVFP4 SF encoding.
        amax = tl.max(tl.abs(x), 0)
        scale = tl.where(amax > 0, amax / 6, 1.0)
        a = tl.abs(x / scale)
        q = tl.where(
            a <= 0.25,
            0.0,
            tl.where(
                a < 0.75,
                0.5,
                tl.where(
                    a <= 1.25,
                    1.0,
                    tl.where(
                        a < 1.75, 1.5, tl.where(a <= 2.5, 2.0, tl.where(a < 3.5, 3.0, tl.where(a <= 5.0, 4.0, 6.0)))
                    ),
                ),
            ),
        )
        out = tl.where(x < 0, -q, q) * scale
    tl.store(Y + r * N + j, out, j < N)


def row(x, kind):
    x = x.contiguous()
    out = torch.empty_like(x)
    _row[(x.shape[0],)](x, out, x.shape[1], triton.next_power_of_2(x.shape[1]), kind)
    return out


@triton.jit
def _mm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.program_id(0) * 16 + tl.arange(0, 16)
    n = tl.program_id(1) * 16 + tl.arange(0, 16)
    k = tl.arange(0, 32)
    acc = tl.full((16, 16), 0, tl.float32)
    for start in range(tl.cdiv(K, 32)):
        kk = start * 32 + k
        a = tl.load(A + m[:, None] * K + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), 0)
        b = tl.load(B + kk[:, None] * N + n[None, :], (kk[:, None] < K) & (n[None, :] < N), 0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(C + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def mm(a, b):
    a, b = a.contiguous(), b.contiguous()
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
    _mm[(triton.cdiv(a.shape[0], 16), triton.cdiv(b.shape[1], 16))](a, b, out, a.shape[0], b.shape[1], a.shape[1])
    return out


@triton.jit
def _gather(X, Indices, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    index = tl.load(Indices + i, i < N, 0)
    x = tl.load(X + index, i < N, 0)
    tl.store(Y + i, x, i < N)


def gather(x, index):
    index = index.contiguous()
    out = torch.empty(index.shape, device=x.device, dtype=x.dtype)
    _gather[(triton.cdiv(index.numel(), 128),)](x.contiguous(), index, out, index.numel(), 128)
    return out


@triton.jit
def _hist(X, Y, N: tl.constexpr):
    i = tl.program_id(0) * 128 + tl.arange(0, 128)
    v = tl.load(X + i, i < N, 0)
    tl.atomic_add(Y + v, 1, i < N)


def histogram(x, bins):
    out = torch.zeros(bins, device=x.device, dtype=torch.int32)
    _hist[(triton.cdiv(x.numel(), 128),)](x, out, x.numel())
    return out
