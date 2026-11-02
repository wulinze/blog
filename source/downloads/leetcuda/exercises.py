"""Shared input contracts, readable DSL compositions, independent PyTorch references.

DSL baselines are intentionally small and unfused. They do not claim to implement
TMA/WGMMA/persistent scheduling or hardware NVFP4. See each article's pinned CUDA lab.
"""

import math

import torch

LAYOUT_CHAPTERS = {8, 13, 21, 22, 24, 28, 29}
MATMUL_CHAPTERS = {9, 10, 11, 12, 14, 15, 23, 25}


def make_problem(chapter, variant=0, device="cuda"):
    """Finite float32 inputs; contiguous storage; sizes deliberately include tails."""
    if chapter not in range(4, 41) or variant not in range(4):
        raise ValueError("chapter 4..40 and variant 0..3 required")
    generator = torch.Generator().manual_seed(20261102 + variant)
    m, k, n = [(1, 1, 1), (3, 17, 5), (17, 33, 19), (7, 65, 9)][variant]
    if chapter == 37:
        k = [64, 128, 256, 512][variant]

    def random(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)

    p = {"chapter": chapter, "variant": variant, "x": random((m, k)), "b": random((k, n))}
    if variant == 0:
        p["x"].zero_()
    p["keys"] = random((n, k))
    p["values"] = random((n, 5))
    p["bias"] = random((n,))
    if chapter == 4:
        p["ids"] = torch.tensor(
            [i % 7 if variant < 2 else 0 for i in range([1, 17, 129, 257][variant])], dtype=torch.int32, device=device
        )
    if chapter in LAYOUT_CHAPTERS:
        # Each chapter owns a different concrete address mapping.
        if chapter == 8:
            index = torch.arange(m * k).reshape(m, k).T.flatten()
        elif chapter == 13 or chapter == 24:
            p["x"] = torch.arange(64, dtype=torch.float32, device=device).reshape(8, 8)
            index = torch.tensor([r * 8 + (c ^ r) for r in range(8) for c in range(8)])
        elif chapter == 21:
            p["x"] = torch.arange(15, dtype=torch.float32, device=device).reshape(3, 5)
            index = torch.tensor([r * 5 + c for r in range(3) for c in range(4)])
        elif chapter == 22:
            index = torch.arange(m * k - 1, -1, -1)  # bijective copy with a tail
        elif chapter == 28:
            index = torch.tensor([i for worker in range(3) for i in range(worker, m * k, 3)])
        else:
            index = torch.tensor([i for phase in range(2) for i in range(phase, m * k, 2)])
        p["index"] = index.to(device=device, dtype=torch.int64)
    return p


def attention_scores(p, backend):
    x, keys = p["x"], p["keys"]
    chapter = p["chapter"]
    if chapter in {20, 27, 29, 37}:
        scores = torch.zeros((x.shape[0], keys.shape[0]), device=x.device)
        for start in range(0, x.shape[1], 16):
            scores = scores + backend.mm(x[:, start : start + 16], keys[:, start : start + 16].T)
    elif chapter == 35:
        scores = backend.mm(x, backend.row(keys.T, 3))
    elif chapter == 38:
        qmean = x.mean(dim=0, keepdim=True)
        kcenter = backend.row(keys.T, 3).T
        scores = backend.mm(x - qmean, kcenter.T) + backend.mm(qmean, kcenter.T)
    else:
        scores = backend.mm(x, keys.T)
    scores = scores / math.sqrt(x.shape[1])
    if p["variant"] % 2:
        # Explicit top-left rectangular causal convention; every row has key 0.
        i = torch.arange(x.shape[0], device=x.device)[:, None]
        j = torch.arange(keys.shape[0], device=x.device)[None, :]
        scores = scores.masked_fill(j > i, -float("inf"))
    return scores


def run_reference(p, backend):
    c = p["chapter"]
    x = p["x"]
    b = p["b"]
    if c == 4:
        return backend.histogram(p["ids"], 7)
    if c in LAYOUT_CHAPTERS:
        return backend.gather(x.flatten(), p["index"])
    if c == 5:
        return backend.row(x, 0)
    if c == 7:
        return backend.row(x, 1 if p["variant"] % 2 == 0 else 2)
    if c in MATMUL_CHAPTERS:
        if c == 9:
            b = b[:, :1]
        out = backend.mm(x, b)
        if c == 25:
            out = (out + p["bias"]).clamp_min(0)
        return out
    if c in {30, 32}:
        return backend.row(x, 4 if c == 30 else 5)
    if c in {31, 33}:
        # Per-K-block dequantized baseline. Scale is applied INSIDE each partial.
        out = torch.zeros((x.shape[0], b.shape[1]), device=x.device)
        kind = 4 if c == 31 else 5
        for start in range(0, x.shape[1], 16):
            aa = backend.row(x[:, start : start + 16], kind)
            bb = backend.row(b[start : start + 16, :].T, kind).T
            out = out + backend.mm(aa, bb)
        return out
    scores = attention_scores(p, backend)
    if c in {6, 19}:
        # Numerically stable split-KV merge. Framework ops expose states clearly.
        outputs = []
        logs = []
        for start in range(0, scores.shape[1], 2):
            s = scores[:, start : start + 2]
            valid = torch.isfinite(s).any(dim=1)
            safe = torch.where(valid[:, None], s, torch.zeros_like(s))
            probs = backend.row(safe, 0)
            outputs.append(backend.mm(probs, p["values"][start : start + 2, :]))
            logs.append(torch.where(valid, torch.logsumexp(s, dim=1), -torch.inf))
        ls = torch.stack(logs, dim=1)
        weights = backend.row(ls, 0)
        return sum(out * weights[:, i : i + 1] for i, out in enumerate(outputs))
    probs = backend.row(scores, 0)
    if c == 34:
        return backend.mm(probs, backend.row(p["values"].T, 4).T)
    if c == 36:
        # Demonstrate restoring QK scale before softmax; scalar 2 is explicit.
        return backend.mm(backend.row(scores * 2, 0), p["values"])
    if c == 39:
        # Data-only E2M1 baseline; CPU lab covers two-level SF and online state.
        probs = backend.row(probs, 5)
    return backend.mm(probs, p["values"])


def quant_reference(x, kind):
    """Independent framework reference for baseline data quantizers."""
    # Match float32 scale arithmetic, then compute quantization independently.
    maximum = x.abs().amax(dim=1, keepdim=True)
    limit = 448 if kind == 4 else 6
    scale = torch.where(maximum > 0, maximum / limit, torch.ones_like(maximum))
    normalized = x / scale
    if kind == 4:
        return normalized.to(torch.float8_e4m3fn).float() * scale
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    distances = (normalized.abs()[..., None] - table).abs()
    # Stable even-code tie break by visiting even encodings first.
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=x.device)
    idx = order[distances[..., order].argmin(dim=-1)]
    return table[idx] * torch.where(normalized < 0, -1.0, 1.0) * scale


def expected(p):
    """Mathematical reference independent of DSL implementations, CPU FP64 where appropriate."""
    c = p["chapter"]
    x = p["x"].cpu()
    b = p["b"].cpu()
    if c == 4:
        return torch.bincount(p["ids"].cpu().long(), minlength=7).to(torch.int32)
    if c in LAYOUT_CHAPTERS:
        return x.flatten()[p["index"].cpu()]
    if c == 5:
        return torch.softmax(x.double(), dim=1)
    if c == 7:
        xx = x.double()
        if p["variant"] % 2 == 0:
            return xx / torch.sqrt((xx * xx).mean(dim=1, keepdim=True) + 1e-5)
        centered = xx - xx.mean(dim=1, keepdim=True)
        return centered / torch.sqrt((centered * centered).mean(dim=1, keepdim=True) + 1e-5)
    if c in MATMUL_CHAPTERS:
        out = x.double() @ (b[:, :1] if c == 9 else b).double()
        return (out + p["bias"].cpu().double()).clamp_min(0) if c == 25 else out
    if c in {30, 32}:
        return quant_reference(x, 4 if c == 30 else 5)
    if c in {31, 33}:
        out = torch.zeros((x.shape[0], b.shape[1]), dtype=torch.float64)
        for start in range(0, x.shape[1], 16):
            aa = quant_reference(x[:, start : start + 16], 4 if c == 31 else 5)
            bb = quant_reference(b[start : start + 16, :].T, 4 if c == 31 else 5).T
            out += aa.double() @ bb.double()
        return out
    scores = x.double() @ p["keys"].cpu().double().T / math.sqrt(x.shape[1])
    if c == 36:
        scores *= 2
    if p["variant"] % 2:
        scores = scores.masked_fill(
            torch.arange(scores.shape[1])[None, :] > torch.arange(scores.shape[0])[:, None], -torch.inf
        )
    probs = torch.softmax(scores, dim=1)
    values = p["values"].cpu()
    if c == 34:
        values = quant_reference(values.T, 4).T
    if c == 39:
        probs = quant_reference(probs.float(), 5).double()
    return probs @ values.double()
