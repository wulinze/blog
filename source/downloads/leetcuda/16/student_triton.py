"""Chapter 16: FlashAttention：为什么可以不写出分数矩阵

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 和 TileLang 从公开的 fused attention 教程出发，先固定单头、小尺寸。测试 N=1/17/65，dense/causal 与极端分数；每一 KV tile 后与 CPU 的 m/l/u 状态对照，再做性能实验。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
