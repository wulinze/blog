"""Chapter 7: RMSNorm 与 LayerNorm：统计量和广播

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: 两种语言分别实现 RMSNorm 和两遍 LayerNorm，统计使用 float32。输入包含常数行、全零、小方差和 129 长度；gamma/beta 使用非均匀值，避免测试漏掉广播维度错误。扩展对照 Welford。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's tilelang exercise")
