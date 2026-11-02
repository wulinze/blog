"""Chapter 34: 量化 Attention：误差在哪里被放大

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: 两语言均提供开关只量化 Q/K、V 或 P，每组保存中间输出。输入包含均匀分数、单个尖峰、全负、长序列；输出按行报告误差，特别检查 causal 第一行。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's tilelang exercise")
