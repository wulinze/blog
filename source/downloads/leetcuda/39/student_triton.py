"""Chapter 39: FP4 Attention：P 的两级量化

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 先软件量化 P 并用高精度 PV 定位误差；TileLang 再接 block-scaled PV。测试后续 tile 才出现大最大值、均匀长行、causal 首行与极小概率，报告每个阶段的误差。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
