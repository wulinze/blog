"""Chapter 35: FP8 Attention 的前处理链

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 先分别写均值、平滑、量化；TileLang 尝试把局部变换与转置融合。分别验证均值、重建 Q/K/V、最终 O；同时输出预处理占总时间比例。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
