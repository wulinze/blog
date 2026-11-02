"""Chapter 23: MMA 的归属：从原子到输出 fragment

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 追踪 dot 前后布局转换的编译器 IR；TileLang 可视化 accumulator 和输入 fragment。用唯一编号填每个输入坐标，先验证小矩阵，再扩展到两种 M/N warp 排布。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
