"""Chapter 8: RoPE 与转置：数据重排为何影响性能

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 实现两种 RoPE 配对和带 mask 转置；TileLang 实现 shared 中转，比较 stride 32/33。先检查范数，再检查相对位置点积；转置用 17×35，做两次必须回到原矩阵。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's tilelang exercise")
