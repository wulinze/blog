"""Chapter 5: 稳定 Softmax：从整行到在线状态

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 用 tl.max/tl.sum 做单行版本，尾部载入 -inf；TileLang 用二维 fragment 和 dim=1 归约。先验证宽度 1、17、127、129，再把极大偏置加到输入上，检查概率不变。扩展实现长行两遍 online，并把第二遍成本算入时间。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's tilelang exercise")
