"""Chapter 6: LSE 与注意力状态合并

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 与 TileLang 都实现 merge(O_A,L_A,O_B,L_B)。每行一组 LSE，沿 D 广播权重。测试空/非空、双空、LSE 相差 1000、三个不等长段；参考从原始 scores/V 一次计算，不能只拿同一合并函数作参考。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's tilelang exercise")
