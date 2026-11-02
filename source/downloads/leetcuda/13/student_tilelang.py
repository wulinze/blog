"""Chapter 13: 布局与调度：三种不同的 swizzle

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 只改 grouped program ordering，保持 BM/BN/BK 相同；TileLang 只改 shared layout，再独立改流水。逐项保存时间和 NCU 指标，不能只报告全部开关都开后的一个数字。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's tilelang exercise")
