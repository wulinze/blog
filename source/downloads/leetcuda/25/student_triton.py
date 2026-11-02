"""Chapter 25: 布局驱动的 GEMM 工程

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton tl.dot 后融合列 bias+ReLU；TileLang T.gemm 后显式 epilogue。先测非方阵和尾块，再比较是否 staging、不同 stages；保留原始输入，验证 beta 非零时没有错误覆盖。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
