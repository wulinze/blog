"""Chapter 28: Persistent Attention：任务循环与尾波

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 用 program ID 加固定 grid stride；TileLang 用同样任务循环。先用整数 task ID 写 marker，检查每个 marker 恰好一次，再接 attention。测试 tasks=1/P-1/P/P+1，包含 causal 不均匀负载。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's tilelang exercise")
