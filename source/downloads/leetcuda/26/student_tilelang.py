"""Chapter 26: Attention 实现对照：统一流水、角色分工与双 consumer

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: 两语言均先跑同一个测试矩阵：N=1/65/257、D=64/128、dense/causal。只改变一项流水开关，记录实际选择的内核，不允许静默 fallback 冒充指定路径。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's tilelang exercise")
