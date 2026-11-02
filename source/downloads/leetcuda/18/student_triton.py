"""Chapter 18: 注意力的双流水：K、V 与角色分工

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: 两语言都先使用一条统一流水作为正确性基线。扩展到受支持的专门化路径时，独立扫 K/V stages；记录 single-tile、odd-tile 的通过状态及等待事件。浏览器中的时间线只表示依赖模型。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
