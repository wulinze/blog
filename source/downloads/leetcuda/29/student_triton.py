"""Chapter 29: 大 D 的 non-WS 流水：让所有线程参与 MMA

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 与 TileLang 分别保留普通流水和专门化基线。选择 chunk 数 1/3/5、stage 2/3 的组合做协议测试；记录每个 slot 的最近 writer 和最后 reader，不满足就阻止复用。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
