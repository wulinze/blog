"""Chapter 9: GEMV：如何把一行交给 warp

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 写一行与四行 program；TileLang 改 row×K 的 fragment 分工。测 M=1/17/1024，K=16/33/1025；最后一行和 K 尾部必须正确。先用 FP64 参考，再比较不同划分。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
