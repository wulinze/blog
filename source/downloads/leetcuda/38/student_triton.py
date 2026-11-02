"""Chapter 38: FP4 Attention 的量化链

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 和 TileLang 都先验证未量化的平滑恒等式，再插入 FP4。使用非零 Q/K 均值与 causal mask，故意省略 delta S 应被测试判错；然后验证数据与 SF 的置换还原。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
