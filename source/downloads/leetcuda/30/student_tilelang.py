"""Chapter 30: FP8 的格式与量化前处理

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 写 amax+缩放+转换，TileLang 融合同样前处理。与软件可表示值表逐项比较，覆盖零、次正规附近、舍入中点、448 和离群值。报告 max_abs 与饱和计数，不只 allclose。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's tilelang exercise")
