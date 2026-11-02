"""Chapter 31: FP8 GEMM：scale 如何进入累加

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 先做支持的 FP8 dot+行列 scale，再加 K-block scale；TileLang 同样分两步。输入为 2 个 scale 不同的 K 块，确保错误外提会被测试捕获；再扫真实形状并计入量化。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's tilelang exercise")
