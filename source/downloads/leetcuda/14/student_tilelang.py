"""Chapter 14: TMA 与 WGMMA：Hopper 的完成协议

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 先做 descriptor copy 的输入输出等价测试，再研究 Hopper matmul 教程；TileLang 先 TMA copy 再接 gemm。均先验证 1 个 tile、2 个 tile、奇数 tile，记录每次 slot/phase。原书 WGMMA 最小测试只在 sm_90a 目标运行。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's tilelang exercise")
