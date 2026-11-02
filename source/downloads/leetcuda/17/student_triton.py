"""Chapter 17: Split-Q：让每组线程拥有自己的查询行

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 改 BQ 而保持 BK 与数值路径一致；TileLang 改 GEMM warp policy 前先枚举行归约范围。检查每行写入一次、causal 对齐与尾行；扩展比较 Split-KV 的合并成本。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
