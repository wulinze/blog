"""Chapter 10: GEMM：block tile 与寄存器 tile

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton tl.dot 与 TileLang T.gemm 都先做 FP16 输入、FP32 累加基线。用 M=33,N=49,K=65 验证尾块，再扫 BM/BN/BK。额外写 thread tile 的 CPU 索引枚举，确认每个输出只有一个 owner。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's tilelang exercise")
