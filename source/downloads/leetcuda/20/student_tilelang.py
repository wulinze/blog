"""Chapter 20: 大 head_dim：Split-D 的两次分块

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 分块累加 QK 后再 softmax；TileLang 为 QK/PV 使用不同 fragment 形状。测试 D=64/128/256/非整除值，所有输出 chunk 与完整参考比较；记录编译器寄存器和 spill。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's tilelang exercise")
