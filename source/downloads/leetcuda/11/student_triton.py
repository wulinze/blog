"""Chapter 11: 精度与异步搬运：TF32 和双缓冲

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 比较 input_precision="ieee" 与 "tf32"，固定形状分别报误差；TileLang 固定数学类型先扫 num_stages。记录生成代码、寄存器、shared 和时间，短 K 与长 K 各测一组。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
