"""Chapter 19: 多 consumer 与分段合并

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 先做两个 kernel 分段加 merge；TileLang 也先用分 kernel 等价实现验证数学。再尝试单 CTA 多 consumer，测试 1/2/3/5 个 KV tile，包含一组为空的情形。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
