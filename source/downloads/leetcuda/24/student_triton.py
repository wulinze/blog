"""Chapter 24: Swizzle 与 TMA 的地址契约

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton descriptor copy 与普通 load 做等价验证；TileLang 显式 swizzle 和 TMA 搬运配对。小 tile、不同 stage、非零 base offset 各测一组。原书 ch23_swizzle_tma 是 host 布局断言，不能当硬件 copy 测试。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
