"""Chapter 21: Layout：坐标、地址与可逆映射

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 构造带 stride 的 view copy，和连续张量对照；TileLang 显式建立布局并枚举。CPU 先穷举 (2,3)、padding、重复 stride，GPU 练习只接受无别名且边界合法的输入。原书 ch20_layout 是 host 断言程序。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's tilelang exercise")
