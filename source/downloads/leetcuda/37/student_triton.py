"""Chapter 37: FP8 大 D：Split-D 与并行布局

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 对不同 Br 与 D chunk 做资源扫描；TileLang 对 M/N policy 变化画出 row ownership。至少包含 D=128/256/512 的语义参考和目标支持的大 D 路径，记录无法编译和 spill 配置。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
