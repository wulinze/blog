"""Chapter 36: FP8 persist-D 与 scale 折叠

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 和 TileLang 分别写显式反量化版与 scale 折叠版。先在无量化舍入的小例子中证明等价，再用 FP8 输入比较误差；加入跨 key 改变 scale 的反例，防止非法外提。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
