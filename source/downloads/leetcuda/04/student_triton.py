"""Chapter 4: 访问粒度与原子冲突：向量化和直方图

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton：先用 tl.arange 和 mask 写 stride copy，再用整数 atomic_add 做 histogram。TileLang：先写同样的全局原子基线，再引入 shared 私有计数。输入 [0,1,1,3,3,3] 的输出应为 [1,2,0,3]；均匀与全零两类输入都检查 sum(hist)==N。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
