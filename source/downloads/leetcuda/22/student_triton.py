"""Chapter 22: Copy 的归属：谁搬什么，搬到哪里

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 比较连续与交错索引载入；TileLang 实现 global→shared→global 往返。输入用 arange 编码位置以定位搬错，测试 N=1/29/32/33，确保不存在重复写和越界。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
