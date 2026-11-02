"""Chapter 33: Block-scaled MMA：scale 也是操作数

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 先用反量化 dot 参考，再接目标支持的 dot_scaled；TileLang 按目标后端的 block-scaled 示例实现。每组不同 scale、最后不足一组、数据/SF padding 都要验证；设备不支持时明确跳过硬件部分。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's tilelang exercise")
