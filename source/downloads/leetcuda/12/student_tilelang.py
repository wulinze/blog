"""Chapter 12: MMA 与 fragment：理解协作计算的数据归属

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 打印编译后的 PTX，识别 dot lowering 的实际形状；TileLang 查看生成的 CUDA，打印/可视化 fragment。先完成 128 个 C 坐标的唯一覆盖，再比较 M=16,N=8,K=16 和多个 atom 的结果。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's tilelang exercise")
