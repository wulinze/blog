"""Chapter 27: 大 D 的布局代数与寄存器预算

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 对不同 D chunk 方案比较编译资源；TileLang 保存两套 fragment 布局图，禁用未证明的 reinterpret。验收 D=64/192/512，加一个非整除的语义基线；专用 kernel 不支持的形状明确拒绝。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
