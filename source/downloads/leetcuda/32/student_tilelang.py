"""Chapter 32: NVFP4：两级 scale 与打包

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 与 TileLang 先做软件码点选择、uint8 pack/unpack；再用受支持的 scaled dot。测试 0、正负中点、极值、奇数长度及零组。字节一致与允许的重建误差分别验收。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's tilelang exercise")
