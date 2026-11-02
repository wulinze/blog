"""Chapter 40: 基准与精度：把学习结果变成可复现实验

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: Triton 与 TileLang 接入同一个 harness，导出 JSON：设备、版本、shape、dtype、误差、kernel 与 e2e 时间。固定参照后端，保留失败配置；禁止把 CPU 时间或未同步 wall time 标成 CUDA kernel 延迟。
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's triton exercise")
