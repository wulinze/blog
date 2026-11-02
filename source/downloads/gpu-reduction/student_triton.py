"""Implement sum, max and dot using Triton, then combine partials on the device."""
# Contract: prepare returns launch(x, y, out), out has shape (1,), float32.
# Preallocate intermediate buffers in prepare, not in the timed launch.
# Keep partials on the device; do not use .item(), CPU sum or torch reduction.
# max needs -inf for masked elements; sum/dot need zero.
# Dot applies x*y ONLY at the first level. Later levels sum the partials.
# n=0 returns 0 for sum/dot and -inf for max.


def prepare(n: int, block: int, workers: int, operation: str):
    """workers = Triton num_warps; block is a logical power-of-two tile."""
    raise NotImplementedError("Implement the chapter exercises before running --impl student")
