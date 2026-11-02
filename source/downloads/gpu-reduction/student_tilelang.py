"""Implement the same hierarchical sum/max/dot contract in TileLang."""
# Use a (1, block) fragment and reduce dim=1 into a (1,) fragment.
# Fill ALL fragment positions, including neutral values in the tail.
# clear=True is correct for each independent tile; accumulation across a loop
# requires a separately reasoned initialization/clear policy.
# Launch successive levels on the same stream; allocate partials in prepare.
# n=0: sum/dot -> 0, max -> -inf; empty data does not launch a reduction grid.


def prepare(n: int, block: int, workers: int, operation: str):
    """workers = CUDA threads per TileLang block; return launch(x, y, out)."""
    raise NotImplementedError("Implement the chapter exercises before running --impl student")
