"""CPU demonstrations of XOR all-reduce, hierarchical reduction and fp32 rounding."""

import argparse
import math
import struct


def fp32(x: float) -> float:
    """Round once to IEEE binary32; intended for finite teaching inputs."""
    return struct.unpack("f", struct.pack("f", x))[0]


def xor_rounds(values: list) -> list:
    """Return snapshots; every lane reads the previous round, never an in-place update."""
    width = len(values)
    if width == 0 or width & (width - 1):
        raise ValueError("width must be a nonzero power of two")
    rounds = [list(values)]
    mask = width // 2
    while mask:
        old = rounds[-1]
        rounds.append([old[i] + old[i ^ mask] for i in range(width)])
        mask //= 2
    return rounds


def tree_sum(values: list) -> float:
    """Pair adjacent values using binary32 additions; pad an odd tail with zero."""
    work = [fp32(x) for x in values]
    while len(work) > 1:
        work = [fp32(work[i] + (work[i + 1] if i + 1 < len(work) else 0.0)) for i in range(0, len(work), 2)]
    return work[0] if work else 0.0


def hierarchy(values: list, block: int = 4) -> list:
    """Return each level of partials; models arithmetic, not GPU scheduling."""
    if block < 2:
        raise ValueError("block must be at least two")
    levels = [list(values)]
    while len(levels[-1]) > 1:
        old = levels[-1]
        levels.append([sum(old[i : i + block]) for i in range(0, len(old), block)])
    return levels


def self_check() -> None:
    """Check invariants, tails and a known floating-point non-associativity example."""
    assert xor_rounds(list(range(1, 9)))[-1] == [36] * 8
    assert xor_rounds([7]) == [[7]]
    assert hierarchy(list(range(1, 11)), 4) == [list(range(1, 11)), [10, 26, 19], [55]]
    assert hierarchy([], 4) == [[]]
    x = [2**25, 1, -(2**25), 1]
    serial = 0.0
    for v in x:
        serial = fp32(serial + v)
    assert serial == 1 and tree_sum(x) == 0 and math.fsum(x) == 2
    print("PASS: XOR snapshots, coverage, partials, empty input and float32 rounding")


def main() -> None:
    """Print a small trace or run analytical tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    for step, values in enumerate(xor_rounds(list(range(1, 9)))):
        print(f"xor round {step}: {values}")
    print("two-level example:", hierarchy(list(range(1, 11))))
    x = [2**25, 1, -(2**25), 1]
    serial = 0.0
    for value in x:
        serial = fp32(serial + value)
    print("rounding example:", dict(serial_fp32=serial, pair_tree_fp32=tree_sum(x), reference=math.fsum(x)))


if __name__ == "__main__":
    main()
