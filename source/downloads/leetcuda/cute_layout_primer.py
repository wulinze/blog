def layout(row, col, strides):
    return row * strides[0] + col * strides[1]


def check():
    # 第一模式优先的坐标展开；验证 coalesce 的连续特例。
    values = [layout(i % 2, i // 2, (1, 2)) for i in range(6)]
    assert values == list(range(6))
    # padding 布局有洞，不能当作相同连续数组。
    padded = [layout(i % 2, i // 2, (1, 4)) for i in range(6)]
    assert padded == [0, 1, 4, 5, 8, 9]
    # divide/product 的小整数特例，验证无遗漏、无重算。
    coordinates = [(i % 4, i // 4) for i in range(12)]
    rebuilt = [inner + 4 * outer for inner, outer in coordinates]
    assert rebuilt == list(range(12))

    # 函数复合先 B 后 A。
    def a(i):
        return 2 * i

    def b(i):
        return 3 * i

    assert [a(b(i)) for i in range(4)] == [0, 6, 12, 18]
    # 广播布局不是单射，逆不唯一。
    aliases = [layout(r, c, (0, 1)) for r in range(2) for c in range(3)]
    assert len(set(aliases)) == 3 and len(aliases) == 6
    print("PASS: coordinates, composition, divide/product, aliases")


if __name__ == "__main__":
    check()
