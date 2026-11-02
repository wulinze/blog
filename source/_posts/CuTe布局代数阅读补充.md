---
title: CUDA 学习补充：CuTe 白皮书怎样读——坐标、函数与布局代数
description: 梳理 LeetCUDA 中独立 CuTe 导读的层级 shape、复合、合并、补、乘积与切分，用小模型和对应练习连接后续布局章节。
permalink: llm/cute-layout-primer/
date: 2026-11-02 17:01:00
categories:
  - LLM 推理
tags:
  - CUDA
  - CuTe
  - 算子学习
---

## 为什么单独保留这篇导读

原 PDF 第 277–309 页是独立的 CuTe 白皮书中文译注，不是新的编号章节。它位于 Attention 与 CuTe 章节之间。本补充保持这个位置，不挤占（一）至（四十一）的连续编号，也不把 33 页理论直接当成一段附注略过。

白皮书的价值在于给复杂线程/数据布局提供统一语言。学习时先理解“布局是函数”，再看哪些函数组合仍可表达为 CuTe layout，最后回到真实 copy 和 MMA。原论文为 Cris Cecka 的 [CuTe Layout Representation and Algebra](https://arxiv.org/abs/2603.02298v2)，这里是学习导读与原创小例子，不重复发布整份译文。

## 术语先分三层

| 层次 | 对象 | 要问的问题 |
| --- | --- | --- |
| 坐标结构 | tuple、层级 tuple、shape、rank、size | 合法坐标长什么样，元素有多少？ |
| 映射 | stride、layout、tensor/accessor | 坐标变成哪个 offset，再怎样取值？ |
| 代数操作 | composition、coalesce、complement、product、divide、inverse | 怎样组合或重组映射，并保留需要的性质？ |

`rank` 是顶层模式数量，不是所有嵌套叶子的总数。`size` 是 shape 的叶子大小乘积，不是所占地址空间字节数。`Tensor` 不只是 shape；还要知道数据引擎或访问器和 layout。

## 用同一组坐标理解平坦与层级形状

普通二维布局可以写为 `L(r,c)=r*s_r+c*s_c`。shape=(2,3)、stride=(1,2) 时，L(1,2)=5。若用整数坐标按 CuTe 的第一模式优先展开：

```text
i = r + 2*c
r = i % 2
c = i // 2
L(i) = (i%2)*1 + (i//2)*2 = i
```

不要把 Python 常见行主序的展平顺序自动套给 CuTe。先看具体 shape/stride 与坐标分解规则。

层级 shape 比如 `((2,3),4)` 把前两根轴组合为一个模式，但仍可保留内部结构。算法可以把这一模式当整体操作，也能进一步展开；“元素数量相同”并不保证两种层级结构处处可互换。正式的相容性条件要回到白皮书定义。

## 六类操作分别解决什么

### Coalesce：消去可合并的坐标层

若相邻模式在指定展开顺序下连续，多个模式可以合成一个。例如 `(2,3):(1,2)` 可视为 `6:1`。若 stride 变成 `(1,4)`，中间有洞，就不能直接宣称同一个连续布局。Coalesce 改变表示，不等于拷贝数据，也不能随意跨越不满足条件的模式。

### Composition：先内层，后外层

`R(c)=A(B(c))`。例：A(i)=2*i，B(j)=3*j，则 R(j)=6*j。真实层级 layout 中必须核对 B 的输出是否是 A 能解释的坐标，合法坐标范围是否匹配。函数式写法是直觉入口，不意味着任意 Python 函数组合都能化为合法 CuTe 类型。

### Complement：补全尚未覆盖的方向

一个 tile 布局只描述了部分地址方向，complement 提供与它配合的其余方向，使 tile 可以铺到更大范围。它不是简单的集合差，也不是按位取反。阅读时用小整数布局枚举两者组合后的覆盖范围，比直接背符号更可靠。

### Product：把小布局平移铺开

例如四元素 tile `0,1,2,3` 沿外层坐标以步长 4 重复，得到 0…11 的三块。product 家族会以不同模式组合方式组织内外坐标。更换组合顺序会改变逻辑视图，未必改变底层数据。

### Divide：取出局部 tile 与外部坐标

对长度 12、tile=4 的例子，`i=inner+4*outer`，inner 在 0…3，outer 在 0…2。每个全局 i 唯一分解为局部元素和 tile 编号；这正是线程块取 global tile 的基础。实际 logical/zipped/tiled divide 的模式结构不同，要打印结果再用。

### Inverse：先检查是否一一对应

如果多个逻辑坐标映到同一地址，不能要求一个全局唯一逆。stride=0 的广播是最简单反例。讨论 left/right inverse 时必须注明定义域与满足的组合关系，不能因为有 `inverse` 名称就把任意映射当双射。

## 图与交互：从坐标推到地址，再检查别名

<figure class="cuda-diagram">
<img src="/images/leetcuda/cuda-cute-layout.svg" alt="shape stride到offset再到覆盖和组合的流程" loading="lazy">
<figcaption>这里复用布局章节的图：它是本导读的最小可计算实例。<a href="/images/leetcuda/cuda-cute-layout.excalidraw" download>编辑原图</a></figcaption>
</figure>

<div class="leet-lab" data-leetcuda-lab data-source="/downloads/leetcuda/21/scenarios.json">
<label>选择连续、padding、列主序或别名布局<select aria-label="布局导读案例"></select></label>
<p data-progress aria-live="polite"></p><div data-output></div><p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div><p data-check></p>
</div>

先预测 shape=(2,3)、stride=(4,1) 的最大 offset 和逻辑元素数量，再操作。然后切到 stride=(0,1)，解释为什么它不是可用于唯一写出的布局。快照来自已运行 CPU 模型，不是硬件布局自动推断器。

## 一个完整的代数小 demo

下面只需 Python 标准库，执行后所有断言应通过。它是整数模型，不依赖 GPU，也不声称覆盖 CuTe 所有层级代数规则。

```python
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
    def a(i): return 2 * i
    def b(i): return 3 * i
    assert [a(b(i)) for i in range(4)] == [0, 6, 12, 18]
    # 广播布局不是单射，逆不唯一。
    aliases = [layout(r, c, (0, 1)) for r in range(2) for c in range(3)]
    assert len(set(aliases)) == 3 and len(aliases) == 6
    print("PASS: coordinates, composition, divide/product, aliases")


if __name__ == "__main__":
    check()
```

[完整布局实验包与 CuTe host 测试入口](/downloads/leetcuda/21/cuda-cute-layout.zip) 还包含网页同款模型和两种 DSL 练习。

## Triton 与 TileLang 怎么验证这份理解

Triton 先用显式 stride 构造两种 view 的 copy，和连续逻辑参考比较；不要以 `.reshape` 的形状相同代替地址验证。TileLang 先枚举 thread/value 对应坐标，再实现 global→shared→global 往返。两者先保证覆盖无遗漏、无非法别名和尾部不越界，再讨论 bank 与向量宽度。

在上面的布局实验包中执行：

```bash
python run_gpu.py --chapter 21 --backend triton
python run_gpu.py --chapter 21 --backend tilelang
```

这些是语义基线，GPU 编译与运行在当前无 NVIDIA GPU 的编辑环境中未实测。CuTe 的具体类型计算继续到 [（二十二）Layout](/llm/cuda-cute-layout/)、[（二十三）TiledCopy](/llm/cuda-cute-tiled-copy/)、[（二十四）TiledMMA](/llm/cuda-cute-tiled-mma/) 和 [（二十五）Swizzle/TMA](/llm/cuda-cute-tma-layout/)。

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
