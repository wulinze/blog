"""Render authored chapter data into standalone Hexo posts and runnable lab bundles.

Edit docs/learning/chapter-notes.json for prose; source/downloads/leetcuda for code.
The generated posts remain ordinary Markdown files, with stable permalinks.
"""

import ast
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
DATA = ROOT / "source/downloads/leetcuda"
notes = json.loads((ROOT / "docs/learning/chapter-notes.json").read_text())
manifest = json.loads((ROOT / "docs/learning/leetcuda-source-map.json").read_text())
spec = importlib.util.spec_from_file_location("learning_lab", DATA / "learning_lab.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
source = (DATA / "learning_lab.py").read_text()
tree = ast.parse(source)
functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
LEET = "3b52c5cd3332c5e992c7aef63087d43f67f23b8e"
LEET_FP4 = "d32a58f53e4122dc68d660698ee025751a596fc5"
FFPA = "861d75ee87824a14a9ebfe295d54c7dac3e3cabc"
TESTS = {
    4: "ch03_elementwise",
    5: "ch04_softmax",
    6: "ch05_merge_attn",
    7: "ch06_norm",
    8: "ch07_rope_transpose",
    9: "ch08_sgemv",
    10: "ch09_sgemm",
    11: "ch10_sgemm_tf32",
    12: "ch11_hgemm_mma",
    13: "ch12_hgemm_swizzle",
    14: "ch13_hgemm_wgmma",
    15: "ch14_hgemm_tma_ws",
    16: "ch16_fa2_mma",
    17: "ch16_fa2_mma",
    18: "ch17_fa2_tma_ws",
    19: "ch18_fa3",
    20: "ch19_ffpa_split_d",
    21: "ch20_layout",
    22: "ch21_tiled_copy",
    23: "ch22_tiled_mma",
    24: "ch23_swizzle_tma",
    25: "ch24_hgemm_cute",
    26: "ch25_fa_cute",
    27: "ch26_ffpa_cute",
    28: "ch26b_fa_persist_d",
    29: "ch26c_ffpa_split_d_non_ws",
    30: "ch34_fp8_quantize",
    31: "ch35_fp8_gemm",
}
FFPA_FILES = {
    34: "fp8/fp8_pscale.cuh",
    35: "fp8/quantize_fp8.cuh",
    36: "fp8/sm_120/persist_d.cuh",
    37: "fp8/sm_120/split_d_m4n2.cuh",
    38: "fp4/quantize_fp4.cuh",
    39: "fp4/sm_120/persist_d.cuh",
    40: "fp8/fp8_pscale.cuh",
}
BASIC = {
    4: "整数 histogram：全局 atomic_add，覆盖均匀与单热点；shared 私有化是扩展题。",
    5: "短行稳定 softmax；长行 online 两遍实现是扩展题。",
    6: "分段 attention + LSE 权重合并；状态归约由框架操作显式串联，尚未融合成单 kernel。",
    7: "RMSNorm / 两遍 LayerNorm；gamma、beta 和 Welford 是扩展题。",
    8: "矩形转置的 gather 语义；RoPE 和 shared padding 是扩展题。",
    9: "GEMV，通过矩阵乘基线的单输出列运行；短 K 子组是扩展题。",
    10: "有尾块的 FP32 GEMM；Triton 使用 dot，TileLang 使用直白标量 FMA。",
    11: "相同 FP32 GEMM 正确性基线；TF32 与异步 stages 的改变是扩展题。",
    12: "完整矩阵乘语义；fragment 的精确 lane 映射由 CPU demo 与原书 CUDA 例子验证。",
    13: "8×8 XOR gather 映射；CTA swizzle 与双缓冲的性能实验是扩展题。",
    14: "同步 GEMM 数值基线；真正 TMA/WGMMA 由下面 Hopper 专用 CUDA demo 提供。",
    15: "同步 GEMM 数值基线；SM120 角色分工由下面 CUDA demo 提供。",
    21: "padding stride 的 gather，显式区分逻辑元素与地址空间。",
    22: "含尾部的双射 gather；改变 thread/value 分工与 shared 中转是扩展题。",
    23: "矩阵乘数值基线；TiledMMA 的 host 映射测试见原书专用程序。",
    24: "匹配 XOR 布局的 gather；真实 TMA round-trip 需要专用硬件路径。",
    25: "GEMM + 逐列 bias + ReLU；staging 和向量写回是扩展题。",
    28: "persistent 静态任务顺序的 gather；真正循环 CTA attention 见原书 CUDA demo。",
    29: "搬运序号分组的 gather；non-WS Split-D 的硬件实现见下面 CUDA demo。",
    30: "逐行 E4M3FN 量化并反量化，返回 FP32 重建值；保留编码和分组 scale 是扩展题。",
    31: "逐 K 块 FP8 重建后 GEMM；scale 不非法外提，硬件 FP8 MMA 见 CUDA demo。",
    32: "E2M1 数据舍入基线，scale 暂用 FP32；CPU demo 另验证两级 SF 舍入与 packing，硬件 NVFP4 见下面专用 demo。",
    33: "逐 K 块 E2M1 重建后 GEMM；真实 block-scaled 指令与 SF 布局见专用 CUDA demo。",
    34: "单独量化 V 的 attention，帮助定位输出误差；多阶段开关是扩展题。",
    35: "K 平滑后 attention；均值与量化融合、V 转置是扩展题。",
    36: "把 QK scale 正确放进 softmax 之前的 attention；FP8 专用折叠实现见 CUDA demo。",
    37: "D=64/128/256/512 的分块 QK attention；M4N2 排布由专用 CUDA 路径提供。",
    38: "Q/K 平滑 + deltaS 修正的未量化 attention；数据/SF packing 与置换见 CPU 和专用 CUDA demo。",
    39: "E2M1 量化 P 的物化 attention；CPU demo 另验证在线两级尺度，专用 CUDA demo 使用硬件 FP4。",
    40: "相同输入契约的 attention + 设备事件计时，输出误差和完整语义基线时间。",
}
for n in [16, 17, 18, 19, 20, 26, 27]:
    BASIC.setdefault(
        n,
        "小尺寸 attention 语义基线，包含明确的 top-left causal mask；各章专用流水、布局与分工见下面的固定版本 CUDA demo。",
    )


def standalone(n):
    required = {f"lab{n:02}"}
    while True:
        old = required.copy()
        for name in old:
            required.update(
                node.id for node in ast.walk(functions[name]) if isinstance(node, ast.Name) and node.id in functions
            )
        if old == required:
            break
    if "quant4" in required or "quant8" in required:
        required.add("e4m3_values")
    used_names = {node.id for name in required for node in ast.walk(functions[name]) if isinstance(node, ast.Name)}
    imports = "import argparse\nimport json"
    for module in ["math", "statistics"]:
        if module in used_names:
            imports += "\nimport " + module
    parts = ['"""Standalone CPU chapter demo. Python 3.9+; no GPU or third-party dependency."""', imports]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in required:
            parts.append(ast.get_source_segment(source, node))
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in {"FP8", "FP4"} for t in node.targets
        ):
            if "quant4" in required or "quant8" in required:
                parts.append(ast.get_source_segment(source, node))
    parts.append(f"""if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    print(json.dumps(lab{n:02}(args.case), ensure_ascii=False, indent=2, allow_nan=False))""")
    return "\n\n\n".join(parts) + "\n"


def upstream(n):
    if n in TESTS:
        name = TESTS[n]
        config = {
            "repo": "https://github.com/xlite-dev/LeetCUDA",
            "commit": LEET,
            "kind": "book-test",
            "test": name.split("_")[0],
        }
        if n == 14:
            config["requires_arch"] = "sm_90a"
        url = f"{config['repo']}/blob/{LEET}/kernels/interview/book/tests/{name}.cu"
        arch = "sm_90a" if n == 14 else ("sm_120a" if n in {15, 18, 19, 28, 29, 30, 31} else "sm_80")
        note = "CUDA C++20、NVCC、对应目标 GPU 与 CUTLASS submodule。TMA 相关例子还需满足该版本头文件的 CUDA toolkit 要求。"
        if n in {21, 22, 23}:
            note = "此原书程序主要做 CuTe host 布局断言，仍使用 NVCC/CUTLASS 编译；它不是 GPU kernel 性能测试。"
        if n == 24:
            note = "原书程序包含 host 布局断言和可用设备上的检查；无设备的分支会 SKIP，必须逐项阅读输出。"
        if n == 16:
            note += " 原书理论章没有独立 kernel，使用紧接的 FA2 最小实现作为数学落地入口。"
    elif n in {32, 33}:
        config = {"repo": "https://github.com/xlite-dev/LeetCUDA", "commit": LEET_FP4, "kind": "fp4-gemm"}
        url = f"{config['repo']}/blob/{LEET_FP4}/kernels/interview/bench/fp4_gemm.cu"
        arch = "sm_120a"
        note = "这个 FP4 GEMM 在较新的固定提交中，不能用前面旧提交的测试编号寻找它。需要支持 SM120 FP4 的 CUDA toolkit、CUTLASS、cuDNN/cuBLAS 等该构建脚本依赖。运行 --fp4-gemm 会执行量化与主 GEMM 的组合验证。"
    else:
        config = {
            "repo": "https://github.com/vipshop/ffpa-attn",
            "commit": FFPA,
            "kind": "ffpa",
            "implementation": "fp4" if n in {38, 39} else "fp8",
            "head_dim": 512 if n == 37 else 128,
        }
        url = f"{config['repo']}/blob/{FFPA}/csrc/cuffpa/cute/{FFPA_FILES[n]}"
        arch = "sm_120a"
        note = "Linux NVIDIA SM120 环境；先按固定版本 README 准备 CUDA PyTorch、Python≥3.10、构建依赖和 CUDA 扩展。此入口运行整条量化 attention，前处理、主循环、误差报告一起观察。大 D 路径是否编入以构建的 headdim 配置为准。"
    return config, url, arch, note


def commands(config, arch, n):
    checkout = "LeetCUDA-lab" if config["kind"] != "ffpa" else "ffpa-attn-lab"
    lines = [
        f"git clone {config['repo']}.git {checkout}",
        f"git -C {checkout} checkout --detach {config['commit']}",
        f"git -C {checkout} submodule update --init --recursive",
    ]
    if config["kind"] == "ffpa":
        lines += [
            f"cd {checkout}",
            "# 在已装好匹配 CUDA PyTorch 的隔离环境中安装项目依赖",
            "python -m pip install packaging ninja wheel setuptools setuptools-scm matplotlib",
            'FFPA_SKIP_CUDA_EXT=1 python -m pip install -e ".[bench]" --no-build-isolation',
            f"bash build.sh --arch {arch} --ext all --headdim 128,512 --jobs 4",
            "cd ..",
        ]
    lines += [f"python3 run_upstream.py --repo ./{checkout} --arch {arch}"]
    return "\n".join(lines)


COMMON = [
    "learning_lab.py",
    "reference_triton.py",
    "reference_tilelang.py",
    "exercises.py",
    "run_gpu.py",
    "run_upstream.py",
]
scenes = []
for topic in manifest["topics"]:
    n = topic["source_chapters"][0]
    if n < 4:
        continue
    note = notes[str(n)]
    slug = note["slug"]
    label = topic["blog_label"]
    number = topic["blog_number"]
    out = DATA / f"{n:02}"
    out.mkdir(parents=True, exist_ok=True)
    cpu = standalone(n)
    (out / "demo.py").write_text(cpu)
    snapshots = [lab.run(n, c) for c in range(4)]
    (out / "scenarios.json").write_text(json.dumps(snapshots, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    config, url, arch, hardware = upstream(n)
    (out / "upstream.json").write_text(json.dumps(config, indent=2) + "\n")
    cmd = commands(config, arch, n)
    for backend in ["triton", "tilelang"]:
        (out / f"student_{backend}.py").write_text(f'''"""Chapter {n}: {topic["title"]}

Contract and reference: exercises.make_problem / expected / run_reference.
Hint: {note["hint"]}
"""


def solve(problem, backend):
    """Return a new result tensor; never modify input tensors.

    First read run_reference(problem, backend), then implement independently.
    """
    raise NotImplementedError("Implement this chapter's {backend} exercise")
''')
    (out / "EXERCISES.txt").write_text(
        f"{number}. {topic['title']}\n\n本章可运行语义基线：{BASIC[n]}\n\n{note['hint']}\n\n进阶验收：{topic['acceptance']}\n\n独立学生入口：student_triton.py / student_tilelang.py 的 solve(problem,backend)。\n输入由 exercises.make_problem(chapter,variant) 生成；参考由 expected 独立计算。\n4 个 case 包含小形状、零输入、非整除维度与 causal/布局变化。\n初始学生入口明确抛 NotImplementedError，不会冒充已完成练习。\n基线通过不代表进阶硬件机制已经实现。\n"
    )
    readme = f"""CUDA 算子学习（{label}）：{topic["title"]}
来源：leetcuda-20261029.pdf，第 {topic["pdf_pages"][0]}–{topic["pdf_pages"][1]} 页。

CPU：Python 3.9+ 标准库，解压后运行：
python3 demo.py --case 0
python3 demo.py --case 3
python3 learning_lab.py --self-test

GPU 语义基线：Linux NVIDIA GPU；Python>=3.10；匹配 CUDA 的 PyTorch；选装 triton 或 tilelang。
PyTorch 安装 https://pytorch.org/get-started/locally/
Triton https://triton-lang.org/main/getting-started/installation.html
TileLang https://tilelang.com/get_started/installation.html
在隔离环境里安装所选 DSL（例如 python -m pip install triton 或 tilelang）。
具体版本由 run_gpu.py 写入 result.json；当前未做 GPU 后端编译/设备/性能验证。

python run_gpu.py --chapter {n} --backend triton
python run_gpu.py --chapter {n} --backend tilelang
python run_gpu.py --chapter {n} --backend triton --student student_triton
python run_gpu.py --chapter {n} --backend tilelang --student student_tilelang
python run_gpu.py --chapter {n} --backend triton --benchmark --output result.json

基线范围：{BASIC[n]}
参考实现使用共享原语组合，包含框架编排和若干独立 kernel，不能当成已优化的 fused kernel。
--benchmark 报整个语义基线组合的设备事件时间，包含必要清零和中间操作，不是主 kernel-only。
输入必须为测试契约中的有限 float32/合法索引；具体整数 histogram 是 int32。
GPU 不可用退出 77，明确 SKIP；不静默改跑 CPU。学生未实现会失败。
FP4 DSL 基线是 E2M1+FP32 scale；CPU demo 提供 FP8 SF 舍入、pack/unpack；真实 NVFP4 见下方。

架构专用完整源码与测试：{url}
{hardware}
以下命令在本解压目录执行，新建独立上游 checkout：
{cmd}

run_upstream.py 检查完整提交号；SKIP 不算 GPU PASS。
原作者 CUDA 代码未复制进此压缩包；通过固定版本 checkout 获得完整依赖与许可。
阅读上游 LICENSE；博客讲解和 CPU/DSL 教学代码为本学习资料新增。
"""
    (out / "README.txt").write_text(readme)
    package = out / f"{slug}.zip"
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as z:
        for name in COMMON:
            z.write(DATA / name, name)
        for name in [
            "demo.py",
            "student_triton.py",
            "student_tilelang.py",
            "EXERCISES.txt",
            "README.txt",
            "upstream.json",
            "scenarios.json",
        ]:
            z.write(out / name, name)
    terms = "| 术语 | 本篇含义 |\n| --- | --- |\n" + "\n".join(
        "| " + row.replace("|", " | ") + " |" for row in note["terms"].splitlines()
    )
    previous = "/llm/cuda-reduction/" if n == 4 else "/llm/" + notes[str(n - 1)]["slug"] + "/"
    next_link = (
        ""
        if n == 40
        else f" · 下一篇：[（{manifest['topics'][n - 1]['blog_label']}）{manifest['topics'][n - 1]['title']}](/llm/{notes[str(n + 1)]['slug']}/)"
    )
    base = f"/downloads/leetcuda/{n:02}"
    title = f"CUDA 算子学习（{label}）：{topic['title']}"
    flowtext = " → ".join(note["flow"])
    body = f'''---
title: {title}
description: {topic["goal"]}。附可编辑流程图、四组逐步交互、独立 CPU demo、Triton 与 TileLang 语义基线及进阶练习。
permalink: llm/{slug}/
date: 2026-11-02 17:00:{n:02}
categories:
  - LLM 推理
tags:
  - CUDA
  - 算子学习
  - Triton
  - TileLang
---

## 先知道这篇要解决什么

[LLM 学习入口](/llm/) · [全部编号目录](/systems/gpu-learning-path/) · [上一篇]({previous}){next_link}

{note["intro"]}

本篇接续系列第 {number} 篇，对应用户提供的 `leetcuda-20261029.pdf` 原文第 {n} 章（PDF 阅读器第 {topic["pdf_pages"][0]}–{topic["pdf_pages"][1]} 页）。以下按学习问题展开，用自己的小例子连接原理、图、代码与练习；原书性能数字不作为本机实测。

**完成后应能做到：** {topic["goal"]}。先手算一个小例子，再运行同一模型，最后把正确性基线改成适合目标 GPU 的实现。

## 本篇术语与输入约定

{terms}

矩阵公式中 `@` 表示矩阵乘法，`.T` 表示转置；形状与归约方向必须匹配。数值例子默认有限输入，空集合、mask、尾部以及低精度格式的特殊约定在使用处明确说明。本文的协议与时间模型用于解释依赖，不能把一步直接解释成一个 GPU 时钟。

## 原理与结构

{note["sections"]}

## 公式：从定义到可核对的小例子

```text
{note["formula"]}
```

这里的 tile、stage、线程和 dtype 都是公式的条件。先确认单位是元素、字节、寄存器个数还是模型时间，再代入。若改变归约方向、量化组或数据布局，应重新推导，而不是只修改常数。

## 流程图：沿数据依赖逐步看

<figure class="cuda-diagram">
<a href="/images/leetcuda/{slug}.svg" target="_blank" rel="noopener"><img src="/images/leetcuda/{slug}.svg" alt="{topic["title"]}的数据与控制依赖流程图" loading="lazy"></a>
<figcaption>从上到下沿箭头读：箭头表示后一步依赖前一步的数据或完成条件，不表示真实周期。<a href="/images/leetcuda/{slug}.excalidraw" download>下载可编辑 Excalidraw 原图</a></figcaption>
</figure>

图的主线是：**{flowtext}**。蓝色框给出输入和映射，绿色框表示核心变换，橙色框表示输出或下一轮复用的边界。后一步不能因为“已经发出前一步指令”就绕过其完成条件。对照下面交互，每一步都能看到当前数值或状态。

## 交互实验：改变案例，观察中间状态

选择四个案例之一，先预测结果，再用“下一步”推进。“上一步”和“重置”可用来比较变化。演示数据由本文 Python 模型生成并执行断言校验；网页展示这些可复现的计算快照，**不在浏览器执行 CUDA，也不提供 GPU 测速**。

<div class="leet-lab" data-leetcuda-lab data-source="{base}/scenarios.json">
<label>输入 / 配置案例<select aria-label="选择本章实验案例"></select></label>
<p data-progress aria-live="polite">正在载入实验…</p>
<div data-output></div>
<p data-detail></p>
<div class="leet-actions"><button type="button" data-previous>上一步</button><button type="button" data-next>下一步</button><button type="button" data-reset>重置</button></div>
<p data-check></p>
</div>

第一组案例：`{snapshots[0]["label"]}`。到最后一步应满足：{snapshots[0]["check"]} 改变案例后，注意哪一个中间量发生变化，哪些不变量仍然成立。不能只看最终结果相近就跳过中间的布局或同步约束。

## 可以直接运行的代码 demo

[下载本篇完整实验包]({base}/{slug}.zip) · [独立 CPU 源码]({base}/demo.py) · [运行说明]({base}/README.txt) · [练习说明]({base}/EXERCISES.txt)

解压后先运行 CPU demo，不需安装第三方库，Python 3.9 及以上即可：

```bash
python3 demo.py --case 0
python3 demo.py --case 3
```

程序打印与网页相同的逐步状态，并执行模型内的断言。下面是**完整可独立运行**的本篇 CPU 文件；它验证数值、索引或协议，不代替 GPU 实现。

<details>
<summary>展开完整 Python demo（可直接复制运行）</summary>

```python
{cpu.rstrip()}
```

</details>

### Triton 与 TileLang 的可运行基线

实验包同时包含 `reference_triton.py`、`reference_tilelang.py`、`exercises.py` 与 `run_gpu.py`。依赖为 Linux NVIDIA GPU、匹配 CUDA 的 PyTorch、所选 DSL；安装说明见包内 README。两个后端使用相同输入与独立参考；不会在没有 GPU 时悄悄回退到 CPU。

**本篇基线实际实现：** {BASIC[n]}

```bash
python run_gpu.py --chapter {n} --backend triton
python run_gpu.py --chapter {n} --backend tilelang
```

默认四组输入包含零值、小形状、非整除维度，相关 attention 案例还含明确的矩形 top-left causal 约定。输出必须有限、形状正确且不修改输入。参考采用 CPU 高精度数学或独立的格式舍入；低精度近似另外报告误差。共享基线把难题拆成若干 kernel 和框架编排，便于阅读和定位，不把它称为已优化的 fused kernel。

### 对照原书的架构专用 CUDA 实现

[打开固定版本完整源码 / 最小测试]({url})。{hardware}

实验包的 `run_upstream.py` 会核对提交号，执行本篇对应的入口；源码文件的 `chNN` 是原仓库编号，**不等于 PDF 章节号或本博客编号**。按下面命令获取完整项目，不能只下载一个 `.cu` 就忽略头文件和子模块。

```bash
{cmd}
```

`SKIP`、编译通过、GPU 正确性通过和性能改进是四种不同结论。上游某些程序遇到不支持的架构会返回 0 并输出 SKIP，因此不能只用退出码认定测试成功。FP4 和 SM120 专用路径尤其要先确认设备、toolkit、指令与编译配置。

## 独立练习：先正确，再验证优化假设

**Triton 任务：** {topic["triton"]}。

**TileLang 任务：** {topic["tilelang"]}。

{note["hint"]}

实验包已提供独立学生入口 `student_triton.py` 和 `student_tilelang.py`。实现 `solve(problem, backend)`，返回新的结果张量；输入契约看 `exercises.make_problem`，独立参考看 `expected`，提示实现看 `run_reference`。初始学生入口主动报未实现，避免误把参考实现当作自己完成的练习。

```bash
python run_gpu.py --chapter {n} --backend triton --student student_triton
python run_gpu.py --chapter {n} --backend tilelang --student student_tilelang
# 正确后，才记录完整基线组合的设备事件时间
python run_gpu.py --chapter {n} --backend triton --benchmark --output result.json
```

**进阶通过条件：** {topic["acceptance"]}。基线通过只证明对应语义，不能证明进阶的布局、指令或 overlap 已实现。进阶记录至少包含：形状与 dtype、实际后端、编译器版本、误差、寄存器/shared、相关指令或 profiler 证据，以及必要预处理/初始化/合并在内的时间。不规定必须加速，变慢但解释清楚原因也属于有效实验。

## 常见误解与判断边界

{note["pitfall"]}

定位性能时，回到 [第二篇的完整分析流程](/systems/cuda-overlap/#performance-analysis)：先定位值得优化的 kernel，再看计算/存储吞吐、eligible warp、stall 与资源限制，最后针对本章机制做单变量实验。stall 比例不等于可直接消除的时间，低命中率也不自动意味着缺少复用。

## 验证状态与延伸阅读

| 项目 | 本次交付状态 |
| --- | --- |
| 本篇 CPU 的 4 组案例 | 已执行，数值/布局/协议断言通过 |
| 网页交互 | 使用已校验快照，支持切换案例、前进、后退与重置 |
| Python GPU 文件 | 完整源码与统一入口，已做语法/静态检查 |
| CUDA / DSL 后端编译 | 当前编辑环境无 NVIDIA GPU，未做对应目标编译 |
| GPU 正确性与性能 | 未实测；不提供虚构加速比 |
| 原书专用实现 | 固定源码版本与运行入口，按硬件条件复现 |

主要学习来源是用户提供的 LeetCUDA 文档及 [LeetCUDA 项目](https://github.com/xlite-dev/LeetCUDA)、[FFPA 项目](https://github.com/vipshop/ffpa-attn)。这是围绕原文主题重新组织的学习讲解与实验，不直接复制整章。硬件指令语义核对 [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)；布局抽象参考 [CUTLASS CuTe 文档](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cpp/cute)；两种 DSL 继续参考 [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html) 与 [TileLang 文档](https://tilelang.com/)。

[回到 LLM 学习目录](/llm/) · [完整编号顺序](/systems/gpu-learning-path/){next_link}

<link rel="stylesheet" href="/css/leetcuda-labs.css">
<script src="/js/leetcuda-labs.js" defer></script>
'''
    (ROOT / "source/_posts" / f"CUDA算子学习{number:02}-{slug}.md").write_text(body)
    topic.update(
        slug=slug,
        status="published" if topic.get("status") == "published" else "written",
        article=f"CUDA算子学习{number:02}-{slug}.md",
        cpu_cases=4,
        interactive=f"{base}/scenarios.json",
        gpu_validation="not-run",
        upstream=config,
    )
    scenes.append(
        {
            "slug": slug,
            "title": f"（{label}）" + topic["title"].split("：")[0],
            "subtitle": "箭头表示数据 / 完成依赖；不是 GPU 时钟。按从上到下的顺序阅读。",
            "flow": note["flow"],
            "boundary": note["pitfall"],
        }
    )
(ROOT / "docs/learning/leetcuda-source-map.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
(ROOT / "docs/learning/chapter-scenes.json").write_text(json.dumps(scenes, ensure_ascii=False, indent=2) + "\n")
print(f"Generated {len(scenes)} full articles, standalone demos, exercise packs and scenario files.")
