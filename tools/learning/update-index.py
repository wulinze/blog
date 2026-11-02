"""Build ordered navigation from the source map; never collapse chapters into groups."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = ROOT / "docs/learning/leetcuda-source-map.json"
data = json.loads(p.read_text())
for item in data.get("auxiliary", []):
    print("Auxiliary:", item)
aux_status = {item["slug"]: item.get("status", "written") for item in data.get("auxiliary", []) if "slug" in item}
data["auxiliary"] = [
    {"title": "CuTe 白皮书导读", "pdf_pages": [277, 309], "slug": "cute-layout-primer", "status": "written"},
    {"title": "附录 A–E：实验工具箱与复现", "pdf_pages": [571, 591], "slug": "cuda-lab-toolbox", "status": "written"},
]
data["coverage"] = {
    "numbered_pdf_chapters": 40,
    "numbered_blog_articles": 41,
    "new_articles_this_batch": 37,
    "supplementary_articles": 2,
    "unwritten_numbered_chapters": [],
    "gpu_validation": "not-run",
    "note": "原文第 1 章由系列（二）性能分析覆盖；（一）为额外 CUDA 入门。",
}
p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
base = [
    ("一", "CUDA 基础：执行模型与内存", "/systems/cuda-basics/"),
    ("二", "性能分析：SM 调度、warp stall 与流水线", "/systems/cuda-overlap/"),
]
entries = base + [
    (x["blog_label"], x["title"], ("/systems/" if x["source_chapters"][0] == 2 else "/llm/") + x["slug"] + "/")
    for x in data["topics"]
]
llm = ROOT / "source/llm/index.md"
text = llm.read_text()
start = text.index("CUDA／LeetCUDA 系列")
end = text.index("\n\n---", start)
new = """CUDA／LeetCUDA 系列按原文章节顺序整理，**一章一篇博客**，接续（一）至（四十一），覆盖 PDF 的 40 个编号章节。CuTe 独立导读与附录另列为补充资料。全部归入「LLM 推理」。

每篇有原理与术语、流程图、交互演示、具体代码与 Triton／TileLang 练习。新增篇章的 CPU 案例已经执行；GPU 基线与原书专用代码提供运行入口，尚未在当前环境做 GPU 编译、正确性及性能实测。

- [完整学习目录、来源页码与实验入口](/systems/gpu-learning-path/)
"""
new += "\n".join(f"- [（{label}）{title}]({url})" for label, title, url in entries)
new += "\n\n补充阅读：\n\n- [CuTe 白皮书：坐标、函数与布局代数](/llm/cute-layout-primer/)\n- [附录：实验工具箱、构建入口与性能结果复现](/llm/cuda-lab-toolbox/)\n"
llm.write_text(text[:start] + new + text[end:])
road = ROOT / "source/_posts/GPU算子学习路线.md"
head = road.read_text().split("---", 2)[1]
body = """## 从这里开始

本系列覆盖 `leetcuda-20261029.pdf` 的 **40 个编号章节**，按原文顺序一章对应一篇博客。额外的 CUDA 入门是（一），原文性能分析是（二），随后接续到（四十一）。分组仅帮助导航，不合并原文章节。

全部文章归入 [LLM 推理](/llm/)。本轮已将后续 37 章展开为独立文章，并补上原 PDF 的 CuTe 导读与附录。下面每个标题都能打开正文。

每篇包含术语、原理、公式、可编辑流程图、可切换案例的逐步交互、完整 CPU demo、GPU 语义基线或原书专用代码入口，以及 Triton／TileLang 练习。交互使用已运行校验的 CPU 计算快照。**当前没有 NVIDIA GPU：后端编译、设备正确性与性能仍为未实测，不把静态检查当设备验收。**

## 建议的学习动作

1. 先读术语和输入输出，手算文中的小例子。
2. 沿流程图看每一步的数据、结果归属与完成条件。
3. 切换交互案例，预测下一步，再观察中间状态。
4. 下载该篇实验包，先跑 CPU，再按硬件条件跑语义基线与原书专用代码。
5. 在独立学生入口中实现练习；正确后才用资源与 profiler 证据验证优化假设。

语义基线、架构专用路径与进阶优化有不同目标。比如一个普通 GEMM 基线验证结果，并不声称实现了 WGMMA；原书固定版本的 Hopper 实验才验证对应指令路径。每篇都列出两者的范围。

## 入门与性能分析

- [（一）CUDA 基础：GPU 结构、执行与内存](/systems/cuda-basics/)：额外入门篇，建立物理与逻辑映射。
- [（二）SM 调度、warp stall、访存优化与流水线](/systems/cuda-overlap/)：对应原文第 1 章，含性能分析流程与可交互实验。

"""
last = None
for topic in data["topics"]:
    if topic["group"] != last:
        body += f"\n## {topic['group']}\n\n"
        last = topic["group"]
    ch = topic["source_chapters"][0]
    url = ("/systems/" if ch == 2 else "/llm/") + topic["slug"] + "/"
    body += f"### [（{topic['blog_label']}）{topic['title']}]({url})\n\n"
    body += f"原文第 {ch} 章 · PDF 第 {topic['pdf_pages'][0]}–{topic['pdf_pages'][1]} 页。{topic['goal']}。\n\n"
    if ch >= 4:
        body += f"[直接打开实验包](/downloads/leetcuda/{ch:02}/{topic['slug']}.zip) · 4 组逐步交互 · 可编辑图 · 独立 Python demo。\n\n"
    body += f"- **Triton：** {topic['triton']}。\n- **TileLang：** {topic['tilelang']}。\n- **验收：** {topic['acceptance']}。\n\n"
body += """## 独立导读与附录

- [CuTe 白皮书学习导读](/llm/cute-layout-primer/)：PDF 第 277–309 页；层级 shape、composition、coalesce、complement、product、divide 与 inverse，配布局交互和代数 demo。
- [实验工具箱、构建入口与复现](/llm/cuda-lab-toolbox/)：PDF 第 571–591 页，覆盖附录 A–E；公共工具、性能表口径、两类构建入口、固定源码索引和参考资料，配端到端摊销交互与记录校验 demo。

附录中的原作者实测数据只作为原资料的实验记录；本系列没有把它们改写成自己的测量结果。封面、目录、术语前言中的公共约定分散到各篇开头，不另制造空内容文章。

## 下载与维护

实验包按篇独立下载，解压后可直接运行 `demo.py`，再阅读 `README.txt` 选择 GPU 后端与固定版本上游实验。每篇的输入契约、基线范围、扩展题与验证状态均保留在正文中。

源码中的 `docs/learning/leetcuda-source-map.json` 维护 PDF 章节、页码与博客编号的映射；`docs/learning/chapter-notes.json` 保存本轮正文的结构化原稿。网页中的图可下载 `.excalidraw` 编辑，Python、CUDA 源码入口与案例都随文章版本保存。
"""
road.write_text("---" + head + "---\n\n" + body)
home = ROOT / "source/index.html"
s = home.read_text().replace(
    "已更新至（四）归约与点积。", "已覆盖 40 个原文章节，接续至（四十一），另附 CuTe 导读与实验工具箱。"
)
home.write_text(s)
print("Updated all 41 numbered entries and two supplements under LLM.")
