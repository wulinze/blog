# LeetCUDA 全书学习系列交付验证

本轮补充原 PDF 第 4–40 章，共 37 篇独立正文；博客编号（五）至（四十一）。原 PDF 独立 CuTe 导读与附录 A–E 另有 2 篇补充。分类全部为 LLM 推理。

## 本地已执行

- 37 个 CPU 数值/布局/协议模型，4 个配置各运行一次，共 148 个案例。
- 37 份独立 `demo.py`，各 4 个配置实际运行，共 148 次；无需第三方依赖。
- FP8 格点和 RNE 中点、FP4 奇偶长度 pack/unpack 边界断言。
- GPU harness 的输入生成、语义组合与独立参考，在 CPU 适配器上验证 148 个案例；这项没有执行 Triton/TileLang DSL。
- 37 篇网页共 148 个案例、497 个步骤；切换、前进、后退、重置、终态校验文案、图片解码、Excalidraw JSON 与实验 ZIP 字节核对。
- 手机宽度下无页面横向溢出；检查目录遮罩不挡住实验操作。
- 全站 `./blog verify`：内容、单元测试、Hexo 构建、内部链接与历史地址。
- 补充导读中的代数 Python demo；附录结果检查程序的正反例（构造的元数据测试，不是 GPU 测量）。

## 未执行

本机没有 NVIDIA CUDA GPU。没有宣称 CUDA/Triton/TileLang 后端编译通过，没有设备正确性结果，也没有性能数字。Python AST 与 Ruff 检查仅验证语法/静态代码。

架构专用路径使用固定版本上游源码与对应 runner。正文明确区分语义基线与 TMA/WGMMA/NVFP4 等真实硬件路径。上游运行中出现 SKIP 不记为 GPU PASS。

## 维护

- 原稿：`docs/learning/chapter-notes.json`。
- 章节/页码/博客映射：`docs/learning/leetcuda-source-map.json`。
- 场景定义：`docs/learning/chapter-scenes.json`，已导出 SVG 与 Excalidraw。
- 教学代码：`source/downloads/leetcuda/`。
- 内容生成：`python3 tools/learning/build-series.py`，然后 `python3 tools/learning/update-index.py`。
- 生成器会重建对应 Markdown 与 ZIP；修改正文应同步结构化原稿。
