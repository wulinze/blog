# Wu's Blog

个人技术博客，记录 JVM、C++、算法、系统工程和数学计算中的思考与实践。

- 网站：**https://wulinze.github.io/**
- 源码：https://github.com/wulinze/blog
- 发布产物：https://github.com/wulinze/wulinze.github.io

## 日常编辑与一键上传

首次克隆后，在项目目录运行 `./blog setup`。它会把 Node.js 22、npm 缓存和依赖安装到项目本地目录，并配置当前仓库通过 GitHub SSH 443 端口推送；不会修改系统 Node.js。`.local/` 已被 Git 忽略，不能提交其中的密钥或运行环境。

```bash
./blog dev                         # http://127.0.0.1:4000，保存文件后刷新浏览器
./blog draft -- "文章标题"          # 新建 source/_drafts/文章标题.md
./blog ready -- "文章标题"          # 完成后移入 source/_posts/
./blog publish -- "docs: 更新文章"  # 校验、提交源码、推送 GitHub、发布网站
```

用任意编辑器打开项目，直接修改 `source/_posts/` 中的 Markdown、`source/images/` 中的图片或站点配置。预览包含草稿，正式构建不包含草稿；发布前停止预览（Ctrl+C），避免两个 Hexo 进程同时更新缓存。草稿文件也会随源码上传到公开仓库，因此不要在草稿中保存私密内容。

`./blog publish` 会先运行内容检查、单元测试、构建及全站检查，全部通过后，把当前仓库内**全部未被忽略的改动**提交并推送至 `wulinze/blog` 的 `master`，再将网站产物发布至 `wulinze/wulinze.github.io`。执行前可用 `git diff` 和 `git status` 检查改动。默认提交说明为 `docs: update blog`，也可像示例一样传入自定义说明。

- 只上传源码：`./blog sync -- "更新说明"`。
- 只检查：`./blog verify`。
- 源码已上传，但网站发布失败：修复认证或网络后执行 `./blog deploy` 重试。
- 远端已有新提交时，脚本会停止；先提交或暂存本地改动，再运行 `git pull --rebase` 解决差异后重试。脚本不强制推送，不自动丢弃本地改动。
- 仅在 `master` 分支支持一键上传；其他分支请使用常规 Git 工作流。

本地上传需要同时拥有两个仓库的写入权限。运行 `./blog setup` 后用 `git push --dry-run origin HEAD:master` 检查源码写入认证。如果提示 `Permission denied (publickey)`，将本机 SSH 公钥添加到 GitHub 账号的 [SSH keys](https://github.com/settings/keys)。如果项目内存在 `.local/github_ed25519`，配置会使用这个专用密钥；只将对应 `.pub` 文件内容添加到 GitHub。本地一键发布直接使用 SSH，无需配置 Actions 的 `BLOG_DEPLOY_KEY`。

VS Code 用户可通过“终端 → 运行任务”启动预览、检查或一键发布。

## 本地预览

使用 Node.js 22（版本记录在 `.nvmrc`）：

```bash
npm ci
npm run check
npm run build
npm run server
```

浏览器打开 http://localhost:4000 。构建命令会先清除旧缓存，确保主题修改即时生效；`npm run clean` 可单独清除生成缓存；不要直接编辑 `public/` 或 `node_modules/`。

## 新建与修改文章

```bash
npx hexo new post "文章标题"
```

文章存放在 `source/_posts/`，按模板填写标题、固定日期、一句话摘要、分类和标签。选择一个主分类：`LeetCode 刷题`、`系统知识`、`LLM 推理`、`数学与计算`。标签用于细分内容，例如 `垃圾回收`、`动态规划`、`二分查找`。

- 正文从 `##` 标题开始，代码围栏标注 `cpp`、`java`、`bash` 等语言。
- 数学公式使用 `$...$` 和独立行的 `$$...$$`，构建时由 KaTeX 渲染。
- 图片放入 `source/images/`，引用写成 `![说明](/images/example.png)`。
- 已发布文章的 `permalink` 不要随标题修改，以免旧链接失效。
- 尚未完成的新文章可使用 `npx hexo new draft "标题"`，完成后 `npx hexo publish "标题"`。

## 站点与个人入口

`_config.yml` 管理站点名称、地址、分类；`tools/publish.mjs` 管理发布目标；`_config.next.yml` 管理 NexT 菜单、搜索和外观；`theme-overrides/` 保存文章页样式。首页为 `source/index.html`，样式为 `source/css/home.css`，只展示个人介绍、GitHub 和紧凑的分类入口；完整文章列表位于 `/blog/`。LLM 栏目入口位于 `source/llm/index.md`，按并行方式、Cache 池化、算子学习组织；公开笔记后在对应段落补上文章链接。个人简介位于 `source/about/index.md`，后续可在这里添加履历和项目链接。主题通过 npm 安装，无需初始化 Git 子模块。`scripts/markdown.js` 使用 Markdown-it 的公开接口和 KaTeX 渲染正文，`scripts/local-assets.js` 随站点发布搜索、图标和公式资源。

## 检查与发布

全站默认使用深色背景，不随系统外观切换；`source/css/palette.css` 统一维护首页、阅读页、搜索和 404 页的颜色，代码高亮使用 GitHub Dark。原始系统设计图保留图片自身底色与标注。文章页采用左侧章节导航与右侧正文布局。`source/css/reading.css` 管理阅读版式，`source/js/reading-guide.js` 补充章节信息、当前章节标识和手机目录交互；NexT Pisces 提供目录生成与滚动定位。目录在桌面常驻左侧，在手机通过“文章目录”按钮展开。

```bash
npm run check          # 文章元数据与历史地址
npm test               # Markdown 渲染与安全发布流程
npm run build          # 生成 public/
npm run check:site     # 本地链接、图片、公式与历史页面
npx playwright install chromium
npm run test:browser   # 搜索、文章渲染和移动端
npm run deploy        # 校验、构建并通过 SSH 发布
```

本地部署需要 GitHub SSH 写入权限；默认使用 `ssh.github.com:443`，便于在 22 端口受限的网络访问。部署目标为发布仓库的 `master` 分支。该仓库 Pages 的发布来源应为 **Deploy from a branch → master → /(root)**。

修改源码后提交并推送到本仓库 `master`。GitHub Actions 自动检查和构建，并保存 `public-site` 下载产物。

如需每次更新后自动上线，配置一次专用部署 key：

1. 为发布仓库生成专用 SSH key，将公钥添加到 `wulinze/wulinze.github.io` 的 **Settings → Deploy keys**，勾选写入权限。
2. 将对应私钥保存在本源码仓库 **Settings → Secrets and variables → Actions** 的 `BLOG_DEPLOY_KEY` secret 中。
3. 推送源码或手动运行 **Validate and publish blog**。未配置该 secret 时，工作流只构建，不宣称已经发布。

私钥不要提交到仓库。本机已有的个人 SSH key 无需放进 Actions；自动发布应使用单独的部署 key。

## 迁移说明

原始迁移包含 71 篇文章。内容清理后删除 9 篇未完成笔记，并新增 Qwen3.5 Context Parallel 系统设计文章，当前保留 63 篇。保留文章的历史地址不变；删除条目及原因见 [清理清单](docs/removed-posts.json)，迁移过程见 [内容迁移记录](docs/content-migration.md)。未完成的新内容应放在 `source/_drafts/`，完成后再发布。
