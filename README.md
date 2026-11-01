# Wu's Blog

个人技术博客，记录 JVM、C++、算法、系统工程和数学计算中的思考与实践。

- 网站：**https://wulinze.github.io/**
- 源码：https://github.com/wulinze/blog
- 发布产物：https://github.com/wulinze/wulinze.github.io

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

71 篇原文章均保留；70 条已有文章地址固定在 front matter 中。部分原文尚未完成，两篇空笔记和两张缺失图片均保留明确说明。详见 [内容迁移记录](docs/content-migration.md)。
