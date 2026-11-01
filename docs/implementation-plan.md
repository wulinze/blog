# Hexo Blog Refresh Implementation Plan

**Goal:** Organize 71 existing posts, rebuild the personal technical blog with Hexo/NexT, and publish source and generated site.

**Architecture:** Markdown in `source/_posts` is the source of truth. npm-managed NexT renders Chinese navigation, taxonomy, search and an about page. GitHub Actions validates the build and deploys generated files to the existing website repository with a dedicated deploy key.

**Tech Stack:** Node.js 22, Hexo 8, NexT 8, Markdown-it, KaTeX, GitHub Actions.

- [x] Normalize all 71 posts, classify topics/tags, preserve old permalinks, fix available image references, and record unrecoverable images. Validate counts and metadata against the original checkout.
- [x] Replace broken theme submodules with npm dependencies; configure Chinese site identity, canonical URL, search, navigation, local styles and mathematical rendering. Build with `npm run build`.
- [x] Add source/build checks for missing metadata, duplicate URLs, missing local resources and preserved legacy routes. Run checks before and after migration.
- [x] Add publish workflow and Chinese contributor documentation covering new posts, taxonomy, previews and credentials.
- [x] Inspect desktop/mobile previews, search and sample posts; run `npm run check`, `npm run build`, generated-site validation and `git diff --check`.
- [ ] Review changes, commit and push source; publish static files without overwriting repository history; verify the live website. Report any authentication blocker accurately.
