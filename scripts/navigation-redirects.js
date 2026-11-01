'use strict';

// Keep links to earlier category names and article-list pages working.
hexo.extend.generator.register('navigation-redirects', locals => {
  const redirects = [];
  const add = (from, to) => redirects.push({
    path: `${from}index.html`,
    data: `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>栏目已整理 · Wu's Blog</title><meta http-equiv="refresh" content="0;url=/${to}"><link rel="canonical" href="https://wulinze.github.io/${to}"></head><body><p>栏目已整理，请前往<a href="/${to}">新的文章目录</a>。</p></body></html>`
  });
  const pages = Math.ceil(locals.posts.length / hexo.config.index_generator.per_page);
  for (let n = 2; n <= pages; n++) add(`page/${n}/`, `blog/page/${n}/`);
  for (const [old, current] of Object.entries({ 'jvm-java': 'systems', cpp: 'systems', algorithms: 'leetcode', jvm: 'systems' })) {
    add(`categories/${old}/`, `categories/${current}/`);
  }
  // Keep historical pagination links valid after unfinished posts are removed.
  for (let n = 2; n <= 6; n++) add(`categories/algorithms/page/${n}/`, `categories/leetcode/page/${n}/`);
  add('categories/jvm-java/page/2/', 'categories/systems/');
  return redirects;
});
