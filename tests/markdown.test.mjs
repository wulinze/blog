import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { createRequire } from 'node:module';

test('Markdown renders math and stable, readable Chinese heading anchors', () => {
  const renderers = {};
  vm.runInNewContext(fs.readFileSync('scripts/markdown.js', 'utf8'), {
    require: createRequire(import.meta.url),
    hexo: { extend: { renderer: { register: (ext, output, fn) => renderers[ext] = fn } } }
  });
  const text = '## 标题\n\n公式 $x^2$\n\n[文章](/categories/)';
  const html = renderers.md({ text });
  assert(html.includes('id="标题"'));
  assert(html.includes('class="katex"'));
  assert(html.includes('href="/categories/"'));
  assert.equal(html, renderers.md({ text }));
});
