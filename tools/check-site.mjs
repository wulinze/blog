import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { load } from 'cheerio';

const root = path.resolve('public');
const origin = 'https://wulinze.github.io';
const errors = new Set();
let pages = 0;
function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(item => {
    const file = path.join(dir, item.name);
    return item.isDirectory() ? walk(file) : [file];
  });
}
const files = walk(root);
const generated = new Set(files);
for (const file of files.filter(file => file.endsWith('.html'))) {
  pages++;
  const relative = path.relative(root, file);
  const html = fs.readFileSync(file, 'utf8');
  const $ = load(html);
  if (!$('title').text().trim()) errors.add(`${relative}: no title`);
  if (/https?:\/\/example\.com/.test(html)) errors.add(`${relative}: placeholder domain`);
  if ($('.katex-error').length) errors.add(`${relative}: invalid math expression`);
  for (const element of $('a[href], img[src], script[src], link[href]').toArray()) {
    const value = $(element).attr('src') || $(element).attr('href');
    if (!value || value.startsWith('#') || /^(mailto:|tel:|data:|javascript:)/.test(value)) continue;
    const url = new URL(value, `${origin}/${relative}`);
    if (url.origin !== origin) continue;
    const target = path.join(root, decodeURIComponent(url.pathname));
    if (!generated.has(target) && !generated.has(path.join(target, 'index.html'))) errors.add(`${relative}: missing ${url.pathname}`);
  }
}
for (const file of ['index.html', 'blog/index.html', 'about/index.html', 'categories/index.html', 'tags/index.html', 'archives/index.html', 'search.xml', 'atom.xml', 'sitemap.xml', '.nojekyll']) {
  if (!fs.existsSync(path.join(root, file))) errors.add(`Missing output: ${file}`);
}
const removed = JSON.parse(fs.readFileSync('docs/removed-posts.json', 'utf8'));
const removedUrls = new Set(removed.map(post => post.permalink));
const legacyUrls = JSON.parse(fs.readFileSync('docs/legacy-routes.json', 'utf8')).filter(url => !removedUrls.has(url));
for (const url of legacyUrls) {
  const file = path.join(root, url, 'index.html');
  if (!fs.existsSync(file) || fs.statSync(file).size < 1000) errors.add(`Legacy URL missing/empty: ${url}`);
}
for (const post of removed) {
  if (fs.existsSync(path.join(root, post.permalink, 'index.html'))) errors.add(`Removed draft remains in build: ${post.file}`);
}
assert(pages > legacyUrls.length, 'Expected article and navigation pages');
if (errors.size) {
  console.error([...errors].join('\n'));
  process.exitCode = 1;
} else console.log(`Site OK: ${pages} HTML pages; local links, assets, ${legacyUrls.length} retained legacy routes and ${removed.length} removals verified.`);
