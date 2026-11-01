import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { parse } from 'yaml';

const categories = new Set(['JVM 与 Java', 'C++', '算法与数据结构', '系统工程', '数学与计算']);
const files = fs.readdirSync('source/_posts').filter(name => name.endsWith('.md'));
const urls = new Set();
const errors = [];
for (const name of files) {
  try {
    const text = fs.readFileSync(path.join('source/_posts', name), 'utf8');
    const match = text.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/);
    assert(match, 'front matter must start and end with ---');
    const data = parse(match[1]);
    for (const key of ['title', 'date', 'description']) assert(data[key], `missing ${key}`);
    assert(Number.isFinite(Date.parse(data.date)), 'invalid date');
    assert(Array.isArray(data.categories) && data.categories.length === 1 && categories.has(data.categories[0]), 'choose one documented category');
    assert(Array.isArray(data.tags) && data.tags.length > 0, 'tags must be a nonempty list');
    assert(match[2].trim().length > 0, 'empty article body');
    assert(!/!\[[^\]]*\]\([A-Za-z]:[\\/]/.test(match[2]), 'image points to a local computer');
    if (data.permalink) {
      const url = '/' + data.permalink.replace(/^\/+/, '');
      assert(url.endsWith('/'), 'permalink must end with /');
      assert(!urls.has(url), `duplicate permalink ${url}`);
      urls.add(url);
    }
  } catch (error) { errors.push(`${name}: ${error.message}`); }
}
assert(files.length >= 71, `Expected at least the 71 original articles, found ${files.length}`);
if (fs.existsSync('docs/legacy-routes.json')) {
  for (const url of JSON.parse(fs.readFileSync('docs/legacy-routes.json', 'utf8'))) {
    if (!urls.has(url)) errors.push(`Legacy article URL lost: ${url}`);
  }
}
if (errors.length) {
  console.error(errors.join('\n'));
  process.exitCode = 1;
} else console.log(`Content OK: ${files.length} articles, ${urls.size} explicit permalinks.`);
