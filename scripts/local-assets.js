'use strict';

const fs = require('node:fs');
const path = require('node:path');

// Only ship the libraries the site uses. No external CDN is needed to read/search.
hexo.extend.generator.register('local-vendor-assets', () => {
  const assets = [];
  function addDirectory(source, destination) {
    for (const entry of fs.readdirSync(source, { withFileTypes: true })) {
      const input = path.join(source, entry.name);
      const output = `${destination}/${entry.name}`;
      if (entry.isDirectory()) addDirectory(input, output);
      else assets.push({ path: output, data: () => fs.createReadStream(input) });
    }
  }
  const modules = path.join(hexo.base_dir, 'node_modules');
  addDirectory(path.join(modules, '@fortawesome/fontawesome-free/css'), 'assets/fontawesome/css');
  addDirectory(path.join(modules, '@fortawesome/fontawesome-free/webfonts'), 'assets/fontawesome/webfonts');
  addDirectory(path.join(modules, 'katex/dist/fonts'), 'assets/katex/fonts');
  for (const [source, destination] of [
    ['katex/dist/katex.min.css', 'assets/katex/katex.min.css'],
    ['hexo-generator-searchdb/dist/search.js', 'assets/local-search/search.js']
  ]) {
    assets.push({ path: destination, data: () => fs.createReadStream(path.join(modules, source)) });
  }
  return assets;
});
