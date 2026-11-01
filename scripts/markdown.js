'use strict';

const MarkdownIt = require('markdown-it');
const anchor = require('markdown-it-anchor');
const math = require('@vscode/markdown-it-katex').default;

// Use public parser APIs. Hexo handles fenced-code highlighting before rendering.
const parser = new MarkdownIt({
  html: true,
  breaks: false,
  linkify: true,
  typographer: false
});
parser.use(anchor, {
  level: [2, 3, 4, 5, 6],
  slugify: heading => heading.trim().replace(/\s+/g, '-')
});
parser.use(math, { katex: require('katex'), throwOnError: false });

function render(data, options = {}) {
  return options.inline ? parser.renderInline(data.text) : parser.render(data.text);
}

for (const extension of ['md', 'markdown', 'mkd']) {
  hexo.extend.renderer.register(extension, 'html', render, true);
}
