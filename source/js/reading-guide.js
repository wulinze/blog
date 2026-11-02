'use strict';

document.addEventListener('DOMContentLoaded', () => {
  const article = document.querySelector('.main-inner.post');
  const sidebar = document.querySelector('.sidebar');
  const toc = sidebar?.querySelector('.post-toc');
  const toggle = document.querySelector('.sidebar-toggle');

  if (article) {
    const header = article.querySelector('.post-header');
    const category = header?.querySelector('[itemprop="about"] a');
    const breadcrumb = document.createElement('nav');
    breadcrumb.className = 'reading-breadcrumb';
    breadcrumb.setAttribute('aria-label', '当前位置');
    const allPosts = document.createElement('a');
    allPosts.href = '/blog/';
    allPosts.textContent = '全部文章';
    breadcrumb.append(allPosts);
    if (category) {
      const separator = document.createElement('span');
      separator.textContent = '/';
      separator.setAttribute('aria-hidden', 'true');
      breadcrumb.append(separator, category.cloneNode(true));
    }
    header?.prepend(breadcrumb);
  }

  if (!sidebar || !toc || !toggle) return;
  sidebar.id = 'reading-guide';
  sidebar.setAttribute('aria-label', '文章章节导航');
  sidebar.classList.add('reading-guide');
  const context = document.createElement('div');
  context.className = 'guide-context';
  const label = document.createElement('span');
  label.className = 'guide-eyebrow';
  label.textContent = '阅读导航';
  const name = document.createElement('p');
  name.textContent = document.querySelector('.post-title')?.textContent.trim() || document.title;
  const count = document.createElement('span');
  count.className = 'guide-count';
  count.textContent = `${toc.querySelectorAll('.nav > .nav-item').length} 个章节`;
  context.append(label, name, count);
  sidebar.querySelector('.sidebar-inner').prepend(context);
  sidebar.querySelector('.sidebar-nav-toc').textContent = '本文导读';

  // Keep NexT's scroll tracking and expose the active section to screen readers.
  const updateCurrent = () => {
    toc.querySelectorAll('a').forEach(link => {
      if (link.parentElement.classList.contains('active-current')) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  };
  new MutationObserver(updateCurrent).observe(toc, { attributes: true, attributeFilter: ['class'], subtree: true });
  updateCurrent();

  toggle.setAttribute('aria-controls', sidebar.id);
  toggle.setAttribute('tabindex', '0');
  const syncToggle = () => {
    const open = document.body.classList.contains('sidebar-active');
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? '关闭文章目录' : '打开文章目录');
  };
  new MutationObserver(syncToggle).observe(document.body, { attributes: true, attributeFilter: ['class'] });
  syncToggle();
  toggle.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      toggle.click();
    }
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && document.body.classList.contains('sidebar-active')) {
      window.dispatchEvent(new Event('sidebar:hide'));
      toggle.focus();
    }
  });
  toc.addEventListener('click', event => {
    if (event.target.closest('a') && window.innerWidth < 992) window.dispatchEvent(new Event('sidebar:hide'));
  });
  window.matchMedia('(min-width: 992px)').addEventListener('change', () => {
    window.dispatchEvent(new Event('sidebar:hide'));
  });
});
