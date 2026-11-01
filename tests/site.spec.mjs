import { test, expect } from '@playwright/test';

test('homepage is a compact profile with topic links, not a post feed', async ({ page }, testInfo) => {
  await page.goto('/');
  await expect(page.locator('.profile-intro h1')).toContainText('wulinze');
  await expect(page.locator('.post-title')).toHaveCount(0);
  await expect(page.locator('.topic-link')).toHaveCount(4);
  await expect(page.locator('.github-link')).toHaveAttribute('href', 'https://github.com/wulinze');
  await page.screenshot({ path: testInfo.outputPath('home-desktop.png'), fullPage: true });
  await page.getByRole('link', { name: '全部文章' }).click();
  await expect(page).toHaveURL(/\/blog\//);
  await expect(page.locator('.post-title')).toHaveCount(10);
});

test('blog search and reorganized topics work', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/blog/');
  await page.locator('.menu-item-search').click();
  await page.locator('.search-input').fill('CRaC');
  await expect(page.locator('.search-result-list li').first()).toBeVisible();
  await page.goto('/categories/');
  await expect(page.getByRole('link', { name: 'LeetCode 刷题', exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: '系统知识', exact: true })).toBeVisible();
  await page.getByRole('link', { name: 'LLM 推理', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'LLM 推理', exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test('article has highlighted code, rendered formulas and preserved route', async ({ page }) => {
  await page.goto('/2025/10/28/1004.最大连续1的个数III/');
  await expect(page.locator('h1.post-title')).toContainText('最大连续');
  await expect(page.locator('.post-body .katex').first()).toBeVisible();
  await expect(page.locator('.post-body figure.highlight').first()).toBeVisible();
  expect(await page.locator('.katex-error').count()).toBe(0);
});

test('mobile homepage and about page fit the viewport', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.locator('.profile-intro')).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('home-mobile.png'), fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(391);
  await page.getByRole('link', { name: '更多关于我' }).click();
  await expect(page).toHaveURL(/\/about\//);
  await expect(page.getByRole('heading', { name: '你好，我是 wulinze' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(391);
});

test('legacy category and pagination links redirect to the new directory', async ({ page }) => {
  await page.goto('/categories/algorithms/');
  await expect(page).toHaveURL(/\/categories\/leetcode\//);
  await page.goto('/page/2/');
  await expect(page).toHaveURL(/\/blog\/page\/2\//);
});

test('desktop reading guide stays on the left and tracks chapter navigation', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/llm/qwen35-context-parallel/');
  const guide = page.locator('#reading-guide');
  expect((await page.locator('header.header').boundingBox()).y).toBe(0);
  await expect(guide.locator('.guide-context')).toContainText('Qwen3.5');
  const sidebar = await guide.boundingBox();
  const article = await page.locator('.main-inner.post').boundingBox();
  expect(sidebar.x + sidebar.width).toBeLessThan(article.x);
  await expect(guide.locator('.nav-child').first()).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('reading-desktop.png') });
  const target = guide.getByRole('link', { name: '3. GDN：分别处理卷积 halo 与递推状态', exact: true });
  await target.click();
  await expect(target).toHaveAttribute('aria-current', 'location');
  const pinned = await guide.boundingBox();
  expect(pinned.y).toBeGreaterThanOrEqual(0);
  expect(pinned.y).toBeLessThan(30);
  const code = page.locator('.post-body figure.highlight .code pre').first();
  expect(await code.evaluate(el => getComputedStyle(el).fontSize)).toBe('16px');
  expect(await page.locator('.post-body img').count()).toBe(16);
});

test('mobile reading guide opens with keyboard, follows links and closes with Escape', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/llm/qwen35-context-parallel/');
  const toggle = page.getByRole('button', { name: '打开文章目录', exact: true });
  await expect(toggle).toHaveAttribute('aria-expanded', 'false');
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(391);
  await page.screenshot({ path: testInfo.outputPath('reading-mobile.png') });
  await toggle.focus();
  await page.keyboard.press('Enter');
  const close = page.getByRole('button', { name: '关闭文章目录', exact: true });
  await expect(close).toHaveAttribute('aria-expanded', 'true');
  await expect.poll(async () => Math.round((await page.locator('#reading-guide').boundingBox()).x)).toBe(0);
  await page.screenshot({ path: testInfo.outputPath('reading-mobile-guide.png') });
  await page.locator('#reading-guide').getByRole('link', { name: '整体计算流程', exact: true }).click();
  await expect(toggle).toHaveAttribute('aria-expanded', 'false');
  await expect(page).toHaveURL(/#%E6%95%B4%E4%BD%93/);
  await toggle.click();
  await page.keyboard.press('Escape');
  await expect(toggle).toHaveAttribute('aria-expanded', 'false');
  await expect(toggle).toBeFocused();
});
