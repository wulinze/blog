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
