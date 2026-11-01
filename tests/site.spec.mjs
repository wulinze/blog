import { test, expect } from '@playwright/test';

test('homepage, categories and search work on desktop', async ({ page }, testInfo) => {
  const failures = [];
  page.on('pageerror', error => failures.push(error.message));
  await page.goto('/');
  await expect(page.locator('.journal-intro h1')).toContainText('在问题中探索');
  await expect(page.locator('.post-title')).toHaveCount(10);
  const brand = await page.locator('.site-title').boundingBox();
  const intro = await page.locator('.journal-intro').boundingBox();
  expect(intro.y).toBeGreaterThan(brand.y + brand.height);
  await page.screenshot({ path: testInfo.outputPath('home-desktop.png'), fullPage: true });
  await page.locator('.menu-item-search').click();
  await page.locator('.search-input').fill('CRaC');
  await expect(page.locator('.search-result-list li').first()).toBeVisible();
  await page.locator('.popup-btn-close').click();
  await page.goto('/categories/');
  await expect(page.locator('.category-list-item')).toHaveCount(5);
  expect(failures).toEqual([]);
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
  await expect(page.locator('.journal-intro')).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('home-mobile.png'), fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(391);
  await page.locator('.site-nav-toggle').click();
  await page.locator('.menu-item-about a').click();
  await expect(page).toHaveURL(/\/about\//);
  await expect(page.getByRole('heading', { name: '你好，我是 wulinze' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(391);
});
