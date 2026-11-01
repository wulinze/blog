import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: '**/*.spec.mjs',
  use: { baseURL: 'http://127.0.0.1:4000', headless: true },
  webServer: {
    command: 'python3 -m http.server 4000 --bind 127.0.0.1 --directory public',
    url: 'http://127.0.0.1:4000',
    reuseExistingServer: !process.env.CI,
    timeout: 30000
  }
});
