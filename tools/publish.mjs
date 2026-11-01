import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

const git = (args, cwd) => execFileSync('git', args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();

export function publishSite({ source, repository, branch = 'master', dryRun = false }) {
  if (!fs.existsSync(path.join(source, 'index.html')) || fs.statSync(path.join(source, 'index.html')).size === 0) {
    throw new Error('Build output is missing or empty; run npm run build first.');
  }
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'wulinze-blog-'));
  const checkout = path.join(temporary, 'site');
  try {
    git(['clone', '--depth', '1', '--branch', branch, repository, checkout]);
    // Preserve repository metadata and custom domain configuration.
    const preserve = new Set(['.git', '.github', 'CNAME', 'README.md']);
    for (const name of fs.readdirSync(checkout)) {
      if (!preserve.has(name)) fs.rmSync(path.join(checkout, name), { recursive: true, force: true });
    }
    for (const name of fs.readdirSync(source)) {
      if (!preserve.has(name)) fs.cpSync(path.join(source, name), path.join(checkout, name), { recursive: true });
    }
    fs.writeFileSync(path.join(checkout, '.nojekyll'), '');
    if (process.env.GITHUB_ACTIONS) {
      git(['config', 'user.name', 'github-actions[bot]'], checkout);
      git(['config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com'], checkout);
    }
    git(['add', '--all'], checkout);
    const changes = git(['diff', '--cached', '--name-only'], checkout);
    if (!changes) {
      console.log('Published files already match this build.');
      return;
    }
    git(['commit', '-m', `deploy: refresh technical blog${process.env.GITHUB_SHA ? ` (${process.env.GITHUB_SHA.slice(0, 12)})` : ''}`], checkout);
    if (dryRun) {
      console.log(`Dry run: prepared ${changes.split('\n').length} changed files; no push performed.`);
      return;
    }
    // A concurrent update causes a normal non-fast-forward rejection, never an overwrite.
    git(['push', 'origin', `HEAD:${branch}`], checkout);
    console.log(`Published ${changes.split('\n').length} changed files to ${repository}.`);
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    publishSite({
      source: path.resolve('public'),
      repository: 'ssh://git@ssh.github.com:443/wulinze/wulinze.github.io.git',
      dryRun: process.argv.includes('--dry-run')
    });
  } catch (error) {
    console.error(error.stderr?.toString().trim() || error.message);
    process.exitCode = 1;
  }
}
