import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { publishSite } from '../tools/publish.mjs';

test('publishing retains history and custom domain, removes stale output, and is repeatable', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'blog-publish-test-'));
  const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
  const previous = [process.env.GIT_AUTHOR_NAME, process.env.GIT_AUTHOR_EMAIL, process.env.GIT_COMMITTER_NAME, process.env.GIT_COMMITTER_EMAIL];
  const keys = ['GIT_AUTHOR_NAME', 'GIT_AUTHOR_EMAIL', 'GIT_COMMITTER_NAME', 'GIT_COMMITTER_EMAIL'];
  keys.forEach((key, i) => process.env[key] = i % 2 ? 'test@example.org' : 'Test');
  try {
    git('init', '--bare', '--initial-branch=master', 'remote.git');
    git('clone', 'remote.git', 'seed');
    fs.writeFileSync(path.join(root, 'seed', 'old.html'), 'old');
    fs.writeFileSync(path.join(root, 'seed', 'CNAME'), 'example.org');
    git('-C', 'seed', 'add', '.');
    git('-C', 'seed', 'commit', '-m', 'original');
    git('-C', 'seed', 'push', 'origin', 'master');
    const before = git('--git-dir=remote.git', 'rev-parse', 'master');
    const source = path.join(root, 'public');
    fs.mkdirSync(source);
    fs.writeFileSync(path.join(source, 'index.html'), '<h1>New site</h1>');
    const options = { source, repository: path.join(root, 'remote.git') };
    publishSite(options);
    assert.equal(git('--git-dir=remote.git', 'rev-parse', 'master^'), before);
    assert.equal(git('--git-dir=remote.git', 'show', 'master:CNAME'), 'example.org');
    const files = git('--git-dir=remote.git', 'ls-tree', '--name-only', 'master');
    assert(!files.includes('old.html'));
    assert(files.includes('.nojekyll'));
    const after = git('--git-dir=remote.git', 'rev-parse', 'master');
    publishSite(options);
    assert.equal(git('--git-dir=remote.git', 'rev-parse', 'master'), after);
    fs.writeFileSync(path.join(source, 'index.html'), '<h1>Dry run</h1>');
    publishSite({ ...options, dryRun: true });
    assert.equal(git('--git-dir=remote.git', 'rev-parse', 'master'), after);
  } finally {
    keys.forEach((key, i) => previous[i] === undefined ? delete process.env[key] : process.env[key] = previous[i]);
    fs.rmSync(root, { recursive: true, force: true });
  }
});
