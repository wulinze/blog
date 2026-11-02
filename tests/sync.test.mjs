import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { syncSource } from '../tools/sync.mjs';

test('source upload respects ignored files, retries unchanged uploads, and rejects remote advances', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'blog-sync-test-'));
  const git = (cwd, ...args) => execFileSync('git', args, {
    cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe']
  }).trim();
  const local = path.join(root, 'local');
  const remote = path.join(root, 'remote.git');
  const other = path.join(root, 'other');
  const configure = cwd => {
    git(cwd, 'config', 'user.name', 'Test');
    git(cwd, 'config', 'user.email', 'test@example.org');
    git(cwd, 'config', 'commit.gpgsign', 'false');
  };
  try {
    git(root, 'init', '--bare', '--initial-branch=master', remote);
    git(root, 'clone', remote, local);
    configure(local);
    fs.writeFileSync(path.join(local, '.gitignore'), '.local/\n');
    fs.writeFileSync(path.join(local, 'article.md'), 'initial');
    git(local, 'add', '.');
    git(local, 'commit', '-m', 'initial');
    git(local, 'push', 'origin', 'master');

    fs.mkdirSync(path.join(local, '.local'));
    fs.writeFileSync(path.join(local, '.local', 'private-key'), 'must stay local');
    fs.writeFileSync(path.join(local, 'article.md'), 'edited');
    const commit = syncSource({ cwd: local, message: 'update article' });
    assert.equal(git(root, `--git-dir=${remote}`, 'rev-parse', 'master'), commit);
    assert.equal(git(root, `--git-dir=${remote}`, 'show', 'master:article.md'), 'edited');
    assert(!git(root, `--git-dir=${remote}`, 'ls-tree', '-r', '--name-only', 'master').includes('private-key'));
    assert.equal(syncSource({ cwd: local }), commit);

    git(local, 'checkout', '-b', 'draft-work');
    assert.throws(() => syncSource({ cwd: local }), /Switch to master/);
    git(local, 'checkout', 'master');

    git(root, 'clone', remote, other);
    configure(other);
    fs.writeFileSync(path.join(other, 'article.md'), 'remote edit');
    git(other, 'add', '.');
    git(other, 'commit', '-m', 'remote edit');
    git(other, 'push', 'origin', 'master');
    fs.writeFileSync(path.join(local, 'article.md'), 'local uncommitted edit');
    assert.throws(() => syncSource({ cwd: local }), /Remote master has new commits/);
    assert.equal(fs.readFileSync(path.join(local, 'article.md'), 'utf8'), 'local uncommitted edit');
    assert.equal(git(local, 'diff', '--cached', '--name-only'), '');
    assert.equal(git(local, 'rev-parse', 'HEAD'), commit);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
