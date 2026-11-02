import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

export function syncSource({ cwd = process.cwd(), message = 'docs: update blog' } = {}) {
  const git = (...args) => execFileSync('git', args, {
    cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe']
  }).trim();
  if (git('branch', '--show-current') !== 'master') {
    throw new Error('Switch to master before uploading. No changes were staged.');
  }
  if (git('ls-files', '--unmerged')) throw new Error('Resolve Git conflicts before uploading.');
  git('fetch', 'origin', 'master');
  const behind = Number(git('rev-list', '--count', 'HEAD..origin/master'));
  if (behind) {
    throw new Error('Remote master has new commits. Commit/stash your edits, then git pull --rebase before retrying.');
  }
  // Check credentials and branch permissions before creating a local commit.
  git('push', '--dry-run', 'origin', 'HEAD:master');
  git('add', '--all');
  if (git('diff', '--cached', '--name-only')) git('commit', '-m', message);
  // Never force-push: concurrent remote edits must be reconciled explicitly.
  git('push', 'origin', 'HEAD:master');
  const commit = git('rev-parse', 'HEAD');
  console.log(`Source uploaded: ${commit}`);
  return commit;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    const args = process.argv.slice(2);
    const deploy = args[0] === '--deploy';
    if (deploy) args.shift();
    const commit = syncSource({ message: args.join(' ') || 'docs: update blog' });
    if (deploy) {
      try {
        execFileSync(process.execPath, ['tools/publish.mjs'], {
          stdio: 'inherit', env: { ...process.env, GITHUB_SHA: commit }
        });
      } catch {
        throw new Error('Source was uploaded, but site publishing failed. Fix the error above and retry ./blog deploy.');
      }
    }
  } catch (error) {
    console.error(error.stderr?.toString().trim() || error.message);
    process.exitCode = 1;
  }
}
