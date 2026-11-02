import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

// Fetch host keys over HTTPS from GitHub itself; never disable host verification.
const metadata = JSON.parse(execFileSync('curl', [
  '--fail', '--silent', '--show-error', '--location', '--max-time', '30',
  'https://api.github.com/meta'
], { encoding: 'utf8' }));
if (!metadata.ssh_keys?.length) throw new Error('GitHub did not return SSH host keys.');
fs.mkdirSync('.local', { recursive: true });
const knownHosts = path.resolve('.local/github-known-hosts');
fs.writeFileSync(knownHosts, metadata.ssh_keys.map(key =>
  `github.com,ssh.github.com,[ssh.github.com]:443 ${key}\n`
).join(''));
execFileSync('git', ['config', 'core.sshCommand',
  `ssh -o StrictHostKeyChecking=yes -o UserKnownHostsFile="${knownHosts}"` +
  (fs.existsSync('.local/github_ed25519')
    ? ` -o IdentitiesOnly=yes -i "${path.resolve('.local/github_ed25519')}"` : '')
]);
execFileSync('git', ['remote', 'set-url', '--push', 'origin',
  'ssh://git@ssh.github.com:443/wulinze/blog.git'
]);
console.log('GitHub SSH configured for this checkout.');
