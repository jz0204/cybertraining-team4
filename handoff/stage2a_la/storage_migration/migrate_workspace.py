"""Move the stopped training workspace to /home, retaining its local-PV path."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import tempfile

SOURCE = Path('/var/lib/rancher/k3s/storage/pvc-03b30e17-a4d2-4680-897b-d054dab4017c_instance-impact-stage3_harvey-workspace')
DEST = Path('/home/zhjyu11/stage2a-workspace')
AUDIT = Path('/home/zhjyu11/stage2a-storage-migration-20261024/audit')
FSTAB = Path('/etc/fstab')
HEADROOM = 8 * 1024**3


def inventory(root):
    result = {}
    for path in [root, *root.rglob('*')]:
        s = path.lstat()
        entry = [stat.S_IFMT(s.st_mode), stat.S_IMODE(s.st_mode), s.st_uid, s.st_gid]
        if stat.S_ISREG(s.st_mode):
            entry.append(s.st_size)
        elif stat.S_ISLNK(s.st_mode):
            entry.append(os.readlink(path))
        elif not stat.S_ISDIR(s.st_mode):
            raise RuntimeError(f'Unexpected special file: {path}')
        result[str(path.relative_to(root))] = entry
    return result


def active_users():
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            links = [proc/'cwd', *list((proc/'fd').iterdir())]
            for link in links:
                try:
                    target = os.readlink(link)
                except FileNotFoundError:
                    continue
                if any(target == base or target.startswith(base+'/') for base in [str(SOURCE), '/stage3']):
                    found.append((proc.name, target))
        except FileNotFoundError:
            continue
    return found


def self_test():
    root = Path(tempfile.mkdtemp(prefix='stage2a-migration-check-'))
    a = root/'a'; b = root/'b'; a.mkdir(); b.mkdir()
    (a/'weights.pt').write_bytes(b'example'); shutil.copy2(a/'weights.pt', b/'weights.pt')
    (a/'link').symlink_to('weights.pt'); (b/'link').symlink_to('weights.pt')
    assert inventory(a) == inventory(b)
    (b/'weights.pt').write_bytes(b'truncated')
    assert inventory(a) != inventory(b)
    assert DEST.parent == Path('/home/zhjyu11') and SOURCE != DEST
    print('PASS: metadata parity, changed-size detection, fixed migration paths; fixture:', root)


def main(execute):
    if socket.gethostname().split('.')[0] != 'geog-zk8-wk3':
        raise SystemExit('This migration is restricted to the wk3 GPU node.')
    if os.geteuid() != 0:
        raise SystemExit('Run on wk3 through sudo; do not send your password to anyone.')
    if any(shutil.which(command) is None for command in ['du', 'mv', 'mount', 'findmnt', 'systemctl']):
        raise SystemExit('Required host utilities are unavailable; no migration performed.')
    if not SOURCE.is_dir() or SOURCE.is_symlink() or os.path.ismount(SOURCE):
        raise SystemExit('Source must be the original, unmounted workspace directory.')
    if DEST.exists() or DEST.is_symlink() or AUDIT.exists():
        raise SystemExit('Destination/audit already exists. Inspect previous attempt; do not overwrite.')
    if os.stat(SOURCE).st_dev == os.stat(DEST.parent).st_dev:
        raise SystemExit('/home must be on a different filesystem.')
    old = FSTAB.read_text()
    if any(len(parts := line.split()) > 1 and parts[1] == str(SOURCE) for line in old.splitlines() if not line.lstrip().startswith('#')):
        raise SystemExit('An fstab entry already manages the source path; inspect it first.')
    busy = active_users()
    if busy:
        raise SystemExit(f'Workspace is in use; stop before migration: {busy[:10]}')
    size = int(subprocess.check_output(['du', '-sb', str(SOURCE)], text=True).split()[0])
    free = shutil.disk_usage(DEST.parent).free
    print(json.dumps(dict(source=str(SOURCE), destination=str(DEST), required_bytes=size,
                          home_free_bytes=free, reserved_headroom_bytes=HEADROOM), indent=2), flush=True)
    if size + HEADROOM > free:
        subprocess.run(['du', '-xh', '--max-depth=1', str(SOURCE)], check=True)
        raise SystemExit('Insufficient /home headroom. No migration performed; select a smaller scope.')
    if not execute:
        return
    before = inventory(SOURCE)
    AUDIT.mkdir(mode=0o755)
    (AUDIT/'before.json').write_text(json.dumps(before, sort_keys=True)+'\n')
    shutil.copy2(FSTAB, AUDIT/'fstab.before')
    # mv retains ownership and metadata; interrupted moves need inspection of BOTH paths.
    subprocess.run(['mv', '--', str(SOURCE), str(DEST)], check=True)
    after = inventory(DEST)
    (AUDIT/'after.json').write_text(json.dumps(after, sort_keys=True)+'\n')
    if before != after:
        raise SystemExit('Metadata verification failed. Data is at destination; preserve both paths and inspect.')
    SOURCE.mkdir()
    subprocess.run(['mount', '--bind', str(DEST), str(SOURCE)], check=True)
    assert os.path.samefile(SOURCE, DEST)
    if FSTAB.read_text() != old:
        raise SystemExit('fstab changed concurrently. Current bind works; inspect before adding persistence.')
    entry = f'{DEST} {SOURCE} none bind,nofail,x-systemd.requires-mounts-for=/home,x-systemd.before=k3s-agent.service 0 0\n'
    fd, temporary = tempfile.mkstemp(prefix='.stage2a-fstab-', dir='/etc')
    os.close(fd)
    shutil.copy2(FSTAB, temporary)
    with open(temporary, 'a') as stream:
        stream.write(('' if old.endswith('\n') else '\n') + entry)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, FSTAB)
    subprocess.run(['systemctl', 'daemon-reload'], check=True)
    subprocess.run(['findmnt', '--fstab', '--target', str(SOURCE)], check=True)
    result = dict(status='moved_and_bound', entries=len(before), source_bytes=size,
                  metadata_verified=True, content_hashes_checked=False,
                  root_free_bytes=shutil.disk_usage('/').free, home_free_bytes=shutil.disk_usage(DEST).free)
    (AUDIT/'COMPLETE.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main(args.execute)
