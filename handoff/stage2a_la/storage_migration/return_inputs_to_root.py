"""Keep experiment outputs on /home; return only II_package inputs to root."""
import argparse
import json
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import migrate_workspace as migration

SOURCE = Path('/home/zhjyu11/active-II_package')
DEST = migration.SOURCE / 'II_package'
AUDIT = migration.AUDIT.parent / 'return-inputs-audit'


def without_input_mount(text):
    lines = text.splitlines(keepends=True)
    matched = [line for line in lines if not line.lstrip().startswith('#')
               and len(line.split()) > 1 and line.split()[1] == str(DEST)]
    assert len(matched) == 1 and matched[0].split()[0] == str(SOURCE)
    return ''.join(line for line in lines if line != matched[0])


def main():
    assert os.geteuid() == 0 and socket.gethostname().split('.')[0] == 'geog-zk8-wk3'
    if AUDIT.exists():
        receipt = AUDIT / 'COMPLETE.json'
        if receipt.is_file():
            result = json.loads(receipt.read_text())
            if (result.get('status') == 'inputs_returned_outputs_retained'
                    and result.get('metadata_verified') is True
                    and not SOURCE.exists() and not SOURCE.is_symlink()
                    and DEST.is_dir() and not os.path.ismount(DEST)):
                print('Already completed; no files moved. Completion record:', receipt)
                return
        raise SystemExit(f'An earlier attempt exists at {AUDIT}; inspect it before retrying. No files moved.')
    if not SOURCE.is_dir() or SOURCE.is_symlink():
        raise SystemExit(f'Expected input directory is missing or is a symlink: {SOURCE}. No files moved.')
    assert os.path.ismount(DEST) and os.path.samefile(SOURCE, DEST)
    assert not migration.active_users(), 'Workspace is still in use'
    required = int(subprocess.check_output(['du', '-sb', str(SOURCE)], text=True).split()[0])
    free = shutil.disk_usage(DEST.parent).free
    assert free >= required + migration.HEADROOM, (required, free)
    old = migration.FSTAB.read_text()
    updated = without_input_mount(old)
    before = migration.inventory(SOURCE)
    AUDIT.mkdir()
    (AUDIT / 'before.json').write_text(json.dumps(before))
    shutil.copy2(migration.FSTAB, AUDIT / 'fstab.before')
    subprocess.run(['umount', str(DEST)], check=True)
    assert not any(DEST.iterdir()), 'Underlying mountpoint is not empty; preserve both paths'
    subprocess.run(['mv', '-T', '--', str(SOURCE), str(DEST)], check=True)
    after = migration.inventory(DEST)
    (AUDIT / 'after.json').write_text(json.dumps(after))
    assert before == after, 'Inventory mismatch; preserve both paths and inspect'
    assert migration.FSTAB.read_text() == old, 'Concurrent fstab change; inspect persistence'
    fd, temp = tempfile.mkstemp(prefix='.stage2a-fstab-', dir='/etc')
    os.close(fd)
    shutil.copy2(migration.FSTAB, temp)
    with open(temp, 'w') as f:
        f.write(updated)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, migration.FSTAB)
    subprocess.run(['systemctl', 'daemon-reload'], check=True)
    result = dict(status='inputs_returned_outputs_retained', metadata_verified=True,
                  root_free_bytes=shutil.disk_usage(DEST).free,
                  home_free_bytes=shutil.disk_usage('/home').free)
    (AUDIT / 'COMPLETE.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        line = f'{SOURCE} {DEST} none bind 0 0\n'
        assert without_input_mount('# keep\n' + line + '/a /b none bind 0 0\n') == '# keep\n/a /b none bind 0 0\n'
        for invalid in ['', line + line]:
            try:
                without_input_mount(invalid)
            except AssertionError:
                pass
            else:
                raise AssertionError('Ambiguous mount accepted')
        print('PASS: exact fstab entry removal; absent/duplicate entries rejected')
    elif args.execute:
        main()
    else:
        p.error('Choose --execute or --self-test')
