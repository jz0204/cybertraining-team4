"""Move only the active Stage2a data and output directories on wk3."""
import argparse
import json
import os
import shutil
import socket
import subprocess

import migrate_workspace as migration

ROOT = migration.SOURCE
AUDIT_ROOT = migration.AUDIT.parent
NAMES = ('II_package', 'stage2a_la_explore_20261023', 'stage2a_la_data_20261024')
TARGETS = tuple((ROOT / name, migration.DEST.parent / ('active-' + name),
                 AUDIT_ROOT / ('audit-' + name)) for name in NAMES)


def main(execute):
    if os.geteuid() != 0 or socket.gethostname().split('.')[0] != 'geog-zk8-wk3':
        raise SystemExit('Run through sudo on the wk3 GPU node.')
    total = 0
    for source, destination, audit in TARGETS:
        migration.SOURCE, migration.DEST, migration.AUDIT = source, destination, audit
        migration.main(False)
        total += int(subprocess.check_output(['du', '-sb', str(source)], text=True).split()[0])
    free = shutil.disk_usage(TARGETS[0][1].parent).free
    print(json.dumps(dict(total_required_bytes=total, home_free_bytes=free,
                          reserved_headroom_bytes=migration.HEADROOM)), flush=True)
    if total + migration.HEADROOM > free:
        raise SystemExit('Combined directories exceed safe capacity; nothing moved.')
    if execute:
        for source, destination, audit in TARGETS:
            migration.SOURCE, migration.DEST, migration.AUDIT = source, destination, audit
            migration.main(True)
        print('COMPLETE: all three active directories moved and bound; historical directories retained.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        migration.self_test()
        assert len(TARGETS) == len(set(t[1] for t in TARGETS)) == 3
        assert all(s.parent == ROOT and d.parent == migration.DEST.parent for s, d, _ in TARGETS)
        assert ROOT not in [s for s, _, _ in TARGETS]
        print('PASS: three distinct child directories; whole PVC excluded.')
    else:
        main(args.execute)
