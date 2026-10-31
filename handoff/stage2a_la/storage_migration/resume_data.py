"""Recover the disk-pressure interruption without overwriting its evidence."""
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import run_data_experiment as experiment


def validate_export(frame, ids):
    assert frame.building_uid.is_unique
    assert set(frame.building_uid) == set(ids)
    probs = frame[['prob_' + name for name in experiment.source.NAMES]].to_numpy()
    assert np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()
    assert np.allclose(probs.sum(1), 1, atol=1e-5)


def main():
    root = experiment.ROOT
    assert not (root / 'FINAL_SELECTION.json').exists()
    assert shutil.disk_usage(root).free >= 8 * 1024**3
    rows, groups = experiment.preflight()
    a1, a2, n2 = experiment.MODELS
    assert (root / f'runs/{a1}/summary.json').is_file()
    for label, group in groups.items():
        validate_export(pd.read_csv(root / f'predictions/{a1}/{label}.csv'),
                        [row['building_uid'] for row in group])
    checked = []
    for folder in [experiment.EXPLORE, root]:
        for path in sorted((folder / 'runs').glob('*/stage2a_best_model.pt')):
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            assert checkpoint['model_state']
            assert all(torch.isfinite(t).all() for t in checkpoint['model_state'].values())
            checked.append(str(path))
            del checkpoint
    interrupted = root / f'runs/{a2}'
    archive = root / 'interrupted_20261024'
    assert interrupted.is_dir() and not (interrupted / 'summary.json').exists()
    assert not archive.exists() and not (root / f'runs/{n2}').exists()
    assert not (root / f'predictions/{a2}').exists()
    archive.mkdir()
    for name in ('BATCH.json', 'train.log'):
        shutil.copy2(root / name, archive / name)
    interrupted.rename(archive / a2)
    experiment.save('MIGRATION_RUNTIME_CHECK.json', dict(
        checked_checkpoints=checked, a1_exports_valid=True, restart_a2_from_seed=42,
        interrupted_archive=str(archive), checked_at=datetime.now(timezone.utc).isoformat()))
    state = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                 completed=[a1], reused=[a1], la_training_rows=0, sandiego_training_rows=0)
    start = time.monotonic()
    try:
        for name in (a2, n2):
            assert not (root / f'runs/{name}').exists()
            state['current'] = name
            experiment.save('BATCH.json', state)
            subprocess.run([sys.executable, '-u', str(root / 'runtime/II_package/scripts/stage2a/training/train_stage2a.py'),
                            '--config', str(root / f'configs/{name}.json')], check=True)
            experiment.source.previous.export(name, root / f'runs/{name}/stage2a_best_model.pt', groups)
            state['completed'].append(name)
            experiment.save('BATCH.json', state)
        state.update(status='complete', selected=experiment.select(rows))
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start)
        experiment.save('BATCH.json', state)
    print('RECOVERED DATA EXPERIMENT COMPLETE', state, flush=True)


if __name__ == '__main__':
    main()
