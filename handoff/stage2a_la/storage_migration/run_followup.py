"""Run the frozen seed/context controls; keep every prior experiment untouched."""
import gc
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score
from scripts.stage2a.common import Stage2aDataset, build_model_from_config, collate_stage2a, read_csv_rows, seed_all

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
SOURCE = Path('/stage3/stage2a_la_explore_20261023')
sys.path.insert(0, str(SOURCE))
import run_exploration as source
source.previous.ROOT = ROOT


def save(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def required_paths(row, pooling):
    keys = ['crop_path', 'mask_path']
    if pooling == 'mask_m_ring':
        keys.append('mask_R')
    if row.get('scale_valid') == '1':
        keys += ['scale_crop_path', 'scale_mask_path']
        if pooling == 'mask_m_ring':
            keys.append('scale_ring_path')
    return keys


def main():
    protocol = json.loads((ROOT / 'PROTOCOL.json').read_text())
    assert not (ROOT / 'BATCH.json').exists(), 'Do not overwrite an earlier attempt'
    rows = read_csv_rows(SOURCE / 'data/labels_explore.csv')
    groups = {'la': read_csv_rows(SOURCE / 'data/la_input.csv'),
              **{f'harris_{split}': [r for r in rows if r['rebuild_split'] == split] for split in ['val', 'test']}}
    assert len(rows) == 18012 and len(groups['la']) == 21888
    assert len(groups['harris_val']) == 2026 and len(groups['harris_test']) == 2653
    assert torch.cuda.is_available()
    for name in protocol['models']:
        cfg = json.loads((ROOT / f'configs/{name}.json').read_text())
        candidates = read_csv_rows(Path(cfg['labels_csv']))
        assert len({r['building_uid'] for r in candidates}) == len(candidates)
        assert {r['building_uid'] for r in candidates if r['rebuild_split'] != 'train'} == {
            r['building_uid'] for r in rows if r['rebuild_split'] != 'train'}
        assert all(r.get('event_id', '') in ('harvey', 'nyc2018', '') for r in candidates)
        for r in candidates:
            for key in required_paths(r, cfg['pooling_mode']):
                assert Path(r[key]).is_file(), (name, r['building_uid'], key)
    cfg = json.loads((ROOT / 'configs/R1_natural_ring_seed42.json').read_text())
    selected = [next(r for r in rows if r['rebuild_split'] == 'train' and r['type_class'] == str(c)) for c in range(3)]
    ds = Stage2aDataset(selected, train=False, input_mode=cfg['input_mode'], geometry_cols=cfg['geometry_cols'].split(','))
    batch = collate_stage2a([ds[i] for i in range(3)])
    assert batch['ring_mask'].sum() > 0
    seed_all(0)
    model = build_model_from_config(cfg, pretrained=False).cuda().train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(batch['x'].cuda(), geometry=batch['geometry'].cuda(), mask=batch['mask'].cuda(),
                    ring_mask=batch['ring_mask'].cuda(), return_dict=True)
        loss = torch.nn.functional.cross_entropy(out['type_logits'], batch['type_idx'].cuda())
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    save('PREFLIGHT.json', dict(passed=True, ring_nonempty=True, finite_gradients=True, loss=float(loss)))
    del model, batch, out, loss
    gc.collect()
    torch.cuda.empty_cache()
    state = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(), completed=[])
    start = time.monotonic()
    try:
        for name in protocol['models']:
            assert not (ROOT / f'runs/{name}').exists()
            state['current'] = name
            save('BATCH.json', state)
            print('START MODEL', name, flush=True)
            subprocess.run([sys.executable, '-u', str(DATA / 'runtime/II_package/scripts/stage2a/training/train_stage2a.py'),
                            '--config', str(ROOT / f'configs/{name}.json')], check=True)
            source.previous.export(name, ROOT / f'runs/{name}/stage2a_best_model.pt', groups)
            state['completed'].append(name)
            save('BATCH.json', state)
        reference = pd.read_csv(SOURCE / 'data/la_reference.csv')
        reference = reference[reference.primary_single_outline_iou50]
        val = pd.DataFrame(groups['harris_val'])[['building_uid', 'type_class_name']]
        metrics = {}
        for name, folder in [(n, ROOT) for n in protocol['models']] + [
                ('N1_natural_ce_seed42', SOURCE), ('N2_nyc_natural_seed42', DATA)]:
            f = reference.merge(pd.read_csv(folder / f'predictions/{name}/la.csv'), on='building_uid', validate='one_to_one')
            v = val.merge(pd.read_csv(folder / f'predictions/{name}/harris_val.csv'), on='building_uid', validate='one_to_one')
            assert len(f) == 3332 and len(v) == 2026
            truth = f.assessor_type.eq('commercial').to_numpy()
            metrics[name] = {**source.operating_point(truth, f.prob_commercial),
                             'ap': float(average_precision_score(truth, f.prob_commercial)),
                             'harris_val_f1': float(f1_score(v.type_class_name, v.pred_type_class, labels=source.NAMES, average='macro'))}
        paired = {str(seed): {arm: metrics[f'{arm}_seed{seed}']['precision'] - metrics[f'N1_natural_ce_seed{seed}']['precision']
                             for arm in ['N2_nyc_natural', 'R1_natural_ring']} for seed in [42, 43]}
        save('RESULTS.json', dict(metrics=metrics, paired_precision_deltas=paired,
                                 sd_opened=False, inference_precision='FP32', limits=protocol['limitations']))
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start)
        save('BATCH.json', state)
    print('FOLLOWUP COMPLETE', state, flush=True)


if __name__ == '__main__':
    main()
