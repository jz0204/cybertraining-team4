"""Run the frozen OOF classifiers and matched exposure heads in one isolated job."""
import gc
import json
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scripts.stage2a.common import (
    Stage2aDataset, build_model_from_config, classification_metrics, collate_stage2a,
    read_csv_rows, regression_metrics, seed_all,
)
from scripts.stage2a.training import package_stage2a_native3_population as population

from prepare_exposure_oof import GROUPS, SEEDS, check_groups

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/stage3/stage2a_la_explore_20261023')
NAMES = ['residential_small', 'residential_multi', 'commercial']
PROB_COLS = [f'prob_{name}' for name in NAMES]


def main():
    protocol = json.loads((ROOT / 'PROTOCOL.json').read_text())
    assert not (ROOT / 'BATCH.json').exists(), 'Do not overwrite or silently restart an attempt'
    state = dict(status='running', started_at=now(), completed=[], reused_exports=[], current='preflight')
    save('BATCH.json', state)
    start = time.monotonic()
    try:
        rows = read_csv_rows(ROOT / 'data/labels_clean.csv')
        groups = {s: [r for r in rows if r['rebuild_split'] == s] for s in ('train', 'val', 'test')}
        preflight(rows, protocol)
        sys.path.insert(0, '/stage3/stage2a_la_adapt_20261023')
        import run_batch
        run_batch.ROOT = ROOT
        evaluation = {s: groups[s] for s in ('val', 'test')}
        for seed, checkpoint in protocol['reused_full_models'].items():
            name = f'N1_full_seed{seed}'
            state['current'] = f'export_{name}'
            save('BATCH.json', state)
            run_batch.export(name, Path(checkpoint), evaluation)
            state['reused_exports'].append(name)
            save('BATCH.json', state)
        for name in protocol['models']:
            check_storage(protocol)
            assert not (ROOT / f'runs/{name}').exists(), name
            state['current'] = name
            save('BATCH.json', state)
            print('START MODEL', name, flush=True)
            subprocess.run([sys.executable, '-u', f'{protocol["runtime"]}/scripts/stage2a/training/train_stage2a.py',
                            '--config', str(ROOT / f'configs/{name}.json')], check=True)
            cfg = json.loads((ROOT / f'configs/{name}.json').read_text())
            fold_rows = read_csv_rows(Path(cfg['labels_csv']))
            export_rows = {'oof': [r for r in fold_rows if r['rebuild_split'] == 'test']} if '_oof_' in name else evaluation
            run_batch.export(name, ROOT / f'runs/{name}/stage2a_best_model.pt', export_rows)
            state['completed'].append(name)
            save('BATCH.json', state)
        state['current'] = 'matched_exposure_heads'
        save('BATCH.json', state)
        evaluate_exposure(groups, protocol)
        check_checkpoints(protocol)
        state.update(status='complete', current=None)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state.update(finished_at=now(), elapsed_seconds=time.monotonic() - start)
        save('BATCH.json', state)
    print('EXPOSURE COMPARISON COMPLETE', flush=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check_storage(protocol):
    space = {key: shutil.disk_usage(path).free / 2**30 for key, path in [('home', ROOT), ('root', '/stage3')]}
    for key, free in space.items():
        assert free >= protocol['storage'][f'minimum_{key}_free_gib'], (key, free)
    return space


def preflight(rows, protocol):
    assert torch.cuda.is_available(), 'Cluster CUDA required'
    assert len(rows) == len({r['building_uid'] for r in rows}) == 18012
    check_groups(rows)
    assert all(np.isfinite(float(r['estimated_population'])) and float(r['estimated_population']) > 0 for r in rows)
    assert all(np.isclose(float(r['log_footprint_m2']), np.log1p(float(r['footprint_m2']))) for r in rows)
    for row in rows:
        keys = ['crop_path', 'mask_path']
        if row['scale_valid'] == '1':
            keys += ['scale_crop_path', 'scale_mask_path']
        for key in keys:
            assert Path(row[key]).is_file(), (row['building_uid'], key)
    holdouts = []
    for fold in range(5):
        folded = read_csv_rows(ROOT / f'data/oof_fold{fold}.csv')
        check_groups(folded)
        holdouts += [r['building_uid'] for r in folded if r['rebuild_split'] == 'test']
    assert len(holdouts) == len(set(holdouts)) == 13333
    assert set(holdouts) == {r['building_uid'] for r in rows if r['rebuild_split'] == 'train'}
    reference_cfg = json.loads((SOURCE / 'configs/N1_natural_ce_seed42.json').read_text())
    for name in protocol['models']:
        cfg = json.loads((ROOT / f'configs/{name}.json').read_text())
        assert {k for k in cfg if cfg[k] != reference_cfg[k]} <= {'labels_csv', 'config', 'out_dir', 'seed'}
    cfg = json.loads((ROOT / 'configs/N1_full_seed44.json').read_text())
    selected = [next(r for r in rows if r['rebuild_split'] == 'train' and r['type_class'] == str(c)) for c in range(3)]
    dataset = Stage2aDataset(selected, train=False, img_size=224, input_mode=cfg['input_mode'],
                            geometry_cols=cfg['geometry_cols'].split(','), require_ring_mask=False)
    batch = collate_stage2a([dataset[i] for i in range(3)])
    seed_all(0)
    model = build_model_from_config(cfg, pretrained=False).cuda().train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(batch['x'].cuda(), geometry=batch['geometry'].cuda(), mask=batch['mask'].cuda(),
                    ring_mask=batch['ring_mask'].cuda(), return_dict=True)
        loss = torch.nn.functional.cross_entropy(out['type_logits'], batch['type_idx'].cuda())
    loss.backward()
    assert out['type_logits'].shape == (3, 3) and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    import sklearn
    save('PREFLIGHT.json', dict(passed=True, checked_rows=len(rows), groups=list(GROUPS), oof_once=13333,
         finite_gradients=True, cuda_loss=float(loss), gpu=torch.cuda.get_device_name(),
         versions=dict(torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__),
         free_gib=check_storage(protocol), no_la_or_sd=True))
    del model, out, loss, batch
    gc.collect()
    torch.cuda.empty_cache()
    print('PREFLIGHT PASS', flush=True)


def aligned_probabilities(rows, predictions):
    ids = [r['building_uid'] for r in rows]
    assert len(ids) == len(set(ids)) and predictions.building_uid.is_unique
    assert set(predictions.building_uid) == set(ids)
    probs = predictions.set_index('building_uid').loc[ids, PROB_COLS].to_numpy(dtype=np.float64)
    assert np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()
    assert np.allclose(probs.sum(1), 1, atol=1e-5)
    return probs


def head_features(area, probs):
    if probs is None:
        return area, ['log_footprint_m2']
    soft, names = population._type_soft_features(probs, NAMES)
    return np.column_stack([area, soft]), ['log_footprint_m2', *names]


def predict_head(artifact, area, probs):
    features, names = head_features(area, probs)
    assert names == artifact['feature_names']
    base = population._predict_ridge(artifact['ridge_model'], area, artifact['ridge_mean'], artifact['ridge_std'])
    return base + artifact['residual_model'].predict(features)


def spatial_blocks(frame):
    parent = {tile: tile for tile in sorted(frame.tile_base.unique())}

    def root(tile):
        while parent[tile] != tile:
            tile = parent[tile]
        return tile

    for col in ('physical_building_id', 'parcel_key'):
        for tiles in frame.groupby(col).tile_base.unique():
            for tile in tiles[1:]:
                a, b = sorted([root(tiles[0]), root(tile)])
                parent[b] = a
    return frame.tile_base.map(lambda tile: root(tile)).to_numpy()


def paired_interval(delta, blocks, protocol):
    _, inverse = np.unique(blocks, return_inverse=True)
    count = np.bincount(inverse)
    totals = np.bincount(inverse, weights=delta)
    rng = np.random.default_rng(protocol['bootstrap']['seed'])
    weights = rng.multinomial(len(count), np.full(len(count), 1 / len(count)), size=protocol['bootstrap']['replicates'])
    samples = (weights @ totals) / (weights @ count)
    return dict(delta_b_minus_a=float(np.mean(delta)), ci95=np.quantile(samples, [.025, .975]).tolist(),
                spatial_components=len(count), replicates=len(samples))


def evaluate_exposure(groups, protocol):
    probabilities, type_metrics = {}, {}
    for seed in SEEDS:
        frames = [pd.read_csv(ROOT / f'predictions/N1_oof_fold{fold}_seed{seed}/oof.csv', dtype={'building_uid': str}) for fold in range(5)]
        oof = pd.concat(frames, ignore_index=True)
        oof.to_csv(ROOT / f'predictions/oof_seed{seed}.csv', index=False)
        probabilities[seed] = {'train': aligned_probabilities(groups['train'], oof)}
        for split in ('val', 'test'):
            predictions = pd.read_csv(ROOT / f'predictions/N1_full_seed{seed}/{split}.csv', dtype={'building_uid': str})
            probabilities[seed][split] = aligned_probabilities(groups[split], predictions)
        type_metrics[str(seed)] = {s: classification_metrics([int(r['type_class']) for r in groups[s]], probs=p, class_names=NAMES)
                                   for s, p in probabilities[seed].items()}
    probabilities['mean3'] = {s: np.mean([probabilities[seed][s] for seed in SEEDS], axis=0) for s in groups}
    area = {s: population._geometry_features(rows, ['log_footprint_m2']) for s, rows in groups.items()}
    target = {s: population._target_log(rows) for s, rows in groups.items()}
    ridge, mean, std = population._fit_ridge(area['train'], target['train'], protocol['head']['ridge_alphas'])
    residual = target['train'] - population._predict_ridge(ridge, area['train'], mean, std)
    (ROOT / 'heads').mkdir()
    (ROOT / 'exposure_predictions').mkdir()
    (ROOT / 'features').mkdir()
    for split in groups:
        np.savez_compressed(ROOT / f'features/{split}.npz', area=area[split], target_log=target[split],
                            **{f'probs_{seed}': probabilities[seed][split] for seed in [*SEEDS, 'mean3']})
    predictions, head_info = {}, {}
    for name, seed in [('A_area', None), ('B_semantics_mean3', 'mean3'), *[(f'B_semantics_seed{s}', s) for s in SEEDS]]:
        p = None if seed is None else probabilities[seed]['train']
        train_x, names = head_features(area['train'], p)
        hgb, loss = population._fit_hgb(train_x, residual, protocol['head']['hgb_seed'], protocol['head']['max_iter'])
        artifact = dict(format='stage2a_exposure_ablation_v1', name=name, semantic_seed=seed, feature_names=names,
                        ridge_model=ridge, ridge_mean=mean, ridge_std=std, residual_model=hgb)
        with (ROOT / f'heads/{name}.pkl').open('wb') as stream:
            pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with (ROOT / f'heads/{name}.pkl').open('rb') as stream:
            loaded = pickle.load(stream)
        predictions[name] = {}
        for split in ('val', 'test'):
            p = None if seed is None else probabilities[seed][split]
            pred = predict_head(artifact, area[split], p)
            assert np.isfinite(pred).all()
            assert np.array_equal(pred, predict_head(loaded, area[split], p))
            predictions[name][split] = pred
        head_info[name] = dict(features=names, ridge_alpha=float(ridge.alpha_), hgb_loss=loss,
                               hgb_params=hgb.get_params(), hgb_fitted_iterations=int(hgb.n_iter_), pickle_roundtrip=True)
        print('FITTED HEAD', name, 'iterations', hgb.n_iter_, flush=True)
    scores, intervals, tile_scores = {}, {}, {}
    for split in ('val', 'test'):
        frame = pd.DataFrame(groups[split])[['building_uid', 'physical_building_id', 'parcel_key', 'tile_base', 'GEOID', 'type_class_name']]
        frame['spatial_component'] = spatial_blocks(frame)
        frame['target'] = np.expm1(target[split])
        frame['target_log'] = target[split]
        scores[split], intervals[split], tile_scores[split] = {}, {}, {}
        for name, by_split in predictions.items():
            pred = by_split[split]
            raw = np.maximum(0, np.expm1(pred))
            frame[f'{name}_log'] = pred
            frame[f'{name}_raw'] = raw
            scores[split][name] = {}
            for subgroup in ['all', *NAMES]:
                mask = np.ones(len(frame), dtype=bool) if subgroup == 'all' else frame.type_class_name.eq(subgroup).to_numpy()
                yt = frame.target.to_numpy()[mask]
                scores[split][name][subgroup] = dict(n=int(mask.sum()), **regression_metrics(target[split][mask], pred[mask]),
                    total_target=float(yt.sum()), total_pred=float(raw[mask].sum()),
                    relative_bias=float((raw[mask].sum() - yt.sum()) / yt.sum()))
        frame.to_csv(ROOT / f'exposure_predictions/{split}.csv', index=False)
        sum_cols = ['target', *[f'{name}_raw' for name in predictions]]
        tiles = frame.groupby(['tile_base', 'spatial_component'], as_index=False)[sum_cols].sum()
        tiles.to_csv(ROOT / f'exposure_predictions/{split}_tile_sums.csv', index=False)
        for name in predictions:
            raw = tiles[f'{name}_raw'].to_numpy()
            yt = tiles.target.to_numpy()
            tile_scores[split][name] = dict(n=len(tiles), mae=float(np.abs(raw - yt).mean()),
                rmse=float(np.sqrt(np.mean((raw - yt)**2))), wape=float(np.abs(raw - yt).sum() / yt.sum()),
                relative_bias=float((raw.sum() - yt.sum()) / yt.sum()))
            if name == 'A_area':
                continue
            intervals[split][name] = dict(
                log_mae=paired_interval(np.abs(frame[f'{name}_log'] - frame.target_log) - np.abs(frame.A_area_log - frame.target_log), frame.spatial_component, protocol),
                raw_mae=paired_interval(np.abs(frame[f'{name}_raw'] - frame.target) - np.abs(frame.A_area_raw - frame.target), frame.spatial_component, protocol),
                tile_sum_mae=paired_interval(np.abs(tiles[f'{name}_raw'] - tiles.target) - np.abs(tiles.A_area_raw - tiles.target), tiles.spatial_component, protocol))
    result = dict(completed_at=now(), primary='B_semantics_mean3 vs A_area', metrics=scores, paired_bootstrap=intervals,
                  tile_sum_metrics=tile_scores, heads=head_info, semantic_metrics=type_metrics, limitations=protocol['limitations'],
                  no_la_or_sd=True, manuscript_updated=False, model_promoted=False)
    save('RESULTS.json', result)
    lines = ['# Repaired-data exposure comparison', '', 'Fixed diagnostic comparison against estimated population proxies; no actual-occupancy or unseen-city claim.', '',
             '| Split / head | log1p MAE | raw MAE | RMSE | +1 factor2 | relative bias |', '|---|---:|---:|---:|---:|---:|']
    for split, heads in scores.items():
        for name, values in heads.items():
            m = values['all']
            lines.append(f"| {split} / {name} | {m['population_log_mae']:.6f} | {m['population_mae']:.4f} | {m['population_rmse']:.4f} | {m['population_factor2_hit_rate']:.2%} | {m['relative_bias']:+.2%} |")
    lines += ['', 'Paired spatial bootstrap: negative B-A favors semantics. Intervals condition on these fitted models.']
    for split in ('val', 'test'):
        item = intervals[split]['B_semantics_mean3']['log_mae']
        lines.append(f"- {split}: difference {item['delta_b_minus_a']:+.6f}, 95% interval {item['ci95']}, {item['spatial_components']} components.")
    lines += ['', 'Full per-class metrics, seed comparisons and partial-inventory tile sums: RESULTS.json and exposure_predictions/.', '', *[f'- {s}' for s in protocol['limitations']]]
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def check_checkpoints(protocol):
    checked = []
    for name in protocol['models']:
        for filename in ('stage2a_best_model.pt', 'stage2a_last_model.pt'):
            path = ROOT / f'runs/{name}/{filename}'
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            state = checkpoint['model_state']
            assert state and all(torch.isfinite(t).all() for t in state.values() if torch.is_floating_point(t))
            checked.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, finite=True))
            del checkpoint, state
    save('CHECKPOINT_CHECK.json', dict(passed=True, checkpoints=checked, location='cluster original outputs; recheck downloaded copies separately'))


if __name__ == '__main__':
    main()
