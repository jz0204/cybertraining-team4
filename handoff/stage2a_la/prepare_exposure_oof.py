"""Freeze the repaired-data, matched exposure comparison without changing old runs."""
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

STAGE = Path('/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a')
ROOT = STAGE / 'la_data_20261024/exposure_oof_20261026'
SOURCE = STAGE / 'la_explore_20261023'
ROLES = ('test', 'val', 'train')
GROUPS = ('physical_building_id', 'parcel_key', 'Prop_ID', 'tile_base')
SEEDS = (42, 43, 44)


def main():
    assert not ROOT.exists(), 'Preserve existing attempts; do not regenerate over an experiment'
    rows = read_csv(SOURCE / 'data/labels_explore.csv')
    assert Counter(r['rebuild_split'] for r in rows) == {'train': 13333, 'val': 2026, 'test': 2653}
    check_groups(rows)
    clean = {r['building_uid']: r for r in rows}
    assert len(clean) == len(rows)
    folds, audits, quarantined, holdouts = [], [], [], []
    for fold in range(5):
        old = read_csv(STAGE / f'poolfix_20261015/training/oof_fold{fold}.csv')
        candidates = [{**clean[r['building_uid']], 'rebuild_split': r['rebuild_split']}
                      for r in old if r['building_uid'] in clean]
        assert all(clean[r['building_uid']]['rebuild_split'] == 'train' for r in candidates)
        kept, removed = quarantine_physical(candidates)
        check_groups(kept)
        assert all({r['type_class'] for r in kept if r['rebuild_split'] == role} == {'0', '1', '2'} for role in ROLES)
        folds.append(kept)
        quarantined += [dict(fold=fold, **r) for r in removed]
        holdouts += [r['building_uid'] for r in kept if r['rebuild_split'] == 'test']
        audits.append(dict(fold=fold, before=dict(Counter(r['rebuild_split'] for r in candidates)),
                           after=dict(Counter(r['rebuild_split'] for r in kept)), quarantined=len(removed)))
    assert len(holdouts) == len(set(holdouts)) == 13333
    assert set(holdouts) == {r['building_uid'] for r in rows if r['rebuild_split'] == 'train'}
    ROOT.mkdir()
    for name in ('data', 'configs'):
        (ROOT / name).mkdir()
    write_csv(ROOT / 'data/labels_clean.csv', rows)
    for fold, kept in enumerate(folds):
        write_csv(ROOT / f'data/oof_fold{fold}.csv', kept)
    write_csv(ROOT / 'data/oof_quarantine.csv', quarantined)
    base = json.loads((SOURCE / 'configs/N1_natural_ce_seed42.json').read_text())
    models = [f'N1_oof_fold{fold}_seed{seed}' for seed in SEEDS for fold in range(5)] + ['N1_full_seed44']
    for name in models:
        seed = int(name.rsplit('seed', 1)[1])
        fold = int(name.split('fold')[1].split('_')[0]) if '_oof_' in name else None
        cfg = dict(base, seed=seed, labels_csv=f'/experiment/data/oof_fold{fold}.csv' if fold is not None else '/experiment/data/labels_clean.csv',
                   out_dir=f'/experiment/runs/{name}', config=f'/experiment/configs/{name}.json')
        save(ROOT / f'configs/{name}.json', cfg)
    protocol = dict(
        created_at=datetime.now(timezone.utc).isoformat(), experiment='repaired_proxy_area_vs_oof_semantics',
        seeds=list(SEEDS), models=models, fold_audit=audits, quarantined_rows=len(quarantined),
        source_labels=str(SOURCE / 'data/labels_explore.csv'), outer_counts=dict(Counter(r['rebuild_split'] for r in rows)),
        source_folds=str(STAGE / 'poolfix_20261015/training/oof_fold{0..4}.csv'),
        oof_policy='Preserve original held-out tiles and inner validation; filter clean outer train; physical quarantine test > val > train. Each train UID held out exactly once per seed.',
        checkpoint_selection='Inner validation type_macro_f1 only; OOF holdout is test, never used for selection.',
        reused_full_models={
            '42': '/stage3/stage2a_la_explore_20261023/runs/N1_natural_ce_seed42/stage2a_best_model.pt',
            '43': '/stage3/stage2a_la_data_20261024/followup_20261025/runs/N1_natural_ce_seed43/stage2a_best_model.pt'},
        runtime='/stage3/stage2a_la_explore_20261023/runtime/II_package',
        runtime_note='Seed43 trainer adds optional NYC sample-budget support; natural sampler with default samples_per_epoch=0 follows the same path. Model and population helpers match.',
        arms={'A_area': 'Shared Ridge(log footprint) + residual HGB(log footprint)',
              'B_semantics_mean3': 'Same Ridge + same HGB settings, log footprint plus soft features from mean probabilities of seeds42/43/44'},
        secondary_arms=[f'B_semantics_seed{s}' for s in SEEDS],
        feature_policy='Train semantics are spatial OOF; val/test semantics use full source-train models. Three-member mean on both sides; no true type as a regressor feature.',
        head=dict(ridge_alphas=[.01, .1, 1., 10., 100.], hgb_seed=43050, max_iter=350,
                  helper='package_stage2a_native3_population._fit_ridge/_fit_hgb/_type_soft_features',
                  internal_validation='Existing RidgeCV and HGB automatic early stopping on outer train only; same row order and seed for both arms.'),
        primary_metric='Test log1p MAE difference B-A (negative favors semantics); report validation separately, not as unseen-city evidence.',
        other_metrics=['raw MAE', 'RMSE', 'R2', 'factor2_on_1_plus_target', 'global relative bias', 'per-class errors', 'tile proxy-sum errors'],
        bootstrap=dict(replicates=5000, seed=20261026, unit='Connected tile components linked by parcel or physical ID within each evaluation split', paired=True),
        inference_precision='FP32, temperature=1', no_la_or_sd=True, transfer_hash_verification=False,
        storage=dict(new_folder='/home/zhjyu11/stage2a-exposure-oof-20261026', estimated_new_gib=5, minimum_home_free_gib=8, minimum_root_free_gib=8),
        limitations=[
            'Estimated population is a repaired area/parcel-use/ACS proxy, not measured building occupancy; type information is partly encoded in the target construction.',
            'N1 consumes imagery and four geometry features. The contrast measures added N1 predicted features, not a pure-image causal effect.',
            'Harris validation and test have already been inspected. This is a frozen diagnostic comparison, not independent external generalization.',
            'Full classifiers use outer validation for checkpoint selection. Only type features are OOF on outer train; exposure training scores are not OOF and are not reported.',
            'OOF members train on fewer buildings than full models; within-class target noise and parcel-to-building allocation remain.',
            'Rows retain the existing building_uid unit, including repeated physical buildings within a split; paired bootstrap groups dependent tiles.',
            'Tile totals cover the sampled inventory only and are not Census population validation.',
            'Bootstrap intervals condition on fitted models; three classifier seeds are reported separately and do not establish broad stability.',
            'Legacy factor2 is abs(log1p(pred)-log1p(target)) <= log(2), not an unshifted population ratio.',
        ])
    save(ROOT / 'PROTOCOL.json', protocol)
    save(ROOT / 'MANIFEST_AUDIT.json', dict(passed=True, folds=audits, quarantined_rows=len(quarantined),
         group_columns=list(GROUPS), zero_cross_role_groups=True, exactly_once_oof_coverage=13333))
    for filename in ('prepare_exposure_oof.py', 'run_exposure_oof.py'):
        shutil.copyfile(Path(__file__).with_name(filename), ROOT / filename)
    print(json.dumps(dict(output=str(ROOT), models=len(models), folds=audits, quarantine=len(quarantined)), indent=2))


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def quarantine_physical(rows):
    seen, kept, removed = {}, [], []
    for role in ROLES:
        selected = [r for r in rows if r['rebuild_split'] == role]
        accepted = []
        for row in selected:
            physical = row['physical_building_id']
            assert physical
            if physical in seen:
                removed.append(dict(building_uid=row['building_uid'], physical_building_id=physical,
                                    removed_role=role, protected_role=seen[physical]))
            else:
                accepted.append(row)
        seen.update({r['physical_building_id']: role for r in accepted})
        kept += accepted
    # Preserve the original training row order for reproducibility.
    ids = {r['building_uid'] for r in kept}
    return [r for r in rows if r['building_uid'] in ids], removed


def check_groups(rows):
    for col in GROUPS:
        assert all(r[col] for r in rows), col
        groups = [{r[col] for r in rows if r['rebuild_split'] == role} for role in ROLES]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]), col


if __name__ == '__main__':
    main()
