"""Prepare the owned 720/174 NYC versus multicity comparison without training."""

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from shapely import wkt
from shapely.geometry import box
from shapely.ops import transform
from shapely.strtree import STRtree

import prepare_reliable_transfer as original

SOURCE = original.ROOT
ROOT = SOURCE.parent / 'multicity_transfer_20261028'
SOURCE_SHA = '0a9fe81ed273067a7a01f962409ce779f9055a501211a2aa33faa6af0161411f'
SEED = 20261027
CITIES = ['nyc', 'cambridge', 'seattle']
STRATA = original.CLASSES + ['auxiliary']
A_QUOTAS = {
    'residential_small': {'1': 60, '2': 30, '3': 30},
    'residential_multi': {'4-8': 60, '9+': 60},
    'commercial': {'retail_service': 60, 'office': 60},
    'institutional': {'religion': 40, 'education': 40, 'civic_cultural_health': 40},
    'other': {'warehouse': 60, 'industrial': 60},
}
B_QUOTAS = {
    'nyc': {
        'residential_small': {'1': 10, '2': 15, '3': 15},
        'residential_multi': {'4-8': 20, '9+': 20},
        'commercial': {'retail_service': 20, 'office': 20},
        'institutional': {'religion': 10, 'education': 10, 'civic_cultural_health': 20},
        'other': {'warehouse': 8, 'industrial': 32},
    },
    'cambridge': {
        'residential_small': {'1': 12, '2': 14, '3': 14},
        'residential_multi': {'4-8': 20, '9+': 20},
        'commercial': {'retail_service': 20, 'office': 20},
        'institutional': {'religion': 5, 'education': 20, 'civic_cultural_health': 15},
        'other': {'warehouse': 26, 'industrial': 14},
        'auxiliary': {'OUTBLDG': 60},
    },
    'seattle': {
        'residential_small': {'1': 38, '2': 1, '3': 1},
        'residential_multi': {'4-8': 20, '9+': 20},
        'commercial': {'retail_service': 20, 'office': 20},
        'institutional': {'religion': 25, 'education': 10, 'civic_cultural_health': 5},
        'other': {'warehouse': 26, 'industrial': 14},
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'acquire', 'freeze', 'self-check'])
    args = parser.parse_args()
    inputs = load_inputs()
    if args.action == 'self-check':
        self_check(inputs)
        return
    if args.action == 'prepare':
        prepare(inputs)
        return
    assert ROOT.is_dir() and not (ROOT / 'PROTOCOL.json').exists(), 'Missing preparation or already frozen'
    try:
        verify_sources()
        {'acquire': acquire, 'freeze': freeze}[args.action](inputs)
    except Exception as exc:
        save(ROOT / 'BLOCKER.json', dict(action=args.action, error=str(exc), utc=now()))
        raise


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def subtype(row):
    city, stratum, evidence = row['city'], row['stratum'], row['evidence']
    if stratum == 'auxiliary':
        return str(evidence['feature_code']) if city == 'nyc' else 'OUTBLDG'
    fields = evidence['pluto'] if city == 'nyc' else evidence['landuse2025'] if city == 'cambridge' else evidence['parcel']
    if stratum.startswith('residential'):
        units = int(float(fields['unitsres'] if city == 'nyc' else fields['residential_units'] if city == 'cambridge' else fields['NR_UNITS']))
        return str(units) if stratum == 'residential_small' else '4-8' if units <= 8 else '9+'
    code = fields['bldgclass'] if city == 'nyc' else fields['use_code'] if city == 'cambridge' else fields['PUC']
    if stratum == 'commercial':
        office = code.startswith('O') if city == 'nyc' else code == '340' if city == 'cambridge' else code == 106
        return 'office' if office else 'retail_service'
    if stratum == 'institutional':
        if city == 'nyc':
            return 'religion' if code.startswith('M') else 'education' if code.startswith('W') else 'civic_cultural_health'
        if city == 'cambridge':
            return 'religion' if code == '960' else 'civic_cultural_health' if code == '930' else 'education'
        return 'religion' if code == 165 else 'education' if code in {184, 185, 193} else 'civic_cultural_health'
    if city == 'nyc':
        return 'warehouse' if code.startswith('E') else 'industrial'
    if city == 'cambridge':
        return 'warehouse' if code in {'316', '401'} else 'industrial' if code in {'400', '407'} else 'utility'
    return 'warehouse' if code in {195, 252} else 'industrial'


def actual_context(metadata):
    transformer = Transformer.from_crs(metadata['geometry_crs'], 32618, always_xy=True)
    return transform(transformer.transform, box(*metadata['context_bounds']))


def planned_context(row):
    center = wkt.loads(row['geometry_wkt']).centroid
    half = row['wide_m'] / 2 + 2
    return box(center.x - half, center.y - half, center.x + half, center.y + half)


def load_inputs():
    assert sha(SOURCE / 'dataset.json') == SOURCE_SHA
    frozen = load(SOURCE / 'dataset.json')
    old = {r['sample_id']: r for r in frozen}
    validation = [r for r in frozen if r['split'] == 'val']
    assert len(validation) == 174 and all(r['city'] == 'nyc' for r in validation)
    parcels = {r['parcel'] for r in validation}
    tree = STRtree([actual_context(r['image_metadata']) for r in validation])
    excluded = set(load(SOURCE / 'DATASET_AUDIT.json')['rejected_imagery'])
    rows, queues, omissions = {}, defaultdict(list), []
    for city in CITIES:
        for row in load(SOURCE / (city + '_pool.json')):
            sid = row['sample_id']
            if sid in excluded or (city == 'nyc' and row['split'] != 'train'):
                continue
            if city == 'nyc' and (row['parcel'] in parcels or len(tree.query(planned_context(row), predicate='intersects'))):
                omissions.append(sid)
                continue
            row['coarse_subtype'] = subtype(row)
            assert row['reference_role'] == ('auxiliary' if row['stratum'] == 'auxiliary' else 'primary')
            assert row['reference_use'] == (None if row['stratum'] == 'auxiliary' else row['stratum'])
            assert sid not in rows
            rows[sid] = row
            queues['|'.join((city, row['stratum'], row['coarse_subtype']))].append(sid)
    for queue in queues.values():
        queue.sort(key=lambda sid: (sid not in old, original.digest(sid)))
    return dict(old=old, validation=validation, rows=rows, queues=dict(queues),
                known_failures=sorted(excluded), spatial_exclusions=omissions)


def select(inputs, excluded=()):
    rows, queues, excluded = inputs['rows'], inputs['queues'], set(excluded)

    def pick(city, stratum, kind, count, used):
        chosen = []
        for sid in queues.get('|'.join((city, stratum, kind)), []):
            parcel = (city, rows[sid]['parcel'])
            if sid in excluded or parcel in used:
                continue
            chosen.append(sid)
            used.add(parcel)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f'Quota shortfall: {city}/{stratum}/{kind}: need {count}, have {len(chosen)}')
        return chosen

    a, used = [], set()
    for kind, count in [('1004', 40), ('5110', 80)]:
        a += pick('nyc', 'auxiliary', kind, count, used)
    b = [sid for kind, count in [('1004', 20), ('5110', 40)]
         for sid in [sid for sid in a if rows[sid]['coarse_subtype'] == kind][:count]]
    # Reserve all A auxiliary parcels before selecting the shared NYC primary rows.
    for city, strata in B_QUOTAS.items():
        for stratum, kinds in strata.items():
            for kind, count in kinds.items():
                b += pick(city, stratum, kind, count, used)
    a += [sid for sid in b if rows[sid]['city'] == 'nyc' and rows[sid]['reference_use']]
    used = {(rows[sid]['city'], rows[sid]['parcel']) for sid in a}
    for stratum, kinds in A_QUOTAS.items():
        for kind, count in kinds.items():
            need = count - sum(rows[sid]['stratum'] == stratum and rows[sid]['coarse_subtype'] == kind for sid in a)
            if need:
                a += pick('nyc', stratum, kind, need, used)
    selected = dict(A_nyc=a, B_multicity=b)
    check_selection(inputs, selected)
    return selected


def check_selection(inputs, selected):
    rows = inputs['rows']
    validation = inputs['validation']
    vkeys = {(r['city'], r['parcel']) for r in validation}
    signatures = []
    for name, ids in selected.items():
        rr = [rows[sid] for sid in ids]
        assert len(ids) == len(set(ids)) == len({(r['city'], r['parcel']) for r in rr}) == 720
        assert Counter(r['stratum'] for r in rr) == dict.fromkeys(STRATA, 120)
        assert Counter(r['reference_role'] for r in rr) == {'primary': 600, 'auxiliary': 120}
        assert not vkeys & {(r['city'], r['parcel']) for r in rr}
        assert not set(ids) & {r['sample_id'] for r in validation}
        assert Counter(r['city'] for r in rr) == ({'nyc': 720} if name == 'A_nyc' else {'nyc': 260, 'cambridge': 260, 'seattle': 200})
        if name == 'B_multicity':
            assert all(sum(r['city'] == c and r['stratum'] == s for r in rr) == 40 for c in CITIES for s in original.CLASSES)
        signatures.append(Counter((r['stratum'], r['coarse_subtype']) for r in rr if r['reference_use']))
    assert signatures[0] == signatures[1]
    shared = set(selected['A_nyc']) & set(selected['B_multicity'])
    assert shared == {sid for sid in selected['B_multicity'] if rows[sid]['city'] == 'nyc'} and len(shared) == 260
    assert validation == [r for r in load(SOURCE / 'dataset.json') if r['split'] == 'val']
    assert Counter(r['stratum'] for r in validation) == dict(zip(STRATA, [30, 30, 30, 30, 29, 25]))


def combined_rows(inputs, selected):
    train = [dict(inputs['rows'][sid], original_split=inputs['rows'][sid]['split'], split='train',
                  experiment_role='development_training') for sid in sorted(set(sum(selected.values(), [])))]
    return train + inputs['validation']


def prepare(inputs):
    assert not ROOT.exists(), f'Preserve existing preparation: {ROOT}'
    self_check(inputs)
    selected = select(inputs)
    for folder in ['', 'raw', 'images', 'source_windows', 'image_metadata', 'logs', 'selection', 'acquisition']:
        (ROOT / folder).mkdir(exist_ok=True)
    paths = [SOURCE / name for name in ['dataset.json', 'PROTOCOL.json', 'INPUT_SHA256.json', 'DATASET_AUDIT.json']]
    paths += [SOURCE / (city + '_pool.json') for city in CITIES]
    for city in CITIES:
        for path in sorted((SOURCE / 'raw').glob('naip_' + city + '*.json')):
            shutil.copy2(path, ROOT / 'raw' / path.name)
            paths.append(path)
    paths += [Path(original.__file__), Path(original.__file__).with_name('vlm_pilot.py'),
              Path(original.__file__).with_name('vlm_multiclass_data.py')]
    save(ROOT / 'PROVENANCE.json', dict(source=str(SOURCE), source_sha256={str(p): sha(p) for p in paths},
        selector_sha256=sha(Path(__file__)), selection_seed=SEED,
        ordering='original cached imagery first, then prepare_reliable_transfer.digest(sample_id)', utc=now()))
    save(ROOT / 'REPLACEMENT_QUEUES.json', inputs['queues'])
    save(ROOT / 'REJECTIONS.json', {})
    save(ROOT / 'INITIAL_SELECTION.json', selected)
    save(ROOT / 'selection' / 'round_000.json', selected)
    save(ROOT / 'RETIRED_TESTS.json', dict(source_dataset_sha256=SOURCE_SHA,
        role='Retired external tests: development/training only; subsequent scores are not heldout.',
        ids=[sid for sid, r in inputs['old'].items() if r['city'] != 'nyc'],
        original_artifacts_unchanged=True, final_test_cities_acquired=[]))
    stage_candidates(inputs, selected)
    missing = [r['sample_id'] for r in combined_rows(inputs, selected) if not (ROOT / 'image_metadata' / (r['sample_id'] + '.json')).exists()]
    save(ROOT / 'PREPARATION.json', dict(status='selected_pending_acquisition', selected_counts={k: len(v) for k, v in selected.items()},
        validation=174, missing_ids=missing, missing_by_city=dict(Counter(inputs['rows'][sid]['city'] for sid in missing)),
        known_image_failures=inputs['known_failures'], validation_context_exclusions=inputs['spatial_exclusions']))
    print('PREPARED', len(missing), 'missing paired-image sets', flush=True)


def stage_candidates(inputs, selected):
    rows = combined_rows(inputs, selected)
    for row in rows:
        sid = row['sample_id']
        if sid not in inputs['old']:
            continue
        relatives = ['image_metadata/' + sid + '.json', 'source_windows/' + sid + '.tif']
        relatives += ['images/' + sid + '_' + view + '_' + kind + '.' + ext
                      for view in ['near', 'wide'] for kind, ext in [('rgb', 'jpg'), ('mask', 'png'), ('marked', 'jpg')]]
        for relative in relatives:
            source, destination = SOURCE / relative, ROOT / relative
            assert source.is_file(), source
            if destination.exists():
                assert destination.resolve() == source.resolve(), destination
            else:
                destination.symlink_to(source)
    for city in CITIES:
        save(ROOT / (city + '_candidates.json'), [r for r in rows if r['city'] == city])


def verify_sources():
    for filename, expected in load(ROOT / 'PROVENANCE.json')['source_sha256'].items():
        assert sha(Path(filename)) == expected, ('Source changed', filename)


def acquire(inputs):
    assert load(ROOT / 'REPLACEMENT_QUEUES.json') == inputs['queues']
    rejected = load(ROOT / 'REJECTIONS.json')
    round_number = len(list((ROOT / 'acquisition').glob('round_*.json')))
    original.ROOT = ROOT
    while True:
        selected = select(inputs, rejected)
        stage_candidates(inputs, selected)
        save(ROOT / 'selection' / f'round_{round_number:03d}.json', selected)
        rows = combined_rows(inputs, selected)
        missing = [r['sample_id'] for r in rows if not (ROOT / 'image_metadata' / (r['sample_id'] + '.json')).exists()]
        failures = []
        for city in CITIES:
            if not any(inputs['rows'][sid]['city'] == city for sid in missing):
                continue
            with rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024):
                original.images(city)
            failures += load(ROOT / 'logs' / (city + '_image_failures.json'))
        quality_errors = {'full context unavailable', 'missing target/context pixels', 'target truncated',
                          'Target mask must be nonempty and aligned with the RGB crop'}
        infrastructure = [f for f in failures if any(e['error'] not in quality_errors for e in f['errors'])]
        save(ROOT / 'acquisition' / f'round_{round_number:03d}.json',
             dict(utc=now(), requested_ids=missing, failures=failures, infrastructure_failures=infrastructure))
        if infrastructure:
            raise RuntimeError('Acquisition infrastructure failure; retain selection and retry after recovery: ' + str(infrastructure[:2]))
        for failure in failures:
            rejected[failure['sample_id']] = failure
        validation_tree = STRtree([actual_context(r['image_metadata']) for r in inputs['validation']])
        for row in rows:
            path = ROOT / 'image_metadata' / (row['sample_id'] + '.json')
            if row['city'] == 'nyc' and row['split'] == 'train' and path.exists():
                if len(validation_tree.query(actual_context(load(path)), predicate='intersects')):
                    rejected[row['sample_id']] = dict(reason='Actual context intersects fixed NYC validation')
        save(ROOT / 'REJECTIONS.json', rejected)
        assert all((ROOT / 'image_metadata' / (sid + '.json')).exists() or sid in rejected for sid in missing)
        if not any(r['sample_id'] in rejected for r in rows):
            save(ROOT / 'FINAL_SELECTION.json', selected)
            save(ROOT / 'ACQUISITION_REPORT.json', dict(status='acquired_pending_final_audit', utc=now(),
                selected_unique=len(rows), cached=sum(r['sample_id'] in inputs['old'] for r in rows),
                acquired_by_city=dict(Counter(r['city'] for r in rows if r['sample_id'] not in inputs['old'])),
                rejected=rejected, rounds=round_number + 1, training_started=False))
            print('ACQUISITION_COMPLETE', len(rows), 'unique selected records', flush=True)
            return
        round_number += 1


def audit_image(row):
    sid, metadata = row['sample_id'], row['image_metadata']
    source = ROOT / 'source_windows' / (sid + '.tif')
    assert not metadata['source_limited'] and sha(source) == metadata['source_sha256'], sid
    with rasterio.open(source) as src:
        assert src.count == 3 and src.dtypes == ('uint8',) * 3, sid
        assert np.allclose(src.bounds, metadata['context_bounds'], rtol=0, atol=1e-6), sid
        assert abs(src.res[0] - metadata['source_gsd']) < 1e-9, sid
        assert abs(src.width * src.res[0] - metadata['wide_m']) < 1e-6, sid
        geometry = transform(Transformer.from_crs(row['geometry_crs'], src.crs, always_xy=True).transform,
                             wkt.loads(row['geometry_wkt']))
        values = src.read(indexes=[1, 2, 3])
        mask = rasterize([(geometry, 255)], out_shape=(src.height, src.width), transform=src.transform, dtype='uint8')
        missing = np.all(values == 0, axis=0)
        assert mask.any() and missing.mean() < .05 and missing[mask > 0].mean() < .01, sid
        center = (~src.transform) * (geometry.centroid.x, geometry.centroid.y)
        near = round(metadata['near_m'] / src.res[0])
        left, top = round(center[0] - near / 2), round(center[1] - near / 2)
        close = Image.fromarray(mask).crop((left, top, left + near, top + near))
        retained = float(np.asarray(close).sum() / mask.sum())
        assert retained > .99, (sid, retained)
    for view in ['near', 'wide']:
        for kind in ['rgb', 'mask', 'marked']:
            with Image.open(ROOT / row[view + '_' + kind]) as im:
                assert im.size == (512, 512), (sid, view, kind)
                im.load()
                if kind == 'mask':
                    assert im.getbbox() is not None, (sid, view)
    return dict(sample_id=sid, target_retained=retained, missing_context_fraction=float(missing.mean()),
                missing_target_fraction=float(missing[mask > 0].mean()))


def freeze(inputs):
    selected = load(ROOT / 'FINAL_SELECTION.json')
    assert selected == select(inputs, load(ROOT / 'REJECTIONS.json'))
    check_selection(inputs, selected)
    original.ROOT = ROOT
    original.freeze()
    rows = load(ROOT / 'dataset.json')
    by_id = {r['sample_id']: r for r in rows}
    assert len(rows) == 1354 and len(by_id) == len(rows)
    assert [r for r in rows if r['split'] == 'val'] == inputs['validation']
    old_hashes = load(SOURCE / 'INPUT_SHA256.json')
    image_checks, hashes = [], {}
    for index, row in enumerate(rows):
        image_checks.append(audit_image(row))
        sid = row['sample_id']
        relatives = [row[v + '_' + k] for v in ['near', 'wide'] for k in ['rgb', 'mask', 'marked']]
        relatives += ['image_metadata/' + sid + '.json', 'source_windows/' + sid + '.tif']
        for relative in relatives:
            hashes[relative] = sha(ROOT / relative)
            if sid in inputs['old'] and relative.startswith('images/'):
                assert hashes[relative] == old_hashes[relative], ('Cached input changed', relative)
        if (index + 1) % 100 == 0:
            print('AUDIT_IMAGES', index + 1, '/', len(rows), flush=True)
    save(ROOT / 'IMAGE_AUDIT.json', image_checks)
    prior_protocol = load(SOURCE / 'execution_v3' / 'PROTOCOL.json')
    arms = {}
    for name, ids in selected.items():
        folder = ROOT / name
        folder.mkdir(exist_ok=True)
        for directory in ['images', 'image_metadata', 'source_windows']:
            link = folder / directory
            if not link.exists():
                link.symlink_to('../' + directory, target_is_directory=True)
            assert link.resolve() == (ROOT / directory).resolve()
        arm_rows = [by_id[sid] for sid in ids] + inputs['validation']
        assert len(arm_rows) == 894
        save(folder / 'dataset.json', arm_rows)
        counts = dict(Counter(r['city'] + ':' + r['split'] + ':' + r['stratum'] for r in arm_rows))
        save(folder / 'MANIFEST.json', dict(train_ids=ids, val_ids=[r['sample_id'] for r in inputs['validation']],
            counts=counts, subtype_counts=dict(Counter(r['city'] + ':' + r['stratum'] + ':' + r['coarse_subtype'] for r in arm_rows if r['split'] == 'train')),
            previous_external_test_role='retired_to_development_training', test_ids=[]))
        wanted = {r[v + '_' + k] for r in arm_rows for v in ['near', 'wide'] for k in ['rgb', 'mask', 'marked']}
        wanted |= {directory + '/' + r['sample_id'] + ext for r in arm_rows
                   for directory, ext in [('image_metadata', '.json'), ('source_windows', '.tif')]}
        inputs_hashes = {p: hashes[p] for p in sorted(wanted)}
        inputs_hashes.update({'dataset.json': sha(folder / 'dataset.json'), 'MANIFEST.json': sha(folder / 'MANIFEST.json')})
        save(folder / 'INPUT_SHA256.json', inputs_hashes)
        protocol = {k: prior_protocol[k] for k in ['seeds', 'role_class_weights', 'dino', 'primary_metrics']}
        protocol['dino'] = dict(protocol['dino'], mode='finetune', encoder_modes=['last2blocks+norm'],
            fixed_epochs=True, finetune_epochs=15, finetune_patience=15, early_stopping=False,
            optimizer_steps_per_epoch=45, optimizer_updates_per_seed=675,
            command_flags=['--mode', 'finetune', '--fixed-epochs'])
        assert 720 // protocol['dino']['batch_size'] // protocol['dino']['accumulation'] == 45
        protocol.update(name='multicity_transfer_20261028/' + name, dataset_sha256=sha(folder / 'dataset.json'),
            input_manifest_sha256=sha(folder / 'INPUT_SHA256.json'), counts=counts,
            training_cities=['nyc'] if name == 'A_nyc' else CITIES, validation_city='nyc', heldout_cities=[],
            checkpoint_selection='Same original 174 NYC validation rows; mean(use macroF1, role macroF1)',
            frozen_before_new_comparison_outputs=True, retired_tests=str(ROOT / 'RETIRED_TESTS.json'),
            training_started=False, independent_human_gold=False,
            auxiliary_subtypes_matched=False, primary_coarse_subtypes_matched=True)
        save(folder / 'PROTOCOL.json', protocol)
        arms[name] = dict(rows=894, train=720, val=174, dataset_sha256=protocol['dataset_sha256'],
                          input_manifest_sha256=protocol['input_manifest_sha256'], counts=counts)
    verify_sources()
    report = load(ROOT / 'ACQUISITION_REPORT.json')
    report.update(status='ready_for_parent_preflight', frozen_utc=now(), arms=arms,
        image_checks_passed=len(image_checks), cross_split_context_intersections=0, cross_split_parcels=0,
        unique_parcels_per_arm=720, shared_nyc_train=260, exact_original_validation=True,
        primary_coarse_subtypes_matched=True, auxiliary_subtypes_matched=False,
        old_source_hashes_unchanged=True, selector_sha256=sha(Path(__file__)), final_test_cities_acquired=[])
    save(ROOT / 'ACQUISITION_REPORT.json', report)
    save(ROOT / 'PROTOCOL.json', dict(name='multicity_transfer_20261028', status=report['status'],
        arms=arms, source_dataset_sha256=SOURCE_SHA, selection_seed=SEED, training_started=False))
    print('FROZEN', json.dumps(arms), flush=True)


def self_check(inputs):
    selected = select(inputs)
    assert selected == select(inputs)
    assert len(combined_rows(inputs, selected)) == 1354
    missing = [r for r in combined_rows(inputs, selected) if r['sample_id'] not in inputs['old']]
    assert Counter(r['city'] for r in missing) == {'nyc': 151, 'cambridge': 24, 'seattle': 2}
    victim = next(sid for sid in selected['B_multicity'] if inputs['rows'][sid]['city'] == 'nyc'
                  and inputs['rows'][sid]['stratum'] == 'commercial')
    replacement = select(inputs, {victim})
    assert all(victim not in ids for ids in replacement.values())
    scarce = next(sid for sid in selected['B_multicity'] if inputs['rows'][sid]['city'] == 'cambridge'
                  and inputs['rows'][sid]['coarse_subtype'] == 'industrial')
    try:
        select(inputs, {scarce})
    except ValueError as exc:
        assert 'cambridge/other/industrial' in str(exc)
    else:
        raise AssertionError('An exhausted real subtype must block instead of weakening labels')
    print('PASS real-row quotas, subtype equality, parcel uniqueness, shared NYC subset, exact validation, deterministic replacement and scarcity blocker', flush=True)


if __name__ == '__main__':
    main()
