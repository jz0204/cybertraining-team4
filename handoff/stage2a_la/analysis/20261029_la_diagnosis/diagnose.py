"""Read-only LA error diagnosis; post-hoc readouts are not newly validated models."""

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from analyze_la_transfer import LABELS, STAGE, cnn_predictions, measure
from reliable_transfer_common import USES, load, save


def main():
    root = STAGE/'la_transfer_20261027'
    rows, protocol = load(root)
    predictions = {'CNN_N2': cnn_predictions(rows)['N2_nyc_natural_two_seed_mean']}
    hashes = {'dataset.json': protocol['dataset_sha256']}
    for name in ['qwen_nf4_zero', 'qwen_qlora_s42', 'qwen_qlora_s43']:
        path = root/'execution/remote_la/runs'/name/'predictions.json'
        predictions[name] = read_predictions(path, rows, hashes)
    for arm in ['A_nyc', 'B_multicity']:
        for seed in [42, 43]:
            path = STAGE/f'multicity_transfer_20261028/execution/remote_multicity/{arm}/la/runs/dino_finetune_s{seed}/predictions.json'
            predictions[arm+'_s'+str(seed)] = read_predictions(path, rows, hashes)
    predictions['all_small'] = [dict(sample_id=r['sample_id'], use=LABELS[0]) for r in rows]
    controls = json.loads((HERE.parent/'20261028_la_transfer/RESULTS.json').read_text())
    controls_ab = json.loads((HERE.parent/'20261028_multicity_transfer/RESULTS.json').read_text())
    for name, values in list(predictions.items()):
        if name.startswith(('A_nyc', 'B_multicity')):
            predictions[name+'_three_way_readout'] = project(values)
    for arm in ['A_nyc', 'B_multicity']:
        ensemble = []
        for row, a, b in zip(rows, predictions[arm+'_s42'], predictions[arm+'_s43']):
            if not row['input_available']:
                ensemble.append(dict(sample_id=row['sample_id'], input_available=False, use='unknown'))
                continue
            p = (np.array(a['use_probabilities'])+np.array(b['use_probabilities']))/2
            ensemble.append(dict(sample_id=row['sample_id'], input_available=True,
                                 use=USES[int(p.argmax())], use_probabilities=p.tolist()))
        predictions[arm+'_probability_ensemble'] = ensemble
        predictions[arm+'_probability_ensemble_three_way'] = project(ensemble)
    predictions['qwen_zero_missing_input_CNN_fallback'] = [q if r['input_available'] else c
        for r, q, c in zip(rows, predictions['qwen_nf4_zero'], predictions['CNN_N2'])]
    results, table = {}, []
    for subset, mask in [('full3332', [True]*len(rows)),
                         ('common3157', [r['input_available'] for r in rows])]:
        rr = [r for r, keep in zip(rows, mask) if keep]
        results[subset] = {}
        for name, values in predictions.items():
            m = measure(rr, [v for v, keep in zip(values, mask) if keep])
            results[subset][name] = m
            table.append(dict(subset=subset, model=name, n=m['n'], accuracy=m['accuracy'], macro_f1=m['macro_f1'],
                small_f1=m['per_class'][LABELS[0]]['f1'], multi_f1=m['per_class'][LABELS[1]]['f1'],
                commercial_f1=m['per_class'][LABELS[2]]['f1']))
    for name in ['CNN_N2', 'qwen_nf4_zero', 'qwen_qlora_s42', 'qwen_qlora_s43', 'A_nyc_s42', 'A_nyc_s43', 'B_multicity_s42', 'B_multicity_s43']:
        old = controls_ab if name.startswith(('A_', 'B_')) else controls
        old_name = 'N2_nyc_natural_two_seed_mean' if name == 'CNN_N2' else name
        for metric in ['accuracy', 'macro_f1']:
            assert abs(results['common3157'][name][metric]-old['results']['common_native_context'][old_name][metric]) < 1e-12
    seed_stats = {}
    for arm in ['A_nyc', 'B_multicity']:
        for suffix in ['', '_three_way_readout']:
            members = [results['common3157'][arm+'_s'+str(s)+suffix] for s in [42, 43]]
            seed_stats[arm+suffix] = {key: dict(mean=float(np.mean([m[key] for m in members])),
                sample_std=float(np.std([m[key] for m in members], ddof=1))) for key in ['accuracy', 'macro_f1']}
    overlap = {}
    for label in LABELS:
        pairs = [(c['use']==label, q['use']==label) for r, c, q in zip(rows, predictions['CNN_N2'], predictions['qwen_nf4_zero'])
                 if r['input_available'] and r['reference_use']==label]
        overlap[label] = {name: pairs.count(key) for name, key in
            [('both_correct', (True, True)), ('cnn_only', (True, False)), ('qwen_only', (False, True)), ('neither_correct', (False, False))]}
        assert sum(overlap[label].values()) == len(pairs)
    assessor_path = HERE.parents[1]/'analysis/20261015_gsd_assessor/la_assessor_matched.csv'
    assessors = {}
    for a in csv.DictReader(assessor_path.open()):
        assessors.setdefault((a['cell_id'], a['bldg_uid']), []).append(a)
    cases = []
    for i, row in enumerate(rows):
        if row['reference_use'] not in LABELS[1:]:
            continue
        matched = assessors[tuple(json.loads(row['building_uid']))]
        assert len(matched) == 1
        a = matched[0]
        cases.append(dict(sample_id=row['sample_id'], reference_use=row['reference_use'],
            assessor_units=a['Units'], assessor_building_records=a['totBuildingDataLines'],
            use_code=a['UseCode'], use_description=a['UseCodeDescChar2'],
            strict_single_assessor_building=row['strict_single_assessor_building'],
            predicted_polygon_m2=row['footprint_m2'], near_image=str(root/row['near_marked']),
            wide_image=str(root/row['wide_marked']),
            **{name: predictions[name][i]['use'] for name in ['CNN_N2', 'qwen_nf4_zero', 'qwen_qlora_s42', 'qwen_qlora_s43', 'B_multicity_s42', 'B_multicity_s43']}))
    assert len(cases) == 56
    missing = [r for r in rows if not r['input_available']]
    assert len(rows) == 3332 and len(missing) == 175
    report = dict(metrics=results, seed_stats=seed_stats, cnn_qwen_overlap=overlap,
        cohort_counts=dict(Counter(r['reference_use'] for r in rows)),
        common_counts=dict(Counter(r['reference_use'] for r in rows if r['input_available'])),
        missing_input_cells=dict(Counter(r['cell_id'] for r in missing)),
        full_accuracy_ceiling_without_input_fallback=(len(rows)-len(missing))/len(rows),
        multi_units=dict(Counter(c['assessor_units'] for c in cases if c['reference_use']==LABELS[1])),
        commercial_subtypes=dict(Counter(c['use_description'] for c in cases if c['reference_use']==LABELS[2])),
        source_hashes=hashes,
        limits=['All new readouts, probability ensembles and fallback are post-hoc diagnostics on inspected LA data.',
                'Three-way readout assumes only the three benchmark uses; cannot recognize institutional/other.',
                'No training, new model inference, source data changes or production changes performed.',
                'Assessor references are not independently adjudicated building-use or role truth.'])
    save(HERE/'DIAGNOSIS.json', report)
    for path, records in [(HERE/'METRICS.csv', table), (HERE/'MINORITY_CASES.csv', cases)]:
        with path.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print('PASS frozen cohort, eight raw-metric parity checks, projection invariance,56 unique assessor joins; diagnostics saved')
    print(json.dumps(seed_stats, indent=2))


def read_predictions(path, rows, hashes):
    payload = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(payload).hexdigest()
    values = json.loads(payload)
    assert [v['sample_id'] for v in values] == [r['sample_id'] for r in rows]
    assert [v['input_available'] for v in values] == [r['input_available'] for r in rows]
    return values


def project(values):
    projected = []
    for v in values:
        if not v['input_available']:
            projected.append(v)
            continue
        p = np.array(v['use_probabilities'])
        assert p.shape == (5,) and np.isfinite(p).all() and np.all(p >= 0) and np.isclose(p.sum(), 1)
        use = LABELS[int(p[:3].argmax())]
        if v['use'] in LABELS:
            assert use == v['use']
        projected.append(dict(v, use=use))
    return projected


if __name__ == '__main__':
    main()
