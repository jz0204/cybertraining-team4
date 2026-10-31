"""Compare fixed-budget single-city and multicity DINO on frozen LA development data."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_la_transfer import STAGE, bootstrap, measure
from analyze_reliable_transfer import measure as source_measure
from reliable_transfer_common import ROLES, USES, load, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, default=STAGE/'multicity_transfer_20261028')
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows, _ = load(STAGE/'la_transfer_20261027')
    predictions, source, receipts = {}, {}, {}
    validation = None
    for arm in ['A_nyc', 'B_multicity']:
        dataset, _ = load(a.directory/arm)
        train = [r for r in dataset if r['split'] == 'train']
        val = [r for r in dataset if r['split'] == 'val']
        assert len(train) == 720 and len(val) == 174
        if validation is None:
            validation = val
        assert val == validation, 'Both arms must use identical source validation records'
        complete = json.loads((a.results/('COMPLETE_'+arm+'.json')).read_text())
        for seed in [42, 43]:
            name = 'dino_finetune_s'+str(seed)
            run = a.results/arm/'runs'/name
            training = json.loads((run/'TRAINING.json').read_text())
            assert training['fixed_epochs']
            assert [h['epoch'] for h in training['history']] == list(range(1, 16))
            assert training['train_ids'] == [r['sample_id'] for r in train]
            assert training['val_ids'] == [r['sample_id'] for r in val]
            vp = json.loads((run/'validation_predictions.json').read_text())
            assert [v['sample_id'] for v in vp] == training['val_ids']
            metrics = {field: source_measure(val, vp, field, names)
                       for field, names in [('use', USES), ('role', ROLES)]}
            selection = np.mean([metrics[f]['macro_f1'] for f in ['use', 'role']])
            assert abs(selection-training['selection_score']) < 1e-10
            assert abs(selection-max(h['selection_score'] for h in training['history'])) < 1e-10
            la = a.results/arm/'la/runs'/name
            values = json.loads((la/'predictions.json').read_text())
            assert [v['sample_id'] for v in values] == [r['sample_id'] for r in rows]
            assert [v['input_available'] for v in values] == [r['input_available'] for r in rows]
            assert all(v['use'] == 'unknown' for v in values if not v['input_available'])
            receipt = next(r for r in complete['runs'] if r['name'] == name)
            assert receipt['epochs'] == 15 and receipt['optimizer_updates_from_fixed_schedule'] == 675
            assert receipt['la_predictions_sha256'] == hashlib.sha256((la/'predictions.json').read_bytes()).hexdigest()
            provenance = json.loads((la/'PROVENANCE.json').read_text())
            assert receipt['checkpoint_sha256'] == provenance['checkpoint_sha256']
            assert not provenance['checkpoint_changed'] and not provenance['la_selection']
            assert provenance['threshold_from_source_validation'] == training['threshold_80coverage']
            key = arm+'_s'+str(seed)
            predictions[key] = values
            source[key] = dict(metrics=metrics, best_epoch=training['best_epoch'], seconds=training['seconds'])
            receipts[key] = receipt
    results, paired, table = {}, {}, []
    for subset, mask in [('full3332', [True]*len(rows)),
                         ('common_native_context', [r['input_available'] for r in rows]),
                         ('strict_single_assessor_building', [r['strict_single_assessor_building'] for r in rows])]:
        rr = [r for r, k in zip(rows, mask) if k]
        pp = {n: [v for v, k in zip(values, mask) if k] for n, values in predictions.items()}
        results[subset] = {n: measure(rr, values) for n, values in pp.items()}
        for name, values in pp.items():
            results[subset][name]['source_threshold_selective'] = measure(
                rr, [dict(v, use=v['selective_use']) for v in values])
        for arm in ['A_nyc', 'B_multicity']:
            members = [results[subset][arm+'_s'+str(s)] for s in [42, 43]]
            results[subset][arm+'_seed_mean'] = {
                metric: float(np.mean([m[metric] for m in members]))
                for metric in ['accuracy', 'macro_f1', 'balanced_accuracy', 'five_use_coverage']}
        paired[subset] = bootstrap(rr, pp, 'B_multicity', 'A_nyc')
        for name, metric in results[subset].items():
            table.append(dict(subset=subset, model=name, n=len(rr),
                              **{k: metric[k] for k in ['accuracy', 'macro_f1', 'balanced_accuracy']}))
    a.output.mkdir(parents=True, exist_ok=True)
    save(a.output/'RESULTS.json', dict(results=results, paired_cell_bootstrap=paired,
        source_validation=source, completed_runs=receipts,
        limits=['LA is previously inspected development data, not a fresh final city.',
                'LA has three reference uses, only14 multi references, and no role truth.',
                '720 training samples,15 epochs,675 updates and174 NYC validation samples per run.',
                'Primary-use subtype totals match; auxiliary subtype composition differs.',
                'Seed means average metrics, not prediction probabilities.']))
    with (a.output/'SUMMARY.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    print(json.dumps(dict(results=results, paired=paired), indent=2))


def self_check():
    rows = [dict(sample_id=str(i), cell_id=str(i), reference_use=use)
            for i, use in enumerate(USES[:3])]
    values = [dict(sample_id=r['sample_id'], use=r['reference_use']) for r in rows]
    pp = {arm+'_s'+str(seed): values for arm in ['A_nyc', 'B_multicity'] for seed in [42, 43]}
    result = bootstrap(rows, pp, 'B_multicity', 'A_nyc')
    assert result['delta_macro_f1'] == 0 and result['ci95'] == [0., 0.]
    unknown = [dict(v, use='unknown') for v in values]
    assert measure(rows, unknown)['accuracy'] == measure(rows, unknown)['macro_f1'] == 0
    print('PASS real scoring: identical-arm paired null and unknown-as-error')


if __name__ == '__main__':
    main()
