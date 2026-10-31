"""Score frozen LA predictions on identical full and image-eligible cohorts."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from reliable_transfer_common import USES, save

STAGE = Path('/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a')
LABELS = USES[:3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, default=STAGE/'la_transfer_20261027')
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((a.directory/'dataset.json').read_text())
    protocol = json.loads((a.directory/'PROTOCOL.json').read_text())
    assert hashlib.sha256((a.directory/'dataset.json').read_bytes()).hexdigest() == protocol['dataset_sha256']
    predictions = cnn_predictions(rows)
    names = ['dino_frozen_s42', 'dino_frozen_s43', 'dino_finetune_s42', 'dino_finetune_s43',
             'qwen_nf4_zero', 'qwen_qlora_s42', 'qwen_qlora_s43']
    for name in names:
        values = json.loads((a.results/'runs'/name/'predictions.json').read_text())
        assert [p['sample_id'] for p in values] == [r['sample_id'] for r in rows]
        assert [p['input_available'] for p in values] == [r['input_available'] for r in rows]
        predictions[name] = values
        if name.startswith('qwen'):
            for v in values:
                if not v['input_available']: continue
                raw = v['raw'].strip()
                if raw.startswith('```'): raw='\n'.join(raw.splitlines()[1:-1])
                try: parsed=json.loads(raw)
                except ValueError: parsed={}
                target = parsed.get('type') if isinstance(parsed, dict) else None
                assert v['use'] == (target if target in USES+['unknown'] else 'invalid')
    predictions['all_small'] = [dict(sample_id=r['sample_id'], use=LABELS[0]) for r in rows]
    subsets = {'full3332': np.ones(len(rows), dtype=bool),
               'common_native_context': np.array([r['input_available'] for r in rows]),
               'strict_single_assessor_building': np.array([r['strict_single_assessor_building'] for r in rows])}
    results, table = {}, []
    for subset, mask in subsets.items():
        rr = [r for r, keep in zip(rows, mask) if keep]
        results[subset] = {}
        for name, values in predictions.items():
            vv = [v for v, keep in zip(values, mask) if keep]
            metrics = measure(rr, vv); results[subset][name] = metrics
            table.append(dict(subset=subset, model=name, n=len(rr), **{k: metrics[k] for k in ['accuracy','macro_f1','balanced_accuracy','five_use_coverage']},
                              multi_recall=metrics['per_class']['residential_multi']['recall'],
                              commercial_recall=metrics['per_class']['commercial']['recall']))
            if name.startswith('dino'):
                results[subset][name]['source_threshold_selective'] = measure(rr, [dict(v, use=v['selective_use']) for v in vv])
        for family in ['dino_frozen', 'dino_finetune', 'qwen_qlora']:
            members = [results[subset][family+f'_s{s}'] for s in [42, 43]]
            results[subset][family+'_seed_mean'] = {k: float(np.mean([m[k] for m in members])) for k in ['accuracy','macro_f1','balanced_accuracy','five_use_coverage']}
    previous = list(csv.DictReader((STAGE/'la_data_20261024/generalization_20261026/ALL_CLASS_METRICS.csv').open()))
    for name in ['current_five_seed_ensemble', 'N1_natural_ce_two_seed_mean', 'N2_nyc_natural_two_seed_mean']:
        old = next(r for r in previous if r['domain']=='la' and r['model']==name)
        for metric in ['accuracy', 'macro_f1', 'balanced_accuracy']:
            assert abs(results['full3332'][name][metric]-float(old[metric])) < 1e-10, (name, metric)
    paired = {}
    for subset in ['full3332','common_native_context']:
        mask = subsets[subset]; rr=[r for r,k in zip(rows,mask) if k]
        pp={n:[v for v,k in zip(values,mask) if k] for n,values in predictions.items()}
        for candidate, control in [('dino_finetune','dino_frozen'), ('qwen_qlora','qwen_nf4_zero'),
                                   ('dino_finetune','N2_nyc_natural_two_seed_mean'), ('qwen_qlora','N2_nyc_natural_two_seed_mean')]:
            paired[subset+':'+candidate+'-'+control] = bootstrap(rr, pp, candidate, control)
    report = dict(results=results, paired_cell_bootstrap=paired,
        input_available=dict(Counter(str(r['input_available'])+':'+r['reference_use'] for r in rows)),
        role_prediction_counts={n:dict(Counter(v.get('role') for v in predictions[n])) for n in names},
        protocol_sha256=hashlib.sha256((a.directory/'PROTOCOL.json').read_bytes()).hexdigest(),
        limits=['LA is previously inspected development evidence, not an untouched final city.',
                'Only three assessor reference uses are evaluated; no LA role or five-class accuracy truth.',
                'Field of view and resampling differ from CNN; comparison concerns complete pipelines.',
                'Input failures remain unknown/errors on full3332; common-context scores are separate.',
                'Only14 multifamily references; no detector AP or actual-population accuracy measured.'])
    save(a.output/'RESULTS.json', report)
    with (a.output/'SUMMARY.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    save(a.output/'PREDICTIONS.json', predictions)
    print(json.dumps(dict(full=results['full3332'], paired=paired),indent=2))


def cnn_predictions(rows):
    sources = {'current_five_seed_ensemble': STAGE/'la_eval_20261023/cluster_full/stage2a_predictions.csv',
        'N1_s42': STAGE/'la_explore_20261023/predictions/N1_natural_ce_seed42/la.csv',
        'N1_s43': STAGE/'la_data_20261024/followup_20261025/predictions/N1_natural_ce_seed43/la.csv',
        'N2_s42': STAGE/'la_data_20261024/predictions/N2_nyc_natural_seed42/la.csv',
        'N2_s43': STAGE/'la_data_20261024/followup_20261025/predictions/N2_nyc_natural_seed43/la.csv'}
    probabilities = {}
    for name,path in sources.items():
        values=list(csv.DictReader(path.open())); index={v['building_uid']:v for v in values}
        assert len(index)==len(values)
        probabilities[name]=np.array([[float(index[r['building_uid']]['prob_'+label]) for label in LABELS] for r in rows])
        assert np.isfinite(probabilities[name]).all() and np.allclose(probabilities[name].sum(1),1,atol=1e-5)
    for family, name in [('N1','N1_natural_ce_two_seed_mean'), ('N2','N2_nyc_natural_two_seed_mean')]:
        probabilities[name] = (probabilities[family+'_s42']+probabilities[family+'_s43'])/2
    return {name:[dict(sample_id=r['sample_id'],use=LABELS[int(v.argmax())],use_probabilities=v.tolist()) for r,v in zip(rows,values)]
            for name,values in probabilities.items()}


def measure(rows, values):
    truth=[r['reference_use'] for r in rows]; pred=[v['use'] for v in values]
    precision,recall,f1,support=precision_recall_fscore_support(truth,pred,labels=LABELS,zero_division=0)
    answered=[i for i,p in enumerate(pred) if p in USES]
    columns=USES+['unknown','invalid']
    return dict(n=len(rows), accuracy=float(accuracy_score(truth,pred)), macro_f1=float(f1.mean()),
        balanced_accuracy=float(recall.mean()), five_use_coverage=len(answered)/len(rows),
        selective_accuracy=float(np.mean([truth[i]==pred[i] for i in answered])) if answered else None,
        prediction_counts=dict(Counter(pred)),
        per_class={name:dict(n=int(support[i]),precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i])) for i,name in enumerate(LABELS)},
        confusion=[[sum(y==a and p==b for y,p in zip(truth,pred)) for b in columns] for a in LABELS],
        rows=LABELS, columns=columns)


def bootstrap(rows, predictions, candidate, control):
    def members(name): return [name] if name in predictions else [name+'_s42',name+'_s43']
    cells=sorted({r['cell_id'] for r in rows}); categories=USES+['unknown','invalid']
    matrices={}
    for name in members(candidate)+members(control):
        arr=np.zeros((len(cells),3,len(categories)),dtype=int)
        for r,v in zip(rows,predictions[name]): arr[cells.index(r['cell_id']),LABELS.index(r['reference_use']),categories.index(v['use'])]+=1
        matrices[name]=arr
    def f1(cm):
        tp=np.diag(cm[:,:3]); denominator=cm.sum(1)+cm[:,:3].sum(0)
        return np.divide(2*tp,denominator,out=np.zeros(3,dtype=float),where=denominator>0).mean()
    def delta(weights):
        return float(np.mean([f1(np.tensordot(weights,matrices[n],axes=1)) for n in members(candidate)])-
                     np.mean([f1(np.tensordot(weights,matrices[n],axes=1)) for n in members(control)]))
    rng=np.random.default_rng(20261028); draws=[]
    for _ in range(1000):draws.append(delta(np.bincount(rng.integers(len(cells),size=len(cells)),minlength=len(cells))))
    return dict(delta_macro_f1=delta(np.ones(len(cells),dtype=int)),ci95=np.quantile(draws,[.025,.975]).tolist(),cells=len(cells),draws=1000)


if __name__ == '__main__': main()
