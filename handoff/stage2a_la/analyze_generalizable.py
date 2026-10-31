"""Score identical held-out city IDs, with evidence strata and paired seeds."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from vlm_multiclass import CLASSES, score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    args=p.parse_args();root=args.directory
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    assert hashlib.sha256((root/'DATASET.json').read_bytes()).hexdigest()==protocol['dataset_sha256']
    refs={r['sample_id']:r for r in json.loads((root/'DATASET.json').read_text())}
    major=json.loads((root/'DOMINANT_ROOF_AUDIT.json').read_text())['rows']
    results={};runs={r['name']:r for r in protocol['runs']};table=[];ensembles=defaultdict(list)
    for name,run in runs.items():
        path=root/'runs'/name/'predictions.json'
        if not path.exists():continue
        predictions=json.loads(path.read_text())
        assert [r['sample_id'] for r in predictions]==run['test']
        assert len(predictions)==len({r['sample_id'] for r in predictions})
        rows=[]
        for pred in predictions:
            ref=refs[pred['sample_id']];probs=np.array(pred['probabilities'])
            assert pred['truth']==ref['label'] and np.isfinite(probs).all() and np.all(probs>=0)
            assert abs(probs.sum()-1)<1e-5 and CLASSES[int(probs.argmax())]==pred['prediction']
            assert pred['role']=='unknown'
            rows.append(dict(ref,reference_use=ref['label'],pred_type=pred['prediction'],probabilities=pred['probabilities'],accepted=pred['accepted']))
        results[name]=summarize(rows,major)
        ensembles[(run['mode'],run['encoder'],run['quality'],run['heldout_city'],run['dual'])].append((run['seed'],rows))
        for subset,scopes in results[name].items():
            for scope,value in scopes.items():
                if value['n']:
                    table.append(dict(run=name,city=run['heldout_city'],subset=subset,scope=scope,n=value['n'],
                        accuracy=value['accuracy'],macro_f1=value['macro_f1'],balanced_accuracy=value['balanced_accuracy'],
                        multi_recall=value['per_class'].get('residential_multi',{}).get('recall'),
                        institutional_recall=value['per_class'].get('institutional',{}).get('recall')))
    combined={};combined_rows={}
    for key,members in ensembles.items():
        if {seed for seed,_ in members}!={42,43}:continue
        first,second=sorted(members)
        assert [r['sample_id'] for r in first[1]]==[r['sample_id'] for r in second[1]]
        rows=[]
        for a,b in zip(first[1],second[1]):
            probs=(np.array(a['probabilities'])+np.array(b['probabilities']))/2
            rows.append(dict(a,pred_type=CLASSES[int(probs.argmax())],probabilities=probs.tolist()))
        name='/'.join(map(str,key));combined[name]=summarize(rows,major,selective=False);combined_rows[key]=rows
    contrasts={}
    for city in ['harris','nyc','sandiego']:
        pairs=[(('frozen','convnext','source',city,True),('frozen','convnext','major_roof',city,True)),
            (('frozen','convnext','major_roof',city,True),('frozen','dinov3','major_roof',city,True)),
            (('frozen','convnext','major_roof',city,True),('finetune','convnext','major_roof',city,True))]
        pairs += [(('frozen',encoder,'major_roof',city,False),('frozen',encoder,'major_roof',city,True)) for encoder in ['convnext','dinov3']]
        for before,after in pairs:
            if before not in combined_rows or after not in combined_rows:continue
            a=combined_rows[before];b=combined_rows[after]
            if city=='nyc':
                a=[r for r in a if r['current_single_building_corroborated']]
                b=[r for r in b if r['current_single_building_corroborated']]
            contrasts[str(before)+' -> '+str(after)]=paired(a,b)
    for name,value in [('SCORES.json',results),('ENSEMBLE_SCORES.json',combined),('PAIRED_CONTRASTS.json',contrasts)]:
        (root/name).write_text(json.dumps(value,indent=2)+'\n')
    if table:
        with (root/'SUMMARY.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    receipt=dict(expected_runs=len(runs),completed_runs=len(results),independent_sklearn_verified=True,
        no_role_gold=True,all_role_predictions='unknown',primary_city_scope='NYC current corroborated; Harris/SD source references',
        final_independent_city=False,ensemble_selective_metrics='not reported; individual-seed thresholds are not ensemble calibration')
    (root/'ANALYSIS_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


def summarize(rows,major,selective=True):
    subsets={'all_source':rows}
    if rows[0]['domain']=='nyc':subsets['current_corroborated']=[r for r in rows if r['current_single_building_corroborated']]
    if rows[0]['domain']=='harris':subsets['major_roof_proxy']=[r for r in rows if major[r['sample_id']]['major_roof']]
    if rows[0]['domain']=='sandiego':subsets['official_corroborated']=[r for r in rows if r['evidence_grade'].startswith('A')]
    result={}
    for subset,rr in subsets.items():
        result[subset]={}
        for scope,labels in [('five',CLASSES),('shared_three',CLASSES[:3])]:
            selected=[r for r in rr if r['reference_use'] in labels]
            actual=score(selected,labels)
            if selected:
                y=[r['reference_use'] for r in selected];pred=[r['pred_type'] for r in selected]
                supported=[c for c in labels if c in y]
                _,_,f1,_=precision_recall_fscore_support(y,pred,labels=supported,zero_division=0)
                assert abs(actual['macro_f1']-float(np.mean(f1)))<1e-12
                assert abs(actual['accuracy']-accuracy_score(y,pred))<1e-12
                if selective:
                    actual['selective']={}
                    for coverage in ['0.8','0.9']:
                        abstained=[dict(r,pred_type=r['pred_type'] if r['accepted'][coverage] else 'unknown') for r in selected]
                        stats=score(abstained,labels)
                        stats['class_retention']={c:sum(r['accepted'][coverage] for r in selected if r['reference_use']==c)/sum(r['reference_use']==c for r in selected) for c in supported}
                        actual['selective'][coverage]=stats
            result[subset][scope]=actual
    return result


def paired(a,b):
    assert [r['sample_id'] for r in a]==[r['sample_id'] for r in b]
    fixed=[x['sample_id'] for x,y in zip(a,b) if x['pred_type']!=x['reference_use'] and y['pred_type']==y['reference_use']]
    regressed=[x['sample_id'] for x,y in zip(a,b) if x['pred_type']==x['reference_use'] and y['pred_type']!=y['reference_use']]
    groups=defaultdict(list)
    for i,r in enumerate(a):groups[r['spatial_component']].append(i)
    names=sorted(groups);rng=np.random.default_rng(20261026);deltas=[]
    for _ in range(500):
        ix=[i for g in rng.choice(names,len(names),replace=True) for i in groups[g]]
        aa=[a[i] for i in ix];bb=[b[i] for i in ix]
        # Fixed five-class denominator in resamples, even when a rare class is absent.
        def macro(rr):
            return float(precision_recall_fscore_support([r['reference_use'] for r in rr],[r['pred_type'] for r in rr],labels=CLASSES,zero_division=0)[2].mean())
        deltas.append(macro(bb)-macro(aa))
    return dict(n=len(a),blocks=len(names),fixed=fixed,regressed=regressed,
        macro_f1_delta=score(b,CLASSES)['macro_f1']-score(a,CLASSES)['macro_f1'],
        block_bootstrap_95ci=np.quantile(deltas,[.025,.975]).tolist(),
        inference_scope='Development reference labels only; interval does not cover label error or new-city variation')


if __name__=='__main__':main()
