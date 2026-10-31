"""Independently score the frozen new-city experiment with sklearn."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

USES=['residential_small','residential_multi','commercial','institutional','other']
ROLES=['primary','auxiliary']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads((a.directory/'dataset.json').read_text())
    protocol=json.loads((a.directory/'PROTOCOL.json').read_text())
    assert hashlib.sha256((a.directory/'dataset.json').read_bytes()).hexdigest()==protocol['dataset_sha256']
    refs={r['sample_id']:r for r in rows};results={};predictions={};table=[]
    for name in ['dino_frozen_s42','dino_frozen_s43','dino_finetune_s42','dino_finetune_s43',
                 'qwen_nf4_zero','qwen_qlora_s42','qwen_qlora_s43']:
        folder=a.results/'runs'/name
        filename='test_predictions.json' if name.startswith('dino') else 'evaluation_predictions.json'
        values=json.loads((folder/filename).read_text())
        assert len(values)==len({r['sample_id'] for r in values})
        expected={r['sample_id'] for r in rows if r['split']=='test' or (name.startswith('qwen') and r['split']=='val')}
        assert {r['sample_id'] for r in values}==expected
        if name!='qwen_nf4_zero':
            training=json.loads((folder/'TRAINING.json').read_text())
            for split in ['train','val']:
                assert training[split+'_ids']==[r['sample_id'] for r in rows if r['split']==split]
            assert not any(refs[s]['city']!='nyc' for s in training['train_ids']+training['val_ids'])
            val_name='validation_predictions.json' if name.startswith('dino') else f"validation_epoch{training['best_epoch']}.json"
            val_predictions=json.loads((folder/val_name).read_text())
            val_rows=[refs[p['sample_id']] for p in val_predictions]
            assert [r['sample_id'] for r in val_rows]==training['val_ids']
            selection=np.mean([measure(val_rows,val_predictions,field,names)['macro_f1']
                               for field,names in [('use',USES),('role',ROLES)]])
            assert abs(selection-training['selection_score'])<1e-10
        if name.startswith('qwen'):
            for r in values:
                raw=r['raw'].strip()
                if raw.startswith('```'):
                    raw='\n'.join(raw.splitlines()[1:-1])
                try:parsed=json.loads(raw)
                except ValueError:parsed={}
                for field,key,names in [('use','type',USES),('role','role',ROLES)]:
                    target=parsed.get(key,'invalid') if isinstance(parsed,dict) else 'invalid'
                    if target not in names+['unknown']:target='invalid'
                    assert r[field]==target,(name,r['sample_id'])
        predictions[name]={r['sample_id']:r for r in values};results[name]={}
        saved=json.loads((folder/('TEST.json' if name.startswith('dino') else 'EVALUATION.json')).read_text())
        if name.startswith('dino'):saved=saved['cities']
        for city in ['cambridge','seattle']:
            selected=[r for r in rows if r['city']==city]
            pp=[predictions[name][r['sample_id']] for r in selected]
            metrics={field:measure(selected,pp,field,names) for field,names in [('use',USES),('role',ROLES)]}
            for field in ['use','role']:
                for key in ['accuracy','macro_f1','coverage']:
                    assert abs(metrics[field][key]-saved[city][field][key])<1e-10,(name,city,field,key)
            if name.startswith('dino'):
                alt=[dict(p,use=p['selective_use']) for p in pp]
                metrics['selective_use']=measure(selected,alt,'use',USES)
            results[name][city]=metrics
            table.append(dict(model=name,city=city,n_use=metrics['use']['n'],accuracy=metrics['use']['accuracy'],
                macro_f1=metrics['use']['macro_f1'],multi_recall=metrics['use']['per_class']['residential_multi']['recall'],
                institutional_recall=metrics['use']['per_class']['institutional']['recall'],use_coverage=metrics['use']['coverage'],
                role_accuracy=metrics['role']['accuracy'],auxiliary_n=metrics['role']['per_class']['auxiliary']['n'],
                auxiliary_recall=metrics['role']['per_class']['auxiliary']['recall']))
    paired={}
    for city in ['cambridge','seattle']:
        selected=[r for r in rows if r['city']==city and r['reference_use']]
        for method,baseline in [('dino_finetune','dino_frozen'),('qwen_qlora','qwen_nf4_zero')]:
            paired[city+':'+method]=paired_bootstrap(selected,predictions,method,baseline)
    report=dict(protocol_sha256=hashlib.sha256((a.directory/'PROTOCOL.json').read_bytes()).hexdigest(),
                counts=dict(Counter(r['city']+':'+r['split']+':'+r['stratum'] for r in rows)),
                results=results,paired_spatial_block_bootstrap=paired,
                limits=['Official-source references, not independent human or exact-imagery-date gold.',
                        'Purposive class-stratified samples; accuracy is not city prevalence-weighted accuracy.',
                        'Seattle has primary-role references only; no Seattle auxiliary recall estimate.',
                        'Conditional building-use classification on supplied footprints; no detector AP measured.'])
    (a.output/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    with (a.output/'SUMMARY.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    text=['# Matched DINO / QLoRA transfer experiment','',
          'Training and model selection use NYC only. Cambridge and Seattle are frozen external cities.',
          'Numbers below are building-level official-reference classification results, not independent human gold or detection AP.','',
          '| Model | City | Accuracy | Macro F1 | Multi recall | Institutional recall | Auxiliary recall |',
          '|---|---|---:|---:|---:|---:|---:|']
    for r in table:
        aux='not measured' if r['auxiliary_recall'] is None else f"{r['auxiliary_recall']:.3f}"
        text.append(f"| {r['model']} | {r['city']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['multi_recall']:.3f} | {r['institutional_recall']:.3f} | {aux} |")
    text += ['', 'Paired changes in five-use macro F1 (mean of two seeds; 500 draws of fixed 2km spatial blocks):','']
    for key,v in paired.items():text.append(f"- {key}: {v['delta']:+.3f}, 95% interval [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}].")
    text += ['',*['- '+x for x in report['limits']], '',
             'Checks: prediction IDs and train/validation IDs match the frozen dataset; all city metrics agree with independent sklearn scoring; no held-out city entered training or checkpoint selection.',
             'Bootstrap intervals describe this selected reference cohort, not variability over all cities.']
    (a.output/'RESULTS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(table=table,paired=paired),indent=2))


def measure(rows,predictions,field,names):
    known=[(r['reference_'+field],p[field]) for r,p in zip(rows,predictions) if r['reference_'+field] in names]
    truth,guess=map(list,zip(*known))
    present=[n for n in names if n in truth]
    precision,recall,f1,support=precision_recall_fscore_support(truth,guess,labels=names,zero_division=0)
    return dict(n=len(known),accuracy=float(accuracy_score(truth,guess)),
                macro_f1=float(f1_score(truth,guess,labels=present,average='macro',zero_division=0)),
                coverage=sum(g in names for g in guess)/len(guess),
                per_class={n:dict(n=int(s),precision=float(p),recall=float(r) if s else None,f1=float(f) if s else None)
                           for n,p,r,f,s in zip(names,precision,recall,f1,support)})


def paired_bootstrap(rows,predictions,method,baseline):
    groups=sorted({r['spatial_group'] for r in rows});indices={g:[i for i,r in enumerate(rows) if r['spatial_group']==g] for g in groups}
    truth=np.array([r['reference_use'] for r in rows]);pairs=[]
    for seed in [42,43]:
        model=predictions[f'{method}_s{seed}'];control=predictions[baseline if baseline=='qwen_nf4_zero' else f'{baseline}_s{seed}']
        pairs.append((np.array([model[r['sample_id']]['use'] for r in rows]),np.array([control[r['sample_id']]['use'] for r in rows])))
    def delta(ix):
        return float(np.mean([f1_score(truth[ix],a[ix],labels=USES,average='macro',zero_division=0)-
                              f1_score(truth[ix],b[ix],labels=USES,average='macro',zero_division=0) for a,b in pairs]))
    rng=np.random.default_rng(20261027);draws=[]
    for _ in range(500):
        ix=np.array([i for g in rng.choice(groups,size=len(groups),replace=True) for i in indices[g]])
        draws.append(delta(ix))
    return dict(delta=delta(np.arange(len(rows))),ci95=np.quantile(draws,[.025,.975]).tolist(),
                spatial_blocks=len(groups),buildings=len(rows),draws=500)


def self_check():
    rows=[dict(sample_id='a',spatial_group='g1',reference_use='commercial',reference_role='primary'),
          dict(sample_id='b',spatial_group='g1',reference_use='commercial',reference_role='primary'),
          dict(sample_id='c',spatial_group='g2',reference_use=None,reference_role='auxiliary')]
    pp=[dict(sample_id='a',use='commercial',role='primary'),dict(sample_id='b',use='unknown',role='primary'),
        dict(sample_id='c',use='other',role='auxiliary')]
    s=measure(rows,pp,'use',USES)
    assert s['n']==2 and s['accuracy']==.5 and abs(s['macro_f1']-2/3)<1e-12
    assert measure(rows,pp,'role',ROLES)['accuracy']==1
    pred={name:{p['sample_id']:p for p in pp} for name in ['m_s42','m_s43','b_s42','b_s43']}
    paired=paired_bootstrap(rows[:2],pred,'m','b')
    assert paired['delta']==0 and paired['ci95']==[0.,0.] and paired['spatial_blocks']==1
    print('PASS independent sklearn denominators, sparse fields, unknown and paired bootstrap null')


if __name__=='__main__':main()
