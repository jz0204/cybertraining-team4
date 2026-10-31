"""Compare LA-adapted classifiers with N2 on the unchanged LA reference cohort."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from analyze_la_transfer import LABELS, STAGE, bootstrap, cnn_predictions, measure
from reliable_transfer_common import load, save

ROOT=STAGE/'la_adaptation_20261029'
OUT=Path(__file__).resolve().parent/'analysis/20261029_la_adaptation'


def main():
    rows,old_protocol=load(STAGE/'la_transfer_20261027')
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    assert protocol['old_test_dataset_sha256']==old_protocol['dataset_sha256']
    assert hashlib.sha256((ROOT/'DATASET.json').read_bytes()).hexdigest()==protocol['dataset_sha256']
    data=json.loads((ROOT/'DATASET.json').read_text())
    validation=[r for r in data if r['split']=='val']
    assert [r['sample_id'] for r in data if r['split']=='test']==[r['sample_id'] for r in rows]
    old=cnn_predictions(rows)
    predictions={'CNN_old':old['N2_nyc_natural_two_seed_mean']}
    predictions.update({'CNN_old_s'+str(s):old['N2_s'+str(s)] for s in [42,43]})
    fits={};hashes={};validation_predictions={}
    for encoder in ['cnn','dino','cnn_lowrise']:
        for seed in [42,43]:
            name=f'{encoder}_s{seed}';p=ROOT/'execution/collected/runs'/name
            result=json.loads((p/'RESULT.json').read_text())
            assert result['dataset_sha256']==protocol['dataset_sha256']
            with (p/'best.pt').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==result['checkpoint_sha256']
            if encoder=='cnn_lowrise':
                raw=(ROOT/'execution/collected/LOWRISE_SAMPLING.json').read_bytes()
                assert hashlib.sha256(raw).hexdigest()==result['sampling_factor_sha256']
            history=json.loads((p/'history.json').read_text())
            assert [h['epoch'] for h in history]==list(range(1,7)) and result['optimizer_updates']==1536
            scores=[(h['epoch'],h['validation']['sampling_weighted_macro_f1']) for h in history]
            initial=json.loads((p/'initial_validation.json').read_text())
            if encoder.startswith('cnn'):scores.insert(0,(0,initial['sampling_weighted_macro_f1']))
            assert result['best_epoch']==max(scores,key=lambda v:v[1])[0]
            v=read_predictions(p/'validation_predictions.json',validation,hashes)
            validation_predictions[name]=v
            y=[int(r['type_class']) for r in validation];pred=[LABELS.index(r['use']) for r in v]
            weighted=f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0,
                              sample_weight=[r['validation_weight'] for r in validation])
            assert abs(weighted-result['validation']['sampling_weighted_macro_f1'])<1e-10
            predictions[name]=read_predictions(p/'test_predictions.json',rows,hashes)
            m=measure(rows,predictions[name])
            assert abs(m['macro_f1']-result['test']['type_macro_f1'])<1e-10
            assert abs(m['accuracy']-result['test']['type_accuracy'])<1e-10
            fits[name]=dict(best_epoch=result['best_epoch'],validation=result['validation'],
                initial_validation=initial,seconds=result['seconds'],peak_reserved_gib=result['peak_reserved_gib'],
                draw_hashes=[h['draw_sha256'] for h in history])
        probabilities=np.mean([[v['probabilities'] for v in predictions[f'{encoder}_s{s}']] for s in [42,43]],axis=0)
        predictions[encoder+'_ensemble']=[dict(sample_id=r['sample_id'],use=LABELS[int(v.argmax())],probabilities=v.tolist())
            for r,v in zip(rows,probabilities)]
    for seed in [42,43]:assert fits[f'cnn_s{seed}']['draw_hashes']==fits[f'dino_s{seed}']['draw_hashes']
    flagged={r['sample_id'] for r in json.loads((OUT/'LABEL_AUDIT.json').read_text())['flagged']}
    subsets={'full3332':[True]*len(rows),'common3157':[r['input_available'] for r in rows],
        'strict_single_assessor':[r['strict_single_assessor_building'] for r in rows],
        'without_four_attribute_uncertainties':[r['sample_id'] not in flagged for r in rows]}
    results={};paired={};table=[]
    for subset,mask in subsets.items():
        rr=[r for r,k in zip(rows,mask) if k]
        pp={n:[v for v,k in zip(values,mask) if k] for n,values in predictions.items()}
        results[subset]={n:measure(rr,v) for n,v in pp.items()}
        for name,m in results[subset].items():
            cm=np.array(m['confusion']);m['true_positives']=dict(zip(LABELS,np.diag(cm[:,:3]).tolist()))
            m['false_positives']=dict(zip(LABELS,(cm[:,:3].sum(0)-np.diag(cm[:,:3])).tolist()))
            table.append(dict(subset=subset,model=name,n=len(rr),accuracy=m['accuracy'],macro_f1=m['macro_f1'],
                multi_f1=m['per_class'][LABELS[1]]['f1'],commercial_f1=m['per_class'][LABELS[2]]['f1'],
                multi_tp=m['true_positives'][LABELS[1]],multi_fp=m['false_positives'][LABELS[1]],
                commercial_tp=m['true_positives'][LABELS[2]],commercial_fp=m['false_positives'][LABELS[2]]))
        if subset in ['full3332','common3157']:
            for candidate,control in [('cnn_ensemble','CNN_old'),('dino_ensemble','CNN_old'),('dino_ensemble','cnn_ensemble'),
                                      ('cnn_lowrise_ensemble','cnn_ensemble'),('cnn_lowrise_ensemble','CNN_old')]:
                paired[subset+':'+candidate+'-'+control]=bootstrap(rr,pp,candidate,control)
    assert abs(results['full3332']['CNN_old']['macro_f1']-.4832792226)<1e-9
    cases=list(csv.DictReader((OUT.parent/'20261029_la_diagnosis/MINORITY_CASES.csv').open()))
    index={r['sample_id']:i for i,r in enumerate(rows)}
    for c in cases:
        for name in ['CNN_old','cnn_ensemble','dino_ensemble','cnn_lowrise_ensemble']:c[name]=predictions[name][index[c['sample_id']]]['use']
    for filename,values in [('METRICS.csv',table),('MINORITY_COMPARISON.csv',cases)]:
        with (OUT/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    candidates={r['sample_id']:r for r in json.loads((ROOT/'STABLE_CANDIDATES.json').read_text())}
    strata={}
    for encoder in ['cnn','dino','cnn_lowrise']:
        vp=np.mean([[v['probabilities'] for v in validation_predictions[f'{encoder}_s{s}']] for s in [42,43]],axis=0)
        strata[encoder]={}
        for lo,hi in [(4,4),(5,8),(9,10000)]:
            indices=[i for i,r in enumerate(validation) if int(r['type_class'])==1
                     and lo<=int(candidates[r['sample_id']]['evidence']['Units'])<=hi]
            strata[encoder][f'{lo}-{hi}_units']=dict(n=len(indices),correct=sum(int(vp[i].argmax()==1) for i in indices))
    save(OUT/'RESULTS.json',dict(results=results,paired_cell_bootstrap=paired,fits=fits,prediction_sha256=hashes,
        validation_multifamily_strata=strata,
        dataset_sha256=protocol['dataset_sha256'],limits=protocol['limits']))
    save(OUT/'PREDICTIONS.json',predictions)
    print(json.dumps(dict(full=results['full3332'],paired=paired),indent=2))


def read_predictions(path,rows,hashes):
    raw=path.read_bytes();hashes[str(path)]=hashlib.sha256(raw).hexdigest();values=json.loads(raw)
    assert [v['sample_id'] for v in values]==[r['sample_id'] for r in rows]
    p=np.array([v['probabilities'] for v in values])
    assert p.shape==(len(rows),3) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5)
    assert [v['use'] for v in values]==[LABELS[int(i)] for i in p.argmax(1)]
    return values


if __name__=='__main__':main()
