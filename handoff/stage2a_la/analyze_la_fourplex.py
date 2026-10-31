"""Score the fourplex supplement on the unchanged LA cohort and validation set."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from analyze_la_adaptation import read_predictions
from analyze_la_transfer import LABELS, STAGE, bootstrap, measure
from reliable_transfer_common import load, save

ROOT=STAGE/'la_fourplex_20261029'
OLD=STAGE/'la_adaptation_20261029'
OUT=Path(__file__).resolve().parent/'analysis/20261029_la_fourplex'


def main():
    rows,reference_protocol=load(STAGE/'la_transfer_20261027')
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    assert protocol['old_test_dataset_sha256']==reference_protocol['dataset_sha256']
    assert sha(ROOT/'DATASET.json')==protocol['dataset_sha256']
    assert sha(OLD/'DATASET.json')==protocol['parent_dataset_sha256']
    data=json.loads((ROOT/'DATASET.json').read_text());parent=json.loads((OLD/'DATASET.json').read_text())
    assert data[:len(parent)]==parent and len(data)-len(parent)==28
    for split in ['val','test']:
        selected=[r for r in data if r['split']==split]
        assert selected==[r for r in parent if r['split']==split]
        assert hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()==protocol['unchanged_cohort_sha256'][split]
    validation=[r for r in data if r['split']=='val']
    assert len(validation)==438 and [r['sample_id'] for r in data if r['split']=='test']==[r['sample_id'] for r in rows]
    previous=json.loads((OUT.parent/'20261029_la_adaptation/PREDICTIONS.json').read_text())
    predictions={n:previous[n] for n in ['CNN_old','cnn_ensemble','cnn_lowrise_ensemble']}
    fits={};hashes={};vp={}
    for seed in [42,43]:
        name=f'cnn_fourplex_s{seed}';folder=ROOT/'execution/collected/runs'/f'cnn_s{seed}'
        result=json.loads((folder/'RESULT.json').read_text())
        assert result['dataset_sha256']==protocol['dataset_sha256'] and sha(folder/'best.pt')==result['checkpoint_sha256']
        assert result['epochs']==6 and result['optimizer_updates']==1536
        history=json.loads((folder/'history.json').read_text());assert [h['epoch'] for h in history]==list(range(1,7))
        initial=json.loads((folder/'initial_validation.json').read_text())
        scores=[(0,initial['sampling_weighted_macro_f1'])]+[(h['epoch'],h['validation']['sampling_weighted_macro_f1']) for h in history]
        assert result['best_epoch']==max(scores,key=lambda s:s[1])[0]
        # Initialization and validation are identical to the matched-budget control.
        control=json.loads((OLD/'execution/collected/runs'/f'cnn_s{seed}'/'initial_validation.json').read_text())
        assert abs(initial['sampling_weighted_macro_f1']-control['sampling_weighted_macro_f1'])<1e-10
        vp[name]=read_predictions(folder/'validation_predictions.json',validation,hashes)
        weighted=f1_score([int(r['type_class']) for r in validation],[LABELS.index(v['use']) for v in vp[name]],
            labels=[0,1,2],average='macro',zero_division=0,sample_weight=[r['validation_weight'] for r in validation])
        assert abs(weighted-result['validation']['sampling_weighted_macro_f1'])<1e-10
        predictions[name]=read_predictions(folder/'test_predictions.json',rows,hashes)
        score=measure(rows,predictions[name])
        assert abs(score['macro_f1']-result['test']['type_macro_f1'])<1e-10
        assert abs(score['accuracy']-result['test']['type_accuracy'])<1e-10
        witness=json.loads((ROOT/f'execution/collected/WITNESS_cnn_{seed}.json').read_text())
        assert protocol['witness_multifamily_id'] in witness['sample_ids'] and witness['baseline_probability_max_difference']<3e-4
        fits[name]=dict(best_epoch=result['best_epoch'],validation=result['validation'],seconds=result['seconds'],
            peak_reserved_gib=result['peak_reserved_gib'],draw_hashes=[h['draw_sha256'] for h in history])
    p=np.mean([[r['probabilities'] for r in predictions[f'cnn_fourplex_s{s}']] for s in [42,43]],axis=0)
    predictions['cnn_fourplex_ensemble']=[dict(sample_id=r['sample_id'],use=LABELS[int(v.argmax())],probabilities=v.tolist()) for r,v in zip(rows,p)]
    assert all([v['sample_id'] for v in values]==[r['sample_id'] for r in rows] for values in predictions.values())
    results={};paired={};table=[]
    for subset,mask in [('full3332',[True]*len(rows)),('common3157',[r['input_available'] for r in rows])]:
        rr=[r for r,k in zip(rows,mask) if k];pp={n:[v for v,k in zip(values,mask) if k] for n,values in predictions.items()}
        results[subset]={n:measure(rr,v) for n,v in pp.items()}
        for name,m in results[subset].items():
            cm=np.array(m['confusion']);multi=m['per_class'][LABELS[1]];commercial=m['per_class'][LABELS[2]]
            table.append(dict(subset=subset,model=name,n=len(rr),accuracy=m['accuracy'],macro_f1=m['macro_f1'],
                multi_precision=multi['precision'],multi_recall=multi['recall'],multi_f1=multi['f1'],
                multi_tp=int(cm[1,1]),multi_fp=int(cm[:,1].sum()-cm[1,1]),
                commercial_recall=commercial['recall'],commercial_f1=commercial['f1'],
                commercial_tp=int(cm[2,2]),commercial_fp=int(cm[:,2].sum()-cm[2,2])))
        for control in ['CNN_old','cnn_ensemble','cnn_lowrise_ensemble']:
            name=subset+':cnn_fourplex_ensemble-'+control
            paired[name]=dict(macro_f1=bootstrap(rr,pp,'cnn_fourplex_ensemble',control),
                multifamily=minority_bootstrap(rr,pp['cnn_fourplex_ensemble'],pp[control]))
    old_metrics=json.loads((OUT.parent/'20261029_la_adaptation/RESULTS.json').read_text())['results']
    for subset in results:
        for name in ['CNN_old','cnn_ensemble','cnn_lowrise_ensemble']:
            assert results[subset][name]['confusion']==old_metrics[subset][name]['confusion']
    candidates={r['sample_id']:r for r in json.loads((OLD/'STABLE_CANDIDATES.json').read_text())}
    val_prob=np.mean([[v['probabilities'] for v in vp[f'cnn_fourplex_s{s}']] for s in [42,43]],axis=0)
    strata={}
    for lo,hi in [(4,4),(5,8),(9,10000)]:
        indices=[i for i,r in enumerate(validation) if int(r['type_class'])==1 and lo<=candidates[r['sample_id']]['evidence']['Units']<=hi]
        strata[f'{lo}-{hi}_units']=dict(n=len(indices),correct=sum(int(val_prob[i].argmax()==1) for i in indices))
    cases=list(csv.DictReader((OUT.parent/'20261029_la_diagnosis/MINORITY_CASES.csv').open()));index={r['sample_id']:i for i,r in enumerate(rows)}
    for c in cases:
        for name in predictions:c[name]=predictions[name][index[c['sample_id']]]['use']
    test_strata={}
    for lo,hi in [(4,4),(5,8),(9,10000)]:
        subset=[r for r in cases if r['reference_use']==LABELS[1] and lo<=float(r['assessor_units'])<=hi]
        test_strata[f'{lo}-{hi}_units']=dict(n=len(subset),correct={n:sum(r[n]==LABELS[1] for r in subset) for n in predictions})
    changes=[dict(sample_id=r['sample_id'],truth=r['reference_use'],old=b['use'],new=a['use'],cell_id=r['cell_id'])
        for r,b,a in zip(rows,predictions['cnn_ensemble'],predictions['cnn_fourplex_ensemble']) if b['use']!=a['use']]
    save(OUT/'CHANGED_PREDICTIONS.json',changes)
    for filename,values in [('METRICS.csv',table),('MINORITY_COMPARISON.csv',cases)]:
        with (OUT/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    save(OUT/'RESULTS.json',dict(results=results,paired_cell_bootstrap=paired,fits=fits,prediction_sha256=hashes,
        validation_multifamily_strata=strata,test_multifamily_strata=test_strata,
        dataset_sha256=protocol['dataset_sha256'],limits=protocol['limits']))
    save(OUT/'PREDICTIONS.json',predictions)
    print(json.dumps(dict(metrics=[v for v in table if v['subset']=='full3332'],paired=paired,validation_strata=strata),indent=2))


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def minority_bootstrap(rows,candidate,control):
    cells=sorted({r['cell_id'] for r in rows});cell_index={c:i for i,c in enumerate(cells)}
    counts=np.zeros((2,len(cells),3),dtype=int)
    for model,values in enumerate([candidate,control]):
        for r,v in zip(rows,values):
            truth=r['reference_use']==LABELS[1];pred=v['use']==LABELS[1]
            counts[model,cell_index[r['cell_id']]]+=np.array([truth and pred,not truth and pred,truth and not pred])
    def scores(cm):
        tp,fp,fn=cm.T
        return np.stack([np.divide(tp,tp+fn,out=np.zeros(2),where=tp+fn>0),
                         np.divide(2*tp,2*tp+fp+fn,out=np.zeros(2),where=2*tp+fp+fn>0)],axis=1)
    full=scores(counts.sum(1));rng=np.random.default_rng(20261028);draws=[]
    for _ in range(1000):
        cm=counts[:,rng.integers(len(cells),size=len(cells))].sum(1)
        if cm[0,0]+cm[0,2]==0:continue
        s=scores(cm);draws.append(s[0]-s[1])
    expected=[measure(rows,v)['per_class'][LABELS[1]] for v in [candidate,control]]
    assert np.allclose(full,[[v['recall'],v['f1']] for v in expected])
    return dict(metrics=['recall','f1'],delta=(full[0]-full[1]).tolist(),
        ci95=np.quantile(draws,[.025,.975],axis=0).T.tolist(),cells=len(cells),draws=1000,
        valid_draws=len(draws),excluded_draws_reason='No multifamily reference in resampled cells')


if __name__=='__main__':main()
