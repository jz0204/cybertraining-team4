"""Compare paired fixed/close-up runs on the unchanged LA development cohorts."""
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from prepare_la_closeup import ROOT, PARENT, sha, save

OUT = Path(__file__).parent/'analysis/20261031_closeup'
NAMES = ['residential_small','residential_multi','commercial']


def main():
    rows = read(ROOT/'DATASET.json'); protocol=read(ROOT/'PROTOCOL.json')
    parent_code=read(PARENT/'CODE_SHA256.json')
    for name in ['common.py','train_la_adaptation.py']:
        assert sha(PARENT/name)==parent_code[name]
    parent_protocol=read(PARENT/'C_positive_negative/PROTOCOL.json')
    assert protocol['initial_checkpoints']==parent_protocol['initial_checkpoints']
    assert protocol['cnn_config']==parent_protocol['cnn_config']
    cohorts={s:[r for r in rows if r['split']==s] for s in ['val','test']}
    results={}; predictions={}; checks={}
    for arm,base in [('fixed',PARENT/'collected/C_positive_negative'),('closeup',ROOT/'collected')]:
        members={s:[] for s in cohorts}
        for seed in [42,43]:
            folder=base/f'runs/cnn_lowrise_s{seed}'
            result=read(folder/'RESULT.json'); history=read(folder/'history.json')
            initial=read(folder/'initial_validation.json')
            scores=[initial['sampling_weighted_macro_f1']]+[h['validation']['sampling_weighted_macro_f1'] for h in history]
            assert result['best_epoch']==max(range(len(scores)),key=scores.__getitem__)
            assert len(history)==6 and result['optimizer_updates']==1536
            assert sha(folder/'best.pt')==result['checkpoint_sha256']
            if arm=='closeup':
                assert result['dataset_sha256']==sha(ROOT/'DATASET.json')
                assert result['sampling_factor_sha256']==sha(ROOT/'LOWRISE_SAMPLING.json')
                previous=read(PARENT/f'collected/C_positive_negative/runs/cnn_lowrise_s{seed}/history.json')
                assert [h['draw_sha256'] for h in history]==[h['draw_sha256'] for h in previous]
            checks[f'{arm}_{seed}']=dict(selected_epoch=result['best_epoch'],seconds=result['seconds'],
                peak_reserved_gib=result['peak_reserved_gib'],checkpoint_sha256=result['checkpoint_sha256'])
            for split,rr in cohorts.items():
                values=read(folder/('validation_predictions.json' if split=='val' else 'test_predictions.json'))
                assert [v['sample_id'] for v in values]==[r['sample_id'] for r in rr]
                pp=[v['probabilities'] for v in values]
                assert all(len(p)==3 and all(math.isfinite(x) and 0<=x<=1 for x in p) and abs(sum(p)-1)<1e-5 for p in pp)
                members[split].append(pp)
                results[f'{arm}_{seed}_{split}']=measure(rr,pp,weighted=split=='val')
        for split,rr in cohorts.items():
            pp=[[sum(v)/2 for v in zip(a,b)] for a,b in zip(*members[split])]
            predictions[f'{arm}_{split}']=[dict(sample_id=r['sample_id'],probabilities=p,use=NAMES[argmax(p)]) for r,p in zip(rr,pp)]
            results[f'{arm}_{split}']=measure(rr,pp,weighted=split=='val')
    previous=read(OUT.parent/'20261030_hard_ablation/RESULTS.json')['metrics']['C']
    assert abs(results['fixed_test']['macro_f1']-previous['macro_f1'])<1e-12
    assert abs(results['fixed_test']['accuracy']-previous['accuracy'])<1e-12
    case_sets={}
    for arm in ['fixed','closeup']:
        values=predictions[arm+'_test']
        case_sets[arm]={kind:[r['sample_id'] for r,v in zip(cohorts['test'],values)
                         if (int(r['type_class'])==1,v['use']=='residential_multi')==pair]
                         for kind,pair in [('tp',(True,True)),('fp',(False,True)),('fn',(True,False))]}
    intervals=bootstrap(cohorts['test'],predictions)
    staged=read(ROOT/'collected/STAGED.json'); assert staged['passed']
    report=dict(metrics=results,paired_cell_bootstrap=intervals,multifamily_cases=case_sets,checks=checks,
                input_checks=staged,paired_draws_match=True,
                protocol_sha256=sha(ROOT/'PROTOCOL.json'),unit_panel='Pending independent verification; not used in training')
    OUT.mkdir(exist_ok=True); save(OUT/'RESULTS.json',report); save(OUT/'PREDICTIONS.json',predictions)
    lines=['# Paired visible-target close-up experiment','',
        'The close-up ensemble did not improve the target outcome. Diagnostic macro-F1 fell from 0.610370 to 0.580358; multifamily recall stayed at 3/14 with the same 11 misses, while false positives increased from 15 to 21. Weighted validation macro-F1 also fell, from 0.925022 to 0.902005. Retain C as the research baseline; do not promote this close-up candidate.','',
        'Fixed uses the frozen C two-seed CNN. Close-up changes only the image/mask view, with the same source pixels, labels, split, initial weights, sampling draws and training budget. All four geometry features retain their original augmented-mask values. Each fit uses six epochs and 1,536 optimizer updates; checkpoint selection includes epoch zero and uses only the original weighted validation cohort.','',
        '|Model|LA diagnostic macro-F1|Multifamily TP|FP|FN|Precision|Recall|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ['fixed','closeup']:
        for name in [arm+'_42',arm+'_43',arm]:
            m=results[name+'_test'];u=m['per_class']['residential_multi']
            lines.append(f"|{name}|{m['macro_f1']:.6f}|{u['tp']:g}|{u['fp']:g}|{u['fn']:g}|{u['precision']:.4f}|{u['recall']:.4f}|")
    lines+=['','|Model|Weighted validation macro-F1|Weighted validation multifamily F1|','|---|---:|---:|']
    for arm in ['fixed','closeup']:
        for name in [arm+'_42',arm+'_43',arm]:
            m=results[name+'_val'];lines.append(f"|{name}|{m['macro_f1']:.6f}|{m['per_class']['residential_multi']['f1']:.6f}|")
    lines+=['','Paired spatial-cell bootstrap, 2,000 draws, conditional on these fitted checkpoints (does not include training variability):', '',json.dumps(intervals,indent=2),'',
        'Input audit: 18,480 records (14,710 training, 438 validation, 3,332 diagnostic), 36,960 original file hashes checked. Fixed-view tensors match the prior runtime; augmented geometry is identical across views. Every visible target-mask pixel is retained. 3,204 diagnostic inputs zoom; 121 edge-touching masks retain the original crop. Larger non-edge targets can also retain the original crop.','',
        'This is a same-pixel scale/context experiment, not recovery of complete roofs from wider source imagery. It jointly enlarges the visible target and reduces surrounding context. It cannot identify which of those changes causes a result. Original fixed windows have different physical widths across source domains; the minimum close-up width is 40 m.','',
        'Seed directions are mixed, and the paired macro-F1 interval includes zero; this is no demonstrated improvement, not proof of universal degradation. The conditional context-extension run was not started because the close-up gate did not show a benefit. This does not rule out a separately designed joint-view model, higher native resolution, or full-roof source recovery. The next exposure prerequisite is independently verified per-building unit supervision, with a separate spatial test set. The prepared 51-case unit review package supplies evidence and review fields, not verified unit truth.','',
        'The fixed fits are reused, with source/runtime parity and exact sampling-draw checks, rather than rerun. LA has already been used for development; the cohort has only 14 multifamily references. No unseen-city generalization, dwelling-unit error, observed population, updated population head, or damage-model benefit is measured. Reserved LA inputs remain untouched. No production model is promoted.','',
        f"Training seconds (fixed42/43, close-up42/43): {[round(checks[n]['seconds'],1) for n in ['fixed_42','fixed_43','closeup_42','closeup_43']]}. GPU peak reserved GiB: {[round(checks[n]['peak_reserved_gib'],2) for n in ['fixed_42','fixed_43','closeup_42','closeup_43']]}."
    ]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in results.items() if k in ['fixed_test','closeup_test','fixed_val','closeup_val']},indent=2))


def read(path): return json.loads(path.read_text())
def argmax(values): return max(range(len(values)),key=values.__getitem__)


def measure(rows,probabilities,weighted=False):
    cm=[[0.,0.,0.] for _ in range(3)]
    for r,p in zip(rows,probabilities): cm[int(r['type_class'])][argmax(p)]+=r.get('validation_weight',1.) if weighted else 1.
    return from_confusion(cm)


def from_confusion(cm):
    per={}
    for c,name in enumerate(NAMES):
        tp=cm[c][c];fp=sum(row[c] for row in cm)-tp;fn=sum(cm[c])-tp
        per[name]=dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0.,
                      recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return dict(confusion_matrix=cm,accuracy=sum(cm[c][c] for c in range(3))/sum(map(sum,cm)),
                macro_f1=sum(p['f1'] for p in per.values())/3,per_class=per)


def bootstrap(rows,predictions):
    grouped=defaultdict(list)
    for i,r in enumerate(rows):grouped[r['spatial_group']].append(i)
    cells=list(grouped.values()); matrices=[]
    for cell in cells:
        matrices.append([measure([rows[i] for i in cell],[predictions[a+'_test'][i]['probabilities'] for i in cell])['confusion_matrix'] for a in ['fixed','closeup']])
    rng=random.Random(20261031); draws=defaultdict(list)
    for _ in range(2000):
        cms=[[[0.]*3 for _ in range(3)] for _ in range(2)]
        for k in rng.choices(range(len(cells)),k=len(cells)):
            for a in range(2):
                for i in range(3):
                    for j in range(3):cms[a][i][j]+=matrices[k][a][i][j]
        fixed,close=map(from_confusion,cms)
        draws['macro_f1'].append(close['macro_f1']-fixed['macro_f1'])
        draws['multifamily_recall'].append(close['per_class']['residential_multi']['recall']-fixed['per_class']['residential_multi']['recall'])
    return {k:dict(mean=sum(v)/len(v),percentile95=[sorted(v)[49],sorted(v)[1949]],n=2000) for k,v in draws.items()}


if __name__=='__main__':main()
