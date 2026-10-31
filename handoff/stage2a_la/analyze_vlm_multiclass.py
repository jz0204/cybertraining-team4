"""Verify frozen results and report paired changes, spatial groups and role diagnostics."""
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from vlm_multiclass import OUT,CLASSES,score


def main():
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    for f,digest in protocol['sha256'].items():
        assert hashlib.sha256((OUT/f).read_bytes()).hexdigest()==digest,f
    refs={r['sample_id']:r for r in json.loads((OUT/'reference.json').read_text())}
    predictions=[json.loads(l) for l in (OUT/'predictions.jsonl').read_text().splitlines()]
    assert len(predictions)==820 and len({(r['sample_id'],r['variant']) for r in predictions})==820
    by_variant={v['name']:{p['sample_id']:p for p in predictions if p['variant']==v['name']} for v in protocol['variants']}
    for name,pp in by_variant.items():
        assert set(pp)==set(refs)
        for domain in ['harris','sandiego']:
            for labels in [CLASSES,CLASSES[:3]]:
                rows=[dict(r,**pp[sid]) for sid,r in refs.items() if r['domain']==domain and r['reference_use'] in labels]
                actual=score(rows,labels);supported=[c for c in labels if any(r['reference_use']==c for r in rows)]
                y=[r['reference_use'] for r in rows];pred=[r['pred_type'] for r in rows]
                precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=supported,zero_division=0)
                assert abs(actual['accuracy']-accuracy_score(y,pred))<1e-12
                assert abs(actual['macro_f1']-statistics.mean(f1))<1e-12
                assert abs(actual['balanced_accuracy']-statistics.mean(recall))<1e-12
                for c,pr,re,fi,n in zip(supported,precision,recall,f1,support):
                    assert actual['per_class'][c]['support']==n
                    assert abs(actual['per_class'][c]['f1']-fi)<1e-12
    contrasts={}
    pairs=[('A_three_near','B_five_near'),('C_three_dual','D_five_dual'),('A_three_near','C_three_dual'),('B_five_near','D_five_dual'),('E_three_forced_dual','C_three_dual'),('E_three_forced_dual','D_five_dual')]
    for domain in ['harris','sandiego']:
        for scope,labels in [('shared_three',CLASSES[:3]),('five',CLASSES)]:
            ids=[sid for sid,r in refs.items() if r['domain']==domain and r['reference_use'] in labels]
            for before,after in pairs:
                correct=lambda sid,v:by_variant[v][sid]['pred_type']==refs[sid]['reference_use']
                fixed=[sid for sid in ids if not correct(sid,before) and correct(sid,after)]
                regressed=[sid for sid in ids if correct(sid,before) and not correct(sid,after)]
                contrasts['/'.join([domain,scope,before+'->'+after])]=dict(n=len(ids),fixed=fixed,regressed=regressed,
                    net=len(fixed)-len(regressed),fixed_groups=dict(Counter(refs[s]['spatial_group'] for s in fixed)),regressed_groups=dict(Counter(refs[s]['spatial_group'] for s in regressed)))
    spatial={};sensitivity={};roles={}
    for v,pp in by_variant.items():
        spatial[v]={}
        for g in sorted({r['domain']+'/'+r['spatial_group'] for r in refs.values()}):
            rr=[dict(r,**pp[sid]) for sid,r in refs.items() if r['domain']+'/'+r['spatial_group']==g]
            spatial[v][g]=dict(five=score(rr,CLASSES),shared_three=score(rr,CLASSES[:3]))
        sensitivity[v]={}
        for domain in ['harris','sandiego']:
            rr=[dict(r,**pp[sid]) for sid,r in refs.items() if r['domain']==domain]
            for r in rr:r['reference_use']=r['candidate_reference_use'] or r['parcel_reference_use']
            sensitivity[v][domain]=dict(five=score(rr,CLASSES),shared_three=score(rr,CLASSES[:3]))
        rr=[dict(r,**pp[sid]) for sid,r in refs.items() if r.get('visual_role_hypothesis')]
        supported=any(r['pred_role'] is not None for r in rr)
        roles[v]=dict(n=len(rr),supported=supported,
            hypothesis_agreement=sum(r['visual_role_hypothesis']==r['pred_role'] for r in rr) if supported else None,
            cases=[{k:r[k] for k in ['sample_id','visual_role_hypothesis','pred_role','pred_type','evidence']} for r in rr],
            independent_truth=False)
    records=[]
    for sid,r in refs.items():
        entry={k:r.get(k) for k in ['sample_id','domain','spatial_group','reference_use','reference_grade','candidate_reference_use','scored','adjudication_reason','visual_role_hypothesis']}
        for name,pp in by_variant.items():
            for k in ['pred_type','pred_role','evidence']:entry[name+'_'+k]=pp[sid][k]
        records.append(entry)
    with (OUT/'CASE_RESULTS.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    for file,value in [('CONTRASTS.json',contrasts),('SPATIAL_RESULTS.json',spatial),('CANDIDATE_LABEL_SENSITIVITY.json',sensitivity),('ROLE_DIAGNOSTICS.json',roles)]:
        (OUT/file).write_text(json.dumps(value,indent=2)+'\n')
    (OUT/'SCORER_VERIFICATION.json').write_text(json.dumps(dict(frozen_hashes=len(protocol['sha256']),prediction_pairs=820,sklearn_checks='20 domain/label-scope combinations; exact accuracy, macroF1, balanced accuracy, class supports/F1',passed=True),indent=2)+'\n')
    print('PASS frozen hashes, complete paired matrix, independent sklearn metrics; paired/spatial/role outputs saved')


if __name__=='__main__':
    main()
