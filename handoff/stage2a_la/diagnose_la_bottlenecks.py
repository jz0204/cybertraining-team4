"""Read-only diagnosis of saved LA scores, target sampling and error imagery."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, roc_auc_score

from analyze_la_transfer import LABELS, STAGE
from reliable_transfer_common import load, save

OUT=Path(__file__).resolve().parent/'analysis/20261029_bottlenecks'
ROOT=STAGE/'la_fourplex_20261029'
OLD=STAGE/'la_adaptation_20261029'


def main():
    OUT.mkdir(exist_ok=True)
    rows,reference=load(STAGE/'la_transfer_20261027')
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text());raw=(ROOT/'DATASET.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==protocol['dataset_sha256']
    assert protocol['old_test_dataset_sha256']==reference['dataset_sha256']
    data=json.loads(raw);test=[r for r in data if r['split']=='test'];val=[r for r in data if r['split']=='val']
    assert [r['sample_id'] for r in test]==[r['sample_id'] for r in rows]
    prior=OUT.parent/'20261029_la_adaptation';current=OUT.parent/'20261029_la_fourplex'
    pp=json.loads((current/'PREDICTIONS.json').read_text())
    pp['dino_ensemble']=json.loads((prior/'PREDICTIONS.json').read_text())['dino_ensemble']
    names=['CNN_old','cnn_ensemble','cnn_lowrise_ensemble','dino_ensemble','cnn_fourplex_ensemble']
    truth=np.array([LABELS.index(r['reference_use']) for r in rows]);scores={}
    for name in names:
        v=pp[name];assert [r['sample_id'] for r in v]==[r['sample_id'] for r in rows]
        p=np.array([r.get('probabilities',r.get('use_probabilities')) for r in v]);pred=p.argmax(1)
        assert p.shape==(3332,3) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
        scores[name]={}
        for c in [1,2]:
            missed=(truth==c)&(pred!=c);fp=(truth!=c)&(pred==c)
            scores[name][LABELS[c]]=dict(n=int((truth==c).sum()),fn=int(missed.sum()),fp=int(fp.sum()),
                roc_auc=float(roc_auc_score(truth==c,p[:,c])),average_precision=float(average_precision_score(truth==c,p[:,c])),
                wrong_max_probability_gt90=int((missed&(p.max(1)>.9)).sum()),
                false_positive_probability_gt90=int((fp&(p[:,c]>.9)).sum()))
    train=[r for r in data if r['split']=='train'];source=[r for r in train if r['domain']=='source']
    source_mass=np.bincount([int(r['type_class']) for r in source],minlength=3)/len(source)
    raw_val=np.bincount([int(r['type_class']) for r in val],minlength=3)
    weighted_val=np.bincount([int(r['type_class']) for r in val],weights=[r['validation_weight'] for r in val],minlength=3)
    priors=dict(classes=LABELS,training_effective=(.5*source_mass+.5*np.array(protocol['la_class_mass'])).tolist(),
        validation_raw=(raw_val/raw_val.sum()).tolist(),validation_sampling_weighted=(weighted_val/weighted_val.sum()).tolist(),
        test_observed=(np.bincount(truth,minlength=3)/len(truth)).tolist(),
        note='Observed test prevalence is diagnostic only; not used to tune thresholds or priors.')
    assessor_path=STAGE/'la_eval_20261023/reference/la_assessor_matched.csv'
    assessor_rows=[r for r in csv.DictReader(assessor_path.open()) if r['primary_single_outline_iou50']=='True']
    assessor={'la_'+r['cell_id']+'_'+r['bldg_uid']:r for r in assessor_rows}
    assert len(assessor_rows)==len(assessor)==3332 and set(assessor)=={r['sample_id'] for r in rows}
    image_stats=[]
    for ref,row,old,new in zip(rows,test,pp['cnn_ensemble'],pp['cnn_fourplex_ensemble']):
        a=assessor[ref['sample_id']]
        assert a['assessor_type']==ref['reference_use'] and int(row['type_class'])==LABELS.index(ref['reference_use'])
        with Image.open(local(row['mask_path'])) as im:
            mask=np.array(im.convert('L'))/255.;resized=np.array(im.convert('L').resize((224,224),Image.Resampling.NEAREST))/255.
        w=resized.reshape(7,32,7,32).mean(axis=(1,3))
        assert w.sum()>0 and abs(w.sum()-resized.mean()*49)<1e-10
        item=dict(sample_id=ref['sample_id'],reference=ref['reference_use'],old_prediction=old['use'],new_prediction=new['use'],
            footprint_m2=ref['footprint_m2'],gsd=ref['original_gsd'],legacy_width_m=mask.shape[1]*ref['original_gsd'],
            legacy_mask_area_ratio=float(mask.sum()*ref['original_gsd']**2/ref['footprint_m2']),
            feature_mask_area_cells=float(w.sum()),feature_weight_effective_cells=float(w.sum()**2/(w*w).sum()),
            near_available=ref['input_available'],strict_single_record=ref['strict_single_assessor_building'],
            units=float(a['Units']),assessor_use_code=a['UseCode'],
            assessor_description=' | '.join(a[k].strip() for k in ['UseCodeDescChar2','UseCodeDescChar3','UseCodeDescChar4']),
            saved_pre_zero_fraction=float(a['pre_zero_fraction']),
            new_probabilities=new['probabilities'],legacy_path=str(local(row['crop_path'])),
            near_path=str(STAGE/'la_transfer_20261027'/ref['near_marked']) if ref['input_available'] else None,
            wide_path=str(STAGE/'la_transfer_20261027'/ref['wide_marked']) if ref['input_available'] else None)
        image_stats.append(item)
    groups={
        'all_multifamily':[r for r in image_stats if r['reference']==LABELS[1]],
        'missed_multifamily':[r for r in image_stats if r['reference']==LABELS[1] and r['new_prediction']!=LABELS[1]],
        'new_multifamily_false_positive':[r for r in image_stats if r['reference']!=LABELS[1] and r['old_prediction']!=LABELS[1] and r['new_prediction']==LABELS[1]],
        'missed_commercial':[r for r in image_stats if r['reference']==LABELS[2] and r['new_prediction']!=LABELS[2]],
        'multifamily_false_positive':[r for r in image_stats if r['reference']!=LABELS[1] and r['new_prediction']==LABELS[1]],
    }
    summary={}
    for name,rr in groups.items():
        summary[name]=dict(n=len(rr),predictions=dict(Counter(r['new_prediction'] for r in rr)),
            area_m2_quantiles=np.quantile([r['footprint_m2'] for r in rr],[0,.5,1]).tolist(),
            feature_mask_area_cells_quantiles=np.quantile([r['feature_mask_area_cells'] for r in rr],[0,.5,1]).tolist(),
            mask_area_ratio_quantiles=np.quantile([r['legacy_mask_area_ratio'] for r in rr],[0,.5,1]).tolist(),
            ratio_below95=sum(r['legacy_mask_area_ratio']<.95 for r in rr),
            original_context_unavailable=sum(not r['near_available'] for r in rr),
            saved_rgb_zero_fraction_gt10=sum(r['saved_pre_zero_fraction']>.1 for r in rr),
            assessor_use_counts=dict(Counter(r['assessor_use_code'] for r in rr)))
    availability={}
    for available in [True,False]:
        rr=[r for r in image_stats if r['near_available']==available]
        negatives=[r for r in rr if r['reference']!=LABELS[1]]
        availability[str(available)]=dict(n=len(rr),multi_negatives=len(negatives),
            multi_fp=sum(r['new_prediction']==LABELS[1] for r in negatives),
            saved_rgb_zero_fraction_gt10=sum(r['saved_pre_zero_fraction']>.1 for r in rr))
    candidates_path=OLD/'STABLE_CANDIDATES.json'
    candidates={r['sample_id']:r for r in json.loads(candidates_path.read_text())}
    coverage={}
    for split in ['train','val']:
        rr=[r for r in data if r['split']==split and r['domain']=='la' and int(r['type_class'])==2]
        codes=[candidates[r['sample_id']]['evidence']['UseCode'][:2] for r in rr]
        coverage[split]=dict(n=len(rr),use_prefix_counts=dict(Counter(codes)),
            area_m2_quantiles=np.quantile([r['footprint_m2'] for r in rr],[0,.5,1]).tolist(),
            area_below300=sum(r['footprint_m2']<300 for r in rr),office_professional_count=sum(c in ['17','19'] for c in codes))
    for group,rr in [('test',[r for r in image_stats if r['reference']==LABELS[2]]),('missed',groups['missed_commercial'])]:
        coverage[group]=dict(n=len(rr),use_prefix_counts=dict(Counter(r['assessor_use_code'][:2] for r in rr)),
            office_professional_count=sum(r['assessor_use_code'][:2] in ['17','19'] for r in rr))
    for name in ['all_multifamily','new_multifamily_false_positive','missed_commercial']:
        contact_sheets(name,groups[name])
    save(OUT/'DIAGNOSIS.json',dict(scores=scores,priors=priors,image_groups=summary,context_availability_strata=availability,
        la_commercial_coverage=coverage,
        architecture=dict(feature_shape=[768,7,7],input_size=224,pooling=protocol['cnn_config']['pooling_mode'],
            note='Feature shape confirmed by actual CPU forward of the saved common.py model; weighted cells do not measure independent receptive fields.'),
        reference_sha256=reference['dataset_sha256'],training_sha256=protocol['dataset_sha256'],
        assessor_reference_sha256=hashlib.sha256(assessor_path.read_bytes()).hexdigest(),
        candidates_sha256=hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
        source_prediction_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [current/'PREDICTIONS.json',prior/'PREDICTIONS.json']}))
    save(OUT/'IMAGE_DIAGNOSTICS.json',image_stats)
    save(OUT/'VISUAL_REVIEW_CASES.json',groups)
    print(json.dumps(dict(scores=scores,priors=priors,image_groups=summary),indent=2))


def local(path):
    assert path.startswith('/stage3/')
    result=STAGE/path.split('/stage3/')[1].removeprefix('stage2a_')
    assert result.is_file(),result
    return result


def contact_sheets(name,rows):
    folder=OUT/'contact_sheets';folder.mkdir(exist_ok=True)
    for first in range(0,len(rows),4):
        canvas=Image.new('RGB',(1200,4*450),'white');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(rows[first:first+4]):
            y=j*450
            draw.text((4,y+2),f"{first+j+1} {r['sample_id']}\nref={r['reference']} old={r['old_prediction']} new={r['new_prediction']}\nunits={r['units']} area={r['footprint_m2']:.0f}m2 maskarea/full={r['legacy_mask_area_ratio']:.2f} 7x7-mask-area={r['feature_mask_area_cells']:.2f}",fill='black')
            for k,(label,key) in enumerate([('Actual CNN RGB crop','legacy_path'),('Target near (diagnostic)','near_path'),('Wide (diagnostic)','wide_path')]):
                draw.text((k*400+4,y+48),label,fill='black')
                if r[key]:
                    with Image.open(r[key]) as im:canvas.paste(im.convert('RGB').resize((384,384)),(k*400,y+64))
        canvas.save(folder/f'{name}_{first//4+1:02d}.jpg',quality=95)


if __name__=='__main__':main()
