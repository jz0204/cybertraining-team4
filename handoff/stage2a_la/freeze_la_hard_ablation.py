"""Freeze reviewed additions while preserving the original LA evaluation cohorts."""
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from shapely import wkt
from shapely.geometry import box,shape
from shapely.strtree import STRtree

import prepare_la_adaptation as base

STAGE=base.STAGE
CANDIDATES=STAGE/'la_hard_expansion_20261030'
PARENT=STAGE/'la_augmentation_20261030/execution'
ROOT=STAGE/'la_hard_ablation_20261030/execution'
REMOTE='/stage3/stage2a_la_hard_ablation_20261030'
OUT=Path(__file__).parent/'analysis/20261030_hard_ablation'


def main():
    assert not (ROOT/'FROZEN.json').exists(),'Inputs already frozen'
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'images').mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    parent=json.loads((PARENT/'PROTOCOL.json').read_text());old=json.loads((PARENT/'DATASET.json').read_text())
    assert sha(PARENT/'DATASET.json')==parent['dataset_sha256']
    review=json.loads((CANDIDATES/'REVIEW_SOURCE.json').read_text())
    candidates=json.loads((CANDIDATES/'STABLE_CANDIDATES.json').read_text())
    decisions={r['sample_id']:r for r in review['records']}
    assert set(decisions)=={r['sample_id'] for r in candidates}
    scores={r['sample_id']:r for r in json.loads((CANDIDATES/'MODEL_SCORES.json').read_text())['records']}
    positives=[r for r in candidates if r['reference_use']=='residential_multi' and decisions[r['sample_id']]['admit']]
    negatives=[];grids=Counter()
    for r in sorted(candidates,key=lambda r:-scores[r['sample_id']]['probabilities'][1]):
        v=scores[r['sample_id']]
        if r['reference_use']!='residential_small' or not decisions[r['sample_id']]['admit']:continue
        if v['probabilities'][1]<.05 and max(p[1] for p in v['seed_probabilities'])<.10:continue
        if grids[r['spatial_group']]>=4:continue
        negatives.append(r);grids[r['spatial_group']]+=1
        if len(negatives)==40:break
    assert positives and negatives,('Need eligible positives and confusing negatives',len(positives),len(negatives))
    forbidden=shape(json.loads((base.ROOT/'TEST_EXCLUSION.geojson').read_text())['features'][0]['geometry'])
    validation=[r for r in old if r['split']=='val']
    val_tree=STRtree([box(*json.loads((base.ROOT/'image_metadata'/(r['sample_id']+'.json')).read_text())['context_bounds']) for r in validation])
    val_groups={r['spatial_group'] for r in validation};used={r['parcel'] for r in old if r['domain']!='source'}
    additions=[];hashes={}
    for r in positives+negatives:
        info=json.loads((CANDIDATES/'image_metadata'/(r['sample_id']+'.json')).read_text())
        support=box(*info['context_bounds']);g=wkt.loads(r['geometry_wkt']);c=g.centroid
        gx,gy=math.floor(c.x/2000),math.floor(c.y/2000)
        assert r['parcel'] not in used and r['spatial_group'] not in val_groups
        assert int(base.digest((gx,gy))[:8],16)%5 not in [0,1]
        assert box(gx*2000,gy*2000,(gx+1)*2000,(gy+1)*2000).contains(support)
        assert not support.intersects(forbidden) and len(val_tree.query(support,predicate='intersects'))==0
        assert r['parcel_outline_count']==1 and r['evidence']['totBuildingDataLines']==1 and r['outline_iou_2014_2020']>=.8
        assert info['target_retention']>.99 and info['legacy_target_retention']>.99 and info['missing_fraction']<.05
        assert info['imagery_datetime'][:10]<'2025-01-07' and r['native_gsd']==.6
        paths={k:info['paths']['legacy_'+v] for k,v in [('crop_path','rgb'),('mask_path','mask')]}
        for path in paths.values():
            expected=info['image_sha256'][path];assert sha(CANDIDATES/path)==expected
            shutil.copyfile(CANDIDATES/path,ROOT/path);hashes[path]=expected
        use=r['reference_use'];cls=base.USES.index(use)
        assert (cls==1 and 4<=r['evidence']['Units']<=8) or (cls==0 and r['evidence']['Units']==1)
        additions.append(dict(sample_id=r['sample_id'],building_uid=r['sample_id'],split='train',domain='la',
            parcel=r['parcel'],spatial_group=r['spatial_group'],footprint_m2=r['footprint_m2'],log_footprint_m2=math.log1p(r['footprint_m2']),
            type_class=cls,type_class_name=use,validation_weight=1.,assessor_units=r['evidence']['Units'],reference_role='primary',
            reference_grade='official_single_record_outline_plus_ai_visual_review_not_independent_gold',
            imagery_provider=r['imagery_provider'],imagery_date=info['imagery_datetime'],native_gsd=r['native_gsd'],
            **{k:REMOTE+'/'+v for k,v in paths.items()}))
        used.add(r['parcel'])
    evidence={r['sample_id']:r for r in json.loads((base.ROOT/'STABLE_CANDIDATES.json').read_text())}
    old_factors=json.loads((PARENT/'LOWRISE_SAMPLING.json').read_text())['factors']
    for arm,extra in [('B_positive',additions[:len(positives)]),('C_positive_negative',additions)]:
        folder=ROOT/arm;folder.mkdir(exist_ok=True);rows=old+extra
        assert rows[:len(old)]==old and len({r['sample_id'] for r in rows})==len(rows)
        for split in ['val','test']:assert [r for r in rows if r['split']==split]==[r for r in old if r['split']==split]
        local=[r for r in rows if r['split']=='train' and r['domain']=='la']
        units={r['sample_id']:r.get('assessor_units') or evidence[r['sample_id']]['evidence']['Units'] for r in local if r['type_class']==1}
        bucket={sid:('fourplex' if n==4 else '5to8' if n<=8 else '9plus') for sid,n in units.items()};counts=Counter(bucket.values())
        assert set(counts)=={'fourplex','5to8','9plus'}
        factors={sid:len(bucket)/(3*counts[group]) for sid,group in bucket.items()}
        assert abs(sum(factors.values())-len(factors))<1e-8
        masses=[478/748,93/748,177/748];class_counts=Counter(r['type_class'] for r in local)
        weighted={c:sum(.5*masses[c]/class_counts[c]*factors.get(r['sample_id'],1.) for r in local if r['type_class']==c) for c in range(3)}
        assert all(abs(weighted[c]-.5*masses[c])<1e-10 for c in range(3))
        base.save(folder/'DATASET.json',rows)
        base.save(folder/'PROTOCOL.json',dict(parent,dataset_sha256=sha(folder/'DATASET.json'),la_class_mass=masses,
            parent_dataset_sha256=parent['dataset_sha256'],added_count=len(extra),counts=dict(Counter(r['split'] for r in rows)),
            witness_multifamily_id=extra[0]['sample_id'],sampling_class_mass=weighted,review_source_sha256=sha(CANDIDATES/'REVIEW_SOURCE.json')))
        base.save(folder/'LOWRISE_SAMPLING.json',dict(dataset_sha256=sha(folder/'DATASET.json'),counts=dict(counts),factors=factors,
            policy='Preserve source/LA,class,and three multifamily unit-group masses; change unique training examples.'))
    original_multi=[r for r in old if r['split']=='train' and r['domain']=='la' and r['type_class']==1]
    old_counts=Counter('fourplex' if evidence[r['sample_id']]['evidence']['Units']==4 else '5to8' if evidence[r['sample_id']]['evidence']['Units']<=8 else '9plus' for r in original_multi)
    assert all(abs(old_factors[r['sample_id']]-len(original_multi)/(3*old_counts['fourplex' if evidence[r['sample_id']]['evidence']['Units']==4 else '5to8' if evidence[r['sample_id']]['evidence']['Units']<=8 else '9plus']))<1e-12 for r in original_multi)
    base.save(ROOT/'INPUT_SHA256.json',hashes)
    for name in ['PARENT_INPUT_SHA256.json','MODEL_SHA256.json','common.py']:shutil.copyfile(PARENT/name,ROOT/name)
    trainer=(PARENT/'train_la_adaptation.py').read_text()
    assert trainer.count('        assert not a.lowrise\n')==1
    (ROOT/'train_la_adaptation.py').write_text(trainer.replace('        assert not a.lowrise\n',''))
    audit=dict(positive_count=len(positives),negative_count=len(negatives),
        positives=[r['sample_id'] for r in positives],negatives=[r['sample_id'] for r in negatives],
        positive_grids=dict(Counter(r['spatial_group'] for r in positives)),negative_grids=dict(grids),
        original_test_dataset_sha256=parent['old_test_dataset_sha256'],original_dataset_sha256=parent['dataset_sha256'],
        unchanged_validation_n=438,unchanged_test_n=3332,zero_test_and_val_context_overlap=True,
        no_new_gold=True,baseline_reused=str(PARENT),trainer_change='Allow existing class-mass normalization and lowrise factors together',
        source_change='New NAIP2022 additions have 0.6m native detail; original Maxar inputs unchanged.')
    base.save(ROOT/'FROZEN.json',audit);base.save(OUT/'DATA_AUDIT.json',audit);print(json.dumps(audit,indent=2))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':main()
