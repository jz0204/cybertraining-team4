"""Review the isolated 31 fourplex parcels before admitting individual buildings."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from shapely.geometry import box, shape
from shapely.ops import transform
from shapely import wkt

import prepare_la_adaptation as base
from vlm_multiclass_data import view_sizes

OLD=base.ROOT
ROOT=base.STAGE/'la_fourplex_20261029'
OUT=Path(__file__).resolve().parent/'analysis/20261029_la_fourplex'
QUEUE=OUT.parent/'20261029_la_adaptation/FOURPLEX_REVIEW_QUEUE.csv'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['candidates','images','sheets','freeze'])
    a=p.parse_args()
    for d in [ROOT,ROOT/'raw',ROOT/'images',ROOT/'image_metadata',ROOT/'source_windows',OUT]:d.mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'PROTOCOL.json').exists(),'Preserve frozen inputs'
    base.ROOT=ROOT
    {'candidates':candidates,'images':base.images,'sheets':sheets,'freeze':freeze}[a.action]()


def candidates():
    queue=list(csv.DictReader(QUEUE.open()));ains=sorted(r['parcel'] for r in queue)
    assert len(ains)==len(set(ains))==31
    pool={r['parcel']:r for r in json.loads((OLD/'PARCEL_POOL.json').read_text())}
    assert all(pool[a]['split']=='train' for a in ains)
    where='AIN IN ('+','.join("'"+a+"'" for a in ains)+')'
    detail=base.fetch('assessor_31',base.ASSESSOR+'/query',dict(f='json',where="RollYear='2024' AND "+where,
        outFields=base.FIELDS+',PropertyLocation,Bedrooms,Bathrooms,EffectiveYearBuilt',returnGeometry='false'))
    attrs={f['attributes']['AIN']:f['attributes'] for f in detail['features']}
    assert len(attrs)==len(detail['features'])==31 and not detail.get('exceededTransferLimit')
    ff=base.fetch('footprints_31',base.FOOTPRINT+'/query',dict(f='geojson',where=where,
        outFields='OBJECTID,BLD_ID,AIN,HEIGHT,AREA,DATE_,STATUS',returnGeometry='true',outSR=4326))
    assert not ff.get('exceededTransferLimit')
    old=defaultdict(list)
    for f in ff['features']:old[f['properties']['AIN']].append(f)
    assert all(len(old[a])==2 for a in ains)
    ids=sorted(f['properties']['BLD_ID'] for f in ff['features']);assert len(set(ids))==62
    modern=base.fetch('modern_62',base.MODERN+'/query',dict(f='geojson',
        where='BLD_ID IN ('+','.join("'"+x+"'" for x in ids)+')',
        outFields='OBJECTID,BLD_ID,HEIGHT,AREA,DATE_,STATUS',returnGeometry='true',outSR=4326))
    assert not modern.get('exceededTransferLimit')
    now=defaultdict(list)
    for f in modern['features']:now[f['properties']['BLD_ID']].append(f)
    exclusion=shape(json.loads((OLD/'TEST_EXCLUSION.geojson').read_text())['features'][0]['geometry'])
    items=json.loads((base.STAGE/'la_transfer_20261027/maxar_prefire_items.json').read_text())
    scene_shapes=[(i,shape(i['properties']['proj:geometry'])) for i in items]
    result=[]
    for index,q in enumerate(queue,1):
        a=q['parcel'];e=attrs[a];assert e['UseCode']=='0400' and e['Units']==4 and e['totBuildingDataLines']==1
        assert all(e[k]==pool[a]['evidence'][k] for k in ['UseCode','Units','YearBuilt','SQFTmain'])
        for rank,bld in [('A',q['larger_building_id']),('B',q['smaller_building_id'])]:
            f=next(f for f in old[a] if f['properties']['BLD_ID']==bld);g=transform(base.UTM,shape(f['geometry']))
            assert g.is_valid and g.geom_type=='Polygon'
            near,wide=view_sizes(g.bounds,1);c=g.centroid;h=wide/2+2
            support=box(c.x-h,c.y-h,c.x+h,c.y+h);gx,gy=math.floor(c.x/2000),math.floor(c.y/2000)
            assert f'{gx}_{gy}'==pool[a]['spatial_group'] and not exclusion.intersects(support)
            assert box(gx*2000,gy*2000,(gx+1)*2000,(gy+1)*2000).contains(support)
            scenes=[i for i,s in scene_shapes if s.contains(support)];assert scenes
            item=min(scenes,key=lambda i:(i['properties'].get('view:off_nadir',90),i['properties']['datetime'],i['id']))
            iou=None
            if len(now[bld])==1:
                other=transform(base.UTM,shape(now[bld][0]['geometry']))
                if other.is_valid:iou=g.intersection(other).area/g.union(other).area
            result.append(dict(sample_id=f'la_fourplex_{a}_{bld}',parcel=a,split='train',spatial_group=pool[a]['spatial_group'],
                review_index=index,rank=rank,reference_use=None,reference_role='unreviewed',evidence=e,
                geometry_wkt=g.wkt,geometry_crs='EPSG:32611',footprint_m2=g.area,near_m=near,wide_m=wide,
                floor_area_to_footprint=e['SQFTmain']/(g.area/0.09290304),reported_height=f['properties']['HEIGHT'],
                footprint_evidence=f['properties'],modern_matches=len(now[bld]),outline_iou_2014_2020=iou,
                scene=item,reference_grade='pending_building_level_review',test_distance_m=g.distance(exclusion)+2000))
    base.save(ROOT/'STABLE_CANDIDATES.json',result)
    base.save(ROOT/'CANDIDATE_AUDIT.json',dict(parcels=31,buildings=62,grids=dict(Counter(pool[a]['spatial_group'] for a in ains)),
        stable_2020=sum(r['outline_iou_2014_2020'] is not None and r['outline_iou_2014_2020']>=.8 for r in result),
        minimum_test_distance_m=min(r['test_distance_m'] for r in result),labels_assigned=0,
        queue_sha256=hashlib.sha256(QUEUE.read_bytes()).hexdigest()))
    print('CANDIDATES_PASS',len(result),'buildings; no labels assigned',flush=True)


def sheets():
    from PIL import Image,ImageDraw
    rows=json.loads((ROOT/'STABLE_CANDIDATES.json').read_text())
    assert not json.loads((ROOT/'IMAGE_FAILURES.json').read_text())
    pairs=[rows[i:i+2] for i in range(0,len(rows),2)]
    folder=OUT/'contact_sheets';folder.mkdir(exist_ok=True)
    for first in range(0,len(pairs),4):
        canvas=Image.new('RGB',(1280,4*440),'white');draw=ImageDraw.Draw(canvas)
        for j,pair in enumerate(pairs[first:first+4]):
            a,b=pair;y=j*440;e=a['evidence']
            draw.text((4,y+2),f"#{a['review_index']:02d} AIN={a['parcel']} units=4 beds={e['Bedrooms']} baths={e['Bathrooms']} mainSF={e['SQFTmain']}\nA/B m2={a['footprint_m2']:.0f}/{b['footprint_m2']:.0f} heights={a['reported_height']}/{b['reported_height']} floor/roof A={a['floor_area_to_footprint']:.2f} | {e['PropertyLocation']}",fill='black')
            for k,r in enumerate(pair):
                with Image.open(ROOT/'images'/(r['sample_id']+'_near_marked.png')) as im:
                    canvas.paste(im.resize((384,384)),(k*384,y+50))
                draw.text((k*384+4,y+36),r['rank']+' target',fill='black')
            with Image.open(ROOT/'images'/(a['sample_id']+'_wide_marked.png')) as im:
                canvas.paste(im.resize((384,384)),(832,y+50))
            draw.text((772,y+36),'Context (A outlined)',fill='black')
        canvas.save(folder/f'sheet_{first//4+1:02d}.jpg',quality=95)
    print('SHEETS_READY',len(pairs),'parcels',flush=True)


def freeze():
    from PIL import Image
    from shapely.strtree import STRtree

    parent=json.loads((OLD/'PROTOCOL.json').read_text())
    original=(OLD/'DATASET.json').read_bytes()
    assert hashlib.sha256(original).hexdigest()==parent['dataset_sha256']
    records=json.loads(original);old_records=list(records)
    candidates=json.loads((ROOT/'STABLE_CANDIDATES.json').read_text())
    review_path=OUT/'REVIEW_SOURCE.json';review=json.loads(review_path.read_text())
    assert len(review['parcels'])==31 and {r['index'] for r in review['parcels']}==set(range(1,32))
    decisions={r['index']:r for r in review['parcels']}
    forbidden=shape(json.loads((OLD/'TEST_EXCLUSION.geojson').read_text())['features'][0]['geometry'])
    val=[r for r in records if r['split']=='val'];assert len(val)==438
    tree=STRtree([box(*json.loads((OLD/'image_metadata'/(r['sample_id']+'.json')).read_text())['context_bounds']) for r in val])
    used_parcels={r['parcel'] for r in records if r['domain']!='source'}
    val_grids={r['spatial_group'] for r in val}
    inputs={};audited=[];retention=[]
    for r in candidates:
        decision=decisions[r['review_index']];assert decision['parcel']==r['parcel']
        accepted=decision['admit_primary'] and r['rank']=='A'
        role=('primary' if r['rank']=='A' else 'auxiliary') if decision['admit_primary'] else 'unknown'
        use='residential_multi' if accepted else 'unknown'
        audited.append(dict(r,reference_use=use,reference_role=role,admitted=accepted,
            review_method=review['method'],review_reason=decision['reason'],
            reference_grade='official_plus_ai_visual_review_not_independent_gold',
            auxiliary_subtype='garage_or_carport' if role=='auxiliary' else None))
        if not accepted:continue
        assert r['outline_iou_2014_2020']>=.8 and r['parcel'] not in used_parcels
        assert r['spatial_group'] not in val_grids
        info=json.loads((ROOT/'image_metadata'/(r['sample_id']+'.json')).read_text())
        bounds=box(*info['context_bounds']);g=wkt.loads(r['geometry_wkt']);c=g.centroid
        gx,gy=math.floor(c.x/2000),math.floor(c.y/2000)
        assert r['spatial_group']==f'{gx}_{gy}' and box(gx*2000,gy*2000,(gx+1)*2000,(gy+1)*2000).contains(bounds)
        assert not bounds.intersects(forbidden) and not len(tree.query(bounds,predicate='intersects'))
        assert info['target_retention']>.99 and info['legacy_target_retention']>.99
        assert info['missing_fraction']<.05 and info['imagery_datetime'][:10]<'2025-01-07'
        retention.append(info['legacy_target_retention'])
        crop,mask=[info['paths']['legacy_'+v] for v in ['rgb','mask']]
        for name in [crop,mask]:
            inputs[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            assert inputs[name]==info['image_sha256'][name]
        with Image.open(ROOT/mask) as im:assert im.getextrema()[1]>0
        remote='/stage3/stage2a_la_fourplex_20261029/'
        records.append(dict(sample_id=r['sample_id'],building_uid=r['sample_id'],split='train',domain='la',
            parcel=r['parcel'],spatial_group=r['spatial_group'],crop_path=remote+crop,mask_path=remote+mask,
            footprint_m2=r['footprint_m2'],log_footprint_m2=math.log1p(r['footprint_m2']),type_class=1,
            type_class_name=use,validation_weight=1.,reference_grade=audited[-1]['reference_grade'],
            reference_role=role,assessor_units=4))
        used_parcels.add(r['parcel'])
    assert len(records)-len(old_records)==28 and len(audited)==62
    assert len({r['sample_id'] for r in records})==len(records)
    assert records[:len(old_records)]==old_records
    unchanged={}
    for split,n in [('val',438),('test',3332)]:
        before=[r for r in old_records if r['split']==split];after=[r for r in records if r['split']==split]
        assert before==after and len(after)==n
        unchanged[split]=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest()
    base.save(ROOT/'BUILDING_REVIEW.json',audited)
    summary=dict(reviewed_parcels=31,reviewed_buildings=62,admitted_primary=28,
        roles=dict(Counter(r['reference_role'] for r in audited)),
        added_grids=dict(Counter(r['spatial_group'] for r in records[len(old_records):])),
        unchanged_cohort_sha256=unchanged,parent_dataset_sha256=parent['dataset_sha256'],
        review_source_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
        zero_overlaps=['old_test_2km_buffer','original_val_contexts','train_val_parcels','train_val_grids'],
        added_legacy_retention_min=min(retention),review_limit=review['method'])
    base.save(OUT/'DATA_AUDIT.json',summary)
    base.save(ROOT/'DATASET.json',records);base.save(ROOT/'INPUT_SHA256.json',inputs)
    base.save(ROOT/'PROTOCOL.json',dict(parent,
        dataset_sha256=hashlib.sha256((ROOT/'DATASET.json').read_bytes()).hexdigest(),
        parent_dataset_sha256=parent['dataset_sha256'],review_source_sha256=summary['review_source_sha256'],
        unchanged_cohort_sha256=unchanged,la_class_mass=[478/748,93/748,177/748],
        witness_multifamily_id=records[len(old_records)]['sample_id'],
        counts=dict(Counter(r['split']+':'+r['domain']+':'+r['type_class_name'] for r in records)),
        added_primary_count=28,added_spatial_groups=summary['added_grids'],
        limits=parent['limits']+['All 28 additions are one fourplex tract; count is not diversity.',review['method']]))
    print('FOURPLEX_FREEZE_PASS',json.dumps(summary),flush=True)


if __name__=='__main__':main()
