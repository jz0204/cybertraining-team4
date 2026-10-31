"""Prepare isolated review candidates; never add them to the augmentation experiment."""
import json,math
from collections import Counter,defaultdict
from pathlib import Path
from shapely.geometry import shape,box
from shapely.ops import transform,unary_union
from shapely import wkt
import prepare_la_adaptation as base
from vlm_multiclass_data import view_sizes
OLD=base.ROOT
ROOT=base.STAGE/'la_hard_candidates_20261030'

def main():
    for name in ['', 'raw','images','source_windows','image_metadata']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    forbidden,old_parcels=base.exclusion()
    existing=json.loads((OLD/'STABLE_CANDIDATES.json').read_text())
    val=unary_union([wkt.loads(r['geometry_wkt']).centroid.buffer(r['wide_m']/2*2**.5+2) for r in existing if r['split']=='val'])
    excluded=old_parcels|{r['parcel'] for r in existing}
    excluded|={r['parcel'] for r in json.loads((base.STAGE/'la_fourplex_20261029/STABLE_CANDIDATES.json').read_text())}
    pool=json.loads((OLD/'PARCEL_POOL.json').read_text());queue=[]
    for use in ['residential_multi','residential_small']:
        counts=Counter()
        candidates=[r for r in pool if r['split']=='train' and r['reference_use']==use and r['parcel'] not in excluded
                    and (4<=r['evidence']['Units']<=8 and r['evidence']['SQFTmain']<=6500 if use=='residential_multi'
                         else r['evidence']['Units']==1 and 2500<=r['evidence']['SQFTmain']<=8500)]
        for r in sorted(candidates,key=lambda r:(counts[r['spatial_group']],r['parcel'])):
            if counts[r['spatial_group']]>=4:continue
            queue.append(r);counts[r['spatial_group']]+=1
            if sum(counts.values())>=32:break
    base.ROOT=ROOT;base.save(ROOT/'PARCEL_QUEUE.json',queue)
    ff=[]
    for i in range(0,len(queue),25):
        ains=[r['parcel'] for r in queue[i:i+25]]
        result=base.fetch(f'footprints_{i//25}',base.FOOTPRINT+'/query',dict(f='geojson',where='AIN IN ('+','.join("'"+a+"'" for a in ains)+')',outFields='OBJECTID,BLD_ID,AIN,HEIGHT,AREA,DATE_,STATUS',returnGeometry='true',outSR=4326))
        assert not result.get('exceededTransferLimit');ff.extend(result['features'])
    scenes=json.loads((base.STAGE/'la_transfer_20261027/maxar_prefire_items.json').read_text());accepted=[];rejected=Counter()
    for r in queue:
        buildings=[f for f in ff if f['properties']['AIN']==r['parcel']]
        if not buildings:rejected['missing_outline']+=1;continue
        buildings.sort(key=lambda f:-transform(base.UTM,shape(f['geometry'])).area)
        f=buildings[0];g=transform(base.UTM,shape(f['geometry']));near,wide=view_sizes(g.bounds,1)
        if not 150<=g.area<=700:rejected['outside_target_area_range']+=1;continue
        c=g.centroid;h=wide/2+2;support=box(c.x-h,c.y-h,c.x+h,c.y+h)
        if forbidden.intersects(support) or val.intersects(support):rejected['spatial_overlap']+=1;continue
        scene=[s for s in scenes if shape(s['properties']['proj:geometry']).contains(support)]
        if not scene:rejected['no_scene']+=1;continue
        scene=min(scene,key=lambda s:(s['properties'].get('view:off_nadir',90),s['id']))
        accepted.append(dict(r,sample_id='la_hard_'+r['parcel'],geometry_wkt=g.wkt,geometry_crs='EPSG:32611',footprint_m2=g.area,near_m=near,wide_m=wide,scene=scene,footprint_evidence=f['properties'],parcel_outline_count=len(buildings),reference_grade='candidate_only_pending_main_use_and_roof_review',reference_role=None,test_distance_m=g.distance(forbidden)+2000))
    base.save(ROOT/'STABLE_CANDIDATES.json',accepted)
    assert all(r['parcel'] not in excluded for r in accepted)
    base.save(ROOT/'AUDIT.json',dict(queued=len(queue),accepted=len(accepted),counts=dict(Counter(r['reference_use'] for r in accepted)),groups=dict(Counter(r['spatial_group'] for r in accepted)),rejected=dict(rejected),test_and_validation_context_overlap=0,not_added_to_training=True,labels_are_provisional=True))
    print('CANDIDATES',len(accepted),dict(rejected),flush=True)
    base.images()

if __name__=='__main__':main()
