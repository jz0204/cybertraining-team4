"""Acquire LA adaptation references outside all existing test imagery regions."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from pyproj import Transformer
from shapely import wkt
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import transform, unary_union

STAGE = Path('/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a')
ROOT = STAGE/'la_adaptation_20261029'
ASSESSOR = 'https://services.arcgis.com/RmCCgQtiZLDCtblq/ArcGIS/rest/services/Parcel_Data_2021_Table/FeatureServer/0'
FOOTPRINT = 'https://dpw.gis.lacounty.gov/dpw/rest/services/FEMAFloodzoneAdopted/MapServer/15'
MODERN = 'https://public.gis.lacounty.gov/public/rest/services/LACounty_Dynamic/LARIAC_Buildings_2020/MapServer/0'
FIELDS = 'AIN,RollYear,UseType,UseCode,UseCodeDescChar1,UseCodeDescChar2,UseCodeDescChar3,UseCodeDescChar4,Units,totBuildingDataLines,YearBuilt,SQFTmain,CENTER_LAT,CENTER_LON,OBJECTID'
USES = ['residential_small', 'residential_multi', 'commercial']
UTM = Transformer.from_crs(4326, 32611, always_xy=True).transform


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['pool', 'footprints', 'modern', 'images', 'freeze'])
    a = p.parse_args()
    for name in ['', 'raw', 'images', 'source_windows', 'image_metadata']:
        (ROOT/name).mkdir(parents=True, exist_ok=True)
    assert not (ROOT/'PROTOCOL.json').exists(), 'Preserve frozen inputs'
    {'pool':pool,'footprints':footprints,'modern':modern,'images':images,'freeze':freeze}[a.action]()


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def digest(value):
    return hashlib.sha256(('la-adaptation-20261029:'+str(value)).encode()).hexdigest()


def fetch(name, url, params):
    from audit_sources import get_json
    path = ROOT/'raw'/(name+'.json')
    if path.exists():
        receipt=json.loads(path.with_suffix('.provenance.json').read_text())
        assert receipt['url']==url and receipt['params']==params, 'Cached query differs'
        return json.loads(path.read_text())
    result = get_json(url, params)
    save(path, result)
    save(path.with_suffix('.provenance.json'), dict(url=url, params=params,
        retrieved_utc=datetime.now(timezone.utc).isoformat(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return result


def exclusion():
    product = STAGE/'la_eval_20261023/inputs/building_damage_all_cells.csv'
    grouped = defaultdict(list)
    for r in csv.DictReader(product.open()): grouped[r['cell_id']].append(wkt.loads(r['polygon_wkt_utm']))
    # Exclude every original product cell, not only the 92 cells with scored references.
    bounds = [box(*unary_union(gs).bounds) for gs in grouped.values()]
    old = json.loads((STAGE/'la_transfer_20261027/dataset.json').read_text())
    for r in old:
        c = wkt.loads(r['geometry_wkt']).centroid; h = r['wide_m']/2+2
        bounds.append(box(c.x-h,c.y-h,c.x+h,c.y+h))
    forbidden = unary_union(bounds).buffer(2000)
    save(ROOT/'TEST_EXCLUSION.geojson', dict(type='FeatureCollection',
        crs=dict(type='name',properties=dict(name='EPSG:32611')),
        features=[dict(type='Feature',properties=dict(test_cells=len(grouped),buffer_m=2000),geometry=mapping(forbidden))]))
    save(ROOT/'TEST_SOURCE_HASHES.json', {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [product,STAGE/'la_transfer_20261027/dataset.json',STAGE/'la_transfer_20261027/PROTOCOL.json']})
    return forbidden, {r['parcel'] for r in old}


def pool():
    from evaluate_assessor import map_use
    forbidden, old_parcels = exclusion()
    items = json.loads((STAGE/'la_transfer_20261027/maxar_prefire_items.json').read_text())
    coverage = unary_union([shape(i['properties']['proj:geometry']) for i in items])
    bbox = unary_union([shape(i['geometry']) for i in items]).bounds
    base = (f"RollYear='2024' AND totBuildingDataLines=1 AND SQFTmain>300 AND "
            f"YearBuilt>='1850' AND YearBuilt<='2014' AND CENTER_LON BETWEEN {bbox[0]} AND {bbox[2]} "
            f"AND CENTER_LAT BETWEEN {bbox[1]} AND {bbox[3]}")
    clauses = ["(UseCode LIKE '01%' OR UseCode LIKE '02%' OR UseCode LIKE '03%') AND Units BETWEEN 1 AND 3",
               "(UseCode LIKE '04%' OR UseCode LIKE '05%') AND Units>=4",
               "(UseCode LIKE '1%' OR UseCode LIKE '2%')"]

    def query(job):
        use, clause = job; where = base+' AND ('+clause+')'; values = []
        name=use
        if use=='residential_small':
            where+=' AND MOD(OBJECTID,20)=0'; name+='_mod20'
        count = fetch(name+'_count', ASSESSOR+'/query', dict(f='json',where=where,returnCountOnly='true'))['count']
        for offset in range(0, count, 2000):
            result = fetch(name+f'_{offset:06d}', ASSESSOR+'/query', dict(f='json',where=where,
                outFields=FIELDS, returnGeometry='false', orderByFields='AIN',resultOffset=offset,resultRecordCount=2000))
            values += [f['attributes'] for f in result['features']]
            print('ASSESSOR',use,len(values),count,flush=True)
        assert len(values) == count
        return values

    parcels = sum(list(ThreadPoolExecutor(max_workers=3).map(query,zip(USES,clauses))),[])
    assert len({p['AIN'] for p in parcels}) == len(parcels)
    selected, rejected = [], Counter()
    for p in parcels:
        use = map_use(p)
        if use not in USES: rejected['unmapped_or_mixed']+=1; continue
        point = Point(UTM(p['CENTER_LON'],p['CENTER_LAT']))
        support = box(point.x-250,point.y-250,point.x+250,point.y+250)
        if p['AIN'] in old_parcels or forbidden.intersects(support): rejected['test_buffer']+=1; continue
        if not coverage.contains(support): rejected['outside_prefire_imagery']+=1; continue
        gx,gy = math.floor(point.x/2000),math.floor(point.y/2000)
        if min(point.x%2000,point.y%2000,2000-point.x%2000,2000-point.y%2000)<250:
            rejected['split_boundary']+=1;continue
        fold = int(digest((gx,gy))[:8],16)%5
        split = 'val' if fold==0 else 'reserved' if fold==1 else 'train'
        selected.append(dict(parcel=p['AIN'],reference_use=use,split=split,spatial_group=f'{gx}_{gy}',evidence=p))
    selected.sort(key=lambda r:digest(r['parcel']))
    save(ROOT/'PARCEL_POOL.json',selected)
    save(ROOT/'POOL_AUDIT.json',dict(queried=len(parcels),eligible=len(selected),rejected=dict(rejected),
        counts=dict(Counter(r['split']+':'+r['reference_use'] for r in selected)),
        small_sampling='One in twenty OBJECTIDs for source-pool acquisition only; pool counts are not population prevalences.',
        reservation='No imagery, models or scores will be acquired for reserved cells in this campaign.',
        commercial_units='LA Units is a general unit count, not unitsres; do not exclude a commercial use solely because Units>0.'))
    print('POOL_COMPLETE',json.dumps(json.loads((ROOT/'POOL_AUDIT.json').read_text())),flush=True)


def footprints():
    forbidden, old_parcels = exclusion()
    pool = json.loads((ROOT/'PARCEL_POOL.json').read_text())
    # First batch is a bounded, deterministic candidate queue; quotas are frozen after feasibility checks.
    queue = [r for split in ['train','val'] for use in USES for r in
             [r for r in pool if r['split']==split and r['reference_use']==use][:900 if split=='train' else None]]
    lookup = {r['parcel']:r for r in queue}; ains = sorted(lookup); batches = [ains[i:i+35] for i in range(0,len(ains),35)]

    def get(batch):
        name = 'footprints_'+digest(','.join(batch))[:20]
        result = fetch(name,FOOTPRINT+'/query',dict(f='geojson',where='AIN IN ('+','.join("'"+x+"'" for x in batch)+')',
            outFields='OBJECTID,BLD_ID,AIN,HEIGHT,AREA,DATE_,STATUS',returnGeometry='true',outSR=4326))
        assert result['type']=='FeatureCollection' and not result.get('exceededTransferLimit')
        return result['features']

    all_features = sum(list(ThreadPoolExecutor(max_workers=4).map(get,batches)),[])
    by_parcel = defaultdict(list)
    for f in all_features: by_parcel[f['properties']['AIN']].append(f)
    candidates, rejected = [], Counter()
    items=json.loads((STAGE/'la_transfer_20261027/maxar_prefire_items.json').read_text())
    scene_shapes=[(i,shape(i['properties']['proj:geometry'])) for i in items]
    for ain,r in lookup.items():
        ff=by_parcel[ain]
        if len(ff)!=1: rejected['not_one_countywide_2014_outline']+=1;continue
        f=ff[0];g=transform(UTM,shape(f['geometry']))
        if not g.is_valid or g.geom_type!='Polygon' or g.area<30: rejected['invalid_geometry']+=1;continue
        from vlm_multiclass_data import view_sizes
        near,wide=view_sizes(g.bounds,1);c=g.centroid;h=wide/2+2
        support=box(c.x-h,c.y-h,c.x+h,c.y+h)
        cell=box(math.floor(c.x/2000)*2000,math.floor(c.y/2000)*2000,(math.floor(c.x/2000)+1)*2000,(math.floor(c.y/2000)+1)*2000)
        if forbidden.intersects(support) or not cell.contains(support): rejected['actual_context_overlap']+=1;continue
        group=f'{math.floor(c.x/2000)}_{math.floor(c.y/2000)}'
        if group!=r['spatial_group']: rejected['centroid_cell_change']+=1;continue
        scenes=[i for i,s in scene_shapes if s.contains(support)]
        if not scenes: rejected['no_complete_single_scene']+=1;continue
        item=min(scenes,key=lambda i:(i['properties'].get('view:off_nadir',90),i['properties']['datetime'],i['id']))
        candidates.append(dict(r,sample_id='la_adapt_'+ain,geometry_wkt=g.wkt,geometry_crs='EPSG:32611',
            footprint_m2=g.area,near_m=near,wide_m=wide,footprint_evidence=f['properties'],
            scene=item,reference_grade='2024_single_assessor_record_and_single_countywide_2014_outline_not_human_gold',
            test_distance_m=g.distance(forbidden)+2000))
    save(ROOT/'CANDIDATES.json',candidates)
    save(ROOT/'FOOTPRINT_AUDIT.json',dict(queried_parcels=len(queue),eligible=len(candidates),rejected=dict(rejected),
        counts=dict(Counter(r['split']+':'+r['reference_use'] for r in candidates)),
        minimum_test_distance_m=min([r['test_distance_m'] for r in candidates],default=None)))
    print('FOOTPRINT_COMPLETE',json.dumps(json.loads((ROOT/'FOOTPRINT_AUDIT.json').read_text())),flush=True)


def modern():
    rows=json.loads((ROOT/'CANDIDATES.json').read_text())
    ids=sorted({r['footprint_evidence']['BLD_ID'] for r in rows})
    assert all(x.isdigit() for x in ids)
    def get(batch):
        d=fetch('modern_'+digest(','.join(batch))[:20],MODERN+'/query',dict(f='geojson',
            where='BLD_ID IN ('+','.join("'"+x+"'" for x in batch)+')',
            outFields='OBJECTID,BLD_ID,DATE_,HEIGHT,AREA,STATUS',returnGeometry='true',outSR=4326))
        assert d['type']=='FeatureCollection' and not d.get('exceededTransferLimit')
        return d['features']
    ff=sum(list(ThreadPoolExecutor(max_workers=4).map(get,[ids[i:i+35] for i in range(0,len(ids),35)])),[])
    lookup=defaultdict(list)
    for f in ff:lookup[f['properties']['BLD_ID']].append(f)
    accepted=[];rejected=Counter()
    for r in rows:
        matches=lookup[r['footprint_evidence']['BLD_ID']]
        if len(matches)!=1:rejected['missing_or_nonunique_2020_building_id']+=1;continue
        f=matches[0];g=transform(UTM,shape(f['geometry']));old=wkt.loads(r['geometry_wkt'])
        if not g.is_valid or g.geom_type!='Polygon':rejected['invalid_2020_geometry']+=1;continue
        iou=g.intersection(old).area/g.union(old).area
        if iou<.8:rejected['changed_2014_2020_outline']+=1;continue
        accepted.append(dict(r,modern_footprint_evidence=f['properties'],outline_iou_2014_2020=iou))
    save(ROOT/'STABLE_CANDIDATES.json',accepted)
    save(ROOT/'MODERN_AUDIT.json',dict(accepted=len(accepted),rejected=dict(rejected),
        counts=dict(Counter(r['split']+':'+r['reference_use'] for r in accepted)),
        limitation='2014/2020 outline agreement corroborates geometry, not exact 2024 use or absence of later changes.'))
    print('MODERN_COMPLETE',json.dumps(json.loads((ROOT/'MODERN_AUDIT.json').read_text())),flush=True)


def images():
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from vlm_pilot import mark_target
    from vlm_multiclass_data import view_sizes
    rows=json.loads((ROOT/'STABLE_CANDIDATES.json').read_text())
    opened={}; failures=[]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                      GDAL_HTTP_TIMEOUT=30,GDAL_HTTP_MAX_RETRY=2,GDAL_CACHEMAX=256*1024**2):
        for n,r in enumerate(rows):
            sid=r['sample_id'];out=ROOT/'image_metadata'/(sid+'.json')
            if out.exists():continue
            item=r['scene']; url=item['assets']['visual']['href']
            if url not in opened:opened[url]=rasterio.open(url)
            src=opened[url];assert src.crs.to_epsg()==32611 and src.count==3
            g=wkt.loads(r['geometry_wkt'])
            px=transform(lambda x,y,z=None:(~src.transform)*(x,y),g)
            near,wide=view_sizes(px.bounds,src.res[0]);cx,cy=px.centroid.x,px.centroid.y
            left,top=math.floor(cx-wide/2),math.floor(cy-wide/2)
            assert left>=0 and top>=0 and left+wide<=src.width and top+wide<=src.height
            win=Window(left,top,wide,wide);tr=src.window_transform(win)
            rgb=src.read([1,2,3],window=win);assert rgb.dtype==np.uint8 and rgb.shape==(3,wide,wide)
            mask=rasterize([(g,255)],out_shape=(wide,wide),transform=tr,dtype='uint8')
            missing=np.all(rgb==0,axis=0)
            if not mask.any() or missing.mean()>.05 or missing[mask>0].mean()>.01:
                failures.append(dict(sample_id=sid,reason='missing_target_or_context'));continue
            image=Image.fromarray(np.moveaxis(rgb,0,2));m=Image.fromarray(mask)
            bounds=(round(cx-left-near/2),round(cy-top-near/2),round(cx-left-near/2)+near,round(cy-top-near/2)+near)
            close,cm=image.crop(bounds),m.crop(bounds)
            retention=float(np.asarray(cm).sum()/mask.sum());assert retention>.99
            legacy=(round(cx-left-128),round(cy-top-128),round(cx-left-128)+256,round(cy-top-128)+256)
            legacy_rgb,legacy_mask=image.crop(legacy),m.crop(legacy)
            paths={}
            for name,im,mm in [('near',close,cm),('wide',image,m),('legacy',legacy_rgb,legacy_mask)]:
                size=256 if name=='legacy' else 512
                im=im.resize((size,size),Image.Resampling.LANCZOS);mm=mm.resize((size,size),Image.Resampling.NEAREST)
                _,marked,_=mark_target(im,mm)
                for suffix,value in [('rgb',im),('mask',mm),('marked',marked)]:
                    rel=f'images/{sid}_{name}_{suffix}.png';value.save(ROOT/rel);paths[name+'_'+suffix]=rel
            profile=src.profile.copy();profile.update(width=wide,height=wide,count=3,transform=tr,compress='deflate',photometric='RGB')
            source=ROOT/'source_windows'/(sid+'.tif')
            with rasterio.open(source,'w',**profile) as dst:dst.write(rgb)
            save(out,dict(sample_id=sid,paths=paths,source=url,imagery_datetime=item['properties']['datetime'],
                source_gsd=src.res[0],source_window=list(win.flatten()),context_bounds=list(src.window_bounds(win)),
                geometry_crs=str(src.crs),missing_fraction=float(missing.mean()),target_retention=retention,
                legacy_target_retention=float(np.asarray(legacy_mask).sum()/mask.sum()),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                image_sha256={rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in paths.values()}))
            if (n+1)%10==0:print('IMAGES',n+1,len(rows),'failures',len(failures),flush=True)
        for s in opened.values():s.close()
    save(ROOT/'IMAGE_FAILURES.json',failures)
    print('IMAGES_COMPLETE',len(rows)-len(failures),'failures',len(failures),flush=True)


def freeze():
    import numpy as np
    from PIL import Image
    from shapely.strtree import STRtree
    from reliable_transfer_common import load
    remote='/stage3/stage2a_la_adaptation_20261029'
    candidates=json.loads((ROOT/'STABLE_CANDIDATES.json').read_text())
    failures=json.loads((ROOT/'IMAGE_FAILURES.json').read_text())
    rejected={r['sample_id'] for r in failures};assert not rejected, 'Review acquisition failures before freezing'
    forbidden=shape(json.loads((ROOT/'TEST_EXCLUSION.geojson').read_text())['features'][0]['geometry'])
    old,old_protocol=load(STAGE/'la_transfer_20261027')
    test_parcels={r['parcel'] for r in old}
    records=[];inputs={};contexts=defaultdict(list);retention=[]
    for r in candidates:
        meta=ROOT/'image_metadata'/(r['sample_id']+'.json');info=json.loads(meta.read_text())
        bounds=box(*info['context_bounds']);g=wkt.loads(r['geometry_wkt']);c=g.centroid
        cell=box(math.floor(c.x/2000)*2000,math.floor(c.y/2000)*2000,(math.floor(c.x/2000)+1)*2000,(math.floor(c.y/2000)+1)*2000)
        assert not bounds.intersects(forbidden) and cell.contains(bounds) and r['parcel'] not in test_parcels
        assert info['target_retention']>.99 and info['missing_fraction']<.05 and info['imagery_datetime'][:10]<'2025-01-07'
        contexts[r['split']].append(bounds);retention.append(info['legacy_target_retention'])
        crop,mask=[info['paths']['legacy_'+v] for v in ['rgb','mask']]
        for name in [crop,mask]:
            p=ROOT/name;inputs[name]=hashlib.sha256(p.read_bytes()).hexdigest();assert inputs[name]==info['image_sha256'][name]
        with Image.open(ROOT/mask) as im: assert np.asarray(im).max()>0
        records.append(dict(sample_id=r['sample_id'],building_uid=r['sample_id'],split=r['split'],domain='la',
            parcel=r['parcel'],spatial_group=r['spatial_group'],crop_path=remote+'/'+crop,mask_path=remote+'/'+mask,
            footprint_m2=r['footprint_m2'],log_footprint_m2=math.log1p(r['footprint_m2']),type_class=USES.index(r['reference_use']),
            type_class_name=r['reference_use'],validation_weight=20. if r['reference_use']==USES[0] else 1.,
            reference_grade=r['reference_grade']))
    tree=STRtree(contexts['val']);assert all(not len(tree.query(g,predicate='intersects')) for g in contexts['train'])
    for key in ['parcel','spatial_group']:
        assert not {r[key] for r in records if r['split']=='train'} & {r[key] for r in records if r['split']=='val'}
    source=STAGE/'la_data_20261024/data/N2_nyc_natural_seed42.csv'
    training=[r for r in csv.DictReader(source.open()) if r['rebuild_split']=='train'];assert len(training)==13911
    records += [dict(r,sample_id='source_'+r['building_uid'],split='train',domain='source') for r in training]
    test_path=STAGE/'la_explore_20261023/data/la_input.csv'
    test_inputs={r['building_uid']:r for r in csv.DictReader(test_path.open())}
    records += [dict(test_inputs[r['building_uid']],sample_id=r['sample_id'],split='test',domain='la_test',
        type_class=USES.index(r['reference_use']),type_class_name=r['reference_use'],parcel=r['parcel'],spatial_group=r['cell_id']) for r in old]
    assert len(records)==len({r['sample_id'] for r in records})
    assert all(set(a['sample_id'] for a in records if a['split']==s).isdisjoint({a['sample_id'] for a in records if a['split']==t})
        for s,t in [('train','val'),('train','test'),('val','test')])
    data=ROOT/'DATASET.json';save(data,records)
    cfg=json.loads((STAGE/'la_data_20261024/runs/N2_nyc_natural_seed42/train_config.json').read_text())
    architecture={k:cfg[k] for k in ['num_classes','input_mode','geometry_cols','backbone_name','type_geometry_mode',
        'type_conditioning_mode','pooling_mode','mask_downsample_mode']}
    base={};initial={}
    for seed in [42,43]:
        parent=STAGE/('la_data_20261024' if seed==42 else 'la_data_20261024/followup_20261025')
        name=f'N2_nyc_natural_seed{seed}';values={r['building_uid']:r for r in csv.DictReader((parent/'predictions'/name/'la.csv').open())}
        base[str(seed)]={r['sample_id']:[float(values[r['building_uid']]['prob_'+n]) for n in USES] for r in old}
        initial[str(seed)]='/stage3/stage2a_la_data_20261024/'+('' if seed==42 else 'followup_20261025/')+f'runs/{name}/stage2a_best_model.pt'
    save(ROOT/'INPUT_SHA256.json',inputs)
    save(ROOT/'PROTOCOL.json',dict(dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        old_test_dataset_sha256=old_protocol['dataset_sha256'],cnn_config=architecture,initial_checkpoints=initial,
        baseline_predictions=base,classes=USES,seeds=[42,43],epochs=6,draws_per_epoch=8192,batch_size=32,
        checkpoint_selection='isolated LA val sampling-weighted macro-F1; no old test selection',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,test_path]},
        counts=dict(Counter(r['split']+':'+r['domain']+':'+r['type_class_name'] for r in records)),
        overlap_checks=dict(parcels=0,train_val_contexts=0,test_buffer_contexts=0),
        legacy_target_retention=dict(min=float(min(retention)),below99=sum(v<.99 for v in retention)),
        limits=['Historical official labels are not independent human gold.',
                'Old LA test is development regression evidence; reserved LA cells remain unevaluated.',
                'Legacy fixed256 crops can clip large roofs, matching the prior CNN input contract.']))
    print('FREEZE_PASS',len(records),'new_LA',len(candidates),'zero spatial overlaps',flush=True)


if __name__ == '__main__': main()
