"""Prepare a provenance-preserving, five-use development pilot; no gold labels."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from vlm_pilot import STAGE

OUT = STAGE / 'generalizable_pilot_20261026'
SOURCE = STAGE / 'la_explore_20261023/external/buildingsense'
NAMES = ['residential_small', 'residential_multi', 'commercial', 'institutional', 'other']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['index', 'corroborate', 'prepare', 'self-check'])
    args = p.parse_args()
    {'index': index, 'corroborate': corroborate, 'prepare': prepare, 'self-check': self_check}[args.action]()


def nyc_label(code, category):
    if category == 'Residential':
        if code in {'A0','A1','A2','A3','A4','A5','A6','A7','A9','B1','B2','B3','B9','C0'}:
            return 'residential_small'
        if code in {'C1','C2','C3'}:
            return 'residential_multi'
    if code in {'K1','K2','K3','K5','K6','K7','K8','K9'} and category in {
            'Commercial building','Commercial building:shopping center','Commercial building:bank'}:
        return 'commercial'
    if code in {'O1','O2','O3','O4','O5','O6','O7','O9'} and category in {'Office','Profession service'}:
        return 'commercial'
    if category == 'Hotels' and code in {'H1','H2','H3','H4','H5','HB'}:
        return 'commercial'
    if category == 'Commercial building' and code in {'G2','G3','G4','G8','GW'}:
        return 'commercial'
    if (category == 'Religion' and code in {'M1','M2','M9'} or
        category == 'Education' and code in {'W1','W2','W3','W4','W5','W6','W7','W8','W9'} or
        category == 'Medicine' and code in {'I1','I2'} or
        category == 'Public service' and code in {'P5','P8','W5','Y1','Y2','Y3','Y4','Y6','Z1','Z3'} or
        category == 'Art' and code == 'P7'):
        return 'institutional'
    if (category == 'Factory' and code in {'F1','F2','F4','F5','F9'} or
        category == 'Warehouse' and code in {'E1','E2','E7','E9'} or
        category in {'Garages','Parking lot'} and code in {'G0','G1'}):
        return 'other'
    return None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def index():
    import geopandas as gpd
    from shapely.geometry import box

    self_check()
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'NYC_INDEX.json').exists(), 'Preserve existing preparation'
    folders = {p.name:p for raw in ['raw0','raw1'] for p in (SOURCE/raw).iterdir() if p.is_dir()}
    rows = list(csv.DictReader((SOURCE/'index.csv').open()))
    assert len(rows) == len(folders) == 1000
    records = []
    for item in rows:
        folder = folders[item['id']]
        g = gpd.read_file(folder/'building.shp')
        assert len(g) == 1 and g.crs.to_epsg() == 2263
        geom = g.geometry.iloc[0]
        assert geom.is_valid and geom.area > 0
        geo = g.to_crs(4326).geometry.iloc[0]
        point = geo.representative_point()
        assert box(-74.5,40.3,-73.5,41.1).contains(point)
        records.append(dict(sample_id='N'+item['id'], domain='nyc', source_id=item['id'],
            folder=str(folder), source_code=item['bldclass'], source_category=item['category'],
            label=nyc_label(item['bldclass'],item['category']), role=None,
            geometry_wkt=geom.wkt, geometry_crs='EPSG:2263', geometry_wgs84=geo.wkt,
            longitude=point.x, latitude=point.y, footprint_m2=geom.area*(1200/3937)**2,
            image_year=2018, source_index_sha256=sha(SOURCE/'index.csv'),
            geometry_sha256=sha(folder/'building.shp'), evidence_grade='published_source',
            independent_gold=False))
    save('NYC_INDEX.json', records)
    print('INDEX', len(records), dict(Counter(r['label'] for r in records)), flush=True)


def fetch(name, dataset, params):
    import requests

    path = OUT/'raw'/name
    if not path.exists():
        response = requests.get('https://data.cityofnewyork.us/resource/'+dataset+'.json', params=params, timeout=60)
        response.raise_for_status()
        value = response.json()
        assert isinstance(value, list) and len(value) < int(params['$limit']), 'Pagination needs explicit handling'
        path.write_bytes(response.content)
        save('raw/'+name+'.provenance.json', dict(url=response.url, sha256=sha(path),
             retrieved_utc=datetime.now(timezone.utc).isoformat()))
    return json.loads(path.read_text())


def corroborate():
    from pyproj import Transformer
    from shapely import wkt
    from shapely.geometry import shape
    from shapely.ops import transform
    from shapely.strtree import STRtree

    assert not (OUT/'NYC_CORROBORATED.json').exists(), 'Preserve completed audit'
    rows = json.loads((OUT/'NYC_INDEX.json').read_text())
    (OUT/'raw').mkdir(exist_ok=True)
    buildings = {}
    for start in range(0,len(rows),75):
        points = ','.join(f"({r['longitude']:.9f} {r['latitude']:.9f})" for r in rows[start:start+75])
        values = fetch(f'buildings_{start:04d}.json','5zhs-2jue', {
            '$where':"intersects(the_geom, 'MULTIPOINT ("+points+")')", '$limit':5000,
            '$select':'the_geom,bin,doitt_id,base_bbl,mappluto_bbl,feature_code,last_edited_date'})
        buildings.update({r['doitt_id']:r for r in values})
        print('BUILDINGS',start+len(rows[start:start+75]),len(buildings),flush=True)
    buildings = list(buildings.values())
    project = Transformer.from_crs(4326,2263,always_xy=True).transform
    geometries = [transform(project,shape(r['the_geom'])) for r in buildings]
    tree = STRtree(geometries)
    for row in rows:
        geom = wkt.loads(row['geometry_wkt'])
        matches = sorted([(geom.intersection(geometries[int(i)]).area/geom.union(geometries[int(i)]).area,int(i))
            for i in tree.query(geom,predicate='intersects')],reverse=True)
        row['official_iou'] = matches[0][0] if matches else 0
        if matches and matches[0][0] >= .8:
            b = buildings[matches[0][1]]
            row.update(bin=b.get('bin'),doitt_id=b['doitt_id'],bbl=b.get('mappluto_bbl') or b.get('base_bbl'),
                       official_feature_code=b.get('feature_code'), official_geometry_date=b.get('last_edited_date'))
    bbls = sorted({r['bbl'] for r in rows if r.get('bbl') and r['bbl'].isdigit() and int(r['bbl'])>0})
    parcels = {}
    for start in range(0,len(bbls),100):
        values = fetch(f'pluto_{start:04d}.json','64uk-42ks', {
            '$where':'bbl in ('+','.join(bbls[start:start+100])+')', '$limit':1000,
            '$select':'bbl,bldgclass,numbldgs,unitsres,unitstotal,yearbuilt,yearalter1,yearalter2,version'})
        parcels.update({str(int(float(r['bbl']))):r for r in values})
    for row in rows:
        parcel = parcels.get(row.get('bbl'))
        row['official_parcel'] = parcel
        row['current_single_building_corroborated'] = consistent(row,parcel)
        if row['current_single_building_corroborated']:
            row['evidence_grade'] = 'current_single_building_corroborated'
        row['temporal_limit'] = '2018 imagery; published label date uncertain; current official records are corroboration, not image-date gold'
    save('NYC_CORROBORATED.json',rows)
    report = dict(total=len(rows),matched=sum(r['official_iou']>=.8 for r in rows),
        mapped=dict(Counter(r['label'] for r in rows if r['label'])),
        corroborated=dict(Counter(r['label'] for r in rows if r['current_single_building_corroborated'])),
        role_labels=0, independent_gold=0)
    save('NYC_AUDIT.json',report)
    print(json.dumps(report,indent=2),flush=True)


def consistent(row, parcel):
    if not parcel or not row['label'] or row.get('official_feature_code') != '2100':
        return False
    if float(parcel.get('numbldgs',0)) != 1 or parcel.get('bldgclass') != row['source_code']:
        return False
    units = float(parcel.get('unitsres',0))
    if row['label'] == 'residential_small':
        return 1 <= units <= 3
    if row['label'] == 'residential_multi':
        return units >= 4
    return True


def prepare():
    import math
    import numpy as np
    import rasterio
    from PIL import Image
    from pyproj import Transformer
    from shapely import wkt
    from shapely.geometry import box
    from shapely.ops import transform
    from shapely.strtree import STRtree

    sys.path.insert(0,str(STAGE.parents[2]))
    from II_package.scripts.shared.generate_shared_instance_subimages import clamp_crop_window
    from II_package.scripts.stage2a.training.rebuild_stage2a_dataset import render_mask
    from vlm_multiclass_data import view_sizes

    assert not (OUT/'DATASET.json').exists(), 'Preserve frozen dataset'
    (OUT/'images').mkdir(exist_ok=True)
    (OUT/'image_metadata').mkdir(exist_ok=True)
    nyc = [r for r in json.loads((OUT/'NYC_CORROBORATED.json').read_text()) if r['label']]
    old = json.loads((STAGE/'vlm_multiclass_20261026/reference.json').read_text())
    rows = [dict(r,label=r['reference_use'],role=None,evidence_grade=r['reference_grade']) for r in old
            if r['scored'] and (r['domain']=='sandiego' or r['reference_use'] in NAMES[3:])]
    harris = list(csv.DictReader((STAGE/'la_explore_20261023/data/labels_explore.csv').open()))
    eligible = [r for r in harris if r['rebuild_split']=='train' and r['mask_truncated']=='False'
                and r['parcel_conflicting_use']=='False' and float(r['overlap_fraction'])>=.95
                and box(2,2,1022,1022).contains(wkt.loads(r['poly_px_wkt']))]
    selected = sorted(eligible,key=lambda r:hashlib.sha256(('pilot20261026:'+r['physical_building_id']).encode()).hexdigest())[:1000]
    seen = set()
    for r in selected:
        if r['physical_building_id'] in seen:
            continue
        seen.add(r['physical_building_id'])
        rows.append(dict(sample_id='H'+r['building_uid'],domain='harris',building_uid=r['building_uid'],
            physical_id=r['physical_building_id'],label=r['type_class_name'],role=None,
            evidence_grade='weak_parcel_use',parcel=r['parcel_key'],spatial_group=r['tile_base'],
            geometry_wkt=r['poly_px_wkt'],geometry_crs='pixel',geometry_wgs84=r['poly_geo_wkt'],
            footprint_m2=float(r['footprint_m2']),gsd=float(r['gsd']),image_year=2017,
            temporal_limit='2017 imagery versus 2025 parcel use; units may describe a parcel'))
    all_harris = {r['building_uid']:r for r in csv.DictReader((STAGE/'rebuild_20261015/dataset/all_buildings.csv').open())}
    rows += nyc
    outputs, rejected = [], []
    to_group = Transformer.from_crs(4326,3857,always_xy=True).transform
    for n,row in enumerate(rows):
        sid = row['sample_id']
        meta_path = OUT/'image_metadata'/(sid+'.json')
        if meta_path.exists():
            result=json.loads(meta_path.read_text())
            (outputs if result['image_valid'] else rejected).append(result)
            continue
        geom = wkt.loads(row['geometry_wkt'])
        source = None
        if row['domain'] == 'harris':
            source = STAGE.parent/'stage3_harvey/source_data/xBD/train/images'/(row['spatial_group']+'_pre_disaster.png')
            full = Image.open(source).convert('RGB')
            pixel_geom=geom
            gsd=row['gsd']
            geo = wkt.loads(row.get('geometry_wgs84') or all_harris[row['building_uid']]['poly_geo_wkt'])
        else:
            source = (Path(row['folder'])/'building_buffer_rsi.tif' if row['domain']=='nyc' else
                      STAGE/'vlm_multiclass_20261026/source_windows'/(sid+'.tif'))
            with rasterio.open(source) as src:
                native = transform(Transformer.from_crs(row['geometry_crs'],src.crs,always_xy=True).transform,geom)
                pixel_geom = transform(lambda x,y,z=None:(~src.transform)*(x,y),native)
                full = Image.fromarray(np.moveaxis(src.read([1,2,3]),0,2))
                from pyproj import CRS
                gsd=abs(src.transform.a)*CRS(src.crs).axis_info[0].unit_conversion_factor
                source_transform, source_crs = src.transform, src.crs
            geo = transform(Transformer.from_crs(row['geometry_crs'],4326,always_xy=True).transform,geom)
        near_requested,wide_requested=view_sizes(pixel_geom.bounds,gsd)
        cx,cy=(pixel_geom.bounds[0]+pixel_geom.bounds[2])/2,(pixel_geom.bounds[1]+pixel_geom.bounds[3])/2
        near_box=clamp_crop_window(cx,cy,min(near_requested,*full.size),*full.size)
        wide_box=clamp_crop_window(cx,cy,min(wide_requested,*full.size),*full.size)
        row.update(source_image=str(source),source_image_sha256=sha(source),native_gsd=gsd,
            near_width_m=(near_box[2]-near_box[0])*gsd,wide_width_m=(wide_box[2]-wide_box[0])*gsd,
            wide_limited=wide_requested>min(full.size),near_visible=pixel_geom.intersection(box(*near_box)).area/pixel_geom.area,
            source_visible=pixel_geom.intersection(box(0,0,*full.size)).area/pixel_geom.area,
            longitude=geo.centroid.x,latitude=geo.centroid.y,independent_gold=False)
        row['parcel']=row.get('parcel') or row.get('bbl') or sid
        if row['domain']=='harris':
            center=transform(to_group,geo.centroid)
            # shortcut: no source affine for xBD PNG; conservative full-tile envelope prevents cross-split reuse.
            radius=max(full.size)*gsd/max(.1,math.cos(math.radians(geo.centroid.y)))
            row['coverage_bounds']=[center.x-radius,center.y-radius,center.x+radius,center.y+radius]
        else:
            x0,y0=source_transform*(wide_box[0],wide_box[1]); x1,y1=source_transform*(wide_box[2],wide_box[3])
            cover=transform(Transformer.from_crs(source_crs,3857,always_xy=True).transform,box(x0,y1,x1,y0))
            row['coverage_bounds']=list(cover.buffer(2).bounds)
        for name,win in [('near',near_box),('wide',wide_box)]:
            rgb=full.crop(win).resize((512,512),Image.Resampling.LANCZOS)
            mask=render_mask(pixel_geom,(win[2]-win[0],win[3]-win[1]),win[0],win[1]).resize((512,512),Image.Resampling.NEAREST)
            a=np.asarray(rgb); m=np.asarray(mask)>0
            row[name+'_zero_fraction']=float(np.all(a==0,axis=2).mean())
            row[name+'_target_zero_fraction']=float(np.all(a==0,axis=2)[m].mean()) if m.any() else 1.
            rgb_path=Path('images')/(sid+'_'+name+'.jpg'); mask_path=Path('images')/(sid+'_'+name+'_mask.png')
            rgb.save(OUT/rgb_path,quality=95,subsampling=0);mask.save(OUT/mask_path)
            row[name]=str(rgb_path);row[name+'_mask']=str(mask_path)
            row[name+'_sha256']=sha(OUT/rgb_path);row[name+'_mask_sha256']=sha(OUT/mask_path)
        row['image_valid']=(row['near_visible']>=.999 and row['source_visible']>=.999 and
            row['near_zero_fraction']<.05 and row['wide_zero_fraction']<.05 and
            row['near_target_zero_fraction']<.01 and row['wide_target_zero_fraction']<.01)
        mask=np.asarray(Image.open(OUT/row['near_mask']))>0
        yy,xx=np.where(mask)
        row['geometry_features']=[math.log1p(row['footprint_m2']),float(mask.mean()),
            float((xx.max()-xx.min()+1)/(yy.max()-yy.min()+1)) if len(xx) else 0.,
            float(mask.sum()/((xx.max()-xx.min()+1)*(yy.max()-yy.min()+1))) if len(xx) else 0.]
        meta_path.write_text(json.dumps(row,indent=2)+'\n')
        (outputs if row['image_valid'] else rejected).append(row)
        if (n+1)%100==0: print('IMAGES',n+1,'/',len(rows),flush=True)
    parents=list(range(len(outputs)))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(i,j):
        parents[root(j)]=root(i)
    boxes=[box(*r['coverage_bounds']) for r in outputs]
    tree=STRtree(boxes)
    parcel_groups={}
    for i,row in enumerate(outputs):
        for j in tree.query(boxes[i],predicate='intersects'):
            union(i,int(j))
        key=(row['domain'],row['parcel'])
        if key in parcel_groups:union(i,parcel_groups[key])
        parcel_groups[key]=i
    members={}
    for i,row in enumerate(outputs):members.setdefault(root(i),[]).append(row['sample_id'])
    for i,row in enumerate(outputs):
        row['spatial_component']=hashlib.sha256('|'.join(sorted(members[root(i)])).encode()).hexdigest()[:16]
        row['training_eligible'] = row['domain']!='nyc' or row['current_single_building_corroborated']
    assert len(outputs)==len({r['sample_id'] for r in outputs})
    save('DATASET.json',outputs);save('IMAGE_REJECTED.json',rejected)
    report=dict(n=len(outputs),rejected=len(rejected),city_class_counts={d:dict(Counter(r['label'] for r in outputs if r['domain']==d))
        for d in ['harris','nyc','sandiego']},spatial_components={d:len({r['spatial_component'] for r in outputs if r['domain']==d})
        for d in ['harris','nyc','sandiego']},current_corroborated_nyc=sum(r.get('current_single_building_corroborated',False) for r in outputs),
        role_labels=0,independent_gold=0,images='512px JPEG quality95 no chroma subsampling; masks PNG',
        interpretation='Development source-label pilot, all three cities previously inspected; not independent generalization validation',
        dataset_sha256=sha(OUT/'DATASET.json'))
    save('DATASET_AUDIT.json',report);print(json.dumps(report,indent=2),flush=True)


def self_check():
    assert nyc_label('C0','Residential') == 'residential_small'
    assert nyc_label('C3','Residential') == 'residential_multi'
    assert nyc_label('M1','Religion') == 'institutional'
    assert nyc_label('G0','Garages') == 'other'
    assert nyc_label('F5','Factory') == 'other'
    assert nyc_label('G2','Commercial building') == 'commercial'
    assert all(nyc_label(c,k) is None for c,k in [('M3','Religion_mix'),('I5','Medicine'),
        ('C7','Residential_mix'),('K9','Warehouse'),('F8','Factory'),('A8','Residential')])
    row = dict(label='residential_multi',source_code='C3',official_feature_code='2100')
    parcel = dict(numbldgs='1',bldgclass='C3',unitsres='4')
    assert consistent(row,parcel)
    assert not consistent(row,dict(parcel,numbldgs='2'))
    assert not consistent(row,dict(parcel,unitsres='3'))
    assert not consistent(dict(row,official_feature_code='2101'),parcel)
    assert not consistent(row,None)
    print('PASS taxonomy, ambiguity exclusions, building count and units gates',flush=True)


if __name__ == '__main__':
    main()
