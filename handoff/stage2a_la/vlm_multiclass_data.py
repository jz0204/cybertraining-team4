"""Freeze building-use reference strata and paired native near/context views."""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from vlm_pilot import STAGE, mark_target, read_csv

OUT = STAGE / 'vlm_multiclass_20261026'
SD = STAGE / 'sandiego_pilot_20261015'
HARRIS = STAGE / 'rebuild_20261015/dataset'
INST_ASR = {71, 76, 77, 78, 79, 91}
INST_LU = {6003, 6102, 6103, 6104, 6105, 6108, 6109, 6501, 6502,
           6801, 6802, 6803, 6804, 6805, 6806, 6807, 6809}
OTHER_ASR = {41, 42, 43, 44, 45, 49}
OTHER_LU = {2001, 2101, 2103, 2104, 2201, 2301}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['acquire', 'select', 'permits', 'images', 'sheets', 'self-check'])
    args = parser.parse_args()
    {'acquire': acquire, 'select': select, 'permits': permits, 'images': images, 'sheets': sheets, 'self-check': self_check}[args.action]()


def select():
    from shapely import wkt

    if (OUT / 'PROTOCOL.json').exists():
        raise FileExistsError('Selection frozen by PROTOCOL.json')
    OUT.mkdir(exist_ok=True)
    (OUT / 'raw').mkdir(exist_ok=True)
    harris = {r['building_uid']: r for r in read_csv(HARRIS / 'all_buildings.csv')}
    old = json.loads((STAGE / 'vlm_pilot_20261026/reference.json').read_text())
    rows = []
    selected = set()
    for label in ['residential_small', 'residential_multi', 'commercial']:
        group = sorted([r for r in old if r['domain']=='harris' and r['reference_type']==label],
                       key=lambda r: sample_hash(r['building_uid']))
        selected.update(r['sample_id'] for r in group[:10])
    selected.update([4,239,250])
    for ref in old:
        if ref['domain'] == 'harris' and ref['sample_id'] in selected:
            row = harris[ref['building_uid']]
            entry = harris_record(row, f"H{ref['sample_id']:03d}", 'prior_development')
            entry['old_sample_id'] = ref['sample_id']
            entry['role_reference'] = None
            if ref['sample_id'] in [4, 239, 250]:
                entry.update(reference_grade='C_role_diagnostic', reference_use=None,
                             visual_role_hypothesis='auxiliary', visual_use_hypothesis='other')
            rows.append(entry)
    eligible = defaultdict(list)
    for row in harris.values():
        if (row['STAT_LAND_'] not in {'XJ', 'F2', 'J3', 'J4'}
            or row['parcel_conflicting_use'] != 'False' or float(row['overlap_fraction']) < .95
            or row['mask_truncated'] != 'False' or row['classification_source'] != 'parcel_overlap'):
            continue
        eligible[row['parcel_key']].append(row)
    added = []
    for parcel, candidates in sorted(eligible.items()):
        row = max(candidates, key=lambda r: float(r['footprint_m2']))
        entry = harris_record(row, f'HX{len(added):03d}', 'new_site_reference')
        entry.update(reference_use='institutional' if row['STAT_LAND_'] == 'XJ' else 'other',
                     reference_grade='B_specific_site_use', role_reference=None)
        added.append(entry)
    rows += added
    sd_rows = read_csv(SD / 'label_audit.csv')
    for row in sd_rows:
        if row['label_grade'] in {'A', 'B'}:
            rows.append(sd_record(row))
    extra = select_sd_extra()
    rows += extra
    assert len(rows) == len({r['sample_id'] for r in rows})
    for r in rows:
        assert wkt.loads(r['geometry_wkt']).area > 0
    (OUT / 'reference_candidates.json').write_text(json.dumps(rows, indent=2) + '\n')
    summary = dict(n=len(rows), groups=dict(Counter((r['domain'] + ':' + r['reference_grade']) for r in rows)),
        known_classes=dict(Counter((r['domain'] + ':' + str(r['reference_use'])) for r in rows)),
        extra_sd=len(extra), prior_harris=len(selected), source_labels_unchanged=True,
        no_independent_human_gold=True, selection_before_any_model_outputs=True)
    (OUT / 'SELECTION.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def harris_record(row, sample_id, cohort):
    return dict(sample_id=sample_id, domain='harris', cohort=cohort,
        building_uid=row['building_uid'], spatial_group=row['tile_base'], parcel=row['parcel_key'],
        source_type=row['STAT_LAND_'], reference_use=row['type_class_name'],
        parcel_reference_use=row['type_class_name'], reference_grade='B_parcel_use',
        footprint_m2=float(row['footprint_m2']), gsd=float(row['gsd']),
        geometry_wkt=row['poly_px_wkt'], geometry_crs='pixel', role_reference=None,
        evidence_source=str(HARRIS / 'all_buildings.csv'),
        temporal_limit='2017 image versus later parcel snapshot; actual building use unverified')


def sd_record(row):
    return dict(sample_id=row['sample_id'], domain='sandiego', cohort='heldout_city',
        building_uid=row['physical_id'], spatial_group=row['area_name'], parcel=row['apn'],
        source_type=row['asr_landuse'], reference_use=row['suggested_class'] or None,
        parcel_reference_use=row['candidate_class'], reference_grade=row['label_grade'] + '_prior_review',
        footprint_m2=float(row['footprint_m2']), geometry_wkt=row['geometry_wkt_32611'],
        geometry_crs='EPSG:32611', role_reference=None,
        evidence_source=str(SD / 'label_audit.csv'), official_evidence=row['official_use_evidence_scope'],
        official_evidence_id=row['official_use_evidence_id'],
        temporal_limit=row['time_alignment'])


def select_sd_extra():
    from shapely.strtree import STRtree
    sys.path.insert(0, str(SD))
    import assemble_candidates as source
    source.ROOT = SD
    outlines = source.load_source('buildings')
    parcels = source.load_source('parcels')
    lands = [source.load_source('landuse2023'), source.load_source('landuse2025')]
    source.ROOT = OUT / 'extra_sd'
    for group, name in [(outlines,'buildings'),(parcels,'parcels'),(lands[0],'landuse2023'),(lands[1],'landuse2025')]:
        records = {r['attrs']['objectid']:r for r in group}
        for r in source.load_source(name):
            r['raw_path'] = 'extra_sd/' + r['raw_path']
            records.setdefault(r['attrs']['objectid'],r)
        group[:] = list(records.values())
    outline_tree = STRtree([r['geom'] for r in outlines])
    parcel_tree = STRtree([r['geom'] for r in parcels])
    land_trees = [STRtree([r['geom'] for r in group]) for group in lands]
    pool = []
    for i, building in enumerate(outlines):
        geom = building['geom']
        if geom.area < 20 or len(outline_tree.query(geom, predicate='intersects')) != 1:
            continue
        found = source.matches(geom, parcels, parcel_tree)
        if not found or found[0][0] < .95 or sum(f >= .1 for f, _ in found) != 1:
            continue
        _, parcel = found[0]
        attrs = parcel['attrs']
        code = attrs.get('asr_landuse')
        label = 'institutional' if code in INST_ASR else 'other' if code in OTHER_ASR else None
        if label is None:
            continue
        matches = [source.matches(geom, group, tree) for group, tree in zip(lands, land_trees)]
        allowed = INST_LU if label == 'institutional' else OTHER_LU
        if any(not x or x[0][0] < .9 or x[0][1]['attrs']['lu'] not in allowed for x in matches):
            continue
        if matches[0][0][1]['attrs']['lu'] != matches[1][0][1]['attrs']['lu']:
            continue
        neighbors = [outlines[int(j)] for j in outline_tree.query(parcel['geom'], predicate='intersects')
            if outlines[int(j)]['geom'].area > 20 and
            outlines[int(j)]['geom'].intersection(parcel['geom']).area / outlines[int(j)]['geom'].area >= .5]
        if len(neighbors) != 1:
            continue
        oid = str(building['attrs']['outline_id'])
        pool.append(dict(sample_id='SX' + oid, domain='sandiego', cohort='heldout_city',
            building_uid='outline_' + oid, spatial_group=building['area'], parcel=str(attrs['apn']),
            source_type=code, reference_use=label, parcel_reference_use=label,
            reference_grade='B_single_roof_consistent_sources', role_reference=None,
            footprint_m2=geom.area, geometry_wkt=geom.wkt, geometry_crs='EPSG:32611',
            evidence_source=[building['raw_path'], parcel['raw_path']] + [x[0][1]['raw_path'] for x in matches],
            lu2023=matches[0][0][1]['attrs'], lu2025=matches[1][0][1]['attrs'],
            nucleus_use_cd=attrs.get('nucleus_use_cd'),
            address=' '.join(str(attrs.get(k) or '') for k in ['situs_address','situs_street','situs_suffix']),
            temporal_limit='2022 imagery, 2023/2025 land-use agreement and later parcel; not exact-date building truth'))
    (OUT / 'SD_EXTRA_POOL.json').write_text(json.dumps(pool, indent=2) + '\n')
    chosen, used = [], set()
    for label in ['institutional', 'other']:
        buckets = {a: sorted([r for r in pool if r['spatial_group'] == a and r['reference_use'] == label],
                            key=lambda r: sample_hash(r['building_uid'])) for a in sorted({r['spatial_group'] for r in pool})}
        while sum(r['reference_use'] == label for r in chosen) < 20:
            progress = False
            for values in buckets.values():
                while values:
                    row = values.pop(0)
                    if row['parcel'] not in used:
                        chosen.append(row)
                        used.add(row['parcel'])
                        progress = True
                        break
                if sum(r['reference_use'] == label for r in chosen) == 20:
                    break
            if not progress:
                break
    print('SD extra pool', dict(Counter(r['reference_use'] for r in pool)), flush=True)
    return chosen


def images():
    import numpy as np
    import planetary_computer
    import rasterio
    from PIL import Image
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from shapely import wkt
    from shapely.geometry import Point, shape
    from shapely.ops import transform

    sys.path.insert(0, str(STAGE.parents[2]))
    from II_package.scripts.shared.generate_shared_instance_subimages import clamp_crop_window
    from II_package.scripts.stage2a.training.rebuild_stage2a_dataset import render_mask

    rows = json.loads((OUT / 'reference_candidates.json').read_text())
    for folder in ['images', 'image_metadata', 'source_windows']:
        (OUT / folder).mkdir(exist_ok=True)
    items = json.loads((OUT / 'extra_sd/raw/naip_stac.json').read_text())['features']
    opened, metadata = {}, []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                      GDAL_HTTP_TIMEOUT=30, GDAL_HTTP_MAX_RETRY=2):
        for n, row in enumerate(rows):
            sid = row['sample_id']
            path = OUT / 'image_metadata' / (sid + '.json')
            if path.exists():
                metadata.append(json.loads(path.read_text()))
                continue
            geom = wkt.loads(row['geometry_wkt'])
            if row['domain'] == 'harris':
                source_path = STAGE.parent / 'stage3_harvey/source_data/xBD/train/images' / (row['spatial_group'] + '_pre_disaster.png')
                full = Image.open(source_path).convert('RGB')
                close_size, wide_size = view_sizes(geom.bounds, row['gsd'])
                cx,cy=(geom.bounds[0]+geom.bounds[2])/2,(geom.bounds[1]+geom.bounds[3])/2
                close_box = clamp_crop_window(cx, cy, min(close_size, *full.size), *full.size)
                wide_box = clamp_crop_window(cx, cy, min(wide_size, *full.size), *full.size)
                close, wide = full.crop(close_box), full.crop(wide_box)
                cm = render_mask(geom, close.size, close_box[0], close_box[1])
                wm = render_mask(geom, wide.size, wide_box[0], wide_box[1])
                info = dict(source=str(source_path), source_gsd=row['gsd'], close_box=close_box, wide_box=wide_box,
                            requested_close=close_size, requested_wide=wide_size, source_limited=wide_size > min(full.size))
            else:
                lng, lat = Transformer.from_crs(32611, 4326, always_xy=True).transform(geom.centroid.x, geom.centroid.y)
                candidates = sorted([i for i in items if shape(i['geometry']).covers(Point(lng, lat))], key=lambda i:i['id'])
                success = False
                errors = []
                for item in candidates:
                    try:
                        if item['id'] not in opened:
                            opened[item['id']] = rasterio.open(planetary_computer.sign(item['assets']['image']['href']))
                        src = opened[item['id']]
                        native = transform(Transformer.from_crs(32611, src.crs, always_xy=True).transform, geom)
                        native_px = transform(lambda x,y,z=None: (~src.transform) * (x,y), native)
                        close_size, wide_size = view_sizes(native_px.bounds, src.res[0])
                        cx,cy=(native_px.bounds[0]+native_px.bounds[2])/2,(native_px.bounds[1]+native_px.bounds[3])/2
                        wide_box = clamp_crop_window(cx, cy, wide_size, src.width, src.height)
                        window = Window(wide_box[0], wide_box[1], wide_size, wide_size)
                        values = src.read(indexes=[1,2,3], window=window)
                        assert values.shape == (3,wide_size,wide_size) and values.dtype == np.uint8
                        tr = src.window_transform(window)
                        profile = src.profile.copy()
                        profile.update(width=wide_size,height=wide_size,count=3,transform=tr,compress='deflate')
                        source_path = OUT / 'source_windows' / (sid + '.tif')
                        with rasterio.open(source_path,'w',**profile) as dst: dst.write(values)
                        wide = Image.fromarray(np.moveaxis(values,0,2))
                        wm = Image.fromarray(rasterize([(native,255)], out_shape=(wide_size,wide_size), transform=tr,dtype='uint8'))
                        close_box = clamp_crop_window(cx-wide_box[0], cy-wide_box[1], close_size, wide_size, wide_size)
                        close, cm = wide.crop(close_box), wm.crop(close_box)
                        info = dict(source=item['assets']['image']['href'], item_id=item['id'], imagery_datetime=item['properties']['datetime'],
                            source_gsd=src.res[0], wide_box=wide_box, close_box_relative_to_wide=close_box,
                            requested_close=close_size,requested_wide=wide_size,source_limited=False,
                            native_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())
                        success = True
                        break
                    except Exception as exc:
                        errors.append(dict(item_id=item['id'],error=type(exc).__name__))
                if not success:
                    (OUT / 'raw' / (sid + '_imagery_error.json')).write_text(json.dumps(errors,indent=2))
                    raise RuntimeError('Native imagery unavailable for ' + sid)
            for name, rgb, mask in [('near',close,cm),('wide',wide,wm)]:
                rgb = rgb.resize((512,512),Image.Resampling.LANCZOS)
                mask = mask.resize((512,512),Image.Resampling.NEAREST)
                _, marked, bbox = mark_target(rgb,mask)
                marked.save(OUT / 'images' / (sid + '_' + name + '.png'))
                info[name + '_bbox512'] = bbox
                info[name + '_zero_fraction'] = float(np.all(np.asarray(rgb)==0,axis=2).mean())
                if name == 'near':
                    assert info[name + '_zero_fraction'] < .05
            info['wide_source_missing_pixels_flag'] = info['wide_zero_fraction'] >= .05
            info['sample_id'] = sid
            path.write_text(json.dumps(info,indent=2)+'\n')
            metadata.append(info)
            if (n+1)%10 == 0: print('IMAGES',n+1,'/',len(rows),flush=True)
        for src in opened.values(): src.close()
    (OUT / 'IMAGE_METADATA.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (OUT / 'inputs.json').write_text(json.dumps([dict(sample_id=r['sample_id'],footprint_m2=r['footprint_m2'],
        near='images/'+r['sample_id']+'_near.png',wide='images/'+r['sample_id']+'_wide.png') for r in rows],indent=2)+'\n')
    print('COMPLETE_IMAGES',len(rows),flush=True)


def view_sizes(bounds, gsd):
    long_side = max(bounds[2]-bounds[0],bounds[3]-bounds[1])
    near = max(math.ceil(40/gsd), math.ceil(1.4*long_side))
    return near, max(math.ceil(300/gsd),3*near)


def sample_hash(value):
    return hashlib.sha256(('multiclass20261026:'+value).encode()).hexdigest()


def sheets():
    from PIL import Image,ImageDraw,ImageFont
    rows = json.loads((OUT / 'reference_candidates.json').read_text())
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
    (OUT / 'review_sheets').mkdir(exist_ok=True)
    for part in range(0,len(rows),12):
        canvas=Image.new('RGB',(1040,4*295),'white');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(rows[part:part+12]):
            x,y=(j%3)*346,(j//3)*295
            for dx,view in [(0,'near'),(170,'wide')]:
                with Image.open(OUT/'images'/(r['sample_id']+'_'+view+'.png')) as im:canvas.paste(im.resize((168,168)),(x+dx,y))
            draw.text((x+2,y+227),r['sample_id']+' '+str(r['reference_use']),font=font,fill='black')
            draw.text((x+2,y+246),r['reference_grade'],font=font,fill='black')
            draw.text((x+2,y+265),r['spatial_group'],font=font,fill='black')
        canvas.save(OUT/'review_sheets'/f'{part//12:02d}.jpg',quality=93)


def self_check():
    assert view_sizes((0,0,20,30),.5)==(80,600)
    assert view_sizes((0,0,200,100),.5)==(280,840)
    assert not INST_ASR & OTHER_ASR and not INST_LU & OTHER_LU
    assert sample_hash('a')==sample_hash('a')
    print('PASS view dimensions, disjoint source categories, stable selection')


def acquire():
    import requests
    from datetime import datetime, timezone
    sys.path.insert(0,str(SD))
    from acquire_vectors import BASE, SOURCES
    root=OUT/'extra_sd';(root/'raw').mkdir(parents=True,exist_ok=True)
    areas={'kearny_mesa':[-117.16,32.815,-117.14,32.835],
           'otay_mesa':[-116.976,32.555,-116.946,32.575]}
    manifest_path=root/'ACQUISITION.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for area,bbox in areas.items():
        for name,(service,fields) in SOURCES.items():
            offset=0
            while True:
                filename=f'{area}_{name}_{offset:05d}.json';path=root/'raw'/filename
                params=dict(f='json',where='1=1',geometry=','.join(map(str,bbox)),geometryType='esriGeometryEnvelope',
                    inSR=4326,outSR=32611,spatialRel='esriSpatialRelIntersects',outFields=fields,
                    returnGeometry='true',orderByFields='objectid',resultOffset=offset,resultRecordCount=1000)
                if not path.exists():
                    response=requests.get(BASE+service+'/FeatureServer/0/query',params=params,timeout=60)
                    response.raise_for_status();value=response.json()
                    assert 'features' in value and 'error' not in value and len(response.content)<30_000_000
                    path.write_bytes(response.content)
                    manifest[filename]=dict(url=response.url,sha256=hashlib.sha256(response.content).hexdigest(),
                        features=len(value['features']),retrieved_utc=datetime.now(timezone.utc).isoformat())
                    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
                value=json.loads(path.read_text())
                assert value.get('spatialReference',{}).get('wkid')==32611
                if not value.get('exceededTransferLimit'):break
                assert value['features'];offset+=len(value['features'])
            print('ACQUIRED',area,name,offset+len(value['features']),flush=True)
    path=root/'raw/naip_stac.json'
    if not path.exists():
        response=requests.get('https://planetarycomputer.microsoft.com/api/stac/v1/search',params=dict(collections='naip',
            bbox='-117.23,32.55,-116.93,32.86',datetime='2022-01-01T00:00:00Z/2022-12-31T23:59:59Z',limit=1000),timeout=60)
        response.raise_for_status();value=response.json()
        assert value.get('features') and not any(l.get('rel')=='next' for l in value.get('links',[]))
        path.write_bytes(response.content)
        manifest['naip_stac.json']=dict(url=response.url,sha256=hashlib.sha256(response.content).hexdigest(),
            retrieved_utc=datetime.now(timezone.utc).isoformat())
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')


def permits():
    import requests
    from datetime import datetime,timezone
    rows=json.loads((OUT/'reference_candidates.json').read_text())
    apns=sorted({r['parcel'] for r in rows if r['sample_id'].startswith('SX')})
    assert all(a.isdigit() and len(a)==10 for a in apns)
    fields='OBJECTID,GIS_APN,PROJECT_SCOPE,PROJECT_TITLE,PROJECT_STATUS,APPROVAL_SCOPE,APPROVAL_STATUS,APPROVAL_ISSUE_DATE,APPROVAL_CLOSE_DATE,JOB_BC_CODE_DESCRIPTION'
    records,manifest=[],[]
    for layer in [0,1]:
        offset=0
        while True:
            path=OUT/'raw'/f'permits_{layer}_{offset:05d}.json'
            params=dict(f='json',where='GIS_APN IN ('+','.join("'"+a+"'" for a in apns)+')',outFields=fields,
                        returnGeometry='false',orderByFields='OBJECTID',resultOffset=offset,resultRecordCount=1000)
            url=f'https://webmaps.sandiego.gov/arcgis/rest/services/DoIT_Public/DSDPermits/MapServer/{layer}/query'
            response=requests.get(url,params=params,timeout=60);response.raise_for_status();value=response.json()
            assert 'features' in value and 'error' not in value
            path.write_bytes(response.content)
            manifest.append(dict(url=response.url,file=path.name,sha256=hashlib.sha256(response.content).hexdigest(),
                                 retrieved_utc=datetime.now(timezone.utc).isoformat()))
            records += [dict(x['attributes'],source_layer=layer) for x in value['features']]
            if not value.get('exceededTransferLimit'):break
            assert value['features'];offset+=len(value['features'])
    (OUT/'PERMIT_RECORDS.json').write_text(json.dumps(records,indent=2)+'\n')
    (OUT/'PERMIT_PROVENANCE.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PERMITS',len(records),'parcels',len({r['GIS_APN'] for r in records}),flush=True)


if __name__ == '__main__':
    main()
