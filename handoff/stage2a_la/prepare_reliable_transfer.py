"""Acquire conservative building references and real paired views for a new-city test."""

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from vlm_pilot import STAGE, mark_target
from vlm_multiclass_data import view_sizes

ROOT = STAGE / 'reliable_transfer_20261027'
CLASSES = ['residential_small', 'residential_multi', 'commercial', 'institutional', 'other']
NY_CODES = [set('A1 A2 A3 A4 A5 A6 A7 B1 B2 B3 C0'.split()),
            set('C1 C2 C3 D1 D3'.split()), set('K1 K2 K3 K5 K7 K8 O1 O2 O3 O4'.split()),
            set('M1 M2 W1 W2 W3 W4 P7 P8 I1'.split()), set('E1 E2 E7 F1 F2 F4 F5'.split())]
CB_CODES = [set('101 104 105'.split()), set('111 112'.split()),
            set('324 325 326 340 341'.split()),
            set('930 934 940 9403 941 9410 9413 942 9420 943 956 960 9600'.split()),
            set('316 400 401 407 422 424 430'.split())]
SEA_CODES = [{2, 3, 4}, {5, 11, 17, 18}, {101, 104, 105, 106, 162, 171, 183, 191},
             {165, 172, 173, 184, 185, 189, 193}, {195, 223, 245, 246, 252}]
CB = 'https://services1.arcgis.com/WnzC35krSYGuYov4/arcgis/rest/services/'
SEA = 'https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/services/'
NY = 'https://data.cityofnewyork.us/resource/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['nyc', 'cambridge', 'seattle', 'images', 'freeze', 'self-check'])
    parser.add_argument('--city', choices=['nyc', 'cambridge', 'seattle'])
    args = parser.parse_args()
    if args.action == 'images':
        images(args.city)
    else:
        {'nyc': nyc, 'cambridge': cambridge, 'seattle': seattle,
         'freeze': freeze, 'self-check': self_check}[args.action]()


def digest(value):
    return hashlib.sha256(('reliable-20261027:' + str(value)).encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def fetch(name, url, params=None):
    import requests
    path = ROOT / 'raw' / (name + '.json')
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=90)
            response.raise_for_status()
            value = response.json()
            if isinstance(value, dict) and ('error' in value or 'message' in value):
                raise ValueError(str(value)[:500])
            assert len(response.content) < 80_000_000
            path.write_bytes(response.content)
            save(path.with_suffix('.provenance.json'), dict(url=response.url,
                sha256=hashlib.sha256(response.content).hexdigest(),
                retrieved_utc=datetime.now(timezone.utc).isoformat()))
            return value
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def arc(name, service, fields, where='1=1', cap=None):
    rows, offset = [], 0
    while True:
        params = dict(f='geojson', where=where, outFields=fields, outSR=4326,
                      returnGeometry='true', orderByFields='OBJECTID',
                      resultOffset=offset, resultRecordCount=1000)
        value = fetch(f'{name}_{offset:05d}', service + '/FeatureServer/0/query', params)
        assert value.get('type') == 'FeatureCollection' and isinstance(value['features'], list)
        rows += value['features']
        if len(value['features']) < 1000 or (cap and len(rows) >= cap):
            break
        offset += len(value['features'])
    print('ACQUIRED', name, len(rows), flush=True)
    return rows


def label(code, units, codes):
    found = [CLASSES[i] for i, group in enumerate(codes) if code in group]
    if not found:
        return None
    result = found[0]
    if units is None or (result == CLASSES[0] and not 1 <= units <= 3):
        return None
    if result == CLASSES[1] and units < 4:
        return None
    if result in CLASSES[2:] and units != 0:
        return None
    return result


def number(value):
    return float(value) if value not in ('', None) else None


def metric_geometry(geometry, epsg):
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform
    geom = shape(geometry)
    if geom.is_empty or not geom.is_valid:
        return None
    return transform(Transformer.from_crs(4326, epsg, always_xy=True).transform, geom)


def record(city, identifier, parcel, geom, epsg, use, role, evidence, source):
    near, wide = view_sizes(geom.bounds, 1)
    x, y = geom.centroid.x, geom.centroid.y
    cell = (math.floor(x / 2000), math.floor(y / 2000))
    split = 'test'
    if city == 'nyc':
        # The complete physical context stays inside one source split cell.
        margin = min(x % 2000, y % 2000, 2000 - x % 2000, 2000 - y % 2000)
        if margin <= wide / 2 + 2:
            return None
        split = 'val' if int(digest(str(cell))[:8], 16) % 5 == 0 else 'train'
    return dict(sample_id=city + '_' + str(identifier), city=city, parcel=str(parcel),
                split=split, spatial_group=str(cell), reference_use=use, reference_role=role,
                stratum=use or 'auxiliary', geometry_wkt=geom.wkt, geometry_crs=f'EPSG:{epsg}',
                footprint_m2=geom.area, near_m=near, wide_m=wide,
                evidence=evidence, evidence_source=source,
                reference_grade='official_building_corroboration_not_human_gold',
                temporal_limit='Current official records versus NAIP; no exact-date occupancy verification')


def choose(city, rows):
    assert not (ROOT / 'PROTOCOL.json').exists(), 'Frozen dataset cannot be reselected'
    rows = [r for r in rows if r and r['footprint_m2'] >= 15]
    save(ROOT / (city + '_pool.json'), rows)
    chosen, used = [], set()
    for stratum in CLASSES + ['auxiliary']:
        splits = [('train', 120), ('val', 30)] if city == 'nyc' else [('test', 50)]
        for split, count in splits:
            group = sorted([r for r in rows if r['stratum'] == stratum and r['split'] == split],
                           key=lambda r: digest(r['sample_id']))
            selected = 0
            for row in group:
                key = (row['parcel'], stratum)
                if key in used:
                    continue
                chosen.append(row)
                used.add(key)
                selected += 1
                if selected == count:
                    break
    save(ROOT / (city + '_candidates.json'), chosen)
    print('SELECTED', city, Counter(r['split'] + ':' + r['stratum'] for r in chosen), flush=True)


def nyc():
    fields = 'bbl,bldgclass,numbldgs,unitsres,unitstotal,yearbuilt,yearalter1,yearalter2,version,latitude,longitude,comarea'
    jobs = [(i, boro) for i in range(5) for boro in range(1, 6)]
    def parcels(job):
        i, boro = job
        codes = ','.join("'" + c + "'" for c in sorted(NY_CODES[i]))
        where = f"bldgclass in ({codes}) AND numbldgs=1 AND yearbuilt between 1850 and 2014 AND bbl between {boro*10**9} and {(boro+1)*10**9-1}"
        return fetch(f'nyc_pluto_{i}_{boro}', NY + '64uk-42ks.json',
                     {'$select': fields, '$where': where, '$limit': 400, '$order': 'bbl'})
    with ThreadPoolExecutor(max_workers=3) as pool:
        all_parcels = sum(list(pool.map(parcels, jobs)), [])
    eligible = defaultdict(list)
    for p in all_parcels:
        use = label(p['bldgclass'], number(p.get('unitsres')), NY_CODES)
        if not use or max(number(p.get(k)) or 0 for k in ['yearalter1', 'yearalter2']) > 2021:
            continue
        if use in CLASSES[:2] and (number(p.get('comarea')) != 0 or number(p.get('unitstotal')) != number(p.get('unitsres'))):
            continue
        p['bbl_key'] = str(int(float(p['bbl'])))
        eligible[use].append(p)
    selected = [p for use in CLASSES for p in sorted(eligible[use], key=lambda p: digest(p['bbl']))]
    by_bbl = {p['bbl_key']: p for p in selected}
    for boro in range(1,6):
        codes=','.join("'"+c+"'" for c in sorted(NY_CODES[0]))
        extra=fetch(f'nyc_garage_parcels_{boro}',NY+'64uk-42ks.json',
            {'$select':fields,'$where':f"bldgclass in ({codes}) AND numbldgs between 2 and 3 AND yearbuilt between 1850 and 2014 AND bbl between {boro*10**9} and {(boro+1)*10**9-1}",
             '$limit':500,'$order':'bbl'})
        for p in extra:
            if label(p['bldgclass'],number(p.get('unitsres')),NY_CODES) != CLASSES[0]:continue
            p['bbl_key']=str(int(float(p['bbl'])))
            p['auxiliary_lookup_only']=True
            by_bbl[p['bbl_key']]=p
    footprints = defaultdict(list)
    keys = sorted(by_bbl)
    for i in range(0, len(keys), 75):
        where = 'mappluto_bbl in (' + ','.join("'" + k + "'" for k in keys[i:i+75]) + ')'
        values = fetch(f'nyc_buildings_{digest(where)[:12]}', NY + '5zhs-2jue.json',
                       {'$select': 'the_geom,bin,doitt_id,base_bbl,mappluto_bbl,construction_year,feature_code',
                        '$where': where, '$limit': 5000})
        assert len(values) < 5000
        for f in values:
            footprints[f.get('mappluto_bbl')].append(f)
    rows = []
    for bbl, p in by_bbl.items():
        main = [f for f in footprints[bbl] if f['feature_code'] == '2100']
        if len(main) != 1 or main[0].get('base_bbl') not in (bbl, '0', None):
            continue
        f = main[0]
        if (number(f.get('construction_year')) or 0) > 2021:
            continue
        geom = metric_geometry(f['the_geom'], 32618)
        if geom is None:
            continue
        if not p.get('auxiliary_lookup_only'):
            rows.append(record('nyc', f['doitt_id'], bbl, geom, 32618,
                label(p['bldgclass'], number(p.get('unitsres')), NY_CODES), 'primary',
                dict(pluto=p, building={k:v for k,v in f.items() if k != 'the_geom'},
                     addressable_footprint_count=1, parcel_geometry_overlap=None,
                     linkage='official mappluto_bbl; unique addressable footprint and PLUTO numbldgs=1'),
                ['NYC PLUTO 64uk-42ks', 'NYC Building Footprints 5zhs-2jue']))
        # A detached official garage beside a corroborated residential main building supports role only.
        if p['bldgclass'] in NY_CODES[0] | NY_CODES[1]:
            for garage in footprints[bbl]:
                if garage['feature_code'] != '5110' or (number(garage.get('construction_year')) or 0) > 2021:
                    continue
                gg = metric_geometry(garage['the_geom'], 32618)
                if gg is None or gg.intersects(geom) or gg.distance(geom) > 60 or not 15 <= gg.area <= 150:
                    continue
                rows.append(record('nyc', garage['doitt_id'], bbl, gg, 32618, None, 'auxiliary',
                    dict(feature_code=5110, residential_main_building=f['doitt_id'], pluto=p,
                         official_same_parcel=True, distance_to_main_m=gg.distance(geom)),
                    ['NYC garage footprint 5110', 'NYC residential PLUTO and unique main footprint']))
    aux = fetch('nyc_auxiliary', NY + '5zhs-2jue.json',
                {'$select': 'the_geom,bin,doitt_id,base_bbl,mappluto_bbl,construction_year,feature_code',
                 '$where': 'feature_code=1004', '$limit': 5000, '$order': 'doitt_id'})
    for f in aux:
        geom = metric_geometry(f['the_geom'], 32618)
        bbl = f.get('mappluto_bbl') or f.get('base_bbl')
        if not bbl or bbl == '0' or geom is None or not 15 <= geom.area <= 300:
            continue
        if (number(f.get('construction_year')) or 0) > 2021:
            continue
        rows.append(record('nyc', f['doitt_id'], bbl, geom, 32618, None, 'auxiliary',
                           dict(feature_code=1004, official_definition='nonaddressable auxiliary, excludes garages'),
                           ['NYC Building Footprints 5zhs-2jue']))
    choose('nyc', rows)


def cambridge():
    from shapely.strtree import STRtree
    fields = 'OBJECTID,ml,use_code,use_description,residential_units,mixed_use_type,group_quarters'
    lands = [arc('cb_land_' + str(y), CB + name, fields) for y, name in
             [(2024, 'LandUseFY2024'), (2025, 'Land_Use_FY2025')]]
    buildings = arc('cb_building', CB + 'Buildings', 'OBJECTID,TYPE,BldgID', "TYPE IN ('BLDG','OUTBLDG')")
    outlines = [(b['properties'], metric_geometry(b['geometry'], 32619)) for b in buildings]
    outlines = [(a,g) for a,g in outlines if g is not None]
    tree = STRtree([g for _,g in outlines])
    old = {p['properties']['ml']: p['properties'] for p in lands[0]}
    rows, linked_aux = [], set()
    for p in lands[1]:
        a = p['properties']; before = old.get(a['ml'])
        geom = metric_geometry(p['geometry'], 32619)
        if geom is None:
            continue
        matches = [(outlines[int(i)][0], outlines[int(i)][1]) for i in tree.query(geom, predicate='intersects')
                   if outlines[int(i)][1].intersection(geom).area / outlines[int(i)][1].area >= .95]
        for b, g in matches:
            if b['TYPE'] == 'OUTBLDG' and b['OBJECTID'] not in linked_aux:
                rows.append(record('cambridge', b['OBJECTID'], a['ml'], g, 32619, None, 'auxiliary',
                                   dict(building=b, overlap=g.intersection(geom).area/g.area), ['Cambridge Buildings TYPE=OUTBLDG']))
                linked_aux.add(b['OBJECTID'])
        use = label(a['use_code'], number(a['residential_units']), CB_CODES)
        if not before or not use or any(before[k] != a[k] for k in ['use_code', 'residential_units']):
            continue
        if any(str(q.get('group_quarters') or '').lower() not in ('', 'no', 'n', '0') or
               str(q.get('mixed_use_type') or '').lower() not in ('', 'none', 'no') for q in [a, before]):
            continue
        main = [(b,g) for b,g in matches if b['TYPE'] == 'BLDG']
        # Count any substantial parcel intersection as an ambiguity, even at a boundary.
        intersecting = [int(i) for i in tree.query(geom, predicate='intersects')
                        if outlines[int(i)][0]['TYPE'] == 'BLDG' and
                        outlines[int(i)][1].intersection(geom).area / outlines[int(i)][1].area >= .1]
        if len(main) != 1 or len(intersecting) != 1:
            continue
        b, g = main[0]
        rows.append(record('cambridge', b['OBJECTID'], a['ml'], g, 32619, use, 'primary',
                           dict(landuse2024=before, landuse2025=a, building=b,
                                main_building_count=1, overlap=g.intersection(geom).area/g.area),
                           ['Cambridge LandUseFY2024', 'Cambridge Land Use FY2025', 'Cambridge Buildings']))
    choose('cambridge', rows)


def seattle():
    fields = 'OBJECTID,PIN,PUC,PRES_USE,NR_BLDGS,NR_UNITS,YR_BUILT_MAX,YR_RENOV_MAX,BLDG_DESC'
    parcels = []
    for i, codes in enumerate(SEA_CODES):
        where = 'NR_BLDGS=1 AND PUC IN (' + ','.join(map(str, sorted(codes))) + ')'
        values = arc('sea_parcels_' + str(i), SEA + 'PARCEL_GEO', fields, where)
        values = [p for p in values if label(p['properties']['PUC'], p['properties']['NR_UNITS'], SEA_CODES)
                  and 1800 <= (number(p['properties']['YR_BUILT_MAX']) or 0) < 2022
                  and (number(p['properties']['YR_RENOV_MAX']) or 0) < 2022]
        parcels += sorted(values, key=lambda p: digest(p['properties']['PIN']))[:200]
    pins = sorted({p['properties']['PIN'] for p in parcels})
    footprints = defaultdict(list)
    for i in range(0, len(pins), 75):
        where = 'PIN IN (' + ','.join("'" + k + "'" for k in pins[i:i+75]) + ')'
        for f in arc(f'sea_buildings_{i:04d}', SEA + 'Building_Outlines_2023', 'OBJECTID,OUTLINE_ID,PIN,COMMENT_', where):
            footprints[f['properties']['PIN']].append(f)
    rows = []
    for p in parcels:
        a = p['properties']; fs = footprints[a['PIN']]
        if len(fs) != 1:
            continue
        geom = metric_geometry(fs[0]['geometry'], 32610)
        parcel_geom = metric_geometry(p['geometry'], 32610)
        if geom is None or parcel_geom is None or geom.intersection(parcel_geom).area / geom.area < .95:
            continue
        rows.append(record('seattle', fs[0]['properties']['OUTLINE_ID'], a['PIN'], geom, 32610,
                           label(a['PUC'], a['NR_UNITS'], SEA_CODES), 'primary',
                           dict(parcel=a, building=fs[0]['properties'], footprint_count=1,
                                overlap=geom.intersection(parcel_geom).area/geom.area),
                           ['Seattle PARCEL_GEO assessor summary', 'Seattle Building Outlines 2023']))
    choose('seattle', rows)


def images(city):
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
    assert city, '--city required'
    assert not (ROOT / 'PROTOCOL.json').exists()
    rows = json.loads((ROOT / (city + '_candidates.json')).read_text())
    stac = json.loads((ROOT / 'raw' / ('naip_' + city + '.json')).read_text())
    items, page = list(stac['features']), 1
    while any(x.get('rel') == 'next' for x in stac.get('links', [])):
        url = next(x['href'] for x in stac['links'] if x.get('rel') == 'next')
        stac = fetch(f'naip_{city}_page{page}', url); items += stac['features']; page += 1
    opened, failures = {}, []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                      GDAL_HTTP_TIMEOUT=30, GDAL_HTTP_MAX_RETRY=2):
        for n, row in enumerate(rows):
            sid = row['sample_id']; path = ROOT / 'image_metadata' / (sid + '.json')
            if path.exists():
                continue
            geom = wkt.loads(row['geometry_wkt'])
            lon, lat = Transformer.from_crs(row['geometry_crs'], 4326, always_xy=True).transform(geom.centroid.x, geom.centroid.y)
            choices = sorted([i for i in items if shape(i['geometry']).covers(Point(lon, lat))],
                             key=lambda i: (i['properties']['datetime'], i['id']), reverse=True)
            errors = []
            for item in choices:
                try:
                    if item['id'] not in opened:
                        opened[item['id']] = rasterio.open(planetary_computer.sign(item['assets']['image']['href']))
                    src = opened[item['id']]
                    native = transform(Transformer.from_crs(row['geometry_crs'], src.crs, always_xy=True).transform, geom)
                    pixels = transform(lambda x,y,z=None: (~src.transform) * (x,y), native)
                    close_size, wide_size = view_sizes(pixels.bounds, src.res[0])
                    cx, cy = pixels.centroid.x, pixels.centroid.y
                    left, top = math.floor(cx-wide_size/2), math.floor(cy-wide_size/2)
                    assert left >= 0 and top >= 0 and left+wide_size <= src.width and top+wide_size <= src.height, 'full context unavailable'
                    window = Window(left, top, wide_size, wide_size)
                    values = src.read(indexes=[1,2,3], window=window)
                    assert values.shape == (3,wide_size,wide_size) and values.dtype == np.uint8
                    tr = src.window_transform(window)
                    mask = rasterize([(native,255)], out_shape=(wide_size,wide_size), transform=tr, dtype='uint8')
                    missing = np.all(values == 0, axis=0)
                    assert (mask > 0).sum() > 0 and missing.mean() < .05 and missing[mask > 0].mean() < .01, 'missing target/context pixels'
                    wide = Image.fromarray(np.moveaxis(values,0,2)); wm = Image.fromarray(mask)
                    cl, ct = round(cx-left-close_size/2), round(cy-top-close_size/2)
                    close_box = (cl,ct,cl+close_size,ct+close_size)
                    close, cm = wide.crop(close_box), wm.crop(close_box)
                    assert np.asarray(cm).sum() / mask.sum() > .99, 'target truncated'
                    profile = src.profile.copy()
                    profile.update(width=wide_size,height=wide_size,count=3,transform=tr,compress='deflate')
                    source_path = ROOT / 'source_windows' / (sid + '.tif')
                    with rasterio.open(source_path,'w',**profile) as dst:
                        dst.write(values)
                    info = dict(sample_id=sid, item_id=item['id'], source=item['assets']['image']['href'],
                                imagery_datetime=item['properties']['datetime'], source_gsd=src.res[0],
                                geometry_crs=str(src.crs), context_bounds=list(rasterio.windows.bounds(window,src.transform)),
                                near_m=close_size*src.res[0], wide_m=wide_size*src.res[0], source_limited=False,
                                source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())
                    for name, rgb, m in [('near',close,cm),('wide',wide,wm)]:
                        rgb = rgb.resize((512,512),Image.Resampling.LANCZOS)
                        m = m.resize((512,512),Image.Resampling.NEAREST)
                        _, marked, bbox = mark_target(rgb,m)
                        rgb.save(ROOT/'images'/(sid+'_'+name+'_rgb.jpg'),quality=95)
                        m.save(ROOT/'images'/(sid+'_'+name+'_mask.png'))
                        marked.save(ROOT/'images'/(sid+'_'+name+'_marked.jpg'),quality=95)
                        info[name+'_bbox512'] = bbox
                    save(path, info)
                    break
                except Exception as exc:
                    errors.append(dict(item=item['id'], error=str(exc)[:300]))
            if not path.exists():
                failures.append(dict(sample_id=sid, errors=errors))
            if (n+1)%10 == 0:
                print('IMAGES',city,n+1,'/',len(rows),'failures',len(failures),flush=True)
        for src in opened.values():
            src.close()
    save(ROOT/'logs'/(city+'_image_failures.json'),failures)
    print('IMAGES_COMPLETE',city,len(rows)-len(failures),flush=True)


def freeze():
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform
    from shapely.strtree import STRtree
    assert not (ROOT/'PROTOCOL.json').exists()
    rows, footprints = [], []
    for city in ['nyc','cambridge','seattle']:
        for row in json.loads((ROOT/(city+'_candidates.json')).read_text()):
            path = ROOT/'image_metadata'/(row['sample_id']+'.json')
            if not path.exists():
                continue
            metadata = json.loads(path.read_text())
            row['image_metadata'] = metadata
            for view in ['near','wide']:
                for kind,ext in [('rgb','jpg'),('mask','png'),('marked','jpg')]:
                    row[view+'_'+kind] = 'images/'+row['sample_id']+'_'+view+'_'+kind+'.'+ext
            rows.append(row)
            footprints.append(transform(Transformer.from_crs(metadata['geometry_crs'],32618,always_xy=True).transform,
                                        box(*metadata['context_bounds'])))
    assert len(rows) == len({r['sample_id'] for r in rows})
    tree = STRtree(footprints)
    for i, row in enumerate(rows):
        for j in tree.query(footprints[i],predicate='intersects'):
            assert rows[int(j)]['split'] == row['split'], ('cross-split image overlap',row['sample_id'],rows[int(j)]['sample_id'])
    parcel_splits = defaultdict(set)
    for row in rows:
        parcel_splits[(row['city'],row['parcel'])].add(row['split'])
    assert all(len(s)==1 for s in parcel_splits.values()), 'parcel leaks across splits'
    save(ROOT/'dataset.json',rows)
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [ROOT/'dataset.json',*sorted((ROOT/'images').glob('*'))]}
    save(ROOT/'INPUT_SHA256.json',hashes)
    print('DATASET_READY',len(rows),Counter(r['city']+':'+r['split']+':'+r['stratum'] for r in rows),flush=True)


def self_check():
    assert label('111',6,CB_CODES) == 'residential_multi'
    assert label('0101',1,CB_CODES) is None
    assert label('101',4,CB_CODES) is None
    assert label('942',1,CB_CODES) is None
    assert label('C1',3,NY_CODES) is None
    assert label('D6',12,NY_CODES) is None
    assert label(16,30,SEA_CODES) is None
    assert label(11,None,SEA_CODES) is None
    from shapely.geometry import box
    r = record('cambridge',1,'p',box(0,0,5,5),32619,None,'auxiliary',{},[])
    assert r['reference_use'] is None and r['reference_role']=='auxiliary' and r['split']=='test'
    assert record('nyc',1,'p',box(0,0,5,5),32618,None,'auxiliary',{},[]) is None
    print('PASS source taxonomy, unit gates, sparse role labels, held-out city and spatial boundary')


if __name__ == '__main__':
    main()
