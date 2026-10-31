"""Recover genuine pre-fire context for the unchanged LA regression cohort."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import Affine
from shapely import wkt
from shapely.geometry import box, shape

from reliable_transfer_common import save
from vlm_multiclass_data import view_sizes
from vlm_pilot import STAGE, mark_target

ROOT = STAGE / 'la_transfer_20261027'
OLD = STAGE / 'la_eval_20261023'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['recover', 'images', 'freeze', 'self-check'])
    p.add_argument('--cell')
    a = p.parse_args()
    if a.action == 'self-check':
        self_check(); return
    assert not (ROOT / 'PROTOCOL.json').exists(), 'Inputs are already frozen'
    for name in ['source_cells', 'alignment', 'images', 'image_metadata']:
        (ROOT / name).mkdir(exist_ok=True)
    rows = cohort()
    if a.action == 'recover':
        groups = defaultdict(list)
        for r in rows: groups[r['cell_id']].append(r)
        items = json.loads((ROOT / 'maxar_prefire_items.json').read_text())
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                          CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif', GDAL_HTTP_TIMEOUT=40,
                          GDAL_HTTP_MAX_RETRY=2, GDAL_CACHEMAX=256 * 1024**2):
            for n, (cell, group) in enumerate(sorted(groups.items())):
                if a.cell and cell != a.cell: continue
                recover(cell, group, items)
                print('RECOVER', n + 1, '/', len(groups), cell, flush=True)
    elif a.action == 'images':
        images(rows)
    else:
        freeze(rows)


def cohort():
    refs = [r for r in csv.DictReader((STAGE / 'la_explore_20261023/data/la_reference.csv').open())
            if r['primary_single_outline_iou50'] == 'True']
    products = {(r['cell_id'], r['bldg_uid']): r for r in
                csv.DictReader((OLD / 'inputs/building_damage_all_cells.csv').open())}
    shared = {}
    for cell in sorted({r['cell_id'] for r in refs}):
        shared[cell] = {r['bldg_uid']: r for r in csv.DictReader(
            (OLD / 'inputs' / cell / 'shared_base/shared_instance_samples.csv').open())}
    result = []
    for r in sorted(refs, key=lambda r: r['building_uid']):
        cell, bid = json.loads(r['building_uid']); product = products[(cell, bid)]
        s = shared[cell][bid]; g = wkt.loads(product['polygon_wkt_utm'])
        pixels = wkt.loads(s['polygon_wkt_xy_pre']); gsd = 600 / int(s['width'])
        origin = cell_origin(np.array(pixels.exterior.coords), np.array(g.exterior.coords), gsd)
        near, wide = view_sizes(g.bounds, 1)
        result.append(dict(sample_id='la_' + cell + '_' + bid, building_uid=r['building_uid'],
            city='la', cell_id=cell, parcel=r['AIN'], split='test',
            spatial_group=cell, reference_use=r['assessor_type'], reference_role=None,
            reference_grade='frozen_assessor_parcel_reference_not_building_gold',
            strict_single_assessor_building=r['strict_single_assessor_building'] == 'True',
            geometry_wkt=g.wkt, geometry_crs='EPSG:32611', footprint_m2=g.area,
            near_m=near, wide_m=wide, origin=origin.tolist(), original_gsd=gsd,
            original_crop=str(OLD / 'inputs' / cell / 'shared_base/crops_pre' / Path(s['pre_crop']).name),
            crop_x0=int(s['crop_x0']), crop_y0=int(s['crop_y0'])))
    assert len(result) == len({r['sample_id'] for r in result}) == 3332
    assert Counter(r['reference_use'] for r in result) == dict(
        residential_small=3276, residential_multi=14, commercial=42)
    return result


def cell_origin(pixels, coordinates, gsd):
    assert pixels.shape == coordinates.shape
    offsets = coordinates - pixels * [gsd, -gsd]
    origin = np.round(np.median(offsets, axis=0) / 600) * 600
    assert np.max(np.abs(offsets - origin)) < .002, 'Inconsistent original grid georeference'
    return origin


def image_agreement(original, recovered):
    valid = (original.max(axis=0) > 0) & (recovered.max(axis=0) > 0)
    # Original LA crops can contain a source-image edge; compare observed pixels.
    assert valid.mean() > .5, 'Too few observed pixels to establish scene identity'
    x, y = original[:, valid].astype(float), recovered[:, valid].astype(float)
    side = original.shape[-1] // 4
    # Average interpolation differences over 4 pixels (about 1.2m), not objects.
    small = [np.asarray(Image.fromarray(v.transpose(1, 2, 0)).resize((side, side), Image.Resampling.BOX))
             for v in [original, recovered]]
    common = np.asarray(Image.fromarray(valid.astype('uint8') * 255).resize((side, side), Image.Resampling.BOX)) == 255
    coarse = float(np.corrcoef(small[0][common].ravel(), small[1][common].ravel())[0, 1])
    return dict(mae=float(np.abs(x - y).mean()), correlation=float(np.corrcoef(x.ravel(), y.ravel())[0, 1]),
                correlation_4px=coarse, compared_fraction=float(valid.mean()))


def recover(cell, rows, items):
    receipt = ROOT / 'alignment' / (cell + '.json')
    if receipt.exists():
        assert json.loads(receipt.read_text())['passed'], cell
        return
    gsd = rows[0]['original_gsd']; origin = rows[0]['origin']
    assert all(r['origin'] == origin and r['original_gsd'] == gsd for r in rows)
    transform = Affine(gsd, 0, origin[0], 0, -gsd, origin[1])
    geometries = [wkt.loads(r['geometry_wkt']) for r in rows]
    extents = [(g.centroid.x-r['wide_m']/2-2, g.centroid.y-r['wide_m']/2-2,
                g.centroid.x+r['wide_m']/2+2, g.centroid.y+r['wide_m']/2+2)
               for r, g in zip(rows, geometries)]
    # Align the expanded window to the original grid for direct crop comparison.
    left = min(0, math.floor((min(b[0] for b in extents) - origin[0]) / gsd))
    top = min(0, math.floor((origin[1] - max(b[3] for b in extents)) / gsd))
    right = max(1966, math.ceil((max(b[2] for b in extents) - origin[0]) / gsd))
    bottom = max(1966, math.ceil((origin[1] - min(b[1] for b in extents)) / gsd))
    tr = transform * Affine.translation(left, top)
    bounds = rasterio.transform.array_bounds(bottom-top, right-left, tr)
    probe_rows = sorted(rows, key=lambda r: hashlib.sha256(r['sample_id'].encode()).hexdigest())[:3]
    by_scene = defaultdict(list)
    for item in items:
        assert item['properties']['datetime'][:10] < '2025-01-07'
        if shape(item['properties']['proj:geometry']).intersects(box(*bounds)):
            by_scene[item['properties']['catalog_id']].append(item)
    trials, best = [], None
    for scene, source_items in sorted(by_scene.items()):
        sources = []
        try:
            for item in sorted(source_items, key=lambda i: i['id']):
                sources.append(rasterio.open(item['assets']['visual']['href']))
            assert all(s.crs.to_epsg() == 32611 and s.dtypes[0] == 'uint8' for s in sources)
            values, actual_tr = merge(sources, bounds=bounds, res=gsd, indexes=[1, 2, 3],
                                      nodata=0, resampling=rasterio.enums.Resampling.bilinear)
            assert actual_tr.almost_equals(tr) and values.shape[1:] == (bottom-top, right-left)
            agreements = []
            for r in probe_rows:
                x, y = r['crop_x0']-left, r['crop_y0']-top
                old = np.asarray(Image.open(r['original_crop']).convert('RGB')).transpose(2, 0, 1)
                agreements.append(dict(sample_id=r['sample_id'], **image_agreement(old, values[:, y:y+256, x:x+256])))
            quality = float(np.mean([max(q['correlation'], q['correlation_4px']) for q in agreements]))
            trial = dict(scene=scene, quality=quality, probes=agreements,
                         items=[i['id'] for i in source_items],
                         dates=sorted({i['properties']['datetime'] for i in source_items}),
                         sources=[i['assets']['visual']['href'] for i in source_items])
            trials.append(trial)
            if best is None or quality > best[0]: best = (quality, values, trial)
        except (AssertionError, rasterio.errors.RasterioError) as exc:
            trials.append(dict(scene=scene, error=str(exc)))
        finally:
            for s in sources: s.close()
    passed = best is not None and min(max(q['correlation'], q['correlation_4px']) for q in best[2]['probes']) >= .97
    save(receipt, dict(cell_id=cell, passed=passed, trials=trials,
                       selected=best[2] if best else None, bounds=list(bounds), transform=list(tr)))
    assert passed, ('Scene alignment needs review', cell, trials)
    with rasterio.open(ROOT / 'source_cells' / (cell + '.tif'), 'w', driver='GTiff',
            width=right-left, height=bottom-top, count=3, dtype='uint8', crs='EPSG:32611',
            transform=tr, compress='deflate', tiled=True, nodata=0) as out:
        out.write(best[1])


def images(rows):
    failures = []; opened = {}
    for n, row in enumerate(rows):
        sid = row['sample_id']; path = ROOT / 'image_metadata' / (sid + '.json')
        if path.exists(): continue
        cell = row['cell_id']
        if cell not in opened: opened[cell] = rasterio.open(ROOT / 'source_cells' / (cell + '.tif'))
        src = opened[cell]; geom = wkt.loads(row['geometry_wkt'])
        try:
            metadata = dict(sample_id=sid, source_cell=cell, gsd=src.res[0], views={})
            for view in ['near', 'wide']:
                side = math.ceil(row[view + '_m'] / src.res[0]); cx, cy = (~src.transform) * (geom.centroid.x, geom.centroid.y)
                window = rasterio.windows.Window(math.floor(cx-side/2), math.floor(cy-side/2), side, side)
                assert window.col_off >= 0 and window.row_off >= 0 and window.col_off+side <= src.width and window.row_off+side <= src.height
                values = src.read([1, 2, 3], window=window); tr = src.window_transform(window)
                mask = rasterize([(geom, 255)], out_shape=(side, side), transform=tr, dtype='uint8')
                missing = np.all(values == 0, axis=0)
                assert mask.any() and missing.mean() < .05 and missing[mask > 0].mean() < .01, 'Missing target or context'
                assert abs((mask > 0).sum() * src.res[0]**2 / geom.area - 1) < .08, 'Truncated or misregistered target'
                rgb = Image.fromarray(values.transpose(1, 2, 0)).resize((512, 512), Image.Resampling.LANCZOS)
                m = Image.fromarray(mask).resize((512, 512), Image.Resampling.NEAREST)
                _, marked, bbox = mark_target(rgb, m)
                rgb.save(ROOT / 'images' / (sid + '_' + view + '_rgb.jpg'), quality=95)
                m.save(ROOT / 'images' / (sid + '_' + view + '_mask.png'))
                marked.save(ROOT / 'images' / (sid + '_' + view + '_marked.jpg'), quality=95)
                metadata['views'][view] = dict(bounds=list(rasterio.windows.bounds(window, src.transform)),
                    extent_m=side*src.res[0], missing_fraction=float(missing.mean()), bbox512=bbox)
            save(path, metadata)
        except AssertionError as exc:
            failures.append(dict(sample_id=sid, error=str(exc)))
        if (n+1) % 100 == 0: print('IMAGES', n+1, '/', len(rows), 'failures', len(failures), flush=True)
    for src in opened.values(): src.close()
    save(ROOT / 'IMAGE_FAILURES.json', failures)


def freeze(rows):
    missing = [r['sample_id'] for r in rows if not (ROOT / 'image_metadata' / (r['sample_id'] + '.json')).exists()]
    failures = {r['sample_id']: r['error'] for r in json.loads((ROOT / 'IMAGE_FAILURES.json').read_text())}
    assert set(missing) == set(failures), 'Every unavailable input needs an explicit failure receipt'
    for r in rows:
        r['input_available'] = r['sample_id'] not in failures
        if not r['input_available']:
            r['input_failure'] = failures[r['sample_id']]
            continue
        for view in ['near', 'wide']:
            for kind, ext in [('rgb', 'jpg'), ('mask', 'png'), ('marked', 'jpg')]:
                r[view+'_'+kind] = 'images/' + r['sample_id'] + '_' + view + '_' + kind + '.' + ext
    save(ROOT / 'dataset.json', rows)
    paths = [ROOT / 'dataset.json'] + [ROOT/r[v+'_'+k] for r in rows if r['input_available']
                                      for v in ['near','wide'] for k in ['rgb','mask','marked']]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    save(ROOT / 'INPUT_SHA256.json', hashes)
    print('LA_INPUTS_READY', len(rows), 'available', len(rows)-len(missing), 'files', len(hashes), flush=True)


def self_check():
    pixels = np.array([[0, 0], [12, 7], [7, 15]])
    coords = np.round(pixels * [.3051881993896236, -.3051881993896236] + [348600, 3772800], 3)
    assert np.array_equal(cell_origin(pixels, coords, .3051881993896236), [348600, 3772800])
    try: cell_origin(pixels, coords + [3, 0], .3051881993896236)
    except AssertionError: pass
    else: raise AssertionError('A displaced grid must fail')
    values = np.arange(3*16*16, dtype=np.uint8).reshape(3, 16, 16) + 1
    assert image_agreement(values, values)['mae'] == 0
    assert image_agreement(values, values)['correlation'] > .999999
    assert image_agreement(values, values[:, :, ::-1])['correlation'] < .999
    assert len(cohort()) == 3332
    print('PASS original grid reconstruction, image alignment, unchanged cohort')


if __name__ == '__main__': main()
