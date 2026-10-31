"""Review the fixed best-CNN multifamily errors without changing test labels."""

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely import wkt
from shapely.affinity import translate
from shapely.geometry import shape
from shapely.ops import transform

from prepare_la_adaptation import FOOTPRINT, MODERN, UTM
from run_la import read_csv, sha256, write_csv
from vlm_pilot import STAGE, CLASSES, mark_target

OUT = Path(__file__).resolve().parent / 'analysis/20261030_multifamily_audit'
TRANSFER = STAGE / 'la_transfer_20261027'
ADAPT = STAGE / 'la_adaptation_20261029'
FULL = OUT.parent / '20261030_la_full'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'augmentation-check', 'report'])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.action == 'report':
        report()
        return
    if args.action == 'augmentation-check':
        augmentation_check()
        return
    source_paths = [FULL/'PREDICTION_COMPARISON.csv', TRANSFER/'dataset.json',
                    ADAPT/'DATASET.json', ADAPT/'PROTOCOL.json',
                    STAGE/'la_eval_20261023/reference/la_assessor_matched.csv',
                    OUT.parent/'20261029_la_adaptation/PREDICTIONS.json',
                    OUT.parent/'20261029_la_fourplex/PREDICTIONS.json']
    hashes = {str(p): sha256(p) for p in source_paths}
    comparison = read_csv(source_paths[0])
    refs = {r['sample_id']: r for r in json.loads(source_paths[1].read_text())}
    data = json.loads(source_paths[2].read_text())
    test = {r['sample_id']: r for r in data if r['split'] == 'test'}
    protocol = json.loads(source_paths[3].read_text())
    assert hashes[str(source_paths[1])] == protocol['old_test_dataset_sha256']
    assert hashes[str(source_paths[2])] == protocol['dataset_sha256']
    assert len(test) == len(refs) == 3332 and set(test) == set(refs)
    assessor = {'la_'+r['cell_id']+'_'+r['bldg_uid']: r for r in read_csv(source_paths[4])
                if r['primary_single_outline_iou50'] == 'True'}
    selected = {'FN': [], 'FP': [], 'TP': []}
    for r in comparison:
        if not r['original_test_reference']:
            continue
        assert r['original_test_reference'] == refs[r['sample_id']]['reference_use']
        actual, pred = r['original_test_reference'] == CLASSES[1], r['best_cnn_type'] == CLASSES[1]
        group = 'TP' if actual and pred else 'FN' if actual else 'FP' if pred else None
        if group:
            selected[group].append(r)
    assert {k: len(v) for k, v in selected.items()} == {'FN': 11, 'FP': 27, 'TP': 3}
    review_ids = [r['sample_id'] for rr in selected.values() for r in rr]
    features, historical, modern = countywide(sorted({refs[s]['parcel'] for s in review_ids}),
        sorted({assessor[s]['lariac_bld_id'].removesuffix('.0') for s in review_ids}))
    for name in ['countywide_buildings', 'historical_targets', 'modern_targets']:
        hashes[str(OUT/'raw'/f'{name}.json')] = sha256(OUT/'raw'/f'{name}.json')
    prior = json.loads(source_paths[5].read_text())
    prior['cnn_fourplex_ensemble'] = json.loads(source_paths[6].read_text())['cnn_fourplex_ensemble']
    predictions = {k: {r['sample_id']: r for r in prior[k]} for k in
                   ['cnn_lowrise_s42', 'cnn_lowrise_s43', 'dino_ensemble', 'cnn_fourplex_ensemble']}
    (OUT/'cards').mkdir(exist_ok=True)
    (OUT/'views').mkdir(exist_ok=True)
    cases = []
    for group, rr in selected.items():
        for i, row in enumerate(sorted(rr, key=lambda r: r['sample_id']), 1):
            sid = row['sample_id']; ref = refs[sid]; a = assessor[sid]
            case = dict(case_id=f'{group}{i:02}', group=group, sample_id=sid,
                        reference=row['original_test_reference'], prediction=row['best_cnn_type'],
                        old_prediction=row['old_n2_type'], parcel=ref['parcel'], cell_id=ref['cell_id'],
                        probabilities=[float(row['new_prob_'+c]) for c in CLASSES],
                        comparisons={k: predictions[k][sid] for k in predictions},
                        assessor={k: a[k] for k in ['UseCode', 'Units', 'UseCodeDescChar2', 'UseCodeDescChar3',
                            'UseCodeDescChar4', 'totBuildingDataLines', 'RollYear', 'SQFTmain', 'YearBuilt',
                            'lariac_bld_id', 'parcel_outline_count', 'coregistered_iou', 'shift_dx_m', 'shift_dy_m', 'incident']},
                        original_paired_views_available=ref['input_available'],
                        original_view_failure=ref.get('input_failure'), footprint_m2=ref['footprint_m2'])
            case['countywide'] = footprint_evidence(features, historical, modern, ref, a)
            case['images'] = image_evidence(case, test[sid], ref, hashes)
            cases.append(case)
            render_card(case)
    for group in selected:
        rr = [r for r in cases if r['group'] == group]
        for start in range(0, len(rr), 3):
            page = Image.new('RGB', (1200, 410*len(rr[start:start+3])), 'white')
            for j, c in enumerate(rr[start:start+3]):
                with Image.open(OUT/'cards'/f"{c['case_id']}.jpg") as im: page.paste(im, (0, j*410))
            page.save(OUT/'cards'/f'{group}_page_{start//3+1:02}.jpg', quality=95)
    for p, digest in hashes.items():
        assert sha256(p) == digest, p
    save(OUT/'EVIDENCE.json', cases)
    save(OUT/'SOURCE_SHA256.json', hashes)
    print(json.dumps(dict(prepared=Counter(r['group'] for r in cases), source_files=len(hashes)), indent=2))


def countywide(parcels, ids):
    folder = OUT/'raw'; folder.mkdir(exist_ok=True)
    result = []
    for i in range(0, len(parcels), 25):
        params = dict(f='geojson', where='AIN IN ('+','.join("'"+x+"'" for x in parcels[i:i+25])+')',
                      outFields='OBJECTID,BLD_ID,AIN,HEIGHT,AREA,DATE_,STATUS', returnGeometry='true', outSR=4326)
        result.extend(fetch(f'countywide_{i//25}', FOOTPRINT+'/query', params)['features'])
    save(folder/'countywide_buildings.json', result)
    params = dict(f='geojson', where='BLD_ID IN ('+','.join("'"+x+"'" for x in ids)+')',
                  outFields='OBJECTID,BLD_ID,HEIGHT,AREA,DATE_,STATUS', returnGeometry='true', outSR=4326)
    modern = fetch('modern_targets', MODERN+'/query', params)['features']
    historical = fetch('historical_targets', FOOTPRINT+'/query', dict(params, outFields=params['outFields']+',AIN'))['features']
    return result, historical, modern


def fetch(name, url, params):
    from audit_sources import get_json
    path = OUT/'raw'/f'{name}.json'
    if path.exists():
        receipt = json.loads(path.with_suffix('.provenance.json').read_text())
        assert receipt['url'] == url and receipt['params'] == params and receipt['sha256'] == sha256(path)
        d = json.loads(path.read_text())
    else:
        d = get_json(url, params)
        save(path, d)
        save(path.with_suffix('.provenance.json'), dict(url=url, params=params,
             retrieved_utc=datetime.now(timezone.utc).isoformat(), sha256=sha256(path)))
    assert d['type'] == 'FeatureCollection' and not d.get('exceededTransferLimit')
    return d


def footprint_evidence(features, historical, modern, ref, a):
    ff = [f for f in features if f['properties']['AIN'] == ref['parcel']]
    target = translate(wkt.loads(ref['geometry_wkt']), -float(a['shift_dx_m']), -float(a['shift_dy_m']))
    outlines = []
    for f in ff:
        g = transform(UTM, shape(f['geometry'])); prop = f['properties']
        outlines.append(dict(building_id=str(prop['BLD_ID']), area_m2=g.area,
            matches_original_id=str(prop['BLD_ID']) == a['lariac_bld_id'].removesuffix('.0'),
            target_iou=target.intersection(g).area/target.union(g).area,
            geometry_wkt=g.wkt, height_feet=prop['HEIGHT']))
    outlines.sort(key=lambda r: -r['area_m2'])
    matches = [r for r in outlines if r['matches_original_id']]
    bid = a['lariac_bld_id'].removesuffix('.0')
    older = [f for f in historical if str(f['properties']['BLD_ID']) == bid]
    newer = [f for f in modern if str(f['properties']['BLD_ID']) == bid]
    modern_matches = []
    for f in newer:
        g = transform(UTM, shape(f['geometry']))
        modern_matches.append(dict(properties=f['properties'], geometry_wkt=g.wkt,
            target_iou=target.intersection(g).area/target.union(g).area,
            target_covers_reference_fraction=target.intersection(g).area/g.area))
    return dict(count=len(outlines), target_id_matches=len(matches),
                target_area_rank=next((i+1 for i, r in enumerate(outlines) if r['matches_original_id']), None),
                outlines=outlines, historical_target_parcels=[f['properties']['AIN'] for f in older],
                modern_target_matches=modern_matches,
                note='Countywide historical footprints; catalog year and IDs can differ from frozen references; rank does not determine actual use.')


def image_evidence(case, row, ref, hashes):
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from prepare_la_transfer import image_agreement
    paths = {k: STAGE/row[k].split('/stage3/')[1].removeprefix('stage2a_') for k in ['crop_path', 'mask_path']}
    for p in paths.values(): hashes[str(p)] = sha256(p)
    rgb = Image.open(paths['crop_path']).convert('RGB'); mask = Image.open(paths['mask_path']).convert('L')
    m = np.array(mask)>0; x = np.array(rgb); zeros = np.all(x == 0, axis=-1)
    rgb224 = rgb.resize((224,224), Image.Resampling.BILINEAR)
    mask224 = mask.resize((224,224), Image.Resampling.NEAREST)
    _, overlay, _ = mark_target(rgb224, mask224)
    for name, im in [('cnn_rgb', rgb224), ('cnn_mask', overlay)]: im.save(OUT/'views'/f"{case['case_id']}_{name}.png")
    w = (np.array(mask224)>0).reshape(7,32,7,32).mean(axis=(1,3))
    result = dict(crop_path=str(paths['crop_path']), mask_path=str(paths['mask_path']),
                  target_zero_fraction=float(zeros[m].mean()), whole_crop_zero_fraction=float(zeros.mean()),
                  target_retained_area_ratio=float(m.sum()*ref['original_gsd']**2/ref['footprint_m2']),
                  target_touches_crop_edge=bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any()),
                  feature_mask_area_cells=float(w.sum()), feature_weight_effective_cells=float(w.sum()**2/(w*w).sum()),
                  original_gsd=ref['original_gsd'], cnn_extent_m=rgb.width*ref['original_gsd'], diagnostic_views={})
    source = TRANSFER/'source_cells'/f"{ref['cell_id']}.tif"
    if str(source) not in hashes: hashes[str(source)] = sha256(source)
    geom = wkt.loads(ref['geometry_wkt']); bounds = geom.bounds
    near = max(40., 1.3*max(bounds[2]-bounds[0], bounds[3]-bounds[1]))
    with rasterio.open(source) as src:
        gsd = ref['original_gsd']
        original_tr = Affine(gsd,0,ref['origin'][0]+ref['crop_x0']*gsd,0,-gsd,ref['origin'][1]-ref['crop_y0']*gsd)
        expected_mask = rasterize([(geom,1)], out_shape=(256,256), transform=original_tr)>0
        result['mask_geometry_iou'] = float((m & expected_mask).sum()/(m | expected_mask).sum())
        original_window = rasterio.windows.from_bounds(original_tr.c, original_tr.f-256*gsd,
            original_tr.c+256*gsd, original_tr.f, src.transform)
        recovered = src.read([1,2,3], window=original_window, boundless=True, out_shape=(3,256,256))
        result['original_rgb_source_agreement'] = image_agreement(x.transpose(2,0,1), recovered)
        for name, extent in [('near', near), ('wide', 300.)]:
            side = math.ceil(extent/src.res[0]); cx, cy = (~src.transform)*(geom.centroid.x, geom.centroid.y)
            window = rasterio.windows.Window(math.floor(cx-side/2), math.floor(cy-side/2), side, side)
            values = src.read([1,2,3], window=window, boundless=True, fill_value=0)
            tr = src.window_transform(window)
            target = rasterize([(geom,255)], out_shape=(side,side), transform=tr, dtype='uint8')
            missing = np.all(values == 0, axis=0)
            im = Image.fromarray(values.transpose(1,2,0)).resize((512,512), Image.Resampling.LANCZOS)
            mark = Image.fromarray(target).resize((512,512), Image.Resampling.NEAREST)
            _, marked, _ = mark_target(im, mark)
            marked.save(OUT/'views'/f"{case['case_id']}_{name}.jpg", quality=95)
            result['diagnostic_views'][name] = dict(extent_m=side*src.res[0], missing_fraction=float(missing.mean()),
                target_missing_fraction=float(missing[target>0].mean()), target_area_ratio=float((target>0).sum()*src.res[0]**2/geom.area))
    return result


def render_card(case):
    page = Image.new('RGB', (1200,410), 'white'); draw = ImageDraw.Draw(page)
    font = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 13)
    a = case['assessor']; p = case['probabilities']; im = case['images']
    text = (f"{case['case_id']}  {case['sample_id']}\n"
            f"ref={case['reference']} pred={case['prediction']}  p(S/M/C)={p[0]:.4f}/{p[1]:.4f}/{p[2]:.4f}\n"
            f"units={a['Units']} code={a['UseCode']} area={case['footprint_m2']:.0f}m2  county roofs={case['countywide']['count']} "
            f"rank={case['countywide']['target_area_rank']}  7x7 area={im['feature_mask_area_cells']:.2f} "
            f"target missing={im['target_zero_fraction']:.1%}")
    draw.multiline_text((5,3), text, font=font, fill='black', spacing=3)
    for k, (key, label) in enumerate([('cnn_rgb','Actual CNN RGB 224'), ('cnn_mask','Actual CNN target mask'),
                                    ('near','Diagnostic near, adaptive extent'), ('wide','Diagnostic context, 300m')]):
        draw.text((k*300+3,77), label, font=font, fill='black')
        ext = 'png' if key.startswith('cnn') else 'jpg'
        with Image.open(OUT/'views'/f"{case['case_id']}_{key}.{ext}") as img:
            page.paste(img.resize((300,300)), (k*300,100))
    page.save(OUT/'cards'/f"{case['case_id']}.jpg", quality=95)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def augmentation_check():
    import hashlib
    import importlib.util
    import sys
    path = ADAPT/'execution/common.py'
    spec = importlib.util.spec_from_file_location('common', path)
    common = importlib.util.module_from_spec(spec); sys.modules['common'] = common; spec.loader.exec_module(common)
    rows = [r for r in json.loads((ADAPT/'DATASET.json').read_text()) if r['split'] == 'train']
    for r in rows:
        for key in ['crop_path', 'mask_path']:
            r[key] = str(STAGE/r[key].split('/stage3/')[1].removeprefix('stage2a_'))
    candidates = {r['sample_id']: r for r in json.loads((ADAPT/'STABLE_CANDIDATES.json').read_text())}
    indices = [i for i, r in enumerate(rows) if r['domain'] == 'la' and r['type_class'] == 1
               and candidates[r['sample_id']]['evidence']['Units'] == 4]
    assert len(indices) == 4
    cfg = json.loads((ADAPT/'PROTOCOL.json').read_text())['cnn_config']
    dataset = common.Stage2aDataset(rows, img_size=224, train=True, input_mode=cfg['input_mode'],
        geometry_cols=cfg['geometry_cols'].split(','), require_ring_mask=False, require_targets=False,
        aug_hflip=.5, aug_rot90=.25)
    cases = []
    for i in indices:
        hashes = [hashlib.sha256(dataset[i]['x'].numpy().tobytes()).hexdigest() for _ in range(20)]
        cases.append(dict(sample_id=rows[i]['sample_id'], draws=20, distinct_input_tensors=len(set(hashes))))
    save(OUT/'AUGMENTATION_CHECK.json', dict(common_sha256=sha256(path), dataset_sha256=sha256(ADAPT/'DATASET.json'),
        mode='Actual frozen Stage2aDataset with current trainer augmentation arguments',
        dataset_seed=dataset.seed, domain_augmentation=dataset.domain_augmentation,
        fourplex_cases=cases, note='No trainer or training input modified.'))
    print(json.dumps(cases, indent=2))


def report():
    from html import escape
    from analyze_la_transfer import measure
    cases = json.loads((OUT/'EVIDENCE.json').read_text())
    notes = json.loads((OUT/'REVIEW_NOTES.json').read_text())
    assert notes['no_new_adjudicated_ground_truth']
    assert {r['case_id'] for r in cases} == set(notes['cases']) and len(cases) == 41
    hashes = json.loads((OUT/'SOURCE_SHA256.json').read_text())
    for path, expected in hashes.items(): assert sha256(path) == expected, path
    rows = json.loads((TRANSFER/'dataset.json').read_text())
    reference = {r['sample_id']: r['reference_use'] for r in rows}
    exported = {r['sample_id']: r for r in read_csv(FULL/'PREDICTION_COMPARISON.csv') if r['original_test_reference']}
    expected_errors = {sid for sid, r in exported.items()
        if (r['original_test_reference'] == CLASSES[1]) != (r['best_cnn_type'] == CLASSES[1])}
    assert {r['sample_id'] for r in cases if r['group'] != 'TP'} == expected_errors
    for c in cases:
        assert c['reference'] == reference[c['sample_id']] == exported[c['sample_id']]['original_test_reference']
        c['review'] = notes['cases'][c['case_id']]
    summary = {}
    for group in ['FN', 'FP', 'TP']:
        rr = [r for r in cases if r['group'] == group]
        summary[group] = dict(n=len(rr), cells=dict(Counter(r['cell_id'] for r in rr)),
            incidents=dict(Counter(r['assessor']['incident'] for r in rr)),
            units=dict(Counter(r['assessor']['Units'] for r in rr)),
            area_m2_min_median_max=np.quantile([r['footprint_m2'] for r in rr], [0,.5,1]).tolist(),
            feature_mask_area_cells_min_median_max=np.quantile([r['images']['feature_mask_area_cells'] for r in rr], [0,.5,1]).tolist(),
            high_confidence_wrong_or_correct=sum(max(r['probabilities'])>.9 for r in rr),
            seed_disagreement=sum(r['comparisons']['cnn_lowrise_s42']['use'] != r['comparisons']['cnn_lowrise_s43']['use'] for r in rr),
            dino_reference_correct=sum(r['comparisons']['dino_ensemble']['use'] == r['reference'] for r in rr),
            source_mask_iou_min=min(r['images']['mask_geometry_iou'] for r in rr),
            source_rgb_correlation_4px_min=min(r['images']['original_rgb_source_agreement']['correlation_4px'] for r in rr),
            target_has_missing_pixels=sum(r['images']['target_zero_fraction']>0 for r in rr),
            target_at_crop_edge=[r['case_id'] for r in rr if r['images']['target_touches_crop_edge']],
            original_context_unavailable=[r['case_id'] for r in rr if not r['original_paired_views_available']])
    training = json.loads((ADAPT/'DATASET.json').read_text())
    candidate = {r['sample_id']: r for r in json.loads((ADAPT/'STABLE_CANDIDATES.json').read_text())}
    coverage = {}
    for split in ['train', 'val']:
        for label in [0,1]:
            rr = [r for r in training if r['split'] == split and r['domain'] == 'la' and r['type_class'] == label]
            units = [candidate[r['sample_id']]['evidence']['Units'] for r in rr]
            coverage[f'{split}:{CLASSES[label]}'] = dict(n=len(rr), units=dict(Counter(map(str, units))),
                four_units=sum(u==4 for u in units), spatial_groups=len({r['spatial_group'] for r in rr}),
                area_m2_min_median_max=np.quantile([r['footprint_m2'] for r in rr],[0,.5,1]).tolist(),
                area_ge400=sum(r['footprint_m2']>=400 for r in rr),
                note='LA adaptation portion only; source-domain unit counts are not established here.')
    later = json.loads((STAGE/'la_fourplex_20261029/DATASET.json').read_text())
    original_ids = {r['sample_id'] for r in training}
    added = [r for r in later if r['sample_id'] not in original_ids]
    coverage['subsequent_fourplex_supplement'] = dict(n=len(added), spatial_groups=dict(Counter(r['spatial_group'] for r in added)))
    prior = json.loads((OUT.parent/'20261029_la_adaptation/PREDICTIONS.json').read_text())
    prior['cnn_fourplex_ensemble'] = json.loads((OUT.parent/'20261029_la_fourplex/PREDICTIONS.json').read_text())['cnn_fourplex_ensemble']
    scores = {name: measure(rows, prior[name]) for name in ['CNN_old','cnn_lowrise_ensemble','dino_ensemble','cnn_fourplex_ensemble']}
    assert abs(scores['cnn_lowrise_ensemble']['per_class'][CLASSES[1]]['precision']-.1)<1e-12
    dino_fp = {r['sample_id'] for r,p in zip(rows,prior['dino_ensemble']) if r['reference_use'] != CLASSES[1] and p['use'] == CLASSES[1]}
    current_fp = {r['sample_id'] for r in cases if r['group']=='FP'}
    overlap = dict(dino_current_fp_corrected=len(current_fp-dino_fp), dino_current_fp_retained=len(current_fp&dino_fp),
                   dino_other_fp=len(dino_fp-current_fp), dino_total_fp=len(dino_fp))
    priority = {tag: [r['case_id'] for r in cases if tag in r['review']['tags']]
                for tag in ['label_priority','geometry_review','target_at_edge']}
    augmentation = json.loads((OUT/'AUGMENTATION_CHECK.json').read_text())
    result = dict(groups=summary, training_coverage=coverage, existing_model_scores=scores,
                  dino_overlap=overlap, review_priorities=priority, augmentation=augmentation,
                  no_new_adjudicated_ground_truth=True, unchanged_test_labels=True)
    save(OUT/'SUMMARY.json', result)
    fields = ['case_id','group','sample_id','reference','prediction','parcel','units','p_multi','footprint_m2',
              'label_review','image_review','observation','hypothesis','tags','card']
    table = [dict(case_id=c['case_id'],group=c['group'],sample_id=c['sample_id'],reference=c['reference'],prediction=c['prediction'],
                 parcel=c['parcel'],units=c['assessor']['Units'],p_multi=c['probabilities'][1],footprint_m2=c['footprint_m2'],
                 **{k:c['review'][k] for k in ['label_review','image_review','observation','hypothesis']},
                 tags=';'.join(c['review']['tags']),card=f"cards/{c['case_id']}.jpg") for c in cases]
    write_csv(OUT/'CASE_REVIEW.csv', fields, table)
    html = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>LA多户逐栋复核</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#17202a}nav{position:sticky;top:0;background:white;padding:12px 0;border-bottom:1px solid #bbb}nav a{margin-right:24px}article{margin:40px 0}img{width:100%;height:auto}p{line-height:1.6}h2{margin-bottom:8px}</style>',
        '<h1>11栋漏判 · 27条误报 · 3栋正确对照</h1><p>当前最佳低层重采样CNN。原测试标签不变；图像观察和工作假设不构成新真值。</p>',
        '<nav><a href="#FN">漏判11</a><a href="#FP">误报27</a><a href="#TP">正确对照3</a><a href="REPORT.md">分析报告</a></nav>']
    for group in ['FN','FP','TP']:
        html.append(f'<h2 id="{group}">{group}</h2>')
        for c in [r for r in cases if r['group']==group]:
            html.append(f'<article id="{c["case_id"]}"><h2>{c["case_id"]} · {escape(c["sample_id"])}</h2><img loading="lazy" src="cards/{c["case_id"]}.jpg" alt="{c["case_id"]}四视图">')
            for key,label in [('observation','观察'),('label_review','标签'),('image_review','图像'),('hypothesis','工作判断')]:
                html.append(f'<p><b>{label}：</b>{escape(c["review"][key])}</p>')
            html.append('</article>')
    (OUT/'REVIEW.html').write_text('\n'.join(html)+ '\n</html>\n')
    fn,fp,tp = summary['FN'],summary['FP'],summary['TP']
    lines = ['# 当前最佳CNN的LA多户错误逐栋复核','',
        '2026-10-30。已逐图复核固定3,332条参考测试集中的全部11栋多户漏判和27条多户误报，并查看3栋正确多户作对照。模型是cnn_lowrise_s42/s43概率平均，与此前四户补样模型的10漏判/35误报不是同一清单。',
        '', '**判断：主要证据指向训练形态覆盖不足与模型判别偏差；少数标签归属和几何输入值得单独核实。尚不能将观察性关联当作已确定的因果机制。没有改动原测试标签、模型或推理结果。**','',
        '[逐栋四视图复核页](REVIEW.html) · [38条错误和3条对照明细](CASE_REVIEW.csv) · [原始证据及坐标/影像检查](EVIDENCE.json)','',
        '## 1. 标签：有3个优先疑点，尚未确认新的真值','',
        '11个漏判由9个四户、1个五户、1个八户组成；27个误报由26个1户SFR和1个Office组成。小型住宅的既有定义是1–3户，多户是4户及以上；屋顶形态本身不能可靠显示这个阈值。',
        '', '- **FN01**：原4户来自2条assessor建筑记录，同ID在旧轮廓目录中对应另一个历史AIN。需要把地块户数分配到实际楼栋；目前既不能确认该栋4户，也不能直接改成小型住宅。',
        '- **FP16**：PUD记录为1户，需确认产权/单元记录与整栋轮廓的对应关系，同时目标触裁剪边缘。',
        '- **FP21**：Office记录的主楼面积只有1,730平方英尺，而目标屋顶约1,317平方米，可能存在子单元或地块记录与整栋建筑粒度不一致。商业标签并未因此被证明错误。',
        '', '原parcel_outline_count来自评估范围内的LARIAC/DINS子集，不是全县完整地块建筑数。本次重新查询全县历史轮廓，FN06/FN09/FN10以及FP12/FP13所在的地块各有2个轮廓，当前目标均为较大一个，影像与街前主楼解释相容。不能据此把这些目标当作已确认附属建筑，也不能把原单轮廓筛选称为严格楼栋金标准。',
        '', '按原建筑ID查询较新目录，41条中39条有匹配；FP19、FP25的2023编号未返回。目录缺失/年代变化单独记录，不自动等同于标注错误。',
        '', '## 2. 图像：没有系统性空白或错位，少数边界/轮廓需核查','',
        f"所有41条目标mask内均有有效影像；CNN mask与原产品几何重投影IoU最低{min(v['source_mask_iou_min'] for v in summary.values()):.3f}，原裁剪与恢复同源影像的4像素聚合相关最低{min(v['source_rgb_correlation_4px_min'] for v in summary.values()):.3f}。这检查的是文件/坐标一致性，不是完美实体边界或用途真值。",
        '', '11栋漏判没有目标空白或触裁剪边缘；未见明显漏裁主屋顶。FN02的原产品多边形覆盖约75.9%的县级参考轮廓，仍需考虑屋檐、年代及边界误差。其余问题更像目标在输入中偏小，而非裁剪把屋顶剪没。',
        '', '3条误报FP16/FP18/FP19目标触裁剪边缘。FP17/FP18/FP19的整张RGB分别约6.6%/38.3%/9.8%为空；其中目标mask内没有零像素，不能据此断言整栋屋顶完整。FP03、FP06存在县级轮廓覆盖不足；FP11、FP23可能有内院/屋顶构件并入目标的几何定义问题，需要单独对照而非直接改原输入。',
        '', '7条误报原近景/宽视野导出不可用。本次使用已对齐的本地火前底图另生成自适应近景和300米诊断图；这些宽视野仍存在源影像边缘缺失，全部保留并标记，没有替换CNN输入。近景改变目标占比，不增加原始影像的真实分辨率。',
        '', '## 3. 模型与训练覆盖：最明确的共同模式','',
        '|证据|漏判11栋|正确多户3栋|','|---|---:|---:|',
        f"|占地范围|{fn['area_m2_min_median_max'][0]:.0f}–{fn['area_m2_min_median_max'][2]:.0f}㎡|{tp['area_m2_min_median_max'][0]:.0f}–{tp['area_m2_min_median_max'][2]:.0f}㎡|",
        f"|7×7特征图上mask面积中位数|{fn['feature_mask_area_cells_min_median_max'][1]:.2f}个面积等效单元|{tp['feature_mask_area_cells_min_median_max'][1]:.2f}个面积等效单元|",
        f"|最高类概率>0.9|{fn['high_confidence_wrong_or_correct']}/11（错误）|{tp['high_confidence_wrong_or_correct']}/3（正确）|",
        '', '特征面积不是独立感受野数量；当前CNN也通过感受野看到邻居。因此这支持目标尺度/背景依赖实验，尚不能证明7×7池化就是错误原因。',
        '', '所有漏判均在Eaton；27个误报中20个在Palisades，9个集中于cell_00183。逐图可见大量平屋顶、长屋顶和大型独栋，以及一栋邻接公寓的普通住宅。空间集中本身不证明地理偏差，结合视觉形态说明应补这些形态的训练难负例。',
        '', f"本轮LA适配训练部分：多户93栋，仅4栋四户，面积中位数{coverage['train:residential_multi']['area_m2_min_median_max'][1]:.0f}㎡；小型住宅478栋，均为1户，面积中位数{coverage['train:residential_small']['area_m2_min_median_max'][1]:.0f}㎡。这不是全部源域训练数据的户数统计。当前模型的低层重采样重复利用已有样本，没有新增形态。后续补的28栋四户来自同一个空间网格，增加数量但多样性有限。",
        '', '实际冻结Dataset复核：4个原四户训练样本各连续读取20次，每个只有1种输入张量。当前默认增强按样本索引固定，重采样不会产生新的翻转/旋转；该工程问题已确认，但修复后的增益仍需要训练对照。',
        '', f"两个CNN种子对{fp['seed_disagreement']}/27个误报分类不同，但只对{fn['seed_disagreement']}/11个漏判不同。8个误报也具有>0.9的多户概率；只用低置信度触发复核会遗漏大部分漏判。",
        '', '## 4. 已有模型的同集对照','',
        '|模型|多户TP|多户FP|Precision|Recall|多户F1|整体macro-F1|','|---|---:|---:|---:|---:|---:|---:|']
    names = {'CNN_old':'旧N2','cnn_lowrise_ensemble':'当前最佳CNN','dino_ensemble':'已有DINO微调双种子','cnn_fourplex_ensemble':'后续28栋四户补样CNN'}
    for name, m in scores.items():
        p = m['per_class'][CLASSES[1]]; correct = round(p['recall']*p['n']); positive = m['prediction_counts'].get(CLASSES[1],0)
        lines.append(f"|{names[name]}|{correct}|{positive-correct}|{p['precision']:.2%}|{p['recall']:.2%}|{p['f1']:.4f}|{m['macro_f1']:.4f}|")
    lines += ['', 'DINO没有找回当前CNN漏掉的11栋；它修正当前27个误报中的19个，同时在其他地方产生32个额外误报，合计40个。因此不能只看这38条就认定换编码器能改善precision。所有数字均来自既有冻结预测，没有在此做阈值搜索或新模型融合。',
        '', '## 5. 下一轮建议：按单变量逐步验证','',
        '|步骤|对应证据与改动|成功判断|','|---|---|---|',
        '|A：修复增强|仅让翻转/旋转在重复取样时变化，保持颜色增强、数据、采样权重和模型一致；以当前最佳CNN为对照|两个种子的独立验证集多户P/R/F1及macro-F1，检查有效图像变化而非只看loss|',
        '|B：成对补训练样本|在原测试区及既定隔离缓冲之外，补不同街区的小体量4–8户主楼，同时补相近面积/屋顶形状的1户住宅、办公楼和明确附属建筑难负例；用途不清的屋顶不强贴多户|固定类采样总权重，避免仅增加多户先验；要求precision与recall同时改善|',
        '|C：目标尺度与上下文对照|在B的相同数据上，比较原78米固定crop与完整目标自适应近景；再加入独立上下文表征，避免直接把邻居公寓当成目标用途|分别报告小体量多户、大独栋、商业及边界样本的P/R；记录推理开销|',
        '|D：校准及复核分流|只在独立验证集选择阈值/校准规则，评估种子分歧信号；不对本38条定制规则|报告覆盖率与所有样本指标，不能通过丢弃难例提高表观precision|',
        '', 'A与B分别对照，确认各自贡献后再组合。原3,332条标签、主评估输入和评分口径继续冻结；修复输入或补充人工楼栋级审定必须另存版本，作为附加分析。选模先看独立验证集多户precision、recall及F1，再做一次固定LA对照，并保留商业及整体macro-F1退化检查。跨城市主张仍需要未参与开发的城市数据。',
        '', '本轮没有新训练或验证上述改动的实际提升；也没有把审阅疑点排除后重新报一个更高准确率。标签复核是AI辅助观察和公开记录交叉检查，不是独立人工金标准。',
        '', '## 复现与证据','',
        '脚本：`audit_la_multifamily.py prepare`（py311，首次读取公开县级轮廓）、`augmentation-check`（py311，实际冻结Dataset）、`report`（catcch-env，复用原评分函数）。',
        '', '[完整逐例JSON观察](REVIEW_NOTES.json) · [汇总数值](SUMMARY.json) · [验证记录](VERIFICATION.json)',
        '', '公开轮廓来源：[LA County历史建筑目录](https://dpw.gis.lacounty.gov/dpw/rest/services/FEMAFloodzoneAdopted/MapServer/15)、[LA County较新建筑目录](https://public.gis.lacounty.gov/public/rest/services/LACounty_Dynamic/LARIAC_Buildings_2020/MapServer/0)。实际查询、日期和SHA256在raw/的provenance文件中；用途标签继续来自冻结的2024 Assessor快照。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    extra = [OUT/'REVIEW_NOTES.json', OUT/'AUGMENTATION_CHECK.json', ADAPT/'STABLE_CANDIDATES.json',
             STAGE/'la_fourplex_20261029/DATASET.json', ADAPT/'execution/common.py', Path(__file__),
             Path(__file__).with_name('train_la_adaptation.py')]
    hashes.update({str(p):sha256(p) for p in extra})
    outputs = [OUT/'SUMMARY.json',OUT/'REPORT.md',OUT/'CASE_REVIEW.csv',OUT/'REVIEW.html',OUT/'EVIDENCE.json']
    for c in cases:
        path = OUT/'cards'/f"{c['case_id']}.jpg"; assert path.is_file(); outputs.append(path)
    save(OUT/'VERIFICATION.json',dict(passed=True, original_test_labels_unchanged=True,
        exact_error_ids_verified=True, reviewed_errors=38, reviewed_controls=3, all_errors_have_four_view_cards=True,
        current_baseline_metrics_reproduced=True, no_new_adjudicated_ground_truth=True,
        no_model_or_test_input_modified=True, new_training_run=False, test_threshold_search=False,
        static_review_links_checked=True, browser_interaction_verified=False,
        sources_sha256=hashes, output_sha256={str(p.relative_to(OUT)):sha256(p) for p in outputs}))
    print(json.dumps(dict(groups=summary, priorities=priority, dino_overlap=overlap, coverage=coverage), indent=2))


if __name__ == '__main__':
    main()
