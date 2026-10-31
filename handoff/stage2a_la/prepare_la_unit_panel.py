"""Prepare a model-output-independent unit review panel; do not manufacture unit gold."""
import csv
import hashlib
import html
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode

from prepare_la_closeup import STAGE, sha, save

ROOT = STAGE/'la_unit_panel_20261031'
SOURCE = STAGE/'la_hard_expansion_20261030'
OUT = Path(__file__).parent/'analysis/20261031_unit_panel'


def main():
    assert not ROOT.exists(), 'Never overwrite a review panel'
    rows = json.loads((SOURCE/'STABLE_CANDIDATES.json').read_text())
    reviews = {r['sample_id']:r for r in json.loads((SOURCE/'REVIEW_SOURCE.json').read_text())['records']}
    training = json.loads((STAGE/'la_hard_ablation_20261030/execution/C_positive_negative/DATASET.json').read_text())
    used = {r['sample_id'] for r in training}
    # Select by recorded units, footprint area and stable hash, never model scores.
    multi = [r for r in rows if r['reference_use']=='residential_multi' and reviews[r['sample_id']]['admit'] and r['footprint_m2']<=500]
    chosen = []
    for units, count in [(4,12),(5,4),(6,4),(7,2),(8,2)]:
        pool = sorted((r for r in multi if r['evidence']['Units']==units), key=order)
        assert len(pool)>=count; chosen += [(r,'small_multifamily',None) for r in pool[:count]]
    singles = [r for r in rows if r['reference_use']=='residential_small' and r['sample_id'] not in used]
    matches = []
    for r,_,_ in chosen:
        match = min(singles, key=lambda s:(abs(math.log(s['footprint_m2']/r['footprint_m2'])), order(s)))
        singles.remove(match); matches.append((match,'area_matched_single_family',r['sample_id']))
    uncertain = [(r,'unresolved_roof_attribution',None) for r in rows
                 if r['reference_use']=='residential_multi' and not reviews[r['sample_id']]['admit']]
    selected = chosen+matches+uncertain
    assert len(selected)==51 and len({r['sample_id'] for r,_,_ in selected})==51
    ROOT.mkdir(); (ROOT/'images').mkdir(); OUT.mkdir(parents=True)
    panel=[]; evidence={}; image_hashes={}
    for r,stratum,matched in selected:
        sid=r['sample_id']; meta_path=SOURCE/'image_metadata'/(sid+'.json')
        meta=json.loads(meta_path.read_text()); review=reviews[sid]
        assert r['split']=='train' and r['outline_iou_2014_2020']>=.8
        assert r['evidence']['totBuildingDataLines']==r['parcel_outline_count']==1
        images={}
        for view in ['near','wide']:
            source=SOURCE/meta['paths'][view+'_marked']; expected=meta['image_sha256'][meta['paths'][view+'_marked']]
            assert sha(source)==expected
            dest=ROOT/'images'/source.name; shutil.copyfile(source,dest)
            assert sha(dest)==expected; images[view]=str(dest.relative_to(ROOT)); image_hashes[images[view]]=expected
        query=urlencode(dict(where="AIN='"+r['parcel']+"' AND RollYear='2024'",outFields='AIN,RollYear,Units,totBuildingDataLines,UseCode,PropertyLocation',returnGeometry='false',f='pjson'))
        item=dict(sample_id=sid,stratum=stratum,matched_to=matched or '',parcel=r['parcel'],
            spatial_group=r['spatial_group'],footprint_m2=r['footprint_m2'],
            parcel_units=r['evidence']['Units'],assessor_year=r['evidence']['RollYear'],
            imagery_date=meta['imagery_datetime'],classifier_training_overlap=sid in used,
            independent_evaluation_eligible=False,verified_building_units='',
            verified_building_role='',verification_source='',verification_date='',reviewer='',
            review_status='pending_independent_building_level_verification',
            prior_ai_review=review['reason'],near_image=images['near'],wide_image=images['wide'],
            official_parcel_record='https://services.arcgis.com/RmCCgQtiZLDCtblq/ArcGIS/rest/services/Parcel_Data_2021_Table/FeatureServer/0/query?'+query)
        panel.append(item); evidence[sid]=dict(candidate=r,prior_ai_review=review,image_metadata_sha256=sha(meta_path))
    assert Counter(r['stratum'] for r in panel)=={'small_multifamily':24,'area_matched_single_family':24,'unresolved_roof_attribution':3}
    assert all(r['verified_building_units']=='' and not r['independent_evaluation_eligible'] for r in panel)
    save(ROOT/'PANEL.json',panel); save(ROOT/'EVIDENCE.json',evidence); save(ROOT/'IMAGE_SHA256.json',image_hashes)
    with (ROOT/'REVIEW.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(panel[0])); writer.writeheader(); writer.writerows(panel)
    cards=[]
    for r in panel:
        cards.append(f'''<article><h2>{html.escape(r['sample_id'])}</h2>
<p>{r['stratum'].replace('_',' ')} · {r['footprint_m2']:.0f} m² · parcel record: {r['parcel_units']} units</p>
<div><img src="{r['near_image']}" alt="Marked target roof"><img src="{r['wide_image']}" alt="Surrounding roofs"></div>
<p>{html.escape(r['prior_ai_review'])}</p><a href="{html.escape(r['official_parcel_record'],quote=True)}">Official parcel record</a></article>''')
    (ROOT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>LA dwelling-unit review</title>
<style>body{font:16px system-ui;max-width:1120px;margin:32px auto;background:#f6f7f8;color:#182331}h1{font-size:28px}h2{font-size:18px}article{background:white;padding:20px;margin:24px 0}article div{display:flex;gap:12px}img{width:48%;height:auto}a{color:#164da3}</style>
<h1>LA dwelling-unit review</h1><p>51 development candidates. Parcel counts are unverified for individual roofs. Record independent findings in <a href="REVIEW.csv">REVIEW.csv</a>.</p>
'''+''.join(cards)+'</html>\n')
    report=dict(n=51,strata=dict(Counter(r['stratum'] for r in panel)),
        classifier_training_overlap=sum(r['classifier_training_overlap'] for r in panel),
        verified_building_unit_references=0,independent_test_references=0,
        selection_uses_model_outputs=False,reserved_la_used=False,
        source_sha256={name:sha(SOURCE/name) for name in ['STABLE_CANDIDATES.json','REVIEW_SOURCE.json']},
        missing_strata=['Independently documented inhabited accessory buildings','Independently documented nonresidential accessory buildings'])
    save(ROOT/'AUDIT.json',report); save(OUT/'AUDIT.json',report)
    (OUT/'REPORT.md').write_text(f'''# LA dwelling-unit reference preparation

Prepared [{ROOT.name}]({ROOT}/index.html): 24 small multifamily candidates (12 fourplex, 4 five-unit, 4 six-unit, 2 seven-unit, 2 eight-unit parcel records), 24 footprint-area-matched single-family candidates, and 3 unresolved roof-attribution cases. All are in existing development training cells; no reserved imagery or labels were used. The 24 single-family candidates and 3 unresolved cases are absent from the classifier training manifest. Selection does not read model scores, although this source pool was scored in earlier work.

All 102 marked target/context images were checked against their existing hashes. The panel includes parcel and building-outline evidence, acquisition dates, original AI-review notes, a CSV for independent findings, and an offline HTML review sheet. All records still have blank verified building-unit labels. The official 2024 parcel count, one assessment data line, and one historical outline do not establish per-roof dwelling units or current occupancy. The imagery is from 2022.

Use this as a unit-supervision development panel after independent verification. The 24 multifamily examples already trained the classifier, so they cannot establish independent encoder/generalization performance. No record currently qualifies as a held-out unit test case. Inhabited and nonresidential accessory examples remain a documented gap; do not label an auxiliary roof as zero units from appearance.

For each candidate, confirm that an independently documented unit count belongs to the marked structure, resolve disconnected roofs, and record source/date/reviewer in [REVIEW.csv]({ROOT}/REVIEW.csv). Leave ambiguous or temporally mismatched counts unresolved. Acquire a separate spatially excluded test panel before reporting unit or population accuracy. No direct-unit model was trained on these pending labels.
''')
    print('PANEL_READY',json.dumps(report))


def order(row): return hashlib.sha256(('unit-panel-20261031:'+row['sample_id']).encode()).hexdigest()


if __name__=='__main__': main()
