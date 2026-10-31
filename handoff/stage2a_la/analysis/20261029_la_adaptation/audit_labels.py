"""Audit LA reference evidence without consulting model predictions or relabeling."""

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from evaluate_assessor import map_use

STAGE = Path('/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a')


def main():
    source = HERE.parent/'20261015_gsd_assessor/la_assessor_matched.csv'
    frozen = STAGE/'la_transfer_20261027/dataset.json'
    data = json.loads(frozen.read_text())
    selected = [r for r in data if r['reference_use'] != 'residential_small' or
                'b5669b59-e639-430f-b565-7ef792c54ef5' in r['sample_id']]
    selected.sort(key=lambda r: (r['reference_use'] != 'residential_multi', r['reference_use'], r['sample_id']))
    matches = {}
    for r in csv.DictReader(source.open()):
        matches.setdefault((r['cell_id'], r['bldg_uid']), []).append(r)
    rows = []
    for i, r in enumerate(selected, 1):
        aa = matches[tuple(json.loads(r['building_uid']))]
        assert len(aa) == 1
        a = aa[0]
        units = float(a['Units']); label = r['reference_use']
        assert map_use(a) == label and a['RollYear'] == '2024'
        assert float(a['coregistered_iou']) >= .5 and float(a['matched_gt_per_prediction']) == 1
        assert float(a['parcel_outline_count']) == 1 and float(a['SQFTmain']) > 0
        issues = []
        if float(a['totBuildingDataLines']) != 1: issues.append('multiple_assessor_building_records')
        if label == 'residential_small' and not 1 <= units <= 3: issues.append('use_code_units_conflict')
        if label == 'residential_multi' and units < 4: issues.append('use_code_units_conflict')
        # LA Units is not NYC unitsres; nonzero commercial units alone are not a label conflict.
        rows.append(dict(review_index=i, sample_id=r['sample_id'], reference_use=label,
            AIN=a['AIN'], UseCode=a['UseCode'], Units=units,
            description=' | '.join(a['UseCodeDescChar'+str(n)].strip() for n in [2,3,4]),
            assessor_building_records=a['totBuildingDataLines'], coregistered_iou=a['coregistered_iou'],
            outline_count_in_reference_region=a['parcel_outline_count'], issues=';'.join(issues),
            evidence_status='requires_resolution' if issues else 'consistent_official_parcel_reference',
            near=str(STAGE/'la_transfer_20261027'/r['near_marked']),
            wide=str(STAGE/'la_transfer_20261027'/r['wide_marked'])))
    assert len(rows) == 57 and Counter(r['reference_use'] for r in rows) == {
        'residential_multi':14, 'commercial':42, 'residential_small':1}
    with (HERE/'LABEL_AUDIT.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sheets = HERE/'contact_sheets'; sheets.mkdir(exist_ok=True)
    for start in range(0, len(rows), 6):
        canvas = Image.new('RGB', (1536, 1320), 'white'); draw = ImageDraw.Draw(canvas)
        for j, r in enumerate(rows[start:start+6]):
            x, y = (j%2)*768, (j//2)*440
            text = f"#{r['review_index']:02d} {r['reference_use']} code={r['UseCode']} units={r['Units']:g} AIN={r['AIN']}\n{r['sample_id']}\n{r['description'][:105]}"
            draw.text((x+3,y+3), text, fill='black')
            for k, v in enumerate(['near','wide']):
                with Image.open(r[v]) as im:
                    canvas.paste(im.convert('RGB').resize((384,384)), (x+k*384,y+54))
        canvas.save(sheets/f'sheet_{start//6+1:02d}.jpg', quality=94)
    report = dict(n=57, minority_n=56, status_counts=dict(Counter(r['evidence_status'] for r in rows)),
        flagged=[r for r in rows if r['issues']], original_labels_changed=0,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,frozen]},
        limitation='Consistent parcel attributes and existing outline matches are not independently adjudicated building-use or role truth; parcel outline count is within the old reference region.')
    (HERE/'LABEL_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','flagged']},indent=2))
    print('FLAGGED',[(r['review_index'],r['sample_id'],r['issues']) for r in report['flagged']])


if __name__ == '__main__': main()
