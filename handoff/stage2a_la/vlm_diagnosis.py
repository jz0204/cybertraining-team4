"""Paired input-resolution, prompt and context diagnostics for the frozen VLM pilot."""

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

from vlm_pilot import CLASSES, PROMPT, STAGE, mark_target, metrics, read_csv

PILOT = STAGE / 'vlm_pilot_20261026'
OUT = STAGE / 'vlm_diagnosis_20261026'
PROMPT_P1 = '''Classify the USE of the indicated building in pre-disaster aerial imagery.
Image 1 is the original RGB crop. Image 2 is the SAME crop with the target
building outlined in cyan. Classify only that target, which may be off-center.
The target footprint area is {area:.1f} square metres.
Use the same three definitions:
- residential_small: detached or manufactured housing, duplexes or triplexes (1-3 units).
- residential_multi: residential apartment/multifamily buildings with 4 or more units.
- commercial: buildings used for commerce, retail, offices or services.
Compare the visual evidence for all three uses before choosing. A small footprint,
low height, pitched roof or trees does not by itself establish residential_small.
Consider private access and yards; repeated attached units, shared circulation and
parking; or street-facing premises, service access, paved forecourts and rooftop
equipment. These are uncertain cues, not deterministic rules. Commercial use does
not require readable signage, and apartment use need not expose countable units.
First locate the outlined target, then use its surroundings only as contextual
evidence. Do not replace the target with a more salient neighboring building.
A carport, garage or clubhouse is not automatically a dwelling. Do not invent
occupants, unit counts, business names, activities or invisible structural details.
If the target use cannot be distinguished visually or lies outside these three
classes, still give your best of the three classes and set evidence_sufficient=false.
Return only a JSON object with these keys:
type: one of the three exact class names;
evidence_sufficient: a JSON boolean indicating whether visible evidence resolves use;
evidence: one short sentence stating the discriminating visible cues or ambiguity.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'evaluate', 'self-check'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'evaluate':
        evaluate()
    else:
        self_check()


def prepare():
    from PIL import Image, ImageChops
    from shapely import wkt

    package = STAGE.parents[2]
    sys.path.insert(0, str(package))
    from II_package.scripts.shared.generate_shared_instance_subimages import clamp_crop_window
    from II_package.scripts.stage2a.training.rebuild_stage2a_dataset import render_mask

    protocol_path = OUT / 'PROTOCOL.json'
    if protocol_path.exists():
        raise FileExistsError(protocol_path)
    reference = json.loads((PILOT / 'reference.json').read_text())
    inputs = json.loads((PILOT / 'inputs.json').read_text())
    labels = {r['building_uid']: r for r in read_csv(STAGE / 'la_explore_20261023/data/labels_explore.csv')}
    assert len(inputs) == len(reference) == 296
    assert [r['sample_id'] for r in inputs] == [r['sample_id'] for r in reference]
    images = STAGE.parent / 'stage3_harvey/source_data/xBD/train/images'
    for name in ['native256', 'up512', 'wide512']:
        (OUT / name / 'images').mkdir(parents=True, exist_ok=False)
    provenance = []
    for row, ref in zip(inputs, reference):
        for key in ['image', 'target_image']:
            shutil.copy2(PILOT / row[key], OUT / 'native256' / row[key])
            with Image.open(PILOT / row[key]) as image:
                assert image.size == (256, 256)
                image.resize((512, 512), Image.Resampling.BICUBIC).save(OUT / 'up512' / row[key])
        if ref['domain'] != 'harris':
            continue
        meta = labels[ref['building_uid']]
        source = images / f"{meta['tile_base']}_pre_disaster.png"
        old_x, old_y = int(meta['crop_x0']), int(meta['crop_y0'])
        poly = wkt.loads(meta['poly_px_wkt'])
        with Image.open(source) as full, Image.open(PILOT / row['image']) as old:
            rgb = full.convert('RGB')
            old_box = (old_x, old_y, old_x + 256, old_y + 256)
            assert ImageChops.difference(rgb.crop(old_box), old).getbbox() is None
            box = clamp_crop_window(old_x + 128, old_y + 128, 512, *rgb.size)
            target = render_mask(poly, (512, 512), box[0], box[1])
            crop, marked, bbox = mark_target(rgb.crop(box), target)
            crop.save(OUT / 'wide512' / row['image'])
            marked.save(OUT / 'wide512' / row['target_image'])
            with Image.open(ref['mask_path']) as old_mask:
                original_mask = render_mask(poly, (256, 256), old_x, old_y)
                binary_old = old_mask.convert('L').point(lambda v: 255 if v else 0)
                assert ImageChops.difference(original_mask, binary_old).getbbox() is None
        provenance.append(dict(sample_id=ref['sample_id'], source=str(source), old_box=old_box,
                               wide_box=box, wide_bbox=bbox, original_pixels_and_mask_match=True,
                               native_mask_truncated=meta['mask_truncated']))
    variants = [
        ('replay6', 'native256', 'p0', inputs[:6]),
        ('up512_p0', 'up512', 'p0', inputs),
        ('native_p1', 'native256', 'p1', inputs),
        ('up512_p1', 'up512', 'p1', inputs),
        ('wide512_p1', 'wide512', 'p1', [r for r, ref in zip(inputs, reference) if ref['domain'] == 'harris']),
    ]
    spec = []
    for name, view, prompt, rows in variants:
        directory = OUT / name
        directory.mkdir(exist_ok=False)
        rows = [dict(r, **{k: '../' + view + '/' + r[k] for k in ['image', 'target_image']}) for r in rows]
        (directory / 'inputs.json').write_text(json.dumps(rows, indent=2) + '\n')
        spec.append(dict(name=name, view=view, prompt=prompt, n=len(rows)))
    (OUT / 'prompt_p0.txt').write_text(PROMPT)
    (OUT / 'prompt_p1.txt').write_text(PROMPT_P1)
    (OUT / 'reference.json').write_text(json.dumps(reference, indent=2) + '\n')
    (OUT / 'SOURCE_PARITY.json').write_text(json.dumps(provenance, indent=2) + '\n')
    protocol = dict(variants=spec, model='Qwen/Qwen3-VL-8B-Instruct',
        revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
        prompts=dict(p0=PROMPT, p1=PROMPT_P1), dtype='bfloat16', batch_size=1, seed=20261026,
        max_new_tokens=160, decode='greedy', interpolation='PIL bicubic, no new source pixels',
        purpose='2x2 original/upsampled input versus original/comparative prompt; Harris wider field of view',
        wide_comparison='512 native context vs 256 upsampled to512: equal image dimensions, context/detail tradeoff',
        limits=['All pilot rows are development data; this does not establish unseen-region generalization.',
                'LA native imagery unavailable; wider-context arm is Harris-only.',
                'Parcel-use labels are unchanged; auxiliary-structure mismatch remains unresolved.',
                'No few-shot examples, prompt search, threshold fitting or model fine-tuning.'],
        decision='Report every arm, class recall, new residential errors and paired recoveries; do not select by pooled accuracy.',
        runtime_budget='Same A30 image, 4800s Job deadline, independent restoration within90min bound.')
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(dict(variants=spec, harris_source_parity=len(provenance)), indent=2))


def evaluate():
    baseline = {r['sample_id']: r for r in json.loads((PILOT / 'paired_predictions.json').read_text())}
    protocol = json.loads((OUT / 'PROTOCOL.json').read_text())
    results, paired = {}, {}
    for variant in protocol['variants']:
        name = variant['name']
        predictions = [json.loads(line) for line in (OUT / name / 'predictions.jsonl').read_text().splitlines()]
        expected = {r['sample_id'] for r in json.loads((OUT / name / 'inputs.json').read_text())}
        indexed = {r['sample_id']: r for r in predictions}
        assert len(indexed) == len(predictions) == variant['n'] and set(indexed) == expected
        run = json.loads((OUT / name / 'predictions.run.json').read_text())
        assert run['status'] == 'complete' and run['revision'] == protocol['revision']
        rows = [dict(baseline[i], variant_type=p['pred_type'], variant_evidence=p['evidence'],
                     variant_sufficient=p['evidence_sufficient'], variant_raw=p['raw']) for i,p in indexed.items()]
        paired[name] = rows
        if name == 'replay6':
            results[name] = dict(n=len(rows), types_match=all(r['pred_type']==r['variant_type'] for r in rows),
                                raw_match=sum(r['raw']==r['variant_raw'] for r in rows))
            continue
        groups = {}
        for group in sorted({g for r in rows for g in r['groups']}):
            rr = [r for r in rows if group in r['groups']]
            groups[group] = dict(variant=metrics(rr, 'variant_type'), baseline=metrics(rr, 'pred_type'),
                                current_cnn=metrics(rr, 'baseline_type'), paired=paired_counts(rr))
        groups['runtime'] = dict(n=len(rows), invalid=sum(p['pred_type']=='invalid' for p in predictions),
             seconds=sum(p['seconds'] for p in predictions), median_seconds=statistics.median(p['seconds'] for p in predictions),
             max_reserved_gib=max(p['peak_reserved_gib'] for p in predictions),
             image_grids=sorted({json.dumps(p['image_grid_thw']) for p in predictions}),
             sufficient=sum(r['variant_sufficient'] for r in rows),
             reference_disagreements=sum(r['reference_type']!=r['variant_type'] for r in rows))
        groups['domains'] = {d: paired_counts([r for r in rows if r['domain']==d]) for d in sorted({r['domain'] for r in rows})}
        results[name] = groups
    (OUT / 'RESULTS.json').write_text(json.dumps(results, indent=2) + '\n')
    (OUT / 'paired_predictions.json').write_text(json.dumps(paired, indent=2) + '\n')
    print(json.dumps(results, indent=2))


def paired_counts(rows):
    return dict(n=len(rows),
        corrected=sum(r['pred_type']!=r['reference_type']==r['variant_type'] for r in rows),
        introduced=sum(r['pred_type']==r['reference_type']!=r['variant_type'] for r in rows),
        to_small_errors=sum(r['reference_type']!='residential_small' and r['pred_type']=='residential_small' for r in rows),
        to_small_recovered=sum(r['reference_type']!='residential_small' and r['pred_type']=='residential_small'
                               and r['variant_type']==r['reference_type'] for r in rows),
        small_regressions=sum(r['reference_type']=='residential_small' and r['variant_type']!='residential_small' for r in rows))


def self_check():
    assert '{area}' not in PROMPT_P1.format(area=123.4)
    assert '123.4' in PROMPT_P1.format(area=123.4)
    rows = [dict(reference_type='commercial', pred_type='residential_small', variant_type='commercial'),
            dict(reference_type='residential_small', pred_type='residential_small', variant_type='invalid')]
    assert paired_counts(rows) == dict(n=2, corrected=1, introduced=1, to_small_errors=1,
                                      to_small_recovered=1, small_regressions=1)
    assert metrics(rows, 'variant_type')['accuracy'] == .5
    print('PASS: prompt formatting, paired corrections/regressions, invalid-output denominator')


if __name__ == '__main__':
    main()
