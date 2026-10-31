"""Frozen Qwen3-VL building-type pilot; preparation, inference and paired scoring."""

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

STAGE = Path('/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a')
CLASSES = ['residential_small', 'residential_multi', 'commercial']
MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
SEED = 20261026
PROMPT = '''Classify the USE of the indicated building in pre-disaster aerial imagery.
Image 1 is the original RGB crop. Image 2 is the SAME crop with the target
building outlined in cyan. Classify only that target, which may be off-center.
Use visible building form and surroundings; do not classify the whole scene.
The target footprint area is {area:.1f} square metres.
Use these dataset definitions, which describe use rather than roof size:
- residential_small: detached or manufactured housing, duplexes or triplexes (1-3 units).
- residential_multi: residential apartment/multifamily buildings with 4 or more units.
- commercial: buildings used for commerce, retail, offices or services.
If the use cannot be distinguished visually, still give your best of these three
classes and set evidence_sufficient=false. Do not invent occupants, unit counts,
business names or activities. Set evidence_sufficient=false for uses outside the
three classes. Return only a JSON object, no markdown:
{{"type":"one of the three exact class names","evidence_sufficient":true,
"evidence":"one short sentence describing visible evidence and limitations"}}'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'infer', 'evaluate', 'self-check'])
    parser.add_argument('--directory', type=Path, default=STAGE / 'vlm_pilot_20261026')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--revision')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', default='predictions.jsonl')
    parser.add_argument('--prompt-file', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.directory)
    elif args.action == 'infer':
        infer(args)
    elif args.action == 'evaluate':
        evaluate(args.directory, args.output)
    else:
        self_check()


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def by_id(path):
    rows = read_csv(path)
    indexed = {r['building_uid']: r for r in rows}
    assert len(indexed) == len(rows), f'Duplicate building keys: {path}'
    return indexed


def prepare(out):
    from PIL import Image

    rng = random.Random(SEED)
    source = STAGE / 'la_explore_20261023'
    labels = read_csv(source / 'data/labels_explore.csv')
    la = [r for r in read_csv(source / 'data/la_reference.csv')
          if r['primary_single_outline_iou50'] == 'True']
    assert len(la) == 3332
    assert Counter(r['assessor_type'] for r in la) == dict(residential_small=3276, residential_multi=14, commercial=42)
    inputs = by_id(STAGE / 'la_eval_20261023/prepared/stage2a_input.csv')
    baseline = by_id(STAGE / 'la_eval_20261023/cluster_full/stage2a_predictions.csv')
    n1 = by_id(source / 'predictions/N1_natural_ce_seed42/la.csv')
    selected = {}

    def add(rows, domain, group):
        for row in rows:
            key = domain + '/' + row['building_uid']
            if key not in selected:
                selected[key] = dict(row, domain=domain, groups=[])
            selected[key]['groups'].append(group)

    add(rng.sample(sorted(la, key=lambda r: r['building_uid']), 120), 'la', 'la_random120')
    add([r for r in la if r['assessor_type'] != 'residential_small'], 'la', 'la_all_rare56')
    hard = [r for r in la if r['assessor_type'] == 'residential_small'
            and baseline[r['building_uid']]['pred_type_class'] == 'commercial']
    add(rng.sample(sorted(hard, key=lambda r: r['building_uid']), 34), 'la', 'la_commercial_fp34')
    harris_baseline = by_id(source / 'reference_ensemble/harris_val.csv')
    harris_n1 = by_id(source / 'predictions/N1_natural_ce_seed42/harris_val.csv')
    for label in CLASSES:
        pool = [r for r in labels if r['rebuild_split'] == 'val' and r['type_class_name'] == label]
        add(rng.sample(sorted(pool, key=lambda r: r['building_uid']), 30), 'harris', 'harris_balanced90')
    rows = list(selected.values())
    rng.shuffle(rows)
    # Put one of each city/class first so the six-row smoke actually exercises all classes.
    first = [next(r for r in rows if r['domain'] == d and r.get('assessor_type', r.get('type_class_name')) == c)
             for d in ['la', 'harris'] for c in CLASSES]
    rows = first + [r for r in rows if r not in first]
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    truth, inference = [], []
    for i, row in enumerate(rows):
        uid = row['building_uid']
        if row['domain'] == 'la':
            data = inputs[uid]
            crop, mask = Path(data['crop_path']), Path(data['mask_path'])
            label, group = row['assessor_type'], row['cell_id']
            old, natural = baseline[uid], n1[uid]
        else:
            data = row
            crop = STAGE / 'rebuild_20261015/dataset/crops' / Path(row['crop_path']).name
            mask = STAGE / 'rebuild_20261015/dataset/masks' / Path(row['mask_path']).name
            label, group = row['type_class_name'], row['tile_base']
            old, natural = harris_baseline[uid], harris_n1[uid]
        with Image.open(crop) as image, Image.open(mask) as target:
            rgb, marked, bbox = mark_target(image, target)
            rgb.save(out / f'images/{i:04d}_rgb.png')
            marked.save(out / f'images/{i:04d}_target.png')
        inference.append(dict(sample_id=i, image=f'images/{i:04d}_rgb.png',
                              target_image=f'images/{i:04d}_target.png', footprint_m2=float(data['footprint_m2'])))
        truth.append(dict(sample_id=i, building_uid=uid, domain=row['domain'], spatial_group=group,
                          groups=row['groups'], reference_type=label, baseline_type=old['pred_type_class'],
                          n1_type=natural['pred_type_class'], crop_path=str(crop), mask_path=str(mask),
                          target_bbox=bbox, target_touches_border=bbox[0] == 0 or bbox[1] == 0
                          or bbox[2] == rgb.width or bbox[3] == rgb.height))
    (out / 'inputs.json').write_text(json.dumps(inference, indent=2) + '\n')
    (out / 'reference.json').write_text(json.dumps(truth, indent=2) + '\n')
    protocol = dict(model=MODEL, seed=SEED, n=len(rows), groups=dict(Counter(g for r in truth for g in r['groups'])),
                    prompt=PROMPT, temperature=0, max_new_tokens=160, batch_size=1, dtype='bfloat16',
                    model_inputs='Only two RGB views and footprint area; no labels, city names or CNN predictions.',
                    reference='Frozen parcel-derived use references, not manually confirmed building use.',
                    evaluation='Previously inspected development data. Report groups separately; group membership may overlap.',
                    selection='No prompt/model tuning against pilot outputs. No San Diego access or population inference.')
    (out / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(protocol, indent=2))


def mark_target(image, target):
    from PIL import ImageChops, ImageFilter

    rgb, mask = image.convert('RGB'), target.convert('L').point(lambda v: 255 if v else 0)
    if rgb.size != mask.size or mask.getbbox() is None:
        raise ValueError('Target mask must be nonempty and aligned with the RGB crop')
    edge = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(3)), mask.filter(ImageFilter.MinFilter(3)))
    marked = rgb.copy()
    marked.paste((0, 255, 255), mask=edge)
    return rgb, marked, mask.getbbox()


def infer(args):
    import importlib.metadata as metadata
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = json.loads((args.directory / 'inputs.json').read_text())
    prompt = args.prompt_file.read_text() if args.prompt_file else PROMPT
    if args.limit is not None:
        rows = rows[:args.limit]
    assert rows and len({r['sample_id'] for r in rows}) == len(rows)
    for row in rows:
        assert row['footprint_m2'] > 0
        assert all((args.directory / row[k]).is_file() for k in ['image', 'target_image'])
    target = args.directory / args.output
    if target.exists():
        raise FileExistsError(target)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'This experiment requires CUDA; no implicit CPU fallback.'
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForImageTextToText.from_pretrained(args.model, revision=args.revision,
            dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='sdpa').eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    versions = {p: metadata.version(p) for p in ['torch', 'transformers', 'huggingface-hub', 'Pillow']}
    run = dict(model=args.model, revision=args.revision, versions=versions, device=torch.cuda.get_device_name(),
               n=len(rows), load_seconds=load_seconds, dtype=str(model.dtype), seed=SEED, status='running')
    run_path = target.with_suffix('.run.json')
    run_path.write_text(json.dumps(run, indent=2) + '\n')
    with target.open('x') as handle, torch.inference_mode():
        for row in rows:
            start = time.perf_counter()
            images = [Image.open(args.directory / row[k]).convert('RGB') for k in ['image', 'target_image']]
            content = [{'type': 'image', 'image': image} for image in images]
            content.append({'type': 'text', 'text': prompt.format(area=row['footprint_m2'])})
            inputs = processor.apply_chat_template([{'role': 'user', 'content': content}],
                     tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
            torch.cuda.reset_peak_memory_stats()
            generated = model.generate(**inputs, max_new_tokens=160, do_sample=False)
            torch.cuda.synchronize()
            text = processor.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            result = dict(sample_id=row['sample_id'], raw=text, **parse_answer(text),
                          seconds=time.perf_counter()-start, input_tokens=inputs['input_ids'].shape[1],
                          output_tokens=generated.shape[1]-inputs['input_ids'].shape[1],
                          image_grid_thw=inputs['image_grid_thw'].tolist(),
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            handle.write(json.dumps(result) + '\n')
            handle.flush()
            print(json.dumps({k:v for k,v in result.items() if k != 'raw'}), flush=True)
    run['status'] = 'complete'
    run_path.write_text(json.dumps(run, indent=2) + '\n')


def parse_answer(text):
    try:
        answer = json.loads(text)
        if (not isinstance(answer, dict) or answer.get('type') not in CLASSES
                or type(answer.get('evidence_sufficient')) is not bool
                or not isinstance(answer.get('evidence'), str)):
            raise ValueError('Invalid type, evidence flag or evidence text')
        return dict(pred_type=answer['type'], evidence_sufficient=answer['evidence_sufficient'],
                    evidence=answer['evidence'], parse_error=None)
    except (ValueError, TypeError) as error:
        return dict(pred_type='invalid', evidence_sufficient=False, evidence='', parse_error=str(error))


def metrics(rows, column):
    matrix = [[0]*4 for _ in CLASSES]
    for row in rows:
        prediction = row[column]
        matrix[CLASSES.index(row['reference_type'])][CLASSES.index(prediction) if prediction in CLASSES else 3] += 1
    per_class = {}
    for j, label in enumerate(CLASSES):
        tp, support, predicted = matrix[j][j], sum(matrix[j]), sum(r[j] for r in matrix)
        precision, recall = tp/predicted if predicted else 0., tp/support if support else 0.
        per_class[label] = dict(support=support, precision=precision, recall=recall,
                               f1=2*tp/(support+predicted) if support+predicted else 0.)
    return dict(n=len(rows), accuracy=sum(matrix[j][j] for j in range(3))/len(rows),
                macro_f1=sum(p['f1'] for p in per_class.values())/3, per_class=per_class,
                confusion=matrix, columns=CLASSES+['invalid'])


def evaluate(directory, filename):
    reference = json.loads((directory / 'reference.json').read_text())
    predictions = [json.loads(line) for line in (directory / filename).read_text().splitlines()]
    indexed = {r['sample_id']:r for r in predictions}
    assert len(indexed) == len(predictions)
    assert set(indexed) == {r['sample_id'] for r in reference}, 'Pilot must cover every fixed sample'
    rows = [dict(r, **{k:v for k,v in indexed[r['sample_id']].items() if k != 'sample_id'}) for r in reference]
    results = {}
    for group in sorted({g for r in rows for g in r['groups']}):
        selected = [r for r in rows if group in r['groups']]
        results[group] = {name:metrics(selected, column) for name,column in
                          [('vlm','pred_type'),('current_ensemble','baseline_type'),('n1_seed42','n1_type')]}
        certain = [r for r in selected if r['evidence_sufficient']]
        results[group]['evidence_sufficient_subset'] = dict(n=len(certain), coverage=len(certain)/len(selected),
            metrics=metrics(certain, 'pred_type') if certain else None,
            class_counts=dict(Counter(r['reference_type'] for r in certain)))
        results[group]['paired_vs_current'] = dict(
            corrected=sum(r['baseline_type'] != r['reference_type'] == r['pred_type'] for r in selected),
            introduced=sum(r['baseline_type'] == r['reference_type'] != r['pred_type'] for r in selected))
    results['runtime'] = dict(n=len(rows), invalid=sum(r['pred_type']=='invalid' for r in rows),
        total_inference_seconds=sum(r['seconds'] for r in rows),
        peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),
        peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows))
    (directory / 'RESULTS.json').write_text(json.dumps(results, indent=2) + '\n')
    (directory / 'paired_predictions.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(results, indent=2))


def self_check():
    from PIL import Image, ImageDraw

    rgb = Image.new('RGB', (16,16), (20,30,40))
    mask = Image.new('L', rgb.size)
    ImageDraw.Draw(mask).rectangle((0,5,4,9), fill=1)
    original, marked, bbox = mark_target(rgb, mask)
    assert original.tobytes() == rgb.tobytes() and bbox == (0,5,5,10)
    assert marked.getpixel((4,7)) == (0,255,255) and marked.getpixel((12,12)) == rgb.getpixel((12,12))
    good = parse_answer('{"type":"commercial","evidence_sufficient":false,"evidence":"unclear"}')
    assert good['pred_type'] == 'commercial' and not good['evidence_sufficient']
    assert parse_answer('{"type":"commercial","evidence_sufficient":"false","evidence":"x"}')['pred_type'] == 'invalid'
    assert parse_answer('not JSON')['pred_type'] == 'invalid'
    check = metrics([dict(reference_type='commercial', p='invalid'),
                     dict(reference_type='residential_small', p='residential_small')], 'p')
    assert check['n'] == 2 and check['accuracy'] == .5 and check['confusion'][2][3] == 1
    assert check['per_class']['commercial']['recall'] == 0
    print('PASS: off-center border mask, original pixels, response validation and missing-output penalty')


if __name__ == '__main__':
    main()
