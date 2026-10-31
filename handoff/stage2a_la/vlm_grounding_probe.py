"""Twelve-case localization and building-role probe; not an accuracy benchmark."""

import argparse
import json
import math
import time
from pathlib import Path

from vlm_pilot import CLASSES, SEED

PROMPT = '''Inspect the cyan-outlined target in these aerial images.
Image 1 is RGB; image 2 is the SAME scene with only the target outlined in cyan.
The target footprint area is {area:.1f} square metres. The target may be off-center.
First locate the outlined target, then describe its visible form and role, then
give a likely use if the visible evidence supports one. Do not replace it with
a larger or more salient neighboring building. Do not infer storeys, occupants
or exact dwelling counts from the roof alone.
Return only one JSON object with keys in this order:
target_bbox_1000: [xmin,ymin,xmax,ymax] enclosing the cyan outline, coordinates
normalized from0 to1000 relative to the entire image;
roof_form: flat, pitched, mixed or unclear;
building_role: primary, auxiliary or unclear (auxiliary includes carports,
garages, sheds and similar supporting structures);
visible_evidence: one short sentence describing the outlined target and context;
use: residential_small (1-3 dwelling units), residential_multi (4+ dwelling units),
commercial (commerce, retail, offices or services), other or unknown.
Use unknown if actual use is not visually resolvable. Use other for an evident
function outside the three classes. Do not force an auxiliary roof into a dwelling
class or invent a use from its parcel or neighbors.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['infer', 'self-check'])
    parser.add_argument('--directory', type=Path, default=Path('/experiment'))
    parser.add_argument('--model')
    parser.add_argument('--revision')
    args = parser.parse_args()
    if args.action == 'self-check':
        self_check()
    else:
        infer(args)


def infer(args):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = json.loads((args.directory / 'inputs.json').read_text())
    assert len(rows) == len({r['sample_id'] for r in rows}) == 12
    assert all(Path(r[k]).is_file() for r in rows for k in ['image', 'target_image'])
    assert all(r['footprint_m2'] > 0 for r in rows)
    target = args.directory / 'predictions.jsonl'
    if target.exists():
        raise FileExistsError(target)
    assert torch.cuda.is_available()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForImageTextToText.from_pretrained(args.model, revision=args.revision,
        dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='sdpa').eval()
    with target.open('x') as handle, torch.inference_mode():
        for row in rows:
            t0 = time.perf_counter()
            images = [Image.open(row[k]).convert('RGB') for k in ['image', 'target_image']]
            content = [{'type': 'image', 'image': im} for im in images]
            content.append({'type': 'text', 'text': PROMPT.format(area=row['footprint_m2'])})
            inputs = processor.apply_chat_template([{'role': 'user', 'content': content}],
                tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
            torch.cuda.reset_peak_memory_stats()
            output = model.generate(**inputs, max_new_tokens=160, do_sample=False)
            torch.cuda.synchronize()
            raw = processor.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            parsed, error = parse_probe(raw)
            result = dict(sample_id=row['sample_id'], raw=raw, parsed=parsed, parse_error=error,
                seconds=time.perf_counter()-t0, image_grid_thw=inputs['image_grid_thw'].tolist(),
                input_tokens=inputs['input_ids'].shape[1], output_tokens=output.shape[1]-inputs['input_ids'].shape[1],
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            handle.write(json.dumps(result)+'\n')
            handle.flush()
            print(json.dumps(result), flush=True)
    (args.directory / 'predictions.run.json').write_text(json.dumps(dict(status='complete',n=12,
        model=args.model, revision=args.revision, dtype=str(model.dtype), prompt=PROMPT,
        device=torch.cuda.get_device_name(), max_new_tokens=160),indent=2)+'\n')


def parse_probe(raw):
    try:
        answer = json.loads(raw)
        if not isinstance(answer, dict):
            raise ValueError('Expected JSON object')
        box = answer.get('target_bbox_1000')
        if (not isinstance(box, list) or len(box)!=4
            or not all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1000 for v in box)
            or box[0]>=box[2] or box[1]>=box[3]):
            raise ValueError('Invalid normalized bounding box')
        if (answer.get('roof_form') not in ['flat','pitched','mixed','unclear']
            or answer.get('building_role') not in ['primary','auxiliary','unclear']
            or answer.get('use') not in CLASSES+['other','unknown']
            or not isinstance(answer.get('visible_evidence'),str)):
            raise ValueError('Invalid categorical output or evidence')
        return answer, None
    except (ValueError,TypeError) as error:
        return None, str(error)


def self_check():
    answer=dict(target_bbox_1000=[10,20,100,200],roof_form='flat',building_role='auxiliary',
                visible_evidence='A canopy over a parking row.',use='other')
    assert parse_probe(json.dumps(answer)) == (answer,None)
    answer['target_bbox_1000']=[10,20,5,200]
    assert parse_probe(json.dumps(answer))[0] is None
    assert parse_probe('not JSON')[0] is None
    assert '100.0' in PROMPT.format(area=100)
    print('PASS: normalized box and role/use schema, malformed output, prompt formatting')


if __name__ == '__main__':
    main()
