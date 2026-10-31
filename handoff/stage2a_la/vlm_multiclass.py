"""Matched vocabulary/view experiments with abstention and separate building roles."""

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from vlm_pilot import MODEL, SEED, STAGE

OUT = STAGE / 'vlm_multiclass_20261026'
CLASSES = ['residential_small', 'residential_multi', 'commercial', 'institutional', 'other']
REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
VARIANTS = [
    dict(name='A_three_near', classes=CLASSES[:3], views=['near'], roles=True, unknown=True),
    dict(name='B_five_near', classes=CLASSES, views=['near'], roles=True, unknown=True),
    dict(name='C_three_dual', classes=CLASSES[:3], views=['near','wide'], roles=True, unknown=True),
    dict(name='D_five_dual', classes=CLASSES, views=['near','wide'], roles=True, unknown=True),
    dict(name='E_three_forced_dual', classes=CLASSES[:3], views=['near','wide'], roles=False, unknown=False),
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['infer','evaluate','self-check'])
    parser.add_argument('--directory',type=Path,default=OUT)
    parser.add_argument('--model',default=MODEL)
    parser.add_argument('--revision',default=REVISION)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.action=='infer':infer(args)
    elif args.action=='evaluate':evaluate(args.directory)
    else:self_check()


def prompt_for(variant):
    views=('Image1 is a close view. Image2 is a wider native view of the SAME target and its surroundings.'
           if len(variant['views'])==2 else 'Image1 is a close view of the target building.')
    text='''Classify the actual USE of the cyan-outlined building in aerial imagery.
'''+views+'''
The outline identifies the same target in every image. Classify that individual
structure, not the whole parcel or a neighboring building. Its footprint area
is {area:.1f} square metres. Visible context is evidence, not the target's use.
Allowed uses:
- residential_small: housing with1-3 dwelling units, including detached homes, duplexes and triplexes.
- residential_multi: an apartment/multifamily residential building with4+ dwelling units.
- commercial: retail, offices, restaurants, lodging and business services; ordinary medical offices are included.
'''
    if len(variant['classes'])==5:
        text+='''- institutional: education, worship, hospitals, government, public safety or civic-service buildings.
- other: an identifiable function outside the above uses, such as industrial manufacturing, bulk warehousing,
  utilities, agricultural buildings or supporting parking/storage structures. Other does not mean uncertain.
'''
    text+='''Do not equate a small footprint or pitched roof with housing. Do not infer exact dwelling counts,
occupants, ownership, business names or activities that are not visible. Distinguish the outlined
building's function from the surrounding campus, industrial site or apartment complex.
'''
    if variant['unknown']:
        text+='Use unknown when the target use is outside the allowed uses or visible evidence cannot distinguish them.\n'
    else:
        text+='Choose your best of the three allowed uses even if evidence is insufficient or actual use is outside them.\n'
    text+='Return only one JSON object with type, '
    if variant['roles']:
        text+='''role, and evidence. Role is primary, auxiliary or unknown. Primary is a main occupied or
functional building; auxiliary is a supporting structure such as a detached carport, garage or shed.
Infer role separately from the host site's use; a small building is not automatically auxiliary.
'''
    else:
        text+='evidence.\n'
    text+='Type must be one of the exact allowed names'+(' or unknown' if variant['unknown'] else '')+'. Evidence is one brief sentence of visible cues or ambiguity.\n'
    return text


def parse_answer(raw, variant):
    try:
        answer=json.loads(raw)
        allowed=variant['classes']+(['unknown'] if variant['unknown'] else [])
        if not isinstance(answer,dict) or answer.get('type') not in allowed or not isinstance(answer.get('evidence'),str):
            raise ValueError('Invalid type or evidence')
        role=answer.get('role') if variant['roles'] else None
        if variant['roles'] and role not in ['primary','auxiliary','unknown']:
            raise ValueError('Invalid role')
        return dict(pred_type=answer['type'],pred_role=role,evidence=answer['evidence'],parse_error=None)
    except (ValueError,TypeError) as exc:
        return dict(pred_type='invalid',pred_role='invalid' if variant['roles'] else None,evidence='',parse_error=str(exc))


def infer(args):
    import importlib.metadata
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText,AutoProcessor

    protocol=json.loads((args.directory/'PROTOCOL.json').read_text())
    rows=json.loads((args.directory/'inputs.json').read_text())
    if args.smoke:rows=[r for r in rows if r['sample_id'] in protocol['smoke_ids']]
    assert rows and len(rows)==len({r['sample_id'] for r in rows})
    assert all(set(r)=={'sample_id','near','wide','footprint_m2'} and r['footprint_m2']>0 for r in rows)
    assert all((args.directory/r[k]).is_file() for r in rows for k in ['near','wide'])
    target=args.directory/('smoke.jsonl' if args.smoke else 'predictions.jsonl')
    if target.exists():raise FileExistsError(target)
    assert args.revision==protocol['revision'] and torch.cuda.is_available()
    torch.manual_seed(SEED);torch.set_num_threads(4)
    start=time.perf_counter()
    processor=AutoProcessor.from_pretrained(args.model,revision=args.revision)
    model=AutoModelForImageTextToText.from_pretrained(args.model,revision=args.revision,
        dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    run=dict(status='running',n=len(rows),variants=[v['name'] for v in protocol['variants']],
        model=args.model,revision=args.revision,dtype=str(model.dtype),device=torch.cuda.get_device_name(),
        versions={p:importlib.metadata.version(p) for p in ['torch','transformers','Pillow']},
        protocol_sha256=hashlib.sha256((args.directory/'PROTOCOL.json').read_bytes()).hexdigest(),
        load_seconds=time.perf_counter()-start,max_new_tokens=160,seed=SEED)
    run_path=target.with_suffix('.run.json');run_path.write_text(json.dumps(run,indent=2)+'\n')
    with target.open('x') as handle,torch.inference_mode():
        for row in rows:
            for variant in protocol['variants']:
                t0=time.perf_counter()
                ims=[Image.open(args.directory/row[k]).convert('RGB') for k in variant['views']]
                content=[dict(type='image',image=im) for im in ims]
                content.append(dict(type='text',text=variant['prompt'].format(area=row['footprint_m2'])))
                inputs=processor.apply_chat_template([dict(role='user',content=content)],tokenize=True,
                    add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
                torch.cuda.reset_peak_memory_stats()
                output=model.generate(**inputs,max_new_tokens=160,do_sample=False);torch.cuda.synchronize()
                raw=processor.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
                result=dict(sample_id=row['sample_id'],variant=variant['name'],raw=raw,**parse_answer(raw,variant),
                    seconds=time.perf_counter()-t0,input_tokens=inputs['input_ids'].shape[1],
                    output_tokens=output.shape[1]-inputs['input_ids'].shape[1],image_grid_thw=inputs['image_grid_thw'].tolist(),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                handle.write(json.dumps(result)+'\n');handle.flush()
                print(json.dumps({k:v for k,v in result.items() if k!='raw'}),flush=True)
    run['status']='complete';run_path.write_text(json.dumps(run,indent=2)+'\n')


def score(rows, labels):
    rows=[r for r in rows if r['reference_use'] in labels]
    if not rows:return dict(n=0)
    columns=CLASSES+['unknown','invalid']
    matrix=[[sum(r['reference_use']==truth and r['pred_type']==p for r in rows) for p in columns] for truth in labels]
    per={}
    for truth,line in zip(labels,matrix):
        tp=line[columns.index(truth)];support=sum(line);predicted=sum(r['pred_type']==truth for r in rows)
        per[truth]=dict(support=support,precision=tp/predicted if predicted else 0,
                       recall=tp/support if support else None,
                       f1=2*tp/(support+predicted) if support else None)
    correct=sum(r['reference_use']==r['pred_type'] for r in rows)
    answered=sum(r['pred_type'] in CLASSES for r in rows)
    return dict(n=len(rows),correct=correct,accuracy=correct/len(rows),
        macro_f1=statistics.mean(v['f1'] for v in per.values() if v['support']),
        balanced_accuracy=statistics.mean(v['recall'] for v in per.values() if v['support']),
        coverage=answered/len(rows),selective_accuracy=correct/answered if answered else None,
        unknown=sum(r['pred_type']=='unknown' for r in rows),invalid=sum(r['pred_type']=='invalid' for r in rows),
        per_class=per,confusion=matrix,rows=labels,columns=columns)


def evaluate(directory):
    protocol=json.loads((directory/'PROTOCOL.json').read_text())
    refs={r['sample_id']:r for r in json.loads((directory/'reference.json').read_text())}
    predictions=[json.loads(l) for l in (directory/'predictions.jsonl').read_text().splitlines()]
    assert len(predictions)==len({(r['sample_id'],r['variant']) for r in predictions})==len(refs)*len(protocol['variants'])
    results={}
    for variant in protocol['variants']:
        pp=[r for r in predictions if r['variant']==variant['name']]
        assert {r['sample_id'] for r in pp}==set(refs)
        for p in pp:assert parse_answer(p['raw'],variant)=={k:p[k] for k in ['pred_type','pred_role','evidence','parse_error']}
        rows=[dict(refs[p['sample_id']],**p) for p in pp]
        strata={'all_proxy_labels':rows}
        for domain in ['harris','sandiego']:
            rr=[r for r in rows if r['domain']==domain]
            strata[domain]=rr
            strata[domain+'_official_corroborated']=[r for r in rr if r['reference_grade'].startswith('A')]
            for label in sorted({r['reference_grade'] for r in rr}):strata[domain+'_'+label]=[r for r in rr if r['reference_grade']==label]
        result={s:dict(five=score(rr,CLASSES),shared_three=score(rr,CLASSES[:3])) for s,rr in strata.items()}
        result['runtime']=dict(n=len(pp),invalid=sum(p['parse_error'] is not None for p in pp),
            mean_seconds=statistics.mean(p['seconds'] for p in pp),median_seconds=statistics.median(p['seconds'] for p in pp),
            max_reserved_gib=max(p['peak_reserved_gib'] for p in pp),unknown=sum(p['pred_type']=='unknown' for p in pp),
            predicted_types=dict(Counter(p['pred_type'] for p in pp)),predicted_roles=dict(Counter(str(p['pred_role']) for p in pp)))
        results[variant['name']]=result
    (directory/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({v:{d:results[v][d] for d in ['harris','sandiego']} for v in results},indent=2))


def self_check():
    for v in VARIANTS:
        assert '100.0' in prompt_for(v).format(area=100)
        a=dict(type='commercial',evidence='Visible premises.')
        if v['roles']:a['role']='primary'
        assert parse_answer(json.dumps(a),v)['pred_type']=='commercial'
        a['type']='institutional'
        assert (parse_answer(json.dumps(a),v)['pred_type']=='institutional') == (len(v['classes'])==5)
        a['type']='unknown'
        assert (parse_answer(json.dumps(a),v)['pred_type']=='unknown')==v['unknown']
        assert parse_answer('not JSON',v)['pred_type']=='invalid'
    rows=[dict(reference_use='commercial',pred_type='commercial'),dict(reference_use='commercial',pred_type='unknown'),
          dict(reference_use='other',pred_type='invalid')]
    s=score(rows,CLASSES);assert s['accuracy']==1/3 and s['coverage']==1/3 and s['selective_accuracy']==1
    assert s['per_class']['commercial']['recall']==.5 and s['invalid']==1 and s['unknown']==1
    assert score(rows,CLASSES[:3])['n']==2
    print('PASS variant schemas, unknown/invalid denominators, shared-class scope and prompt formatting')


if __name__=='__main__':
    main()
