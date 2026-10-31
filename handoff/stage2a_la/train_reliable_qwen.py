"""Sparse-label QLoRA and a matched NF4 control on the same transfer dataset."""

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from common import seed_all
from reliable_transfer_common import USES, ROLES, load, save, score, validation_score

PROMPT = '''Classify the actual use and role of the cyan-outlined building. Image 1 is a close view;
image 2 is the wider surroundings of the SAME target. Judge the individual structure.
Uses: residential_small = a building with 1-3 dwellings; residential_multi = an apartment
building with at least 4 dwellings; commercial = shops, offices, restaurants, lodging or
business services; institutional = schools, worship, hospitals, government or civic services;
other = a known industrial, warehouse, utility, agricultural or supporting storage/parking use.
Other does not mean uncertain. A pitched roof or small size alone does not establish housing.
Roles: primary = a main occupied or functional building; auxiliary = a supporting structure,
such as a detached garage, shed or carport. Classify role separately from the host site's use.
Use unknown for a field when visible evidence is insufficient. Do not invent exact dwelling
counts or copy a neighbor's use. Return only JSON with role and type, using the exact names:
{"role":"primary|auxiliary|unknown","type":"residential_small|residential_multi|commercial|institutional|other|unknown"}.'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','evaluate','zero','witness','self-check'])
    p.add_argument('--directory',type=Path,default=Path('/stage3/stage2a_reliable_20261027_v2'))
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();torch.set_num_threads(4);seed_all(a.seed)
    if a.action=='self-check':self_check();return
    rows,protocol=load(a.directory)
    if a.action=='witness':witness(a,rows,protocol);return
    processor,base=base_model()
    out=a.directory/'runs'/('qwen_nf4_zero' if a.action=='zero' else f'qwen_qlora_s{a.seed}')
    if a.action=='train':
        train(a,rows,protocol,processor,base,out)
    else:
        if a.action=='zero':out.mkdir(parents=True,exist_ok=False)
        model=evaluation_model(base,out,zero=a.action=='zero')
        selected=[r for r in rows if r['split'] in ['val','test']]
        predictions=infer(a.directory,selected,processor,model)
        save(out/'evaluation_predictions.json',predictions)
        save(out/'EVALUATION.json',{c:score([r for r in selected if r['city']==c],
             [p for r,p in zip(selected,predictions) if r['city']==c]) for c in ['nyc','cambridge','seattle']})
        print('EVALUATION_COMPLETE',str(out),flush=True)


def base_model():
    from transformers import AutoModelForImageTextToText,AutoProcessor,BitsAndBytesConfig
    processor=AutoProcessor.from_pretrained('/qwen-model',local_files_only=True,min_pixels=128*128,max_pixels=512*512)
    processor.tokenizer.padding_side='left'
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    model=AutoModelForImageTextToText.from_pretrained('/qwen-model',local_files_only=True,
        dtype=torch.bfloat16,device_map={'':'cuda:0'},quantization_config=quant,attn_implementation='sdpa')
    return processor,model


def adapter(base):
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,
                                        gradient_checkpointing_kwargs={'use_reentrant':False})
    config=LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',
                      target_modules=r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$')
    model=get_peft_model(base,config)
    trainable=[name for name,p in model.named_parameters() if p.requires_grad]
    assert trainable and all('language_model' in n and 'lora_' in n for n in trainable)
    model.config.use_cache=False
    return model


def evaluation_model(base,out,zero=False):
    from peft import PeftModel,prepare_model_for_kbit_training
    # Match the non-quantized parameter precision used for checkpoint selection.
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
    return base if zero else PeftModel.from_pretrained(base,out/'adapter')


def message(root,row):
    return dict(role='user',content=[dict(type='image',image=Image.open(root/row[v+'_marked']).convert('RGB')) for v in ['near','wide']]
                +[dict(type='text',text=PROMPT)])


def answer_weights(tokenizer,row,role_weights):
    answer=json.dumps(dict(role=row['reference_role'] or 'unknown',type=row['reference_use'] or 'unknown'),separators=(',',':'))
    encoded=tokenizer(answer,add_special_tokens=False,return_offsets_mapping=True)
    weights=[0.] * len(encoded['input_ids']); occupied=set()
    for field,truth,weight in [('role',row['reference_role'],role_weights[ROLES.index(row['reference_role'])]),
                               ('type',row['reference_use'],1.)]:
        match=re.search('"'+field+'":"([^"]+)"',answer);start,end=match.span(1)
        positions=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if a<end and b>start]
        assert positions
        occupied.update(positions)
        if truth is not None:
            for i in positions:weights[i]=weight/len(positions)
    ids=encoded['input_ids']+[tokenizer.convert_tokens_to_ids('<|im_end|>')]
    weights.append(0.)
    formatting=[i for i in range(len(weights)) if i not in occupied]
    for i in formatting:weights[i]=.1/len(formatting)
    return ids,weights,answer


def training_input(root,row,processor,role_weights,device='cuda'):
    values=processor.apply_chat_template([message(root,row)],tokenize=True,add_generation_prompt=True,
                                          return_dict=True,return_tensors='pt')
    ids,weights,answer=answer_weights(processor.tokenizer,row,role_weights)
    values['input_ids']=torch.cat([values['input_ids'],torch.tensor([ids])],dim=1)
    values['attention_mask']=torch.ones_like(values['input_ids'])
    values['mm_token_type_ids']=torch.cat([values['mm_token_type_ids'],torch.zeros((1,len(ids)),dtype=values['mm_token_type_ids'].dtype)],dim=1)
    assert values['mm_token_type_ids'].shape==values['input_ids'].shape
    assert values['input_ids'].shape[1]<=1600, 'Do not truncate target/context to fit budget'
    return values.to(device),torch.tensor(ids,device=device),torch.tensor(weights,device=device),answer


def loss_for(model,values,ids,weights):
    logits=model(**values,logits_to_keep=len(ids)+1,use_cache=False).logits[0,:-1]
    assert logits.shape[0]==len(ids)
    active=weights>0
    ce=F.cross_entropy(logits[active].float(),ids[active],reduction='none')
    return (ce*weights[active]).sum()/2.1


def parse(raw):
    try:
        cleaned=re.sub(r'^```(?:json)?\s*|\s*```$','',raw.strip())
        result=json.loads(cleaned)
        assert isinstance(result,dict)
        return dict(use=result.get('type') if result.get('type') in USES+['unknown'] else 'invalid',
                    role=result.get('role') if result.get('role') in ROLES+['unknown'] else 'invalid')
    except (ValueError,AssertionError):
        return dict(use='invalid',role='invalid')


def infer(root,rows,processor,model):
    model.eval(); model.gradient_checkpointing_disable(); result=[]
    with torch.inference_mode():
        for start in range(0,len(rows),4):
            group=rows[start:start+4];t=time.monotonic()
            inputs=processor.apply_chat_template([[message(root,r)] for r in group],tokenize=True,
                add_generation_prompt=True,return_dict=True,return_tensors='pt',padding=True).to('cuda')
            output=model.generate(**inputs,max_new_tokens=48,do_sample=False,use_cache=True)
            torch.cuda.synchronize()
            raw=processor.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
            result += [dict(sample_id=r['sample_id'],raw=text,**parse(text),
                            amortized_seconds=(time.monotonic()-t)/len(group)) for r,text in zip(group,raw)]
            if start%40==0: print('INFER',start+len(group),'/',len(rows),flush=True)
    return result


def train(a,all_rows,protocol,processor,base,out):
    out.mkdir(parents=True,exist_ok=False)
    rows=[r for r in all_rows if r['split']=='train']; val=[r for r in all_rows if r['split']=='val']
    model=adapter(base);parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01)
    history=[];best=-1;best_epoch=0;start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,3):
        model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.config.use_cache=False;order=list(range(len(rows)));random.shuffle(order)
        optimizer.zero_grad(set_to_none=True);total=0
        for step,index in enumerate(order):
            values,ids,weights,_=training_input(a.directory,rows[index],processor,protocol['role_class_weights'])
            loss=loss_for(model,values,ids,weights)
            assert torch.isfinite(loss)
            accum=min(16,len(order)-(step//16)*16)
            (loss/accum).backward();total+=loss.item()
            if (step+1)%16==0 or step+1==len(order):
                torch.nn.utils.clip_grad_norm_(parameters,1);optimizer.step();optimizer.zero_grad(set_to_none=True)
            if (step+1)%40==0:
                print('STEP',a.seed,epoch,step+1,'/',len(order),'loss',total/(step+1),'seconds',time.monotonic()-start,flush=True)
        predictions=infer(a.directory,val,processor,model);metrics=score(val,predictions);quality=validation_score(metrics)
        history.append(dict(epoch=epoch,loss=total/len(order),selection_score=quality,validation=metrics,seconds=time.monotonic()-start))
        save(out/f'validation_epoch{epoch}.json',predictions)
        print('EPOCH',a.seed,epoch,'val',quality,flush=True)
        if quality>best+1e-8:
            best=quality;best_epoch=epoch;model.save_pretrained(out/'adapter',safe_serialization=True)
    save(out/'TRAINING.json',dict(best_epoch=best_epoch,selection_score=best,history=history,
        train_ids=[r['sample_id'] for r in rows],val_ids=[r['sample_id'] for r in val],
        seconds=time.monotonic()-start,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        trainable_parameters=sum(p.numel() for p in parameters),role_class_weights=protocol['role_class_weights']))
    print('TRAIN_COMPLETE',str(out),flush=True)


def witness(a,rows,protocol):
    seed_all(20261027);processor,base=base_model();model=adapter(base)
    row=next(r for r in rows if r['split']=='train' and r['reference_use']=='residential_multi')
    aux=next(r for r in rows if r['split']=='train' and r['reference_role']=='auxiliary')
    _,aw,answer=answer_weights(processor.tokenizer,aux,protocol['role_class_weights'])
    encoded=processor.tokenizer(answer,add_special_tokens=False,return_offsets_mapping=True)
    start,end=re.search('"type":"([^"]+)"',answer).span(1)
    assert all(aw[i]==0 for i,(x,y) in enumerate(encoded['offset_mapping']) if x<end and y>start)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    step_seconds=[]
    for item in [row,row,aux]:
        t=time.monotonic()
        values,ids,weights,_=training_input(a.directory,item,processor,protocol['role_class_weights'])
        loss=loss_for(model,values,ids,weights);loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
        step_seconds.append(time.monotonic()-t)
    model.eval();model.gradient_checkpointing_disable()
    with torch.no_grad():expected=model(**values,logits_to_keep=1,use_cache=False).logits.float()
    path=a.directory/'qwen_witness_adapter';model.save_pretrained(path,safe_serialization=True)
    from peft import PeftModel
    base=model.unload();model=PeftModel.from_pretrained(base,path).eval()
    with torch.no_grad():actual=model(**values,logits_to_keep=1,use_cache=False).logits.float()
    error=(expected-actual).abs().max().item();assert error<.05, error
    result=infer(a.directory,[row,aux],processor,model)
    save(a.directory/'QWEN_WITNESS.json',dict(loss=loss.item(),reload_max_error=error,
         forward_backward_step_seconds=step_seconds,
         input_tokens=values['input_ids'].shape[1],image_grid_thw=values['image_grid_thw'].tolist(),
         peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,predictions=result))
    print('QWEN_QLORA_WITNESS_PASS',torch.cuda.get_device_name(),loss.item(),'reload_error',error,flush=True)


def self_check():
    assert parse('{"role":"primary","type":"residential_multi"}')==dict(use='residential_multi',role='primary')
    assert parse('```json\n{"role":"auxiliary","type":"unknown"}\n```')==dict(use='unknown',role='auxiliary')
    assert parse('not json')==dict(use='invalid',role='invalid')
    assert parse('{"role":"guess","type":"commercial"}')==dict(use='commercial',role='invalid')
    assert all(x in PROMPT for x in USES+ROLES+['unknown'])
    print('PASS exact label parser, abstention, format errors and prompt vocabulary')


if __name__=='__main__':main()
