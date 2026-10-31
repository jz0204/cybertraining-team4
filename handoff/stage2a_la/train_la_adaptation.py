"""Matched LA adaptation using the existing N2 dataset, model and metric helpers."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from common import (Stage2aDataset, build_model_from_config, classification_metrics,
                    collate_stage2a, load_state_dict_from_ckpt, macro_f1_from_cm, seed_all)

NAMES=['residential_small','residential_multi','commercial']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['witness','train'])
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--encoder',choices=['cnn','dino'],required=True)
    p.add_argument('--seed',type=int,choices=[42,43],default=42)
    p.add_argument('--lowrise',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    assert not a.lowrise or a.encoder=='cnn'
    a.run_family='cnn_lowrise' if a.lowrise else a.encoder
    protocol=json.loads((a.directory/'PROTOCOL.json').read_text())
    raw=(a.directory/'DATASET.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==protocol['dataset_sha256']
    rows=json.loads(raw);seed_all(a.seed)
    assert torch.cuda.is_available(), 'GPU required'
    model=make_model(a,protocol).cuda()
    if a.action=='witness':witness(a,protocol,rows,model)
    else:train(a,protocol,rows,model)


class DinoFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.encoder=AutoModel.from_pretrained('/dinov3-model',local_files_only=True).float()
        self.encoder.requires_grad_(False)
        layers=[m for m in self.encoder.modules() if type(m).__name__=='DINOv3ViTLayer']
        assert len(layers)==12
        for layer in layers[-2:]:layer.requires_grad_(True)
        self.encoder.norm.requires_grad_(True)

    def forward(self,x):
        tokens=self.encoder(pixel_values=x[:,:3]).last_hidden_state[:,-196:]
        return tokens.transpose(1,2).reshape(len(x),768,14,14)


def make_model(a,protocol):
    cfg=protocol['cnn_config'];assert cfg['mask_downsample_mode']=='area'
    model=build_model_from_config(cfg,pretrained=False)
    if a.encoder=='cnn':model.load_state_dict(load_state_dict_from_ckpt(protocol['initial_checkpoints'][str(a.seed)]))
    else:model.features=DinoFeatures()
    model.pop_head.requires_grad_(False)
    return model


def dataset(rows,protocol,train=False):
    cfg=protocol['cnn_config']
    return Stage2aDataset(rows,img_size=224,train=train,input_mode=cfg['input_mode'],
        geometry_cols=cfg['geometry_cols'].split(','),require_ring_mask=False,aug_hflip=.5,aug_rot90=.25)


def logits(model,batch):
    return model(batch['x'].cuda(),geometry=batch['geometry'].cuda(),mask=batch['mask'].cuda(),return_dict=True)['type_logits']


def predict(model,rows,protocol):
    model.eval();out=[]
    loader=DataLoader(dataset(rows,protocol),batch_size=32,num_workers=4,collate_fn=collate_stage2a)
    with torch.inference_mode():
        for batch in loader:out.append(logits(model,batch).softmax(-1).cpu().numpy())
    return np.concatenate(out)


def metrics(rows,probabilities):
    y=np.array([int(r['type_class']) for r in rows]);pred=probabilities.argmax(1)
    result=classification_metrics(y,probs=probabilities,class_names=NAMES)
    cm=np.zeros((3,3));weights=np.array([r.get('validation_weight',1.) for r in rows])
    np.add.at(cm,(y,pred),weights)
    result['sampling_weighted_macro_f1']=macro_f1_from_cm(cm)[0]
    result['sampling_weighted_confusion']=cm.tolist()
    return result


def predictions(rows,p):
    return [dict(sample_id=r['sample_id'],use=NAMES[int(v.argmax())],probabilities=v.tolist()) for r,v in zip(rows,p)]


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def train(a,protocol,rows,model):
    out=a.directory/'runs'/f'{a.run_family}_s{a.seed}';out.mkdir(parents=True,exist_ok=False)
    groups={s:[r for r in rows if r['split']==s] for s in ['train','val','test']}
    training=groups['train'];source=sum(r['domain']=='source' for r in training);local=len(training)-source
    assert source==13911 and local>0 and len(groups['test'])==3332
    weights=[.5/(source if r['domain']=='source' else local) for r in training]
    if 'la_class_mass' in protocol:
        assert not a.lowrise
        masses=protocol['la_class_mass'];counts=Counter(int(r['type_class']) for r in training if r['domain']=='la')
        assert len(masses)==3 and abs(sum(masses)-1.)<1e-12 and set(counts)=={0,1,2}
        weights=[.5/source if r['domain']=='source' else .5*masses[int(r['type_class'])]/counts[int(r['type_class'])] for r in training]
        for c in range(3):
            assert abs(sum(w for r,w in zip(training,weights) if r['domain']=='la' and int(r['type_class'])==c)-.5*masses[c])<1e-10
    sampling_sha=None
    if a.lowrise:
        raw=(a.directory/'LOWRISE_SAMPLING.json').read_bytes();sampling=json.loads(raw)
        assert sampling['dataset_sha256']==protocol['dataset_sha256']
        factors=sampling['factors'];sampling_sha=hashlib.sha256(raw).hexdigest()
        assert set(factors)=={r['sample_id'] for r in training if r['domain']=='la' and int(r['type_class'])==1}
        assert abs(sum(factors.values())-len(factors))<1e-8 and min(factors.values())>0
        # shortcut: resampling reuses four existing fourplex images; more verified buildings are still needed.
        weights=[w*factors.get(r['sample_id'],1.) for r,w in zip(training,weights)]
    rng=torch.Generator().manual_seed(a.seed)
    sampler=WeightedRandomSampler(weights,8192,replacement=True,generator=rng)
    loader=DataLoader(dataset(training,protocol,True),batch_size=32,sampler=sampler,num_workers=4,
                      collate_fn=collate_stage2a,pin_memory=True)
    if a.encoder=='cnn':params=[dict(params=[p for p in model.parameters() if p.requires_grad],lr=1e-5)]
    else:params=[dict(params=[p for p in model.features.parameters() if p.requires_grad],lr=2e-5),
        dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('features.')],lr=1e-3)]
    opt=torch.optim.AdamW(params,weight_decay=.01)
    trainable={n for n,p in model.named_parameters() if p.requires_grad}
    start=time.monotonic();torch.cuda.reset_peak_memory_stats();history=[];best=-1.;best_epoch=0
    initial=predict(model,groups['val'],protocol);score=metrics(groups['val'],initial)
    if a.encoder=='cnn':
        best=score['sampling_weighted_macro_f1']
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items() if k in trainable},out/'best.pt')
    save(out/'initial_validation.json',score)
    seed_all(a.seed)
    for epoch in range(1,7):
        model.train();total=0.;seen=0;draws=[]
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=logits(model,batch);loss=nn.functional.cross_entropy(z,batch['type_idx'].cuda())
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            opt.step();total+=loss.item()*len(z);seen+=len(z);draws.extend(batch['row_idx'].tolist())
        vp=predict(model,groups['val'],protocol);score=metrics(groups['val'],vp)
        entry=dict(epoch=epoch,loss=total/seen,validation=score,seconds=time.monotonic()-start,
            draw_sha256=hashlib.sha256(np.asarray(draws,dtype=np.int32).tobytes()).hexdigest())
        history.append(entry);save(out/'history.json',history)
        print('EPOCH',a.encoder,a.seed,json.dumps(entry),flush=True)
        if score['sampling_weighted_macro_f1']>best:
            best=score['sampling_weighted_macro_f1'];best_epoch=epoch
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items() if k in trainable},out/'best.pt')
    selected=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    assert set(selected)==trainable
    model.load_state_dict(selected,strict=False)
    vp=predict(model,groups['val'],protocol);tp=predict(model,groups['test'],protocol)
    save(out/'validation_predictions.json',predictions(groups['val'],vp))
    save(out/'test_predictions.json',predictions(groups['test'],tp))
    save(out/'RESULT.json',dict(encoder=a.encoder,seed=a.seed,best_epoch=best_epoch,validation=metrics(groups['val'],vp),
        test=metrics(groups['test'],tp),seconds=time.monotonic()-start,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        epochs=6,optimizer_updates=1536,dataset_sha256=protocol['dataset_sha256'],sampling_factor_sha256=sampling_sha,
        checkpoint_sha256=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest()))
    print('RUN_COMPLETE',a.encoder,a.seed,flush=True)


def witness(a,protocol,rows,model):
    chosen=[next(r for r in rows if r['split']=='train' and r['domain']=='la' and int(r['type_class'])==c) for c in range(3)]
    if 'witness_multifamily_id' in protocol:
        chosen[1]=next(r for r in rows if r['sample_id']==protocol['witness_multifamily_id'])
    batch=collate_stage2a([dataset(chosen,protocol)[i] for i in range(3)])
    assert batch['x'].shape==(3,4,224,224) and (batch['mask'].sum((1,2,3))>0).all()
    model.train();start=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        z=logits(model,batch);loss=nn.functional.cross_entropy(z,batch['type_idx'].cuda())
    loss.backward();torch.cuda.synchronize()
    assert z.shape==(3,3) and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert any(p.grad.abs().sum()>0 for p in model.features.parameters() if p.requires_grad)
    report=dict(encoder=a.encoder,seed=a.seed,device=torch.cuda.get_device_name(),loss=loss.item(),sample_ids=[r['sample_id'] for r in chosen],
        forward_backward_seconds=time.monotonic()-start,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    if a.encoder=='cnn':
        probe=[r for c in range(3) for r in [x for x in rows if x['split']=='test' and int(x['type_class'])==c][:12]]
        actual=predict(model,probe,protocol)
        expected=np.array([protocol['baseline_predictions'][str(a.seed)][r['sample_id']] for r in probe])
        difference=float(np.abs(actual-expected).max())
        assert difference<3e-4 and np.array_equal(actual.argmax(1),expected.argmax(1)),difference
        report['baseline_probe_n']=len(probe);report['baseline_probability_max_difference']=difference
    save(a.directory/f'WITNESS_{a.run_family}_{a.seed}.json',report)
    print('LA_ADAPTATION_GPU_WITNESS_PASS',json.dumps(report),flush=True)


if __name__=='__main__':main()
