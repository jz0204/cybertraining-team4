"""Matched city-held-out development trials with frozen or trainable encoders."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from common import _masked_avg_pool, build_image_feature_extractor, classification_metrics, seed_all

NAMES = ['residential_small','residential_multi','commercial','institutional','other']
REVISION = '5931719e67bbdb9737e363e781fb0c67687896bc'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['freeze','train','witness','self-check'])
    p.add_argument('--directory',type=Path,default=Path('/experiment'))
    p.add_argument('--encoder',choices=['convnext','dinov3'],default='convnext')
    p.add_argument('--run')
    a=p.parse_args()
    if a.action=='self-check':self_check();return
    if a.action=='witness':witness();return
    data=json.loads((a.directory/'DATASET.json').read_text())
    protocol=json.loads((a.directory/'PROTOCOL.json').read_text())
    assert hashlib.sha256((a.directory/'DATASET.json').read_bytes()).hexdigest()==protocol['dataset_sha256']
    torch.set_num_threads(4)
    if a.action=='freeze':freeze(a,data);return
    run=next(r for r in protocol['runs'] if r['name']==a.run)
    train(a,data,run)


def load_images(root,rows):
    images=[]; masks=[]
    for r in rows:
        pair=[]
        for v in ['near','wide']:
            path=root/r[v]
            assert hashlib.sha256(path.read_bytes()).hexdigest()==r[v+'_sha256']
            pair.append(np.asarray(Image.open(path).convert('RGB').resize((224,224),Image.Resampling.BICUBIC)).copy())
        images.append(np.stack(pair))
        path=root/r['near_mask']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['near_mask_sha256']
        mask=np.asarray(Image.open(path).resize((224,224),Image.Resampling.NEAREST)).copy()>0
        assert mask.any(),r['sample_id']
        masks.append(mask)
    return torch.from_numpy(np.stack(images)).permute(0,1,4,2,3),torch.from_numpy(np.stack(masks)).float().unsqueeze(1)


def image_batch(images,indices,device):
    x=images[indices].to(device).float()/255
    return (x-torch.tensor([.485,.456,.406],device=device).view(1,1,3,1,1))/torch.tensor([.229,.224,.225],device=device).view(1,1,3,1,1)


class Classifier(nn.Module):
    def __init__(self,dual,train_encoder,feature_dim=768):
        super().__init__()
        self.dual=dual
        self.encoder=build_image_feature_extractor('convnext_tiny',True,3)[0] if train_encoder else None
        self.head=nn.Sequential(nn.Linear(feature_dim*(2 if dual else 1)+4,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,5))

    def forward(self,x,mask,geometry):
        if self.encoder is not None:
            b=x.shape[0]
            views=2 if self.dual else 1
            feat=self.encoder(x[:,:views].reshape(b*views,3,224,224)).reshape(b,views,768,7,7)
            near=_masked_avg_pool(feat[:,0],mask,mode='area')
            x=torch.cat([near,feat[:,1].mean((2,3))],1) if self.dual else near
        else:
            x=x[:,:1536 if self.dual else 768]
        return self.head(torch.cat([x,geometry],1))


def freeze(a,rows):
    out=a.directory/(a.encoder+'_features.npz')
    assert not out.exists(),'Preserve feature cache'
    images,masks=load_images(a.directory,rows)
    seed_all(42)
    if a.encoder=='convnext':
        encoder=build_image_feature_extractor('convnext_tiny',True,3)[0].cuda().eval()
    else:
        from transformers import AutoModel
        encoder=AutoModel.from_pretrained('/dinov3-model',local_files_only=True,
                                         torch_dtype=torch.bfloat16).cuda().eval()
    features=[]; elapsed=[]; start=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(0,len(rows),32):
            ix=list(range(i,min(i+32,len(rows))));x=image_batch(images,ix,'cuda');b=len(ix)
            torch.cuda.synchronize();t=time.monotonic()
            if a.encoder=='convnext':
                f=encoder(x.reshape(b*2,3,224,224)).reshape(b,2,768,7,7)
            else:
                outputs=encoder(pixel_values=x.reshape(b*2,3,224,224))
                tokens=outputs.last_hidden_state[:,-196:]
                f=tokens.transpose(1,2).reshape(b,2,768,14,14)
            pooled=torch.cat([_masked_avg_pool(f[:,0],masks[ix].cuda(),mode='area'),f[:,1].mean((2,3))],1)
            assert torch.isfinite(pooled).all()
            features.append(pooled.float().cpu().numpy())
            torch.cuda.synchronize();elapsed.append((time.monotonic()-t)/b)
            if i%320==0:print('FEATURES',a.encoder,i+b,len(rows),flush=True)
    np.savez_compressed(out,features=np.concatenate(features),ids=np.array([r['sample_id'] for r in rows]))
    report=dict(encoder=a.encoder,revision=REVISION if a.encoder=='dinov3' else 'torchvision.IMAGENET1K_V1',
        n=len(rows),seconds=time.monotonic()-start,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        batch_amortized_seconds_p50=float(np.median(elapsed)),batch_amortized_seconds_p95=float(np.percentile(elapsed,95)),
        preprocessing='224px RGB, ImageNet normalization, area-weighted target pool + mean context pool',
        file_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FEATURES_COMPLETE',json.dumps(report),flush=True)


def train(a,rows,run):
    out=a.directory/'runs'/run['name'];out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(run,indent=2)+'\n')
    seed_all(run['seed'])
    fine=run['mode']=='finetune';device='cuda' if fine else 'cpu'
    if fine:
        values,masks=load_images(a.directory,rows)
    else:
        cache=np.load(a.directory/(run['encoder']+'_features.npz'))
        assert list(cache['ids'])==[r['sample_id'] for r in rows]
        values=torch.from_numpy(cache['features']);masks=torch.zeros((len(rows),1,1,1))
    ids={r['sample_id']:i for i,r in enumerate(rows)}
    split={k:[ids[x] for x in run[k]] for k in ['train','val','test']}
    assert all(split[k] for k in split)
    groups={k:{rows[i]['spatial_component'] for i in ix} for k,ix in split.items()}
    assert not groups['train'] & groups['val'] and not groups['train'] & groups['test'] and not groups['val'] & groups['test']
    assert all(rows[i]['domain']!=run['heldout_city'] for i in split['train']+split['val'])
    assert not any(rows[i]['role'] is not None for i in split['train']), 'Implement supervised role head when role labels arrive'
    y=torch.tensor([NAMES.index(r['label']) for r in rows])
    g=torch.tensor([r['geometry_features'] for r in rows])
    gmean=g[split['train']].mean(0);gstd=g[split['train']].std(0).clamp_min(1e-6);g=(g-gmean)/gstd
    feature_mean,feature_std=None,None
    if not fine:
        feature_mean=values[split['train']].mean(0);feature_std=values[split['train']].std(0).clamp_min(1e-6)
        values=(values-feature_mean)/feature_std
    model=Classifier(run['dual'],fine).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=run['lr'],weight_decay=.01)
    if fine:torch.cuda.reset_peak_memory_stats()
    best=-1;best_state=None;best_epoch=0;stale=0;history=[];start=time.monotonic()
    for epoch in range(1,run['epochs']+1):
        model.train();order=split['train'].copy();random.shuffle(order);total=0.;count=0
        for offset in range(0,len(order),run['batch_size']):
            ix=order[offset:offset+run['batch_size']]
            x=image_batch(values,ix,device) if fine else values[ix].to(device)
            mask=masks[ix].to(device)
            if fine and random.random()<.5:x=x.flip(-1);mask=mask.flip(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device,dtype=torch.bfloat16,enabled=fine):
                logits=model(x,mask,g[ix].to(device));loss=F.cross_entropy(logits,y[ix].to(device))
            assert torch.isfinite(loss),run['name']
            loss.backward();optimizer.step();total+=loss.item()*len(ix);count+=len(ix)
        val=predict(model,values,masks,g,split['val'],device,fine)
        metrics=classification_metrics(y[split['val']].numpy(),probs=val,class_names=NAMES)
        score=metrics['macro_f1'] if 'macro_f1' in metrics else metrics['type_macro_f1']
        history.append(dict(epoch=epoch,loss=total/count,val_macro_f1=score,elapsed_s=time.monotonic()-start))
        print('EPOCH',run['name'],json.dumps(history[-1]),flush=True)
        if score>best+1e-8:
            best=score;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        if stale>=run['patience']:break
    model.load_state_dict(best_state)
    val=predict(model,values,masks,g,split['val'],device,fine)
    test=predict(model,values,masks,g,split['test'],device,fine)
    thresholds={str(coverage):float(np.quantile(val.max(1),1-coverage,method='higher')) for coverage in [.8,.9]}
    predictions=[dict(sample_id=rows[i]['sample_id'],truth=rows[i]['label'],domain=rows[i]['domain'],
        evidence_grade=rows[i]['evidence_grade'],probabilities=p.tolist(),prediction=NAMES[int(p.argmax())],
        role='unknown',role_reason='no verified role supervision',
        accepted={c:bool(p.max()>=t) for c,t in thresholds.items()}) for i,p in zip(split['test'],test)]
    (out/'predictions.json').write_text(json.dumps(predictions,indent=2)+'\n')
    torch.save(dict(state_dict=best_state,
        geometry_mean=gmean,geometry_std=gstd,feature_mean=feature_mean,feature_std=feature_std,run=run),out/'model.pt')
    report=dict(best_epoch=best_epoch,val_macro_f1=best,thresholds=thresholds,seconds=time.monotonic()-start,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if fine else None,
        test=classification_metrics(y[split['test']].numpy(),probs=test,class_names=NAMES),history=history,
        role_status='untrained; unknown for all; not scored',weights_saved_dtype='float32')
    (out/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print('RUN_COMPLETE',run['name'],json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)


def predict(model,values,masks,g,indices,device,fine):
    model.eval();out=[]
    with torch.inference_mode(),torch.autocast(device,dtype=torch.bfloat16,enabled=fine):
        for offset in range(0,len(indices),32):
            ix=indices[offset:offset+32]
            x=image_batch(values,ix,device) if fine else values[ix].to(device)
            logits=model(x,masks[ix].to(device),g[ix].to(device))
            out.append(logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


def witness():
    seed_all(20261026);model=Classifier(True,True).cuda();model.train()
    x=torch.randn(2,2,3,224,224,device='cuda');m=torch.zeros(2,1,224,224,device='cuda');m[:,:,110,110]=1
    with torch.autocast('cuda',dtype=torch.bfloat16):
        y=model(x,m,torch.ones(2,4,device='cuda'));loss=F.cross_entropy(y,torch.tensor([0,4],device='cuda'))
    loss.backward();torch.cuda.synchronize()
    assert y.shape==(2,5) and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    metrics=classification_metrics([0,3,4],probs=np.eye(5)[[0,3,4]],class_names=NAMES)
    assert metrics['type_accuracy']==1. and metrics['confusion_matrix'][3][3]==1
    print('WITNESS_PASS',torch.cuda.get_device_name(),list(y.shape),loss.item(),flush=True)


def self_check():
    m=torch.zeros(1,1,224,224);m[:,:,110,110]=1
    pooled=_masked_avg_pool(torch.ones(1,3,7,7),m,mode='area')
    assert torch.allclose(pooled,torch.ones_like(pooled))
    model=Classifier(True,False);y=model(torch.randn(3,1536),None,torch.zeros(3,4))
    assert y.shape==(3,5)
    loss=F.cross_entropy(y,torch.tensor([0,3,4]));loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print('PASS area pooling small masks; five-class dual-view head backward',flush=True)


if __name__=='__main__':main()
