"""Matched frozen/last-two-block DINOv3 use and role supervision."""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from common import _masked_avg_pool, seed_all
from reliable_transfer_common import USES, ROLES, load, save, score, validation_score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train','evaluate','witness'])
    p.add_argument('--directory', type=Path, default=Path('/stage3/stage2a_reliable_20261027_v2'))
    p.add_argument('--mode', choices=['frozen','finetune'], default='finetune')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fixed-epochs', action='store_true', help='Use the full epoch budget for matched data comparisons.')
    a = p.parse_args(); torch.set_num_threads(4); seed_all(a.seed)
    rows, protocol = load(a.directory)
    if a.action == 'witness':
        witness(a,rows); return
    out = a.directory / 'runs' / f'dino_{a.mode}_s{a.seed}'
    if a.action == 'train':
        train(a,rows,protocol,out)
    else:
        evaluate(a,rows,out)


class Dino(nn.Module):
    def __init__(self, fine):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained('/dinov3-model',local_files_only=True)
        self.encoder.requires_grad_(False)
        blocks = [m for m in self.encoder.modules() if type(m).__name__ == 'DINOv3ViTLayer']
        assert len(blocks)==12, 'Verify architecture before unfreezing'
        if fine:
            for block in blocks[-2:]: block.requires_grad_(True)
            self.encoder.norm.requires_grad_(True)
        self.head = nn.Sequential(nn.Linear(1536,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,7))

    def features(self,x,mask):
        b,_,_,side,_ = x.shape
        tokens = self.encoder(pixel_values=x.reshape(b*2,3,side,side)).last_hidden_state[:,-(side//16)**2:]
        maps = tokens.transpose(1,2).reshape(b,2,768,side//16,side//16)
        return torch.cat([_masked_avg_pool(maps[:,0],mask,mode='area'),maps[:,1].mean((2,3))],1)

    def forward(self,x,mask):
        return self.head(self.features(x,mask))


def images(root,rows):
    rgb, masks = [], []
    for r in rows:
        rgb.append(np.stack([np.asarray(Image.open(root/r[v+'_rgb']).convert('RGB').resize((448,448),Image.Resampling.BICUBIC)) for v in ['near','wide']]))
        m = np.asarray(Image.open(root/r['near_mask']).resize((448,448),Image.Resampling.NEAREST)) > 0
        assert m.any(), r['sample_id']
        masks.append(m)
    return torch.from_numpy(np.stack(rgb)).permute(0,1,4,2,3), torch.from_numpy(np.stack(masks)).float().unsqueeze(1)


def batch(values,indices):
    x = values[indices].cuda().float()/255
    return (x-torch.tensor([.485,.456,.406],device='cuda').view(1,1,3,1,1))/torch.tensor([.229,.224,.225],device='cuda').view(1,1,3,1,1)


def sparse_loss(logits,y,role_weights):
    use = y[:,0]>=0
    loss = F.cross_entropy(logits[:,5:],y[:,1],weight=role_weights,reduction='sum')/len(y)
    if use.any():
        loss = loss + F.cross_entropy(logits[use,:5],y[use,0],reduction='sum')/len(y)
    return loss


def predict(model,values,masks,indices,rows,features=None):
    model.eval(); result=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for offset in range(0,len(indices),8):
            ix = indices[offset:offset+8]
            logits = model.head(features[ix].cuda()) if features is not None else model(batch(values,ix),masks[ix].cuda())
            up = logits[:,:5].float().softmax(-1).cpu().numpy()
            rp = logits[:,5:].float().softmax(-1).cpu().numpy()
            result += [dict(sample_id=rows[i]['sample_id'],use=USES[int(u.argmax())],role=ROLES[int(r.argmax())],
                            use_probabilities=u.tolist(),role_probabilities=r.tolist()) for i,u,r in zip(ix,up,rp)]
    return result


def train(a,all_rows,protocol,out):
    out.mkdir(parents=True,exist_ok=False)
    rows = [r for r in all_rows if r['split'] in ['train','val']]
    ix = {s:[i for i,r in enumerate(rows) if r['split']==s] for s in ['train','val']}
    model = Dino(a.mode=='finetune').cuda()
    values,masks = images(a.directory,rows)
    y = torch.tensor([[USES.index(r['reference_use']) if r['reference_use'] else -1, ROLES.index(r['reference_role'])] for r in rows],device='cuda')
    role_weights = torch.tensor(protocol['role_class_weights'],device='cuda')
    features = None; start=time.monotonic(); history=[]
    if a.mode=='frozen':
        model.eval(); ff=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for i in range(0,len(rows),8):
                jj=list(range(i,min(i+8,len(rows))));ff.append(model.features(batch(values,jj),masks[jj].cuda()).float().cpu())
        features=torch.cat(ff)
    groups=[dict(params=model.head.parameters(),lr=3e-4)]
    if a.mode=='finetune':groups.append(dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=1e-5))
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    best=-1; best_state=None; stale=0; best_epoch=0
    epochs=40 if a.mode=='frozen' else 15; patience=7 if a.mode=='frozen' else 3
    if a.fixed_epochs: patience=epochs
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,epochs+1):
        model.train(); model.encoder.eval(); order=ix['train'].copy();random.shuffle(order)
        total=0; optimizer.zero_grad(set_to_none=True)
        batches=[order[j:j+8] for j in range(0,len(order),8)]
        for step,jj in enumerate(batches):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if features is None:
                    x=batch(values,jj); m=masks[jj].cuda()
                    if random.random()<.5:x=x.flip(-1);m=m.flip(-1)
                    if random.random()<.5:x=x.flip(-2);m=m.flip(-2)
                    logits=model(x,m)
                else:logits=model.head(features[jj].cuda())
                loss=sparse_loss(logits,y[jj],role_weights)
            assert torch.isfinite(loss)
            # Final short accumulation group retains its actual denominator.
            accum=min(2,len(batches)-(step//2)*2)
            (loss/accum).backward();total+=loss.item()*len(jj)
            if (step+1)%2==0 or step+1==len(batches):
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
        predictions=predict(model,values,masks,ix['val'],rows,features)
        metrics=score([rows[i] for i in ix['val']],predictions); quality=validation_score(metrics)
        history.append(dict(epoch=epoch,loss=total/len(order),selection_score=quality,validation=metrics,seconds=time.monotonic()-start))
        print('EPOCH',a.mode,a.seed,epoch,'loss',total/len(order),'val',quality,flush=True)
        if quality>best+1e-8:
            best=quality;best_epoch=epoch;stale=0
            best_state={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        else:stale+=1
        if stale>=patience:break
    model.load_state_dict(best_state,strict=False)
    val=predict(model,values,masks,ix['val'],rows,features)
    known=[p for i,p in zip(ix['val'],val) if rows[i]['reference_use']]
    threshold=float(np.quantile([max(p['use_probabilities']) for p in known],.2,method='higher'))
    torch.save(dict(trainable_state=best_state,mode=a.mode,seed=a.seed,threshold=threshold),out/'model.pt')
    save(out/'validation_predictions.json',val)
    save(out/'TRAINING.json',dict(best_epoch=best_epoch,selection_score=best,history=history,
        train_ids=[rows[i]['sample_id'] for i in ix['train']],val_ids=[rows[i]['sample_id'] for i in ix['val']],
        seconds=time.monotonic()-start,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),threshold_80coverage=threshold,
        fixed_epochs=a.fixed_epochs))
    print('TRAIN_COMPLETE',str(out),flush=True)


def evaluate(a,rows,out):
    checkpoint=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    model=Dino(a.mode=='finetune').cuda();model.load_state_dict(checkpoint['trainable_state'],strict=False)
    selected=[r for r in rows if r['split']=='test']; values,masks=images(a.directory,selected)
    start=time.monotonic(); predictions=predict(model,values,masks,list(range(len(selected))),selected)
    for p in predictions:
        p['accepted_80']=max(p['use_probabilities'])>=checkpoint['threshold']
        p['selective_use']=p['use'] if p['accepted_80'] else 'unknown'
    save(out/'test_predictions.json',predictions)
    save(out/'TEST.json',dict(seconds=time.monotonic()-start,
         cities={c:score([r for r in selected if r['city']==c], [p for r,p in zip(selected,predictions) if r['city']==c]) for c in ['cambridge','seattle']}))
    print('TEST_COMPLETE',str(out),flush=True)


def witness(a,rows):
    seed_all(20261027);model=Dino(True).cuda().train()
    rr=[next(r for r in rows if r['split']=='train' and r['reference_use']=='residential_multi'),
        next(r for r in rows if r['split']=='train' and r['reference_role']=='auxiliary')]
    values,masks=images(a.directory,rr);y=torch.tensor([[1,0],[-1,1]],device='cuda')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        output=model(batch(values,[0,1]),masks.cuda());loss=sparse_loss(output,y,torch.ones(2,device='cuda'))
    loss.backward();assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step();model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):expected=model(batch(values,[0,1]),masks.cuda())
    path=a.directory/'dino_witness.pt';torch.save({n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},path)
    model.load_state_dict(torch.load(path,weights_only=True),strict=False)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actual=model(batch(values,[0,1]),masks.cuda())
    assert torch.equal(expected,actual)
    # Missing auxiliary use must contribute exactly zero gradient to the use head.
    logits=torch.randn(2,7,device='cuda',requires_grad=True);sparse_loss(logits,y,torch.ones(2,device='cuda')).backward()
    assert logits.grad[1,:5].abs().sum()==0
    print('DINO_TRANSFER_WITNESS_PASS',torch.cuda.get_device_name(),list(output.shape),loss.item(),flush=True)


if __name__=='__main__':main()
