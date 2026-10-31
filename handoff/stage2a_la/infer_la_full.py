"""Evaluate frozen LA CNN checkpoints on every prepared product record."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import Stage2aDataset, build_model_from_config, collate_stage2a, load_state_dict_from_ckpt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=[42,43],required=True)
    args=parser.parse_args();root=args.directory;seed=str(args.seed)
    torch.set_num_threads(8)
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    assert sha(root/'DATASET.json')==protocol['dataset_sha256']
    assert sha(Path(__file__).with_name('common.py'))==protocol['common_sha256']
    for name,expected in protocol['frozen_files'].items():assert sha(root/name)==expected,name
    rows=json.loads((root/'DATASET.json').read_text());assert len(rows)==22024
    out=root/'runs'/('seed'+seed);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();images=json.loads((root/'INPUT_SHA256.json').read_text())
    for i,(path,expected) in enumerate(images.items()):
        assert sha(Path(path))==expected,path
        if (i+1)%12000==0:print('INPUT_HASH',seed,i+1,'/',len(images),flush=True)
    cfg=protocol['cnn_config'];checkpoint=protocol['checkpoints'][seed]
    for name in ['initial','adapted']:assert sha(Path(checkpoint[name]))==checkpoint[name+'_sha256']
    model=build_model_from_config(cfg,pretrained=False)
    model.load_state_dict(load_state_dict_from_ckpt(checkpoint['initial']));model.eval()
    probes=[r for sid in protocol['probe_ids'] for r in rows if r['sample_id']==sid]
    cached=json.loads((root/'OLD_PREDICTIONS.json').read_text())[seed]
    old_actual=predict(model,probes,cfg)
    old_expected=np.asarray([cached[r['building_uid']] for r in probes])
    old_error=float(np.abs(old_actual-old_expected).max())
    assert old_error<3e-4 and np.array_equal(old_actual.argmax(1),old_expected.argmax(1)),old_error
    recovered=[r for r in rows if r['input_status']=='recovered_pixel_geometry']
    save(out/'old_recovered.json',predictions(recovered,predict(model,recovered,cfg)))
    selected=torch.load(checkpoint['adapted'],map_location='cpu',weights_only=True)
    missing=model.load_state_dict(selected,strict=False)
    assert not missing.unexpected_keys and set(missing.missing_keys)=={n for n in model.state_dict() if n.startswith('pop_head.')}
    new_actual=predict(model,probes,cfg)
    expected=json.loads((root/'EXPECTED_PROBES.json').read_text())[seed]
    new_expected=np.asarray([expected[r['sample_id']] for r in probes]);new_error=float(np.abs(new_actual-new_expected).max())
    assert new_error<3e-4 and np.array_equal(new_actual.argmax(1),new_expected.argmax(1)),new_error
    save(out/'WITNESS.json',dict(seed=args.seed,n=len(probes),old_max_probability_error=old_error,new_max_probability_error=new_error,
        torch_version=torch.__version__,device='cpu',threads=8,verified_image_files=len(images),seconds=time.monotonic()-start))
    print('WITNESS_PASS',seed,old_error,new_error,flush=True)
    probabilities=predict(model,rows,cfg,out)
    save(out/'predictions.json',predictions(rows,probabilities))
    save(out/'RESULT.json',dict(seed=args.seed,n=len(rows),seconds=time.monotonic()-start,
        class_counts=np.bincount(probabilities.argmax(1),minlength=3).tolist(),
        dataset_sha256=protocol['dataset_sha256'],checkpoint_sha256=checkpoint['adapted_sha256'],
        predictions_sha256=sha(out/'predictions.json'),old_recovered_sha256=sha(out/'old_recovered.json')))
    print('FULL_INFERENCE_COMPLETE',seed,len(rows),flush=True)


def predict(model,rows,cfg,progress=None):
    dataset=Stage2aDataset(rows,img_size=224,train=False,input_mode=cfg['input_mode'],
        geometry_cols=cfg['geometry_cols'].split(','),require_ring_mask=False,require_targets=False)
    loader=DataLoader(dataset,batch_size=32,num_workers=2,collate_fn=collate_stage2a)
    values=[];done=0;start=time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            logits=model(batch['x'],geometry=batch['geometry'],mask=batch['mask'],return_dict=True)['type_logits']
            values.append(logits.softmax(-1).numpy());done+=len(logits)
            if progress and (done%1024==0 or done==len(rows)):
                save(progress/'PROGRESS.json',dict(done=done,total=len(rows),seconds=time.monotonic()-start))
                print('PREDICT',done,'/',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    p=np.concatenate(values)
    assert p.shape==(len(rows),3) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5)
    return p


def predictions(rows,probabilities):
    return [dict(sample_id=r['sample_id'],probabilities=p.tolist()) for r,p in zip(rows,probabilities)]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
