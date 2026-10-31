"""Rank isolated training candidates using the frozen augmentation ensemble."""
import hashlib
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

STAGE=Path('/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a')
EXPERIMENT=STAGE/'la_augmentation_20261030/execution'
ROOT=STAGE/'la_hard_candidates_20261030'
sys.path.insert(0,str(EXPERIMENT))
from common import build_model_from_config
from infer_la_full import predict


def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT)
    ROOT=parser.parse_args().directory
    torch.set_num_threads(4)
    cfg=json.loads((EXPERIMENT/'PROTOCOL.json').read_text())['cnn_config']
    candidates=json.loads((ROOT/'STABLE_CANDIDATES.json').read_text())
    records=[]
    for r in candidates:
        info=json.loads((ROOT/'image_metadata'/(r['sample_id']+'.json')).read_text())
        paths={k:ROOT/info['paths']['legacy_'+v] for k,v in [('crop_path','rgb'),('mask_path','mask')]}
        for path in paths.values():
            assert digest(path)==info['image_sha256'][str(path.relative_to(ROOT))]
        records.append(dict(sample_id=r['sample_id'],footprint_m2=r['footprint_m2'],
                            log_footprint_m2=math.log1p(r['footprint_m2']),**{k:str(v) for k,v in paths.items()}))
    probes=[r.copy() for r in json.loads((EXPERIMENT/'DATASET.json').read_text()) if r['split']=='test'][:4]
    for r in probes:
        for k in ['crop_path','mask_path']:
            r[k]=str(STAGE/r[k].split('/stage3/')[1].removeprefix('stage2a_'))
    values=[];witness={}
    for seed in [42,43]:
        run=EXPERIMENT/f'collected/runs/cnn_lowrise_s{seed}'
        assert digest(run/'best.pt')==json.loads((run/'RESULT.json').read_text())['checkpoint_sha256']
        model=build_model_from_config(cfg,pretrained=False)
        missing=model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=True),strict=False)
        assert not missing.unexpected_keys and set(missing.missing_keys)=={n for n in model.state_dict() if n.startswith('pop_head.')}
        model.eval()
        expected=json.loads((run/'test_predictions.json').read_text())[:4]
        assert [v['sample_id'] for v in expected]==[r['sample_id'] for r in probes]
        actual=predict(model,probes,cfg);reference=np.array([v['probabilities'] for v in expected])
        error=float(np.abs(actual-reference).max());assert error<3e-4,error
        witness[str(seed)]=dict(max_probe_error=error,checkpoint_sha256=digest(run/'best.pt'))
        values.append(predict(model,records,cfg));print('SCORED',seed,len(records),flush=True)
    mean=np.mean(values,axis=0)
    result=[dict(sample_id=r['sample_id'],reference_use=r['reference_use'],probabilities=p.tolist(),
                 seed_probabilities=[v[i].tolist() for v in values],
                 predicted_use=['residential_small','residential_multi','commercial'][int(p.argmax())])
            for i,(r,p) in enumerate(zip(candidates,mean))]
    result.sort(key=lambda r:-r['probabilities'][1])
    (ROOT/'MODEL_SCORES.json').write_text(json.dumps(dict(witness=witness,records=result,labels_unchanged=True),indent=2)+'\n')
    print('CANDIDATE_SCORES_COMPLETE',len(result),flush=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':main()
