"""Verify source parity, complete visible masks, and unchanged augmented geometry."""
import hashlib
import importlib.util
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import common
from target_closeup import crop_bounds, self_check


def main():
    root = Path(__file__).resolve().parent
    for name, expected in read(root/'CODE_SHA256.json').items(): assert sha(root/name) == expected, name
    for name, expected in read(root/'INPUT_SHA256.json').items(): assert sha(Path(name)) == expected, name
    for name, expected in read(root/'MODEL_SHA256.json').items(): assert sha(Path(name)) == expected, name
    protocol = read(root/'PROTOCOL.json'); rows = read(root/'DATASET.json')
    assert sha(root/'DATASET.json') == protocol['dataset_sha256']
    parent = Path('/stage3/stage2a_la_hard_ablation_20261030')
    original = read(parent/'C_positive_negative/DATASET.json')
    assert [{k:v for k,v in r.items() if k not in ['target_closeup','closeup_gsd']} for r in rows] == original
    assert Counter(r['split'] for r in rows) == {'train':14710,'val':438,'test':3332}
    assert read(root/'LOWRISE_SAMPLING.json')['factors'] == read(parent/'C_positive_negative/LOWRISE_SAMPLING.json')['factors']
    self_check()
    audits = []; summary = defaultdict(Counter)
    for r in rows:
        with Image.open(r['mask_path']) as image: mask = image.convert('L')
        with Image.open(r['crop_path']) as image: assert image.size == mask.size
        bounds = crop_bounds(mask, r['closeup_gsd'])
        before = np.asarray(mask); after = np.asarray(mask.crop(bounds))
        assert np.count_nonzero(before) == np.count_nonzero(after) > 0
        bbox = mask.getbbox(); edge = bbox[0] == 0 or bbox[1] == 0 or bbox[2] == mask.width or bbox[3] == mask.height
        changed = bounds != (0,0,mask.width,mask.height)
        key = r['domain']+':'+r['split']; summary[key]['n'] += 1
        summary[key]['zoomed'] += int(changed); summary[key]['edge_fallback'] += int(edge)
        audits.append(dict(sample_id=r['sample_id'], split=r['split'], bounds=bounds,
            edge_fallback=edge, target_retention=1., linear_enlargement=mask.width/(bounds[2]-bounds[0]),
            footprint_m2=r['footprint_m2']))
    spec = importlib.util.spec_from_file_location('parent_common', parent/'common.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    chosen = [i for domain in ['source','la','la_test'] for c in range(3)
              for i in [next(i for i,r in enumerate(rows) if r['domain']==domain and int(r['type_class'])==c)]]
    cfg = dict(img_size=224, input_mode=protocol['cnn_config']['input_mode'],
               geometry_cols=protocol['cnn_config']['geometry_cols'].split(','),
               require_ring_mask=False, aug_hflip=.5, aug_rot90=.25)
    changes = 0
    for training in [False, True]:
        baseline = old.Stage2aDataset(original, train=training, **cfg)
        unchanged = common.Stage2aDataset(original, train=training, **cfg)
        closeup = common.Stage2aDataset(rows, train=training, **cfg)
        for repeat in range(8):
            for i in chosen:
                samples=[]
                for ds in [baseline, unchanged, closeup]:
                    common.seed_all(42+repeat); samples.append(ds[i])
                a,b,c=samples
                for key in ['x','mask','geometry']: assert torch.equal(a[key],b[key]), (i,key)
                assert torch.equal(a['geometry'],c['geometry']), i
                assert c['x'].shape == (4,224,224) and c['mask'].sum()>0
                changes += int(not torch.equal(a['x'],c['x']))
    assert changes > 0 and not torch.cuda.is_available()
    save(root/'CROP_AUDIT.json', audits)
    report = dict(passed=True, n=len(rows), summary=summary, unchanged_source_hashes=len(read(root/'INPUT_SHA256.json')),
        fixed_runtime_tensor_parity=True, augmented_geometry_parity=True, all_visible_target_pixels_retained=True,
        changed_tensor_checks=changes, free_gib=shutil.disk_usage(root).free/2**30, torch=torch.__version__)
    assert report['free_gib']>5
    save(root/'STAGED.json', report); print('STAGED',json.dumps(report),flush=True)


def read(path): return json.loads(path.read_text())
def save(path,value): path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


if __name__=='__main__': main()
