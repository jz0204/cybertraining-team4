import sys,importlib.util,json,hashlib
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import functional as TF
s=Path('/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a');e=s/'la_augmentation_20261030/execution'
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
old=module(s/'la_adaptation_20261029/execution/common.py','old');fixed=module(e/'common.py','fixed');canonical=module(Path('handoff/stage2a_la/scripts/stage2a/common.py'),'canonical')
r=json.loads((e/'DATASET.json').read_text())[0]
for k in ['crop_path','mask_path']:r[k]=str(s/r[k].split('/stage3/')[1].removeprefix('stage2a_'))
rgb=Image.open(r['crop_path']).convert('RGB');mask=Image.open(r['mask_path']).convert('L')
def signature(pair):return hashlib.sha256(pair[0].tobytes()+pair[1].tobytes()).hexdigest()
allowed={signature((TF.rotate(TF.hflip(rgb) if f else rgb,k*90),TF.rotate(TF.hflip(mask) if f else mask,k*90))) for f in [0,1] for k in range(4)}
results={}
for name,m in [('old',old),('fixed',fixed),('canonical',canonical)]:
 d=m.Stage2aDataset([r],train=True,aug_hflip=.5,aug_rot90=.25,require_ring_mask=False,require_targets=False)
 def sequence():return [signature(d._augment(rgb,mask,None,0)) for _ in range(64)]
 torch.manual_seed(42);a=sequence();torch.manual_seed(42);assert sequence()==a
 assert set(a)<=allowed
 results[name]=len(set(a));assert results[name]==1 if name=='old' else results[name]>1
 d.train=False;assert signature(d._augment(rgb,mask,None,0))==signature((rgb,mask))
print(results);(e/'AUGMENTATION_TEST.json').write_text(json.dumps({'passed':True,'distinct_views_in_64_draws':results,'seed_reproducible':True,'rgb_mask_joint_transform':True,'no_color_change':True,'evaluation_unchanged':True},indent=2)+'\n')
