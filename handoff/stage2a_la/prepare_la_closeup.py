"""Freeze a paired close-up experiment from C, preserving every existing input file."""
import ast
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

STAGE = Path('/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a')
PARENT = STAGE/'la_hard_ablation_20261030/execution'
ROOT = STAGE/'la_closeup_20261031/execution'
REMOTE = '/stage3/stage2a_la_closeup_20261031'
HERE = Path(__file__).resolve().parent


def main():
    assert not ROOT.exists(), 'Never overwrite a frozen experiment'
    ROOT.mkdir(parents=True)
    old = json.loads((PARENT/'C_positive_negative/DATASET.json').read_text())
    protocol = json.loads((PARENT/'C_positive_negative/PROTOCOL.json').read_text())
    assert sha(PARENT/'C_positive_negative/DATASET.json') == protocol['dataset_sha256']
    rows = []
    for original in old:
        row = dict(original)
        if row['sample_id'].startswith('la_adapt_'):
            source = STAGE/'la_adaptation_20261029'
        elif row['sample_id'].startswith('la_expand_'):
            source = STAGE/'la_hard_expansion_20261030'
        else: source = None
        gsd = json.loads((source/'image_metadata'/(row['sample_id']+'.json')).read_text())['source_gsd'] if source else float(row['gsd'])
        row.update(target_closeup=True, closeup_gsd=gsd)
        assert {k:v for k,v in row.items() if k not in ['target_closeup','closeup_gsd']} == original
        rows.append(row)
    save(ROOT/'DATASET.json', rows)
    save(ROOT/'PROTOCOL.json', dict(protocol, dataset_sha256=sha(ROOT/'DATASET.json'),
        parent_dataset_sha256=protocol['dataset_sha256'],
        experiment='Same-pixel target close-up; full visible mask, 40m minimum, 1.4x bounding-box extent; edge targets unchanged',
        geometry_policy='All four features computed from the original augmented mask; physical area unchanged',
        baseline='Frozen C_positive_negative, two seeds, identical training recipe and sampling'))
    sampling = json.loads((PARENT/'C_positive_negative/LOWRISE_SAMPLING.json').read_text())
    save(ROOT/'LOWRISE_SAMPLING.json', dict(sampling, dataset_sha256=sha(ROOT/'DATASET.json')))
    common = (PARENT/'common.py').read_text()
    anchor = '        rgb, mask, ring = self._augment(rgb, mask, ring, local_idx)\n'
    assert common.count(anchor) == 1
    common = common.replace(anchor, anchor + '''        geometry_mask = mask
        if row.get("target_closeup"):
            from target_closeup import crop_bounds
            assert not needs_ring
            bounds = crop_bounds(mask, float(row["closeup_gsd"]))
            rgb, mask = rgb.crop(bounds), mask.crop(bounds)
''')
    anchor = 'mask_for_geometry = np.asarray(mask) if needs_mask_geometry(row, self.geometry_cols) else None'
    assert common.count(anchor) == 1
    common = common.replace(anchor, anchor.replace('np.asarray(mask)', 'np.asarray(geometry_mask)'))
    (ROOT/'common.py').write_text(common)
    trainer = (PARENT/'train_la_adaptation.py').read_text()
    # The original-image parity probe belongs to fixed input, not the changed view.
    anchor = "    if a.encoder=='cnn':\n        probe="
    assert trainer.count(anchor) == 1
    trainer = trainer.replace(anchor, "    if a.encoder=='cnn' and not rows[0].get('target_closeup'):\n        probe=")
    (ROOT/'train_la_adaptation.py').write_text(trainer)
    for name in ['target_closeup.py', 'check_la_closeup.py']: shutil.copyfile(HERE/name, ROOT/name)
    hashes = json.loads((PARENT/'PARENT_INPUT_SHA256.json').read_text())
    hashes.update({'/stage3/stage2a_la_hard_ablation_20261030/'+k:v for k,v in json.loads((PARENT/'INPUT_SHA256.json').read_text()).items()})
    save(ROOT/'INPUT_SHA256.json', hashes)
    shutil.copyfile(PARENT/'MODEL_SHA256.json', ROOT/'MODEL_SHA256.json')
    for name in ['STAGE_POD.json','JOB_cnn.json','launch.py','restore.py','check_restore.py']:
        value = (PARENT/name).read_text().replace('stage2a-la-hard-ablation', 'stage2a-la-closeup').replace('stage2a_la_hard_ablation', 'stage2a_la_closeup').replace('20261030','20261031')
        (ROOT/name).write_text(value)
    # Preserve the already verified mounts: only this experiment's names and paths change.
    for name in ['STAGE_POD.json','JOB_cnn.json']:
        value = json.loads((PARENT/name).read_text())
        value['metadata']['name'] = value['metadata']['name'].replace('hard-ablation','closeup').replace('20261030','20261031')
        spec = value['spec']['template']['spec'] if name.startswith('JOB') else value['spec']
        for c in spec['containers']:
            c['command'] = [x.replace('/stage3/stage2a_la_hard_ablation_20261030', REMOTE) for x in c['command']]
            for env in c.get('env',[]): env['value'] = env['value'].replace('/stage3/stage2a_la_hard_ablation_20261030', REMOTE)
        save(ROOT/name, value)
    (ROOT/'worker.py').write_text('''import json,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent
assert sys.argv[1]=='cnn' and json.loads((p/'STAGED.json').read_text())['passed']
for seed in [42,43]:
    for action in ['witness','train']:
        with (p/f'{seed}_{action}.log').open('x') as log:
            subprocess.run([sys.executable,'-B','-u',str(p/'train_la_adaptation.py'),action,'--directory',str(p),'--encoder','cnn','--seed',str(seed),'--lowrise'],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',seed,action,flush=True)
print('BATCH_COMPLETE',flush=True)
''')
    save(ROOT/'FROZEN.json', dict(parent=str(PARENT), n=len(rows), new_labels=0, reserved_used=False,
        changed_fields=['target_closeup','closeup_gsd'], fixed_checkpoint_reuse=True,
        limits=['Clipped source roofs cannot be recovered by cropping; edge-touching masks retain the entire original crop.',
                'This isolates visible target scale, not information gained from wider source imagery.',
                'The existing LA cohort is development evidence, not untouched-city evidence.']))
    code = ['common.py','train_la_adaptation.py','target_closeup.py','check_la_closeup.py','worker.py',
            'DATASET.json','PROTOCOL.json','LOWRISE_SAMPLING.json','INPUT_SHA256.json','MODEL_SHA256.json','FROZEN.json']
    save(ROOT/'CODE_SHA256.json', {n:sha(ROOT/n) for n in code})
    controller = ['launch.py','restore.py','check_restore.py','JOB_cnn.json','STAGE_POD.json']
    save(ROOT/'SPEC_SHA256.json', dict(file_sha256={n:sha(ROOT/n) for n in controller}))
    for path in ROOT.glob('*.py'): ast.parse(path.read_text())
    for archive, names in [('inputs.tar', code+['CODE_SHA256.json']), ('controller.tar', controller+['SPEC_SHA256.json'])]:
        with tarfile.open(ROOT/archive, 'w') as tar:
            for name in names: tar.add(ROOT/name, arcname=name)
    print('FROZEN', ROOT, len(rows), 'unchanged images', len(hashes))


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value): path.write_text(json.dumps(value, indent=2)+'\n')


if __name__ == '__main__': main()
