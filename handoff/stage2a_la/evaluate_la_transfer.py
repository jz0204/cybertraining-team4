"""Apply frozen NYC checkpoints to LA without training or LA-based selection."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import seed_all
from reliable_transfer_common import load, save, score
import train_reliable_dino as dino
import train_reliable_qwen as qwen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['witness', 'dino', 'qwen_zero', 'qwen42', 'qwen43', 'self-check'])
    p.add_argument('--directory', type=Path, default=Path('/stage3/stage2a_la_transfer_20261027'))
    p.add_argument('--source', type=Path, default=Path('/stage3/stage2a_reliable_20261027_v3'))
    a = p.parse_args(); torch.set_num_threads(4); seed_all(42)
    if a.action == 'self-check':
        self_check(); return
    rows, protocol = load(a.directory)
    for name, expected in protocol['trained_artifact_hashes'].items():
        assert hashlib.sha256((a.source / name).read_bytes()).hexdigest() == expected, name
    if a.action == 'witness':
        witness(a); return
    (a.directory / 'runs').mkdir(exist_ok=True)
    selected = [r for r in rows if r['input_available']]
    if a.action == 'dino':
        values, masks = dino.images(a.directory, selected)
        for mode in ['frozen', 'finetune']:
            for seed in [42, 43]:
                name = f'dino_{mode}_s{seed}'; seed_all(seed)
                model, threshold = load_dino(a.source, name, mode)
                start = time.monotonic()
                predictions = dino.predict(model, values, masks, list(range(len(selected))), selected)
                for pred in predictions:
                    pred['accepted_80'] = max(pred['use_probabilities']) >= threshold
                    pred['selective_use'] = pred['use'] if pred['accepted_80'] else 'unknown'
                finish(a.directory, name, rows, predictions, start)
                del model; torch.cuda.empty_cache()
    else:
        name = 'qwen_nf4_zero' if a.action == 'qwen_zero' else 'qwen_qlora_s' + a.action[-2:]
        processor, base = qwen.base_model()
        model = qwen.evaluation_model(base, a.source / 'runs' / name, zero=a.action == 'qwen_zero')
        start = time.monotonic(); predictions = []
        # Save complete chunks so an interruption never masquerades as a full run.
        for i in range(0, len(selected), 200):
            chunk = qwen.infer(a.directory, selected[i:i+200], processor, model)
            predictions += chunk
            save(a.directory / 'runs' / (name + '_partial.json'), predictions)
            print('LA_PROGRESS', name, len(predictions), '/', len(selected), flush=True)
        finish(a.directory, name, rows, predictions, start)


def load_dino(source, name, mode):
    checkpoint = torch.load(source / 'runs' / name / 'model.pt', map_location='cpu', weights_only=True)
    model = dino.Dino(mode == 'finetune').cuda()
    model.load_state_dict(checkpoint['trainable_state'], strict=False)
    return model, checkpoint['threshold']


def finish(root, name, rows, predictions, start):
    by_id = {p['sample_id']: p for p in predictions}
    assert len(by_id) == len(predictions) == sum(r['input_available'] for r in rows)
    all_predictions = []
    for row in rows:
        if row['input_available']:
            pred = dict(by_id[row['sample_id']], input_available=True)
        else:
            pred = dict(sample_id=row['sample_id'], use='unknown', role='unknown',
                        input_available=False, abstention_reason='missing_native_context')
            if name.startswith('dino'):
                pred.update(accepted_80=False, selective_use='unknown')
        all_predictions.append(pred)
    path = root / 'runs' / name; path.mkdir(exist_ok=False)
    save(path / 'predictions.json', all_predictions)
    save(path / 'EVALUATION.json', dict(seconds=time.monotonic()-start,
        n_attempted=len(rows), n_inferred=len(predictions), metrics=score(rows, all_predictions)))
    print('LA_COMPLETE', name, len(all_predictions), flush=True)


def witness(a):
    source_rows, _ = load(a.source)
    validation = [r for r in source_rows if r['split'] == 'val'][:8]
    values, masks = dino.images(a.source, validation)
    model, _ = load_dino(a.source, 'dino_finetune_s42', 'finetune')
    actual = dino.predict(model, values, masks, list(range(8)), validation)
    expected = json.loads((a.source / 'runs/dino_finetune_s42/validation_predictions.json').read_text())[:8]
    assert [(p['sample_id'], p['use'], p['role']) for p in actual] == [(p['sample_id'], p['use'], p['role']) for p in expected]
    error = float(np.max(np.abs(np.array([p['use_probabilities'] for p in actual]) - np.array([p['use_probabilities'] for p in expected]))))
    assert error < 1e-5, error
    del model, values, masks; torch.cuda.empty_cache()
    processor, base = qwen.base_model()
    model = qwen.evaluation_model(base, a.source / 'runs/qwen_qlora_s42')
    actual = qwen.infer(a.source, validation, processor, model)
    expected = json.loads((a.source / 'runs/qwen_qlora_s42/validation_epoch2.json').read_text())[:8]
    assert [(p['sample_id'], p['raw'], p['use'], p['role']) for p in actual] == [(p['sample_id'], p['raw'], p['use'], p['role']) for p in expected]
    save(a.directory / 'WITNESS.json', dict(source_n=8, dino_probability_max_error=error,
        qwen_raw_use_role_exact=True, peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    print('LA_TRANSFER_WITNESS_PASS', torch.cuda.get_device_name(), 'source_n8', error, flush=True)


def self_check():
    import tempfile
    root = Path(tempfile.mkdtemp(prefix='la-transfer-check-')); (root/'runs').mkdir()
    rows = [dict(sample_id=str(i), input_available=i==0,
                 reference_use='commercial', reference_role=None) for i in range(2)]
    finish(root, 'dino_check', rows, [dict(sample_id='0', use='commercial', role='primary',
           accepted_80=True, selective_use='commercial')], time.monotonic())
    preds = json.loads((root/'runs/dino_check/predictions.json').read_text())
    metrics = json.loads((root/'runs/dino_check/EVALUATION.json').read_text())
    assert len(preds)==2 and preds[1]['use']=='unknown' and not preds[1]['input_available']
    assert metrics['metrics']['use']['accuracy']==.5 and metrics['metrics']['use']['coverage']==.5
    assert metrics['metrics']['role']['n']==0
    print('PASS unavailable inputs remain errors in full cohort; no invented role labels')


if __name__ == '__main__': main()
