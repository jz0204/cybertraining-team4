"""Shared frozen inputs and scoring for the matched transfer experiment."""

import hashlib
import json
from pathlib import Path

USES = ['residential_small', 'residential_multi', 'commercial', 'institutional', 'other']
ROLES = ['primary', 'auxiliary']


def load(root):
    protocol = json.loads((root / 'PROTOCOL.json').read_text())
    assert hashlib.sha256((root / 'dataset.json').read_bytes()).hexdigest() == protocol['dataset_sha256']
    return json.loads((root / 'dataset.json').read_text()), protocol


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def score(rows, predictions):
    assert len(rows) == len(predictions)
    assert [r['sample_id'] for r in rows] == [r['sample_id'] for r in predictions]
    result = {}
    for field, names in [('use', USES), ('role', ROLES)]:
        pairs = [(r['reference_' + field], p[field]) for r, p in zip(rows, predictions)
                 if r['reference_' + field] in names]
        per = {}
        for name in names:
            support = sum(y == name for y, _ in pairs)
            positive = sum(p == name for _, p in pairs)
            tp = sum(y == p == name for y, p in pairs)
            per[name] = dict(n=support, recall=tp/support if support else None,
                             precision=tp/positive if positive else 0,
                             f1=2*tp/(support+positive) if support else None)
        valid = [v['f1'] for v in per.values() if v['n']]
        correct = sum(y == p for y, p in pairs)
        answered = sum(p in names for _, p in pairs)
        result[field] = dict(n=len(pairs), accuracy=correct/len(pairs) if pairs else None,
            macro_f1=sum(valid)/len(valid) if valid else None,
            coverage=answered/len(pairs) if pairs else None,
            selective_accuracy=correct/answered if answered else None, per_class=per,
            confusion=[[sum(y==a and p==b for y,p in pairs) for b in names+['unknown','invalid']] for a in names],
            rows=names, columns=names+['unknown','invalid'])
    return result


def validation_score(metrics):
    return (metrics['use']['macro_f1'] + metrics['role']['macro_f1']) / 2


def self_check():
    rows = [dict(sample_id=str(i),reference_use=u,reference_role=r) for i,(u,r) in
            enumerate([('commercial','primary'),('commercial','primary'),(None,'auxiliary')])]
    predictions = [dict(sample_id=str(i),use=u,role=r) for i,(u,r) in
                   enumerate([('commercial','primary'),('unknown','auxiliary'),('other','auxiliary')])]
    s = score(rows,predictions)
    assert s['use']['n']==2 and s['use']['accuracy']==.5 and s['use']['coverage']==.5
    assert s['role']['accuracy']==2/3 and s['role']['per_class']['auxiliary']['recall']==1
    print('PASS missing labels excluded, unknown counted wrong, role denominator independent')


if __name__ == '__main__':
    self_check()
