"""Apply the frozen OOF exposure head to current C probabilities; no retraining."""
import ast
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import List, Sequence, Tuple
import numpy as np

PACKAGE=Path('/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package')
STAGE=PACKAGE/'outputs/stage2a'
ROOT=STAGE/'la_best_preview_20261031'
HEAD=STAGE/'la_data_20261024/exposure_oof_20261026'
NAMES=['residential_small','residential_multi','commercial']


def main():
    source=PACKAGE/'scripts/stage2a/training/package_stage2a_native3_population.py'
    # Reuse the two pure helpers without importing the training module's GPU stack.
    tree=ast.parse(source.read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_type_soft_features','_predict_ridge']]
    assert len(selected)==2
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),globals())
    path=HEAD/'heads/B_semantics_mean3.pkl'
    with path.open('rb') as f: head=pickle.load(f)
    check=np.load(HEAD/'features/test.npz')
    expected=list(csv.DictReader((HEAD/'exposure_predictions/test.csv').open()))
    error=float(np.max(np.abs(predict(head,check['area'],check['probs_mean3'])-np.array([float(r['B_semantics_mean3_log']) for r in expected]))))
    assert error<1e-6,error
    rows=list(csv.DictReader((ROOT/'viewer/predictions.csv').open()))
    area=np.log1p(np.array([float(r['footprint_m2']) for r in rows]))[:,None]
    probs=np.array([[float(r['prob_'+n]) for n in NAMES] for r in rows])
    assert len(rows)==22024 and np.isfinite(area).all() and np.allclose(probs.sum(1),1)
    values=np.maximum(0,np.expm1(predict(head,area,probs)))
    assert np.isfinite(values).all() and (values>=0).all()
    result=dict(method='Frozen B_semantics_mean3 OOF exposure head with current C ensemble probabilities',head_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),saved_test_max_log_error=error,limitation='Area/parcel-use/ACS population proxy; not observed occupants. Transfer to C probabilities and LA is unvalidated. Damage-associated exposure is not casualties or displacement.',records=[dict(sample_id=r['sample_id'],probabilities=p.tolist(),footprint_m2=float(r['footprint_m2']),estimated_exposure=float(v)) for r,p,v in zip(rows,probs,values)],total=float(values.sum()),damaged_total=float(sum(v for r,v in zip(rows,values) if int(r['existing_damage_vote']) in [1,2,3])))
    (ROOT/'EXPOSURE.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='records'})


def predict(head,area,probs):
    soft,names=_type_soft_features(probs,NAMES)
    assert head['feature_names']==['log_footprint_m2',*names]
    return _predict_ridge(head['ridge_model'],area,head['ridge_mean'],head['ridge_std'])+head['residual_model'].predict(np.column_stack([area,soft]))


if __name__=='__main__':main()
