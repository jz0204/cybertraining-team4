"""Freeze seed and context controls after the NYC development-data diagnosis."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

STAGE = Path('/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a')
DATA = STAGE / 'la_data_20261024'
SOURCE = STAGE / 'la_explore_20261023'
OUT = DATA / 'followup_20261025'
REMOTE = Path('/stage3/stage2a_la_data_20261024/followup_20261025')


def main():
    assert not OUT.exists(), 'Preserve existing follow-up protocol and results'
    OUT.mkdir()
    (OUT / 'configs').mkdir()
    h = pd.read_csv(DATA / 'data/N2_nyc_natural_seed42.csv', low_memory=False)
    train = h[h.rebuild_split.eq('train')]
    distribution = train.groupby(['event_id', 'type_class_name']).footprint_m2.agg(['count', 'median', 'max'])
    distribution.to_csv(OUT / 'training_area_distribution.csv')
    ref = pd.read_csv(SOURCE / 'data/la_reference.csv')
    ref = ref[ref.primary_single_outline_iou50].copy()
    geo = pd.read_csv(SOURCE / 'data/la_input.csv')
    ref = ref.merge(geo[['building_uid', 'footprint_m2']], on='building_uid', validate='one_to_one')
    assert len(ref) == 3332 and ref.assessor_type.eq('commercial').sum() == 42
    comparisons = []
    tails = []
    for name, root in [('N1_natural_ce_seed42', SOURCE), ('N2_nyc_natural_seed42', DATA)]:
        frame = ref.merge(pd.read_csv(root / f'predictions/{name}/la.csv'), on='building_uid', validate='one_to_one')
        assert len(frame) == len(ref)
        positive = frame.assessor_type.eq('commercial')
        threshold = float(frame.loc[positive, 'prob_commercial'].nlargest(38).min())
        predicted = frame.prob_commercial.ge(threshold)
        fp = frame[~positive & predicted]
        bins = pd.cut(fp.footprint_m2, [0, 100, 200, 400, 1000, np.inf]).value_counts(sort=False)
        comparisons.append(dict(model=name, threshold=threshold,
                                tp=int((positive & predicted).sum()), fp=len(fp),
                                precision=float((positive & predicted).sum()/predicted.sum()),
                                fp_area_bins={str(k): int(v) for k, v in bins.items()}))
        tail = frame[positive].sort_values('prob_commercial').head(12).copy()
        tail['model'] = name
        tails.append(tail)
    pd.concat(tails).to_csv(OUT / 'commercial_low_score_tail.csv', index=False)
    (OUT / 'NYC_DIAGNOSIS.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    base = json.loads((SOURCE / 'configs/N1_natural_ce_seed42.json').read_text())
    nyc = json.loads((DATA / 'configs/N2_nyc_natural_seed42.json').read_text())
    arms = [('N1_natural_ce_seed43', base, 43, 'mask_m'),
            ('N2_nyc_natural_seed43', nyc, 43, 'mask_m'),
            ('R1_natural_ring_seed42', base, 42, 'mask_m_ring'),
            ('R1_natural_ring_seed43', base, 43, 'mask_m_ring')]
    for name, template, seed, pooling in arms:
        cfg = dict(template, seed=seed, pooling_mode=pooling,
                   config=str(REMOTE / f'configs/{name}.json'), out_dir=str(REMOTE / f'runs/{name}'))
        expected = {'seed', 'pooling_mode', 'config', 'out_dir'}
        assert {k for k in cfg if cfg[k] != template.get(k)} <= expected
        (OUT / f'configs/{name}.json').write_text(json.dumps(cfg, indent=2)+'\n')
    protocol = dict(recorded_at=datetime.now(timezone.utc).isoformat(), status='frozen_before_training',
                    models=[a[0] for a in arms], seeds=[42, 43],
                    comparisons=['N2 minus N1 paired at seeds42,43; seed42 results retained',
                                 'R1 minus N1 paired at seeds42,43; only pooling changes'],
                    budget='13333 draws,278 steps/epoch,max50 epochs,patience10; unchanged source validation selection',
                    ring='Existing physically constructed20m ring masks including scale views; no new image downloads',
                    selection='Report paired seed deltas at LA P@R>=90%, AP, full PR and argmax; require each ring seed to improve its paired N1 and pass C0-.01 source validation before promoting further study',
                    guardrail=0.9340237963331436-.01,
                    limitations=['Two seeds are a reproducibility diagnostic, not statistical proof',
                                 'LA is already adaptive development data,42 commercial buildings in7 cells',
                                 'Seed comparison also includes changed sampled examples/order, not isolated label causality',
                                 'NYC weak label dates and crop domain are hypotheses; no relabeling based on predictions'],
                    original_five_seed_gate='Unchanged; this is a new authorized two-seed diagnostic, not satisfaction of original gate',
                    la_training_rows=0, sandiego_training_rows=0, sandiego_probe='Closed during all four trials')
    (OUT / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(distribution.to_string())
    print(json.dumps(comparisons, indent=2))
    print('FROZEN', protocol['models'])


if __name__ == '__main__':
    main()
