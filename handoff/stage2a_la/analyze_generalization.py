"""Audit all classes and uncertainty using preserved Stage2a predictions."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, log_loss, roc_auc_score

STAGE = Path('/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a')
SOURCE = STAGE / 'la_explore_20261023'
DATA = STAGE / 'la_data_20261024'
FOLLOW = DATA / 'followup_20261025'
OUT = DATA / 'generalization_20261026'
sys.path.insert(0, str(STAGE / 'la_adapt_20261023'))
from compare import CLASSES, join

PROBS = [f'prob_{name}' for name in CLASSES]
MODELS = {
    'B1_area_hgb': OUT,
    'B2_geometry_hgb': OUT,
    **{n: SOURCE for n in ['C0_scale_clean_seed42', 'C1_ring20m_seed42',
                          'G1_no_geometry_seed42', 'D1_stochastic_photo_seed42', 'N1_natural_ce_seed42']},
    'N2_nyc_natural_seed42': DATA,
    **{n: FOLLOW for n in ['N1_natural_ce_seed43', 'N2_nyc_natural_seed43',
                          'R1_natural_ring_seed42', 'R1_natural_ring_seed43']},
}


def main():
    OUT.mkdir(exist_ok=True)
    labels = pd.read_csv(SOURCE / 'data/labels_explore.csv', usecols=['building_uid', 'rebuild_split', 'type_class_name', 'footprint_m2'])
    la = pd.read_csv(SOURCE / 'data/la_reference.csv')
    la = la[la.primary_single_outline_iou50].copy()
    areas = pd.read_csv(STAGE / 'la_eval_20261023/prepared/stage2a_input.csv')[['building_uid', 'footprint_m2']]
    la = la.merge(areas, on='building_uid', validate='one_to_one')
    refs = {'la': la}
    for split in ['val', 'test']:
        refs[f'harris_{split}'] = labels[labels.rebuild_split.eq(split)][
            ['building_uid', 'type_class_name', 'footprint_m2']].rename(columns={'type_class_name': 'assessor_type'})
    assert [len(refs[n]) for n in ['la', 'harris_val', 'harris_test']] == [3332, 2026, 2653]
    assert all(f.building_uid.is_unique for f in refs.values())
    assert la.assessor_type.value_counts().to_dict() == dict(residential_small=3276, commercial=42, residential_multi=14)
    tables, details, area_rows = [], [], []
    full = {}
    for domain, ref in refs.items():
        predictions = {name: pd.read_csv(folder / f'predictions/{name}/{domain}.csv') for name, folder in MODELS.items()}
        baseline_path = (STAGE / 'la_eval_20261023/cluster_full/stage2a_predictions.csv' if domain == 'la'
                         else SOURCE / f'reference_ensemble/{domain}.csv')
        predictions['current_five_seed_ensemble'] = pd.read_csv(baseline_path)
        for stem in ['N1_natural_ce', 'N2_nyc_natural', 'R1_natural_ring']:
            members = [join(ref[['building_uid']], predictions[f'{stem}_seed{seed}']) for seed in [42, 43]]
            mean = (members[0][PROBS].to_numpy() + members[1][PROBS].to_numpy()) / 2
            f = pd.DataFrame(mean, columns=PROBS)
            f['building_uid'] = ref.building_uid.to_numpy()
            f['pred_type_class'] = np.asarray(CLASSES)[mean.argmax(1)]
            predictions[f'{stem}_two_seed_mean'] = f
        for name, pred in predictions.items():
            f = join(ref, pred[['building_uid', 'pred_type_class', *PROBS]])
            metrics, per_class, review = evaluate(f)
            tables.append(dict(model=name, domain=domain, **metrics))
            details.extend(dict(model=name, domain=domain, **r) for r in per_class)
            full[f'{domain}/{name}'] = dict(metrics=metrics, per_class=per_class, review=review)
            if domain == 'la':
                for j, label in enumerate(CLASSES):
                    truth = f.assessor_type.eq(label)
                    true_area = f.loc[truth, 'footprint_m2'].sum()
                    hard_area = f.loc[f.pred_type_class.eq(label), 'footprint_m2'].sum()
                    soft_area = (f.footprint_m2 * f[PROBS[j]]).sum()
                    area_rows.append(dict(model=name, type=label, reference_m2=true_area,
                                          hard_m2=hard_area, soft_m2=soft_area,
                                          hard_ratio=hard_area/true_area, soft_ratio=soft_area/true_area))
                assert np.isclose(sum(r['soft_m2'] for r in area_rows[-3:]), f.footprint_m2.sum())
                assert np.isclose(sum(r['hard_m2'] for r in area_rows[-3:]), f.footprint_m2.sum())
    table = pd.DataFrame(tables)
    # Compare against independently generated exports, without refitting or selecting thresholds.
    paired = pd.read_csv(FOLLOW / 'PAIRED_COMPARISON.csv')
    for r in paired.itertuples():
        actual = table[(table.model == r.model) & (table.domain == 'harris_val')].iloc[0]
        assert np.isclose(actual.macro_f1, r.harris_val_f1)
        assert np.isclose(full[f'la/{r.model}']['per_class'][2]['ap'], r.ap)
    table.to_csv(OUT / 'ALL_CLASS_METRICS.csv', index=False)
    pd.DataFrame(details).to_csv(OUT / 'PER_CLASS_METRICS.csv', index=False)
    pd.DataFrame(area_rows).to_csv(OUT / 'TYPE_AREA_ASSIGNMENT.csv', index=False)
    (OUT / 'DIAGNOSTICS.json').write_text(json.dumps(full, indent=2, allow_nan=False) + '\n')
    evidence = dict(
        cohort_sizes={k: len(v) for k, v in refs.items()},
        la_cells=int(la.cell_id.nunique()),
        la_cells_by_class=la.groupby('assessor_type').cell_id.nunique().to_dict(),
        la_role='Previously inspected development data, not a blind test.',
        harris_test_role='Repeatedly inspected diagnostic split, not a new independent test.',
        baseline_limit='Current five-seed baseline contains known 22 train physical duplicates of test rows; clean recipes quarantine these.',
        ensemble_limit='Two-seed means are post-hoc diagnostics, no new model selected or deployed.',
        exposure_limit='Type-weighted area uses fixed predicted footprints; neither population nor footprint accuracy.',
        uncertainty_limit='Type uncertainty diagnostics do not validate Stage2b damage uncertainty or joint interval coverage.',
        checks='Exact cohorts and ID joins; normalized finite probabilities; confusion and error totals; parity with six prior source-validation scores.',
    )
    (OUT / 'AUDIT.json').write_text(json.dumps(evidence, indent=2) + '\n')
    names = ['current_five_seed_ensemble', 'N1_natural_ce_seed42', 'N1_natural_ce_seed43',
             'N1_natural_ce_two_seed_mean', 'N2_nyc_natural_two_seed_mean', 'R1_natural_ring_two_seed_mean']
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    for domain, color in [('harris_test', '#33658a'), ('la', '#d66b37')]:
        rows = table[table.domain.eq(domain)].set_index('model').loc[names]
        x = np.arange(len(names)) + (-.18 if domain == 'harris_test' else .18)
        axs[0].bar(x, rows.macro_f1, width=.36, label=domain, color=color)
    axs[0].set(xticks=np.arange(len(names)), xticklabels=['Current 5', 'N1 s42', 'N1 s43', 'N1 mean2', 'NYC mean2', 'Ring mean2'],
               ylabel='Three-class macro-F1', ylim=(0, 1), title='Existing predictions; diagnostic cohorts')
    axs[0].tick_params(axis='x', rotation=30)
    axs[0].legend()
    for name, short in zip(names, ['Current 5', 'N1 s42', 'N1 s43', 'N1 mean2', 'NYC mean2', 'Ring mean2']):
        rows = full[f'la/{name}']['review']
        axs[1].plot([r['retained_fraction'] for r in rows], [r['retained_error_rate'] for r in rows], marker='o', label=short)
    axs[1].set(xlabel='Fraction retained after reviewing most uncertain', ylabel='Error rate in retained set',
               title='LA: entropy review; class coverage reported separately')
    axs[1].legend(fontsize=8)
    fig.savefig(OUT / 'all_class_and_uncertainty.png', dpi=170)
    plt.close(fig)
    print(table[table.domain.eq('la')].to_string(index=False))
    print('PASS: real-data checks and six prior metrics reproduced; outputs:', OUT)


def evaluate(f):
    y = pd.Categorical(f.assessor_type, categories=CLASSES).codes
    assert (y >= 0).all()
    p = f[PROBS].to_numpy()
    z = p.argmax(1)
    cm = confusion_matrix(y, z, labels=range(3))
    assert cm.sum() == len(f) and cm.trace() == (y == z).sum()
    precision = np.divide(cm.diagonal(), cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0)
    recall = cm.diagonal() / cm.sum(1)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(3), where=precision+recall > 0)
    entropy = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
    error = z != y
    high = p.max(1) >= .99
    per_class = [dict(type=c, support=int(cm[j].sum()), precision=float(precision[j]),
                      recall=float(recall[j]), f1=float(f1[j]),
                      ap=float(average_precision_score(y == j, p[:, j])),
                      **{f'predicted_{name}': int(cm[j, k]) for k, name in enumerate(CLASSES)}) for j, c in enumerate(CLASSES)]
    metrics = dict(n=len(f), accuracy=float((~error).mean()), macro_f1=float(f1.mean()),
                   balanced_accuracy=float(recall.mean()), macro_ap=float(np.mean([r['ap'] for r in per_class])),
                   nll=float(log_loss(y, p / p.sum(1, keepdims=True), labels=range(3))),
                   brier=float(np.square(p - np.eye(3)[y]).sum(1).mean()),
                   errors=int(error.sum()), high_confidence_errors=int((error & high).sum()),
                   high_confidence_count=int(high.sum()), entropy_error_auc=float(roc_auc_score(error, entropy)))
    review = []
    order = np.argsort(entropy, kind='stable')
    for fraction in [1., .9, .8, .5]:
        keep = order[:int(np.ceil(len(f) * fraction))]
        removed = np.ones(len(f), dtype=bool)
        removed[keep] = False
        assert int(error[keep].sum()) + int(error[removed].sum()) == int(error.sum())
        review.append(dict(retained_fraction=len(keep)/len(f), retained_error_rate=float(error[keep].mean()),
                           reviewed_error_fraction=float(error[removed].sum()/error.sum()),
                           class_retained_fraction={c: float((y[keep] == j).sum()/(y == j).sum()) for j, c in enumerate(CLASSES)},
                           correct_retained_over_all_class={c: float(((y[keep] == j) & (z[keep] == j)).sum()/(y == j).sum()) for j, c in enumerate(CLASSES)}))
    return metrics, per_class, review


if __name__ == '__main__':
    main()
