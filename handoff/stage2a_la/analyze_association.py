"""Conditional association and fitted-model sensitivity of the saved Harvey product."""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PACKAGE = Path('/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package')
OUT = PACKAGE / 'outputs/stage2a/la_data_20261024/generalization_20261026'


def main():
    source = PACKAGE / 'outputs/stage3_harvey/sam3_full/results_250m'
    frame = gpd.read_file(source / 'buildings.gpkg', ignore_geometry=True)
    areas = gpd.read_file(source / 'areas.gpkg', ignore_geometry=True)
    assert len(frame) == 13806 and len(areas) == 278
    assert frame.bldg_uid.is_unique and areas.area_id.is_unique
    assert set(frame.area_id) == set(areas.area_id)
    ecols = [f'pred_population_member_{j}' for j in range(5)]
    qcols = [f'severe_prob_member_{j}' for j in range(3)]
    values = frame[['pred_population', 'severe_damage_probability', *ecols, *qcols]].to_numpy()
    assert np.isfinite(values).all() and (values >= 0).all()
    assert (frame[['severe_damage_probability', *qcols]].to_numpy() <= 1).all()
    rows = []
    for area, group in frame.groupby('area_id', sort=True):
        e = group.pred_population.to_numpy()
        q = group.severe_damage_probability.to_numpy()
        em = group[ecols].to_numpy()
        qm = group[qcols].to_numpy()
        linked = float(e @ q)
        unlinked = float(e.sum() * q.mean())
        covariance_term = float(((e-e.mean()) * (q-q.mean())).sum())
        assert np.isclose(linked-unlinked, covariance_term, rtol=1e-8, atol=1e-8)
        damage = e @ qm
        semantics = em.T @ q
        joint = em.T @ qm
        rows.append(dict(area_id=area, buildings=len(group), exposure=float(e.sum()), linked=linked,
                         unlinked=unlinked, covariance_term=covariance_term,
                         damage_only_range=float(np.ptp(damage)), semantics_only_range=float(np.ptp(semantics)),
                         combined_range=float(np.ptp(joint)),
                         **{f'combined_{a}_{b}': float(joint[a,b]) for a in range(5) for b in range(3)}))
    table = pd.DataFrame(rows).set_index('area_id')
    saved = areas.set_index('area_id').loc[table.index]
    np.testing.assert_array_equal(table.buildings, saved.building_count)
    np.testing.assert_allclose(table.linked, saved.linked_residential_severe_exposure + saved.linked_commercial_severe_exposure)
    table['unlinked_minus_linked'] = table.unlinked-table.linked
    table['absolute_relative_difference'] = (table.unlinked-table.linked).abs()/table.linked
    table['rank_linked'] = table.linked.rank(method='min', ascending=False)
    table['rank_unlinked'] = table.unlinked.rank(method='min', ascending=False)
    table.to_csv(OUT / 'HARVEY_ASSOCIATION_SENSITIVITY.csv')
    top = set(table.nlargest(10, 'linked').index)
    joint_top = [len(top & set(table.nlargest(10, f'combined_{a}_{b}').index)) for a in range(5) for b in range(3)]
    result = dict(
        source=str(source), buildings=len(frame), areas=len(table),
        fixed_exposure_total=float(table.exposure.sum()), fixed_linked_total=float(table.linked.sum()),
        unlinked_total=float(table.unlinked.sum()),
        total_relative_difference=float(table.unlinked.sum()/table.linked.sum()-1),
        median_area_absolute_relative_difference=float(table.absolute_relative_difference.median()),
        p90_area_absolute_relative_difference=float(table.absolute_relative_difference.quantile(.9)),
        rank_spearman=float(table.linked.corr(table.unlinked, method='spearman')),
        top10_overlap=int(len(top & set(table.nlargest(10, 'unlinked').index))),
        median_ranges={c: float(table[c].median()) for c in ['damage_only_range','semantics_only_range','combined_range']},
        joint_scenario_top10_overlap_min_max=[min(joint_top), max(joint_top)],
        scope='Saved manuscript Harvey outputs only; conditional diagnostic, not ground-truth impact evaluation.',
        formulas=dict(linked='sum_i e_i*q_i', unlinked='sum_i e_i * mean_i q_i',
                      difference='linked-unlinked = n * population_covariance(e,q)'),
        limits=['Exposure is a mixed residential/commercial modeled proxy, not observed event-time occupants.',
                'The old saved product has not been regenerated with current clean models.',
                'Fifteen crossed fitted-model scenarios are sensitivity cases, not independent samples or a calibrated joint posterior.',
                'Inventory omissions, exposure-regressor parameter uncertainty, Census error, and temporal mismatch are held fixed.',
                'Top10 is descriptive rank sensitivity, not a validated response priority or a selection gate.'],
        checks='13806 unique buildings;278 unique areas;all saved counts and linked totals reconcile;covariance identity verified per area.'
    )
    (OUT / 'HARVEY_ASSOCIATION.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
