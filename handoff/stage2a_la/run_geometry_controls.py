"""Fixed, source-only CPU controls using the existing inference geometry."""
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from scripts.stage2a.common import Stage2aDataset, geometry_vector, read_csv_rows

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/stage3/stage2a_la_explore_20261023')
CLASSES = ['residential_small', 'residential_multi', 'commercial']


def main():
    protocol = json.loads((ROOT / 'GEOMETRY_PROTOCOL.json').read_text())
    with (ROOT / 'BATCH.json').open('x') as stream:
        json.dump(dict(status='running', started_at=datetime.now(timezone.utc).isoformat()), stream)
    rows = read_csv_rows(SOURCE / 'data/labels_explore.csv')
    la = read_csv_rows(SOURCE / 'data/la_input.csv')
    frame = pd.DataFrame(rows)
    assert frame.building_uid.is_unique and len({r['building_uid'] for r in la}) == len(la) == 21888
    assert frame.rebuild_split.value_counts().to_dict() == dict(train=13333, val=2026, test=2653)
    for group in ['physical_building_id', 'parcel_key', 'tile_base']:
        assert frame.groupby(group).rebuild_split.nunique().max() == 1
    columns = protocol['feature_columns']
    x = features(rows, columns)
    x_la = features(la, columns)
    # Verify against the real CNN dataset preprocessing on both domains.
    for collection, values in [(rows, x), (la, x_la)]:
        sample = [0, len(collection)//2, len(collection)-1]
        ds = Stage2aDataset(collection, indices=sample, train=False, require_targets=False,
                           require_ring_mask=False, geometry_cols=columns)
        for j, i in enumerate(sample):
            np.testing.assert_allclose(values[i], ds[j]['geometry'].numpy())
    np.savez_compressed(ROOT / 'geometry_features.npz', harris=x, la=x_la)
    pd.DataFrame({'building_uid': frame.building_uid}).to_csv(ROOT / 'harris_feature_ids.csv', index=False)
    pd.DataFrame({'building_uid': [r['building_uid'] for r in la]}).to_csv(ROOT / 'la_feature_ids.csv', index=False)
    (ROOT / 'PREFLIGHT.json').write_text(json.dumps(dict(passed=True, rows=len(rows), la_rows=len(la),
                                                       input_finite=True, physical_tile_parcel_overlap=0,
                                                       actual_dataset_geometry_checks=6), indent=2)+'\n')
    y = np.array([CLASSES.index(r['type_class_name']) for r in rows])
    train = frame.rebuild_split.eq('train').to_numpy()
    completed = []
    for name, selected in protocol['models'].items():
        indices = [columns.index(c) for c in selected]
        model = HistGradientBoostingClassifier(**protocol['hyperparameters'])
        model.fit(x[train][:, indices], y[train])
        assert np.array_equal(model.classes_, [0, 1, 2])
        artifact = dict(model=model, columns=selected, class_names=CLASSES, protocol=protocol)
        with (ROOT / f'{name}.pkl').open('xb') as stream:
            pickle.dump(artifact, stream)
        with (ROOT / f'{name}.pkl').open('rb') as stream:
            loaded = pickle.load(stream)['model']
        np.testing.assert_allclose(model.predict_proba(x[:8, indices]), loaded.predict_proba(x[:8, indices]))
        folder = ROOT / 'predictions' / name
        folder.mkdir(parents=True)
        groups = {'la': (pd.DataFrame(la), x_la)}
        for split in ['val', 'test']:
            selected_rows = frame.rebuild_split.eq(split).to_numpy()
            groups[f'harris_{split}'] = (frame[selected_rows], x[selected_rows])
        for domain, (metadata, matrix) in groups.items():
            probabilities = loaded.predict_proba(matrix[:, indices])
            assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1)
            out = pd.DataFrame(probabilities, columns=[f'prob_{c}' for c in CLASSES])
            out['building_uid'] = metadata.building_uid.to_numpy()
            out['pred_type_class'] = np.asarray(CLASSES)[probabilities.argmax(1)]
            out.to_csv(folder / f'{domain}.csv', index=False)
        completed.append(name)
        print('COMPLETE', name, 'iterations', model.n_iter_, flush=True)
    (ROOT / 'BATCH.json').write_text(json.dumps(dict(status='complete', completed=completed,
                                                   finished_at=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')


def features(rows, columns):
    values = []
    for i, row in enumerate(rows):
        with Image.open(row['mask_path']) as image:
            mask = np.asarray(image.convert('L'))
        assert mask.any() and float(row['footprint_m2']) > 0, row['building_uid']
        values.append(geometry_vector(row, columns, mask=mask))
        if (i + 1) % 5000 == 0:
            print('GEOMETRY', i + 1, '/', len(rows), flush=True)
    result = np.asarray(values)
    assert np.isfinite(result).all() and result.shape == (len(rows), len(columns))
    return result


if __name__ == '__main__':
    main()
