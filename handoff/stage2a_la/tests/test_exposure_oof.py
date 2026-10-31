import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve the frozen runtime before adding handoff's older scripts package.
from scripts.stage2a.training import package_stage2a_native3_population

sys.path.append(str(Path(__file__).resolve().parents[1]))
from prepare_exposure_oof import quarantine_physical
from run_exposure_oof import aligned_probabilities, head_features, paired_interval, spatial_blocks, PROB_COLS


class ExposureComparisonTests(unittest.TestCase):
    def test_quarantine_keeps_holdouts_and_priority_without_reordering(self):
        rows = [dict(building_uid=uid, physical_building_id=physical, rebuild_split=role)
                for uid, physical, role in [('a', 'p', 'train'), ('b', 'q', 'train'), ('c', 'p', 'val'),
                                             ('d', 'p', 'test'), ('e', 'r', 'val'), ('f', 'r', 'train'),
                                             ('g', 'p', 'test')]]
        kept, removed = quarantine_physical(rows)
        self.assertEqual([r['building_uid'] for r in kept], ['b', 'd', 'e', 'g'])
        self.assertEqual({r['building_uid']: r['protected_role'] for r in removed}, {'a': 'test', 'c': 'test', 'f': 'val'})

    def test_prediction_alignment_and_rejection(self):
        rows = [{'building_uid': 'a'}, {'building_uid': 'b'}]
        predictions = pd.DataFrame([['b', .1, .3, .6], ['a', .8, .1, .1]], columns=['building_uid', *PROB_COLS])
        p = aligned_probabilities(rows, predictions)
        np.testing.assert_array_equal(p, [[.8, .1, .1], [.1, .3, .6]])
        for broken in (predictions.iloc[:1], pd.concat([predictions, predictions.iloc[:1]]),
                       predictions.assign(prob_commercial=np.nan), predictions.assign(prob_commercial=0)):
            with self.assertRaises(AssertionError):
                aligned_probabilities(rows, broken)
        area = np.array([[2.], [4.]])
        x_a, names_a = head_features(area, None)
        x_b, names_b = head_features(area, p)
        self.assertEqual(names_a, ['log_footprint_m2'])
        self.assertEqual(x_a.shape, (2, 1))
        self.assertEqual(x_b.shape, (2, 8))
        self.assertEqual(len(names_b), 8)
        np.testing.assert_array_equal(x_b[:, 0], x_a[:, 0])
        np.testing.assert_array_equal(x_b[:, 1:4], p)

    def test_spatial_components_link_physical_and_parcel_transitively(self):
        frame = pd.DataFrame(dict(tile_base=['a', 'b', 'b', 'c', 'd'],
                                  physical_building_id=['p', 'p', 'q', 'r', 's'],
                                  parcel_key=['v', 'w', 'x', 'x', 'y']))
        blocks = spatial_blocks(frame)
        self.assertEqual(blocks.tolist(), ['a', 'a', 'a', 'a', 'd'])
        protocol = dict(bootstrap=dict(seed=42, replicates=100))
        same = paired_interval(np.zeros(5), blocks, protocol)
        self.assertEqual(same['ci95'], [0, 0])
        improved = paired_interval(np.full(5, -.5), blocks, protocol)
        self.assertEqual(improved['ci95'], [-.5, -.5])
        self.assertEqual(improved['spatial_components'], 2)


if __name__ == '__main__':
    unittest.main()
