import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_assessor import map_use, score


class AssessorEvaluationTests(unittest.TestCase):
    def test_use_mapping_respects_training_taxonomy_and_ambiguity(self):
        examples = {"0100": "residential_small", "0200": "residential_small", "0300": "residential_small", "0400": "residential_multi", "0500": "residential_multi", "0700": "residential_small", "1700": "commercial", "7400": "outside_three_class_taxonomy", "010C": "ambiguous_or_vacant", "2700": "ambiguous_or_vacant", "": "unmapped"}
        for code, expected in examples.items():
            with self.subTest(code=code):
                self.assertEqual(map_use({"UseCode": code}), expected)
        self.assertEqual(map_use({"UseCode": "0100", "UseCodeDescChar4": "Other Improvements Only"}), "ambiguous_or_vacant")
        self.assertEqual(map_use({"UseCode": "1200", "UseCodeDescChar2": "Store Combination"}), "ambiguous_or_vacant")
        self.assertEqual(map_use({"UseCode": "010E", "UseType": "CND"}), "ambiguous_or_vacant")
        self.assertEqual(map_use({"UseCode": "010H", "UseCodeDescChar4": "Own-your-own"}), "ambiguous_or_vacant")
        self.assertEqual(map_use({"UseCode": "1720", "UseCodeDescChar3": "Office and Residential"}), "ambiguous_or_vacant")
        self.assertEqual(map_use({"UseCode": "01A6", "UseType": "(missing)", **{f"UseCodeDescChar{i}": "(missing)" for i in range(1,5)}}), "unmapped")
        self.assertEqual(map_use({"UseCode": "01A6", "UseType": " ", **{f"UseCodeDescChar{i}": "(missing)" for i in range(1,5)}}), "unmapped")

    def test_confusion_orientation_and_explicit_absent_class(self):
        data = pd.DataFrame({"assessor_type": ["residential_small", "residential_small", "commercial"], "pred_type_class": ["commercial", "residential_small", "commercial"]})
        result = score(data)
        self.assertEqual(result["confusion_rows_truth_columns_prediction"], [[1, 0, 1], [0, 0, 0], [0, 0, 1]])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["per_class"]["residential_multi"]["support"], 0)
        self.assertAlmostEqual(result["majority_small_accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
