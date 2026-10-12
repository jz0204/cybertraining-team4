"""LA input area, identity, path and failure contracts."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import run_la


class LaHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / "v3"
        self.out = self.root / "output"
        self.out.mkdir()

    def make_cell(self, cell):
        shared = self.run / cell / "shared_base"
        (shared / "crops_pre").mkdir(parents=True)
        (shared / "masks_M").mkdir()
        Image.new("RGB", (256, 256), (80, 90, 100)).save(shared / "crops_pre/crop.png")
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[80:100, 90:110] = 255
        Image.fromarray(mask).save(shared / "masks_M/mask.png")
        run_la.write_csv(shared / "shared_instance_samples.csv", ["bldg_uid", "pre_crop", "mask_M"], [{
            "bldg_uid": "same_uid", "pre_crop": "/old/v2/crops_pre/crop.png",
            "mask_M": "/old/v2/masks_M/mask.png",
        }])
        return {"cell_id": cell, "bldg_uid": "same_uid",
                "polygon_wkt_utm": "POLYGON ((400000 3700000,400010 3700000,400010 3700020,400000 3700020,400000 3700000))"}

    def prepare(self, rows):
        product = self.root / "product.csv"
        run_la.write_csv(product, ["cell_id", "bldg_uid", "polygon_wkt_utm"], rows)
        return run_la.prepare_inputs(product, self.run.resolve(), self.out)

    def test_full_polygon_area_and_cross_cell_identity(self):
        inputs, audit = self.prepare([self.make_cell("cell_00001"), self.make_cell("cell_00002")])
        self.assertEqual(audit["valid_rows"], 2)
        self.assertEqual(audit["excluded_rows"], 0)
        self.assertEqual(inputs[0]["footprint_m2"], 200)
        self.assertNotEqual(inputs[0]["building_uid"], inputs[1]["building_uid"])
        self.assertEqual(json.loads(inputs[0]["building_uid"]), ["cell_00001", "same_uid"])
        self.assertTrue(inputs[0]["crop_path"].startswith(str(self.run)))

    def test_polygon_hole_and_multipolygon_area(self):
        self.assertEqual(run_la.footprint_area(
            "POLYGON ((0 0,10 0,10 10,0 10,0 0),(2 2,2 4,4 4,4 2,2 2))"), 96)
        self.assertEqual(run_la.footprint_area(
            "MULTIPOLYGON (((0 0,2 0,2 2,0 2,0 0)),((5 5,6 5,6 6,5 6,5 5)))"), 5)

    def test_invalid_geometry_is_reported(self):
        row = self.make_cell("cell_00001")
        row["polygon_wkt_utm"] = "not a polygon"
        inputs, audit = self.prepare([row])
        self.assertEqual(inputs, [])
        self.assertEqual(audit["excluded_rows"], 1)
        self.assertEqual(run_la.read_csv(self.out / "excluded_inputs.csv")[0]["bldg_uid"], "same_uid")

    def test_duplicate_product_keys_fail(self):
        row = self.make_cell("cell_00001")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.prepare([row, row])

    def test_wrong_mask_scale_is_reported(self):
        row = self.make_cell("cell_00001")
        Image.fromarray(np.ones((256, 256), dtype=np.uint8)).save(
            self.run / "cell_00001/shared_base/masks_M/mask.png")
        inputs, audit = self.prepare([row])
        self.assertEqual(inputs, [])
        self.assertEqual(audit["excluded_rows"], 1)

    def test_nodata_crop_is_reported(self):
        row = self.make_cell("cell_00001")
        Image.new("RGB", (256, 256)).save(self.run / "cell_00001/shared_base/crops_pre/crop.png")
        inputs, audit = self.prepare([row])
        self.assertEqual(inputs, [])
        self.assertEqual(audit["excluded_rows"], 1)

    def test_prefix_map_does_not_match_partial_directory(self):
        path = self.root / "new/file.png"
        path.parent.mkdir()
        path.touch()
        maps = run_la.parse_path_maps([f"/old/root={path.parent}"])
        shared = self.run / "cell/shared_base/shared_instance_samples.csv"
        self.assertEqual(run_la.resolve_asset("/old/root/file.png", shared, self.run, "crops_pre", maps), path)
        with self.assertRaises(FileNotFoundError):
            run_la.resolve_asset("/old/root_extra/file.png", shared, self.run, "crops_pre", maps)

    def test_predictions_restore_ids_and_reject_missing_rows(self):
        inputs, _ = self.prepare([self.make_cell("cell_00001"), self.make_cell("cell_00002")])
        raw = [{"building_uid": row["building_uid"], "pred_population": "12.5",
                "pred_type_class": "commercial", "prob_residential_small": "0.1",
                "prob_residential_multi": "0.2", "prob_commercial": "0.7"} for row in reversed(inputs)]
        raw_csv = self.out / "raw.csv"
        run_la.write_csv(raw_csv, list(raw[0]), raw)
        summary = run_la.finalize_predictions(raw_csv, inputs, self.out / "final.csv")
        self.assertEqual(summary["prediction_rows"], 2)
        final = run_la.read_csv(self.out / "final.csv")
        self.assertEqual(final[0]["cell_id"], "cell_00002")
        self.assertEqual(final[0]["bldg_uid"], "same_uid")
        run_la.write_csv(raw_csv, list(raw[0]), raw[:1])
        with self.assertRaisesRegex(ValueError, "IDs"):
            run_la.finalize_predictions(raw_csv, inputs, self.out / "bad.csv")


if __name__ == "__main__":
    unittest.main()
