"""Independent LA type evaluation using pre-fire Assessor parcel use codes."""

import json
import hashlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.affinity import translate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from audit_sources import ROOT, OUT
from audit_training import save_json


CLASSES = ["residential_small", "residential_multi", "commercial"]


def map_use(row):
    code = str(row["UseCode"] or "").strip()
    details = " ".join(str(row.get(f"UseCodeDescChar{i}") or "") for i in [2, 3, 4]).lower()
    if "UseType" in row and str(row.get("UseType") or "").strip() in {"", "(missing)"} and all(str(row.get(f"UseCodeDescChar{i}") or "").strip() in {"", "(missing)"} for i in [1, 2, 3, 4]):
        return "unmapped"
    if len(code) != 4 or not code[:2].isdigit():
        return "unmapped"
    if "vacant" in details or "other improvements only" in details or code.endswith("C") or row.get("UseType") == "CND" or "condo" in details or "own-your-own" in details:
        return "ambiguous_or_vacant"
    if code[:2] in {"01", "02", "03", "07"}:
        return "residential_small"
    if code[:2] in {"04", "05"}:
        return "residential_multi"
    if code[0] in {"1", "2"}:
        if "combination" in details or "residential" in details or code[:2] == "27":
            return "ambiguous_or_vacant"
        return "commercial"
    return "outside_three_class_taxonomy"


def score(frame, prediction="pred_type_class"):
    if frame.empty:
        return {"n": 0}
    truth, pred = frame.assessor_type, frame[prediction]
    report = classification_report(truth, pred, labels=CLASSES, target_names=CLASSES, output_dict=True, zero_division=0)
    return {"n": len(frame), "accuracy": float(accuracy_score(truth, pred)), "macro_f1": report["macro avg"]["f1-score"], "confusion_rows_truth_columns_prediction": confusion_matrix(truth, pred, labels=CLASSES).tolist(), "classes": CLASSES, "per_class": {c: report[c] for c in CLASSES}, "truth_counts": truth.value_counts().to_dict(), "prediction_counts": pred.value_counts().to_dict(), "majority_small_accuracy": float((truth == "residential_small").mean()), "majority_small_macro_f1": classification_report(truth, np.repeat("residential_small", len(frame)), labels=CLASSES, output_dict=True, zero_division=0)["macro avg"]["f1-score"]}


def main():
    snapshot = json.loads((OUT / "raw/assessor_2024.json").read_text())
    assessor = pd.DataFrame([f["attributes"] for r in snapshot["requests"] for f in r["response"]["features"]])
    assert assessor.AIN.is_unique and (assessor.RollYear == "2024").all()
    assessor["assessor_type"] = assessor.apply(map_use, axis=1)
    assessor.to_csv(OUT / "assessor_use_mapping.csv", index=False)
    directory = ROOT / "evaluation/la_fire_validation/data"
    gt = gpd.read_file(directory / "lariac_dins_in_region.gpkg")
    gt["gt_index"] = np.arange(len(gt))
    gt["parcel_outline_count"] = gt.groupby("AIN").AIN.transform("size")
    gt["gt_polygon_area_m2"] = gt.geometry.area
    matches = pd.read_csv(directory / "matched_v2.csv", dtype={"cell_id": str, "bldg_uid": str, "lariac_bld_id": str})
    assert matches.gt_index.is_unique
    gt_id = gt.BLD_ID.iloc[matches.gt_index].fillna("").astype(str).to_numpy()
    assert np.array_equal(gt_id, matches.lariac_bld_id.fillna("").to_numpy())
    product = pd.read_csv(ROOT / "results/final_product/building_damage_all_cells.csv", dtype={"cell_id": str, "bldg_uid": str})
    pred = pd.read_csv(Path(__file__).parent / "results/la_v3_full/stage2a_predictions.csv", dtype={"cell_id": str, "bldg_uid": str})
    keys = ["cell_id", "bldg_uid"]
    assert not product.duplicated(keys).any() and not pred.duplicated(keys).any()
    matches["matched_gt_per_prediction"] = matches.groupby(keys).gt_index.transform("size")
    matched = matches.merge(gt[["gt_index", "AIN", "parcel_outline_count", "gt_polygon_area_m2"]], on="gt_index", validate="one_to_one")
    matched = matched.merge(assessor, on="AIN", how="left", validate="many_to_one")
    matched = matched.merge(product[keys + ["polygon_wkt_utm", "m2b_damage_class"]], on=keys, validate="many_to_one")
    matched = matched.merge(pred[keys + ["pred_type_class", "pred_population", "pre_zero_fraction", "footprint_m2"] + [f"prob_{c}" for c in CLASSES]], on=keys, how="left", validate="many_to_one")
    ious = []
    for row in matched.itertuples():
        shifted = translate(wkt.loads(row.polygon_wkt_utm), -row.shift_dx_m, -row.shift_dy_m)
        reference = gt.geometry.iloc[row.gt_index]
        assert reference.is_valid and shifted.is_valid
        intersection = reference.intersection(shifted).area
        ious.append(intersection / (reference.area + shifted.area - intersection))
    matched["coregistered_iou"] = ious
    valid_type = matched.assessor_type.isin(CLASSES)
    residential = matched.assessor_type.isin(CLASSES[:2])
    valid_record = valid_type & (matched.SQFTmain.fillna(0) > 0) & (~residential | (matched.Units.fillna(0) > 0))
    predicted = matched.pred_type_class.notna()
    one_to_one = matched.matched_gt_per_prediction == 1
    overlap = matched.coregistered_iou >= .5
    parcel_single = matched.parcel_outline_count == 1
    assessor_single = matched.totBuildingDataLines == 1
    cohorts = {
        "broad_parcel_labels": predicted & valid_record & one_to_one,
        "iou50": predicted & valid_record & one_to_one & overlap,
        "primary_single_outline_iou50": predicted & valid_record & one_to_one & overlap & parcel_single,
        "strict_single_assessor_building": predicted & valid_record & one_to_one & overlap & parcel_single & assessor_single,
    }
    summary = {"reference_type": "Independent 2024 Assessor parcel-use records inherited by matched building; not observed building occupancy", "source": snapshot["source"], "roll_year": 2024, "requested_ains": len(snapshot["requested_ains"]), "returned_ains": len(assessor), "unreturned_ains": sorted(set(snapshot["requested_ains"]) - set(assessor.AIN)), "gt_buildings": len(gt), "original_matches": len(matches), "multi_gt_prediction_pairs": int((~one_to_one).sum()), "matched_with_predictions": int(predicted.sum()), "matched_with_assessor": int(matched.UseCode.notna().sum()), "valid_type_and_record": int(valid_record.sum()), "cohorts": {name: score(matched[mask]) for name, mask in cohorts.items()}, "mapping_notes": {"residential_small": "01 single family, 02 duplex, 03 triplex, 07 manufactured; follows training definition including up to 3 units", "residential_multi": "04 four units, 05 five+ units/apartments", "commercial": "1x/2x except combination/mixed-residential uses and commercial parking lots", "excluded": "condominiums identified by code, UseType or descriptions; own-your-own; vacant/other-improvement-only records; nonpositive main floor area; residential records without positive unit count; missing UseType plus all missing descriptions; unknown or outside taxonomy"}}
    main_mask = cohorts["primary_single_outline_iou50"]
    summary["by_fire_primary"] = {name: score(g) for name, g in matched[main_mask].groupby("incident")}
    summary["by_area_primary"] = {str(name): score(g) for name, g in matched[main_mask].groupby(pd.cut(matched.loc[main_mask, "footprint_m2"], [0,100,200,400,np.inf]), observed=True)}
    summary["primary_single_family_predictions"] = matched[main_mask & matched.UseCode.str.startswith("01", na=False)].pred_type_class.value_counts().to_dict()
    summary["primary_single_family_proxy_median"] = float(matched.loc[main_mask & matched.UseCode.str.startswith("01", na=False), "pred_population"].median())
    for name, mask in cohorts.items():
        matched[name] = mask
    matched.to_csv(OUT / "la_assessor_matched.csv", index=False)
    save_json("assessor_evaluation.json", summary)
    save_json("assessor_provenance.json", {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [directory / "lariac_dins_in_region.gpkg", directory / "matched_v2.csv", ROOT / "results/final_product/building_damage_all_cells.csv", Path(__file__).parent / "results/la_v3_full/stage2a_predictions.csv", OUT / "raw/assessor_2024.json"]})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
