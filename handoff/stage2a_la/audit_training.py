"""Audit archived labels against their own xBD polygons; preserve originals."""

import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from shapely import wkt
from shapely.ops import transform

from audit_sources import OUT


SOURCE = Path("/Users/zhou/Documents/GitHub/Instance-Impact-IGUIDE-SpatialAIChallenge2026")
LEGACY = SOURCE / "stage2a_old_training_data_codes"
RAW_XBD = SOURCE / "II_package/outputs/stage3_harvey/source_data/xBD/train/labels"


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def quantiles(values):
    return dict(zip(["min", "p10", "p50", "p90", "max"], map(float, np.quantile(values, [0, .1, .5, .9, 1]))))


def load_rules():
    notebook = json.loads((LEGACY / "building_population_pipeline.ipynb").read_text())
    source = "\n\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    names = {"DEFAULT_PPU", "DEFAULT_OCCUPANCY", "POPULATION_CAPS", "SFD_MAX_FOOTPRINT", "SMALL_MF_MAX_FOOTPRINT", "UNIT_AREA_PER_UNIT_SMALL_MF", "UNIT_AREA_PER_UNIT_LARGE_MF", "MAX_UNITS_PER_FOOTPRINT", "RESIDENTIAL_CODES", "NON_RESIDENTIAL_CODES", "NON_RES_DENSITY", "TYPE_TO_CLASS", "CLASS_NAMES", "_estimate_units", "classify_building", "estimate_population", "ACS_VARIABLES", "safe_ratio", "compute_cbg_ratios"}
    nodes = [node for node in ast.parse(source).body if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)) or (isinstance(node, ast.FunctionDef) and node.name in names)]
    namespace = {"pd": pd}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "archived_notebook_rules", "exec"), namespace)
    assert names <= namespace.keys()
    return namespace


def infer_original_code(row, rules):
    # Counterfactual keeps archived parcel-derived use fixed; raw parcel joins are not reconstructed.
    if row["classification_source"] == "area_heuristic":
        return np.nan
    btype = row["building_type"]
    if btype in {"small_multi_family", "large_multi_family"}:
        return "B1"
    return next(code for code, use in {**rules["RESIDENTIAL_CODES"], **rules["NON_RESIDENTIAL_CODES"]}.items() if use == btype)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    package = LEGACY / "Harris_CBG_Building_Population.zip"
    with zipfile.ZipFile(package) as archive:
        df = pd.read_csv(archive.open("ml_dataset/labels.csv"), dtype={"GEOID": str})
    assert len(df) == 23014 and df.building_uid.is_unique
    geoms = df.poly_geo_wkt.map(wkt.loads)
    pixels = df.poly_px_wkt.map(wkt.loads)
    utm = Transformer.from_crs(4326, 32615, always_xy=True)
    equal_area = Transformer.from_crs(4326, 5070, always_xy=True)
    geod = Geod(ellps="WGS84")
    df["area_utm_m2"] = [transform(utm.transform, g).area for g in geoms]
    df["area_equal_area_m2"] = [transform(equal_area.transform, g).area for g in geoms]
    df["area_geodesic_m2"] = [abs(geod.geometry_area_perimeter(g)[0]) for g in geoms]
    df["area_scale_ratio"] = df.footprint_m2 / df.area_equal_area_m2
    df["effective_gsd_m"] = np.sqrt(df.area_equal_area_m2 / np.array([p.area for p in pixels]))
    assert np.allclose(df.footprint_m2, np.array([p.area for p in pixels]) * df.gsd ** 2, rtol=1e-10)
    assert np.max(abs(df.area_equal_area_m2 / df.area_geodesic_m2 - 1)) < .001
    raw_hashes, metadata, joins = {}, [], {}
    for tile in df.tile_base.unique():
        path = RAW_XBD / f"{tile}_pre_disaster.json"
        raw = json.loads(path.read_text())
        raw_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        m = raw["metadata"]
        metadata.append({"tile_base": tile, "metadata_gsd": m["gsd"], "pan_resolution": m.get("pan_resolution"), "capture_date": m.get("capture_date"), "rows": int((df.tile_base == tile).sum())})
        xy = {f["properties"]["uid"]: f["wkt"] for f in raw["features"]["xy"]}
        geo = {f["properties"]["uid"]: f["wkt"] for f in raw["features"]["lng_lat"]}
        joins[tile] = (xy, geo)
    for row, geo, px in zip(df.to_dict("records"), geoms, pixels):
        xy_raw, geo_raw = joins[row["tile_base"]]
        assert px.equals_exact(wkt.loads(xy_raw[row["building_uid"]]), 1e-8)
        assert geo.equals_exact(wkt.loads(geo_raw[row["building_uid"]]), 1e-10)
    rules = load_rules()
    records = []
    for row in df.to_dict("records"):
        baseline = dict(row, STAT_LAND_=infer_original_code(row, rules))
        original_type, original_units, residential = rules["classify_building"](baseline)
        assert original_type == row["building_type"] and original_units == row["estimated_units"]
        baseline.update(building_type=original_type, estimated_units=original_units, is_residential=residential)
        assert abs(rules["estimate_population"](baseline) - row["estimated_population"]) < 1e-7
        fixed = dict(baseline, footprint_m2=row["area_equal_area_m2"])
        btype, units, residential = rules["classify_building"](fixed)
        fixed.update(building_type=btype, estimated_units=units, is_residential=residential)
        area_only_pop = rules["estimate_population"](dict(baseline, footprint_m2=row["area_equal_area_m2"]))
        records.append({"building_uid": row["building_uid"], "tile_base": row["tile_base"], "classification_source": row["classification_source"], "old_type": row["type_class_name"], "counterfactual_type": rules["CLASS_NAMES"][rules["TYPE_TO_CLASS"][btype]], "old_building_type": original_type, "counterfactual_building_type": btype, "old_population": row["estimated_population"], "counterfactual_population": rules["estimate_population"](fixed), "population_area_only_fixed_type": area_only_pop, "old_units": original_units, "counterfactual_units": units})
    cf = pd.DataFrame(records)
    audit = df[["building_uid", "tile_base", "gsd", "footprint_m2", "area_utm_m2", "area_equal_area_m2", "area_geodesic_m2", "effective_gsd_m", "area_scale_ratio", "GEOID", "estimated_population", "type_class_name"]].merge(cf, on=["building_uid", "tile_base"], validate="one_to_one")
    audit.to_csv(OUT / "training_area_label_audit.csv", index=False)
    pd.DataFrame(metadata).to_csv(OUT / "training_tile_metadata.csv", index=False)
    transitions = pd.crosstab(cf.old_type, cf.counterfactual_type)
    transitions.to_csv(OUT / "label_transitions.csv")
    summary = {"n": len(df), "tiles": df.tile_base.nunique(), "raw_uid_geometry_checks": "all_match", "area_scale_ratio": quantiles(df.area_scale_ratio), "metadata_gsd_m": quantiles(df.gsd), "effective_gsd_m": quantiles(df.effective_gsd_m), "stored_area_formula_max_relative_error": float(np.max(abs(df.footprint_m2 / (np.array([p.area for p in pixels]) * df.gsd ** 2) - 1))), "equal_area_geodesic_max_relative_difference": float(np.max(abs(df.area_equal_area_m2 / df.area_geodesic_m2 - 1))), "original_class_counts": df.type_class_name.value_counts().to_dict(), "classification_sources": df.classification_source.value_counts().to_dict(), "baseline_rule_reproduction": "all_23014_type_units_population_match_using_archived_source_family", "counterfactual_definition": "Correct polygon area only; keep archived parcel-derived use family and ACS ratios; reconstruct equivalent codes, not a new raw parcel join; synthetic diagnostic, not ground truth", "counterfactual_class_counts": cf.counterfactual_type.value_counts().to_dict(), "type_changed_n": int((cf.old_type != cf.counterfactual_type).sum()), "population_old_sum": float(cf.old_population.sum()), "population_counterfactual_sum": float(cf.counterfactual_population.sum()), "population_area_only_fixed_type_sum": float(cf.population_area_only_fixed_type.sum()), "units_old_sum": int(cf.old_units.sum()), "units_counterfactual_sum": int(cf.counterfactual_units.sum()), "population_old_median": float(cf.old_population.median()), "population_counterfactual_median": float(cf.counterfactual_population.median()), "per_old_class": {c: {"n": len(g), "old_area_median": float(g.footprint_m2.median()), "physical_area_median": float(g.area_equal_area_m2.median()), "old_population_median": float(g.old_population.median()), "counterfactual_population_median": float(g.counterfactual_population.median()), "type_changed_n": int((g.old_type != g.counterfactual_type).sum())} for c, g in audit.groupby("old_type")}}
    save_json("training_summary.json", summary)
    save_json("training_provenance.json", {"archive": str(package), "archive_sha256": hashlib.sha256(package.read_bytes()).hexdigest(), "notebook": str(LEGACY / "building_population_pipeline.ipynb"), "notebook_sha256": hashlib.sha256((LEGACY / "building_population_pipeline.ipynb").read_bytes()).hexdigest(), "raw_xbd_label_sha256": raw_hashes, "crs": {"primary_area": "EPSG:5070", "crosschecks": ["EPSG:32615", "WGS84 ellipsoid"]}})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
