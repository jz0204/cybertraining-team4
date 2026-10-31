"""Audit Census variable meanings and a separately labelled proxy counterfactual."""

import json
import zipfile

import numpy as np
import pandas as pd

from audit_sources import OUT
from audit_training import LEGACY, load_rules, save_json, infer_original_code


def main():
    rules = load_rules()
    tables = []
    for group in ["B25033", "B25032", "B25002"]:
        table = pd.read_csv(OUT / f"raw/acs2022_{group}_harris.dat", sep="|")
        table["GEOID"] = table.GEO_ID.str.removeprefix("1500000US")
        table = table.rename(columns={f"{group}_E{i:03d}": f"{group}_{i:03d}E" for i in range(1, 24)})
        tables.append(table[["GEOID"] + [c for c in table.columns if c.endswith("E")]])
    acs = tables[0].merge(tables[1], on="GEOID", validate="one_to_one").merge(tables[2], on="GEOID", validate="one_to_one")
    assert len(acs) == 2830 and acs.GEOID.is_unique
    with zipfile.ZipFile(LEGACY / "Harris_CBG_Building_Population.zip") as archive:
        labels = pd.read_csv(archive.open("ml_dataset/labels.csv"), dtype={"GEOID": str})
    old = rules["compute_cbg_ratios"](acs.rename(columns=rules["ACS_VARIABLES"]))
    old = old.set_index("GEOID")
    cols = [c for c in labels if c.startswith("ppu_")] + ["occupancy_rate"]
    old_checks = {}
    for col in cols:
        has = labels.GEOID.notna() & labels[col].notna()
        ref = old.loc[labels.loc[has, "GEOID"], col].to_numpy()
        error = np.abs(labels.loc[has, col].to_numpy() - ref)
        old_checks[col] = {"n": int(has.sum()), "match_n": int((error < 1e-8).sum()), "max_absolute_difference": float(error.max())}
        assert np.all(error < 1e-8), (col, old_checks[col])
    def value(row, group, indices):
        return sum(row[f"{group}_{i:03d}E"] for i in indices)
    corrected = []
    for row in acs.to_dict("records"):
        sf = rules["safe_ratio"](value(row,"B25033",[3,9]), value(row,"B25032",[3,4,14,15]), 2.8)
        two_four = rules["safe_ratio"](value(row,"B25033",[4,10]), value(row,"B25032",[5,6,16,17]), 2.5)
        five_plus = rules["safe_ratio"](value(row,"B25033",[5,11]), value(row,"B25032",[7,8,9,10,18,19,20,21]), 2.0)
        mobile = rules["safe_ratio"](value(row,"B25033",[6,12]), value(row,"B25032",[11,22]), 2.8)
        corrected.append({"GEOID": row["GEOID"], "ppu_single_family_detached": sf, "ppu_mobile_home": mobile, "ppu_duplex": two_four, "ppu_triplex": two_four, "ppu_small_multi_family": five_plus, "ppu_large_multi_family": five_plus, "ppu_multi_family": five_plus, "ppu_unknown": .5 * (sf + five_plus), "occupancy_rate": rules["safe_ratio"](row["B25002_002E"], row["B25002_001E"], .90)})
    correct = pd.DataFrame(corrected).set_index("GEOID")
    correct.to_csv(OUT / "acs_corrected_grouped_ratios.csv")
    audit = pd.read_csv(OUT / "training_area_label_audit.csv")
    labels = labels.merge(audit[["building_uid", "area_equal_area_m2", "counterfactual_population"]], on="building_uid", validate="one_to_one")
    population, ppu_population = [], []
    for row in labels.to_dict("records"):
        ratios = correct.loc[row["GEOID"]].to_dict() if pd.notna(row["GEOID"]) else {}
        new_ratios = dict(row, **ratios)
        ppu_population.append(rules["estimate_population"](new_ratios))
        fixed = dict(new_ratios, footprint_m2=row["area_equal_area_m2"], STAT_LAND_=infer_original_code(row, rules))
        btype, units, residential = rules["classify_building"](fixed)
        fixed.update(building_type=btype, estimated_units=units, is_residential=residential)
        population.append(rules["estimate_population"](fixed))
    labels["population_corrected_acs_only"] = ppu_population
    labels["population_corrected_area_and_acs"] = population
    labels[["building_uid", "GEOID", "estimated_population", "counterfactual_population", "population_corrected_acs_only", "population_corrected_area_and_acs"]].to_csv(OUT / "population_proxy_counterfactuals.csv", index=False)
    variable_audit = []
    for code, assumed in rules["ACS_VARIABLES"].items():
        group = code.split("_")[0]
        definition = json.loads((OUT / f"raw/acs2022_{group}_metadata.json").read_text())["variables"][code]["label"]
        variable_audit.append({"code": code, "notebook_assumed_meaning": assumed, "official_meaning": definition})
    save_json("acs_variable_mapping_audit.json", variable_audit)
    summary = {"reference": "Official 2022 ACS 5-year bulk tables; metadata from Census API", "cbgs": len(acs), "archived_ratio_reproduction": old_checks, "population_scenarios": {col: {"sum": float(labels[col].sum()), "median": float(labels[col].median())} for col in ["estimated_population", "counterfactual_population", "population_corrected_acs_only", "population_corrected_area_and_acs"]}, "limits": "Counterfactual exposure proxies only. B25033 supports 1-unit, 2-4-unit, and 5+-unit groups; cannot separately estimate detached vs attached or duplex vs triplex or 5-9 vs 10+. Existing density assumptions, caps and fallbacks retained; no observed occupancy validation.", "all_population_rows_changed_from_original_n": int((abs(labels.population_corrected_area_and_acs - labels.estimated_population) > .05).sum())}
    save_json("acs_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
