"""Build the source-linked audit report and static figures from saved results."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_sources import OUT, ROOT
from evaluate_assessor import score


def main():
    load = lambda name: json.loads((OUT / name).read_text())
    train = load("training_summary.json")
    acs = load("acs_summary.json")
    evaluation = load("assessor_evaluation.json")
    population = load("population_model_sensitivity.json")
    cnn = load("cnn_area_sensitivity.json")
    primary = evaluation["cohorts"]["primary_single_outline_iou50"]
    data = pd.read_csv(OUT / "la_assessor_matched.csv")
    subset = data[data.primary_single_outline_iou50]
    robustness = {"primary_no_zero_pixels": score(subset[subset.pre_zero_fraction == 0]), "primary_iou70_exploratory": score(subset[subset.coregistered_iou >= .7])}
    (OUT / "assessor_robustness.json").write_text(json.dumps(robustness, indent=2))
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    values = train["original_class_counts"]
    fixed = train["counterfactual_class_counts"]
    classes = ["residential_small", "residential_multi", "commercial", "institutional", "other"]
    y = np.arange(len(classes))
    axes[0].barh(y + .18, [values[c] for c in classes], height=.35, label="Archived proxy")
    axes[0].barh(y - .18, [fixed[c] for c in classes], height=.35, label="Area-only counterfactual")
    axes[0].set(yticks=y, yticklabels=["Small residential", "Multi residential", "Commercial", "Institutional", "Other"], xlabel="Training labels", title="A. Area error changes label semantics")
    axes[0].legend(fontsize=8)
    matrix = np.array(primary["confusion_rows_truth_columns_prediction"])
    axes[1].imshow(matrix / matrix.sum(axis=1, keepdims=True), vmin=0, vmax=1, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i,j] / matrix[i].sum() > .5 else "black")
    axes[1].set(xticks=range(3), yticks=range(3), xticklabels=["Small", "Multi", "Comm."], yticklabels=["Small", "Multi", "Comm."], xlabel="Frozen model prediction", ylabel="2024 Assessor use", title=f"B. LA independent reference (n={primary['n']:,})")
    scenarios = acs["population_scenarios"]
    names = ["estimated_population", "counterfactual_population", "population_corrected_area_and_acs"]
    totals = [scenarios[n]["sum"] for n in names]
    axes[2].bar(range(3), totals, color=["#a04c3b", "#427998", "#469473"])
    axes[2].set(yscale="log", ylim=(1e4, 1e7), xticks=range(3), xticklabels=["Archived", "Area only", "Area + ACS"], ylabel="Sum of exposure proxy (log scale)", title="C. Proxy construction sensitivity")
    for i, value in enumerate(totals):
        axes[2].text(i, value * 1.08, f"{value:,.0f}", ha="center", fontsize=9)
    fig.savefig(figures / "audit_summary.png", dpi=180)
    plt.close(fig)
    metrics_table = "\n".join(f"| {name} | {s['n']:,} | {s['accuracy']:.2%} | {s['macro_f1']:.4f} | {s['majority_small_accuracy']:.2%} |" for name, s in evaluation["cohorts"].items())
    transition = pd.read_csv(OUT / "label_transitions.csv").to_string(index=False)
    use_variable_audit = load("acs_variable_mapping_audit.json")
    mapping_table = "\n".join(f"| `{r['code']}` | `{r['notebook_assumed_meaning']}` | {r['official_meaning'].replace('Estimate!!Total:!!','')} |" for r in use_variable_audit)
    report = f"""# Stage 2A training scale and independent LA type evaluation

Date: 2026-10-15 (America/Chicago). Original results are unchanged. This report audits frozen artifacts; no model was retrained, fine-tuned, recalibrated or deployed.

## Findings

The training label area is systematically inconsistent with its own geographic polygons. The frozen model also disagrees strongly with independent pre-fire LA Assessor use records. These are separate observations: a direct numeric-area intervention measures only one mechanism and does not establish that it explains the entire LA failure.

![Audit summary](figures/audit_summary.png)

## 1. Training area and GSD

All {train['n']:,} archived rows match their original xBD pre-disaster UID, pixel polygon and geographic polygon across {train['tiles']} tiles. Every archived area reproduces `pixel_polygon.area * metadata.gsd**2` (maximum relative error {train['stored_area_formula_max_relative_error']:.2g}). Thus the discrepancy is not caused by reading different buildings or parsing mismatched label geometry.

The coordinate-derived effective pixel scale has median **{train['effective_gsd_m']['p50']:.6f} m/pixel**, while metadata GSD has median **{train['metadata_gsd_m']['p50']:.6f} m/pixel**. Archived area / EPSG:5070 area has median **{train['area_scale_ratio']['p50']:.6f}**, range **{train['area_scale_ratio']['min']:.6f}–{train['area_scale_ratio']['max']:.6f}**. This is a pixel-grid mismatch, not a universal scalar to apply to LA. EPSG:5070 area agrees with WGS84 ellipsoidal area to within {train['equal_area_geodesic_max_relative_difference']:.6%}; EPSG:32615 was also calculated. Metadata `pan_resolution` is retained per tile but is not substituted blindly for the PNG's measured scale.

The same GSD also controls the original crop context buffer (`20 / gsd` pixels), so the crop's nominal metric context is affected. Existing masks and image alignment were not changed in this audit.

All 300 source tiles were captured on 2017-02-16. The demographic reference is ACS 2022 and the supplied Harris parcel archive is dated August 2025. This temporal mismatch is another possible source of use-label error; correcting arithmetic alone does not resolve it.

Evidence: [training summary](training_summary.json), [per-building calculations](training_area_label_audit.csv), [tile metadata](training_tile_metadata.csv), [source hashes](training_provenance.json). Source implementation: `stage2a_old_training_data_codes/building_population_pipeline.ipynb` in the Instance-Impact repository.

## 2. Label and exposure-proxy impact

We reproduce all 23,014 original type, unit-count and population labels before intervention. For the counterfactual, only polygon area changes; archived ACS ratios and parcel-derived use families remain fixed. Equivalent parcel codes are reconstructed from archived families. **This is not a fresh parcel spatial join, independently corrected ground truth, or a replacement training dataset.** The audit does not recover parcels that were originally unmatched or incorrectly matched.

Independent source review identified two additional historical provenance limits in the original notebook: the parcel-join comment promises the smallest-area match, but the code drops duplicate building rows without first sorting parcel area; oversized duplex/triplex parcel codes can be overridden by the area heuristic, while `classification_source` only explicitly recognizes the single-family/mobile override. No affected-row count or historical parcel correctness is established here. The reconstructed source family can reproduce saved labels without proving the original parcel attribution or source flag correct.

Archived source flags identify {train['classification_sources']['area_heuristic']:,} labels ({train['classification_sources']['area_heuristic']/train['n']:.1%}) as `area_heuristic`; because of the legacy flag omission, this is not an independently established exhaustive count. Correct area in the documented reconstructed-code diagnostic changes {train['type_changed_n']:,} / {train['n']:,} class labels ({train['type_changed_n']/train['n']:.2%}). Commercial falls from {train['original_class_counts']['commercial']:,} to {train['counterfactual_class_counts']['commercial']:,}; small residential rises from {train['original_class_counts']['residential_small']:,} to {train['counterfactual_class_counts']['residential_small']:,}. Housing-unit estimates change from {train['units_old_sum']:,} to {train['units_counterfactual_sum']:,}.

```
{transition}
```

Commercial-labelled training buildings have median physical footprint {train['per_old_class']['commercial']['physical_area_median']:.1f} m² (stored area {train['per_old_class']['commercial']['old_area_median']:.1f} m²). Small-residential-labelled buildings have median physical footprint only {train['per_old_class']['residential_small']['physical_area_median']:.1f} m². The original taxonomy therefore does not consistently represent the intended semantic building uses.

Training exposure-proxy sums: archived **{acs['population_scenarios']['estimated_population']['sum']:,.1f}**; area-only counterfactual **{acs['population_scenarios']['counterfactual_population']['sum']:,.1f}**; area plus regrouped ACS counterfactual **{acs['population_scenarios']['population_corrected_area_and_acs']['sum']:,.1f}**. Corresponding medians: **{acs['population_scenarios']['estimated_population']['median']:.1f}**, **{acs['population_scenarios']['counterfactual_population']['median']:.1f}**, **{acs['population_scenarios']['population_corrected_area_and_acs']['median']:.1f}**. These are mixed residential/non-residential exposure values and must not be called resident counts. Fixed-type area-only sensitivity is also saved separately.

Evidence: [class transitions](label_transitions.csv), [population counterfactuals](population_proxy_counterfactuals.csv).

## 3. ACS parameter audit

All archived PPU and occupancy values for the 21,140 rows with a GEOID reproduce from the official 2022 ACS bulk tables under the original code, within floating-point error. This demonstrates that the incorrect variable interpretation actually entered the archived labels; it is not merely an outdated comment.

| Variable | Original interpretation | Official Census meaning |
|---|---|---|
{mapping_table}

The diagnostic regrouping combines owner and renter numerators with matching occupied-unit denominators. B25033 supports **1 unit**, **2–4 units**, **5+ units**, mobile homes and other accommodation. It cannot provide separate detached-only, duplex-only, triplex-only, 5–9-only or 10+-only population ratios. The grouped counterfactual uses one ratio within each supported group; original caps, default bounds and non-residential density assumptions remain. Population accuracy remains unvalidated.

Official definitions: [B25033](https://api.census.gov/data/2022/acs/acs5/groups/B25033.html), [B25032](https://api.census.gov/data/2022/acs/acs5/groups/B25032.html). Values were downloaded from the [official table-based Summary File](https://www.census.gov/programs-surveys/acs/data/summary-file.2022.html), because direct data queries returned a missing-key page. Only Harris County block-group rows were retained; full downloaded response hashes and byte counts are recorded under `raw/`.

Evidence: [ACS summary](acs_summary.json), [mapping audit](acs_variable_mapping_audit.json), [regrouped ratios](acs_corrected_grouped_ratios.csv).

## 4. Frozen model impact

All seven packaged model artifacts passed their recorded checksums. Local CPU population inference reproduces all **{population['n']:,}** returned LA population predictions, maximum absolute error **{population['baseline_reproduction_max_absolute_error']:.3g}**. Holding type probabilities fixed and deliberately multiplying LA area by the historical median scale changes the median output from **{population['baseline_population_quantiles']['p50']:.3f}** to **{population['inflated_area_population_quantiles']['p50']:.3f}**. This proves sensitivity, not a valid correction. LA polygon areas are already metric; inflating them would reintroduce the training error.

The CNN intervention uses **{cnn['n']} training-package examples, 100 per archived active class**, all five frozen checkpoints, identical RGB/masks and identical image features. Replacing only numeric log-area changes **{cnn['physical_area']['argmax_changed_vs_original']} / {cnn['n']}** ensemble class decisions; mean probability L1 change is **{cnn['physical_area']['mean_probability_L1_change']:.4f}**. Zeroing all geometry inputs changes **{cnn['zero_geometry']['argmax_changed_vs_original']} / {cnn['n']}** decisions. These are mechanistic diagnostics on a class-balanced package sample, not held-out accuracy estimates or a full LA CNN rerun. The sample can overlap original training and validation sets.

Evidence: [CNN sensitivity](cnn_area_sensitivity.json), [per-example outputs](cnn_area_sensitivity_predictions.csv), [population sensitivity](population_model_sensitivity.json), [full LA population interventions](la_population_model_sensitivity.csv). Local environment: Python 3.11, torch 2.0.0, torchvision 0.15.1, NumPy 1.26.4, scikit-learn 1.9.0; LA original CUDA run used torch 2.7.1. No backbone weights were downloaded.

## 5. Independent LA Assessor evaluation

Source: [LA County Assessor public feature service](https://services.arcgis.com/RmCCgQtiZLDCtblq/ArcGIS/rest/services/Parcel_Data_2021_Table/FeatureServer/0), roll **2024**, preceding the January 2025 fires. Retrieved **{evaluation['returned_ains']:,} / {evaluation['requested_ains']:,}** requested AINs; four were absent. The original LARIAC/DINS reference contains 14,369 outlines, with 11,403 existing GT-to-prediction pairs. These pairs include 1,810 rows where multiple GT buildings share one prediction; all such pairs are excluded from the evaluated cohorts.

Map use codes independently of the model: 01/02/03/07 → small residential (includes duplex/triplex, consistent with training); 04/05 → multi residential; unambiguous 1x/2x commercial → commercial. Exclude institutional/industrial/other uses, vacant/improvements-only records, condominiums, commercial combinations/parking, nonpositive main floor area, and residential records without a positive unit count.

The primary cohort requires a unique matched prediction, existing registration shifts, IoU ≥0.5, and exactly one LARIAC outline on the parcel within the reference region. A stricter sensitivity also requires one Assessor building data line. Multiple buildings on a parcel can have different functions; the broader cohorts are therefore lower-certainty parcel-use proxies. Assessor floor area and units are parcel attributes, not observations of residents per building. Assessor uses are independent of the Stage 2A training labels and predictions.

| Cohort | Buildings | Accuracy | Fixed three-class macro-F1 | Always-small baseline accuracy |
|---|---:|---:|---:|---:|
{metrics_table}

Primary-cohort confusion matrix (rows reference; columns predictions; order small / multi / commercial):

```
{np.array(primary['confusion_rows_truth_columns_prediction'])}
```

Primary reference counts: small **{primary['truth_counts'].get('residential_small',0):,}**, multi **{primary['truth_counts'].get('residential_multi',0):,}**, commercial **{primary['truth_counts'].get('commercial',0):,}**. Multi and commercial estimates have limited support; the principal evidenced failure is low recall of the dominant small-residential class (**{primary['per_class']['residential_small']['recall']:.2%}**). The majority-small baseline is descriptive, not a deployable classifier.

Supplementary input/match checks: removing all primary-cohort crops with any zero pixels leaves {robustness['primary_no_zero_pixels']['n']:,} buildings and accuracy {robustness['primary_no_zero_pixels']['accuracy']:.2%}. An exploratory stricter IoU≥0.7 subset has {robustness['primary_iou70_exploratory']['n']:,} buildings and accuracy {robustness['primary_iou70_exploratory']['accuracy']:.2%}. These post-evaluation diagnostics do not replace the fixed primary threshold or choose a best-performing subset. See [robustness metrics](assessor_robustness.json).

The primary set covers **{len(subset):,} of 21,888 predictions ({len(subset)/21888:.2%})**, in **{subset.cell_id.nunique()} cells**, and has median footprint **{subset.footprint_m2.median():.1f} m²**. The broader one-to-one set has median footprint **{data[data.broad_parcel_labels].footprint_m2.median():.1f} m²**. Thus the approximately 7% and 31% accuracy values describe different selected cohorts, not competing estimates for all LA buildings. {int((subset.pre_zero_fraction>0).sum())} primary-cohort crops contain some zero pixels and remain included, matching the original frozen run. Type evaluation does not condition on DINS damage category.

Per-fire primary results: {json.dumps({k: {'n':v['n'],'accuracy':v['accuracy'],'macro_f1':v['macro_f1']} for k,v in evaluation['by_fire_primary'].items()})}. Palisades has zero multi-residential and only one commercial reference example in this cohort, so its three-class macro-F1 is not a reliable estimate of those minority-class capabilities.

Evidence: [evaluation metrics](assessor_evaluation.json), [all matched rows and cohort flags](la_assessor_matched.csv), [use-code mapping](assessor_use_mapping.csv), [source hashes](assessor_provenance.json). Geometries were compared in EPSG:32611 after the existing label-independent registration shifts; the 0.5 threshold was fixed before scoring. No threshold tuning against type accuracy was performed.

## Interpretation and remaining scope

The evidence supports correcting training-data construction before considering model adaptation. Correct-area counterfactual labels are not independently validated building uses and should not silently replace the originals. A clean future training dataset needs reliable parcel/building use attribution, physical metric area, matched Census definitions and a fresh held-out evaluation.

This audit did not retrain models, reconstruct the raw Harris parcel join, validate actual building occupancy, or rerun the full LA image classifier (the complete crop tree is on the lab server). The 136 empty-mask inputs remain excluded; no original input, label, weight or prediction was overwritten. No statistical significance claims, spatial regression, MAUP sweep or residual Moran test are made: the task uses individual-building matching, descriptive classification metrics and deterministic interventions rather than a new spatial model. No confidence intervals are reported.

## Reproduction

Verification: all 10 handoff/evaluator tests pass; [validation record](VALIDATION.json). A fresh same-family reviewer independently reproduced the area calculations, original/counterfactual labels, IoUs and final classification metrics. The [independent audit](EXPERIMENT_AUDIT.md) has overall WARN for historical parcel provenance and selected-reference scope, with no unresolved scoring error; review independence is same-family/provisional. [Machine-readable audit](EXPERIMENT_AUDIT.json).

From `handoff/stage2a_la/`, use the existing compatible Python environment:

```bash
python audit_sources.py
python download_audit_acs.py
python audit_training.py
python audit_acs.py
python audit_model.py
python evaluate_assessor.py
python summarize_audit.py
python -m unittest discover -s tests -v
```

`audit_sources.py` refuses to overwrite its Assessor snapshot; reuse the saved snapshot for offline reproduction. Other audit outputs can be regenerated from preserved sources. Existing `geopandas`, `pyproj`, `pandas`, `requests` and `matplotlib` were used; project dependency files were unchanged. Raw xBD and archived training labels are local in the Instance-Impact repository; original paths and hashes are in `training_provenance.json`.
"""
    (OUT / "REPORT.md").write_text(report)
    files = [p for p in OUT.rglob("*") if p.is_file() and "model_sample" not in p.parts and p.name != "artifact_sha256.json"]
    scripts = [p for p in Path(__file__).parent.glob("*.py") if p.name.startswith(("audit_", "evaluate_assessor", "download_audit", "summarize_audit"))]
    (OUT / "artifact_sha256.json").write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files + scripts}, indent=2))
    print("[report]", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
