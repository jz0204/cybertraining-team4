# Stage 2A training scale and independent LA type evaluation

Date: 2026-10-15 (America/Chicago). Original results are unchanged. This report audits frozen artifacts; no model was retrained, fine-tuned, recalibrated or deployed.

## Findings

The training label area is systematically inconsistent with its own geographic polygons. The frozen model also disagrees strongly with independent pre-fire LA Assessor use records. These are separate observations: a direct numeric-area intervention measures only one mechanism and does not establish that it explains the entire LA failure.

![Audit summary](figures/audit_summary.png)

## 1. Training area and GSD

All 23,014 archived rows match their original xBD pre-disaster UID, pixel polygon and geographic polygon across 300 tiles. Every archived area reproduces `pixel_polygon.area * metadata.gsd**2` (maximum relative error 6.7e-16). Thus the discrepancy is not caused by reading different buildings or parsing mismatched label geometry.

The coordinate-derived effective pixel scale has median **0.466859 m/pixel**, while metadata GSD has median **2.201535 m/pixel**. Archived area / EPSG:5070 area has median **22.237019**, range **16.602027–22.287417**. This is a pixel-grid mismatch, not a universal scalar to apply to LA. EPSG:5070 area agrees with WGS84 ellipsoidal area to within 0.001454%; EPSG:32615 was also calculated. Metadata `pan_resolution` is retained per tile but is not substituted blindly for the PNG's measured scale.

The same GSD also controls the original crop context buffer (`20 / gsd` pixels), so the crop's nominal metric context is affected. Existing masks and image alignment were not changed in this audit.

All 300 source tiles were captured on 2017-02-16. The demographic reference is ACS 2022 and the supplied Harris parcel archive is dated August 2025. This temporal mismatch is another possible source of use-label error; correcting arithmetic alone does not resolve it.

Evidence: [training summary](training_summary.json), [per-building calculations](training_area_label_audit.csv), [tile metadata](training_tile_metadata.csv), [source hashes](training_provenance.json). Source implementation: `stage2a_old_training_data_codes/building_population_pipeline.ipynb` in the Instance-Impact repository.

## 2. Label and exposure-proxy impact

We reproduce all 23,014 original type, unit-count and population labels before intervention. For the counterfactual, only polygon area changes; archived ACS ratios and parcel-derived use families remain fixed. Equivalent parcel codes are reconstructed from archived families. **This is not a fresh parcel spatial join, independently corrected ground truth, or a replacement training dataset.** The audit does not recover parcels that were originally unmatched or incorrectly matched.

Independent source review identified two additional historical provenance limits in the original notebook: the parcel-join comment promises the smallest-area match, but the code drops duplicate building rows without first sorting parcel area; oversized duplex/triplex parcel codes can be overridden by the area heuristic, while `classification_source` only explicitly recognizes the single-family/mobile override. No affected-row count or historical parcel correctness is established here. The reconstructed source family can reproduce saved labels without proving the original parcel attribution or source flag correct.

Archived source flags identify 18,460 labels (80.2%) as `area_heuristic`; because of the legacy flag omission, this is not an independently established exhaustive count. Correct area in the documented reconstructed-code diagnostic changes 18,280 / 23,014 class labels (79.43%). Commercial falls from 17,392 to 1,333; small residential rises from 349 to 17,447. Housing-unit estimates change from 86,122 to 35,443.

```
         old_type  commercial  institutional  other  residential_multi  residential_small
       commercial        1333              0      0               1182              14877
    institutional           0            632      0                  0                  0
            other           0              0    954                  0                  0
residential_multi           0              0      0               1466               2221
residential_small           0              0      0                  0                349
```

Commercial-labelled training buildings have median physical footprint 282.4 m² (stored area 6210.8 m²). Small-residential-labelled buildings have median physical footprint only 16.5 m². The original taxonomy therefore does not consistently represent the intended semantic building uses.

Training exposure-proxy sums: archived **6,664,678.2**; area-only counterfactual **183,175.2**; area plus regrouped ACS counterfactual **180,599.0**. Corresponding medians: **215.8**, **2.8**, **2.6**. These are mixed residential/non-residential exposure values and must not be called resident counts. Fixed-type area-only sensitivity is also saved separately.

Evidence: [class transitions](label_transitions.csv), [population counterfactuals](population_proxy_counterfactuals.csv).

## 3. ACS parameter audit

All archived PPU and occupancy values for the 21,140 rows with a GEOID reproduce from the official 2022 ACS bulk tables under the original code, within floating-point error. This demonstrates that the incorrect variable interpretation actually entered the archived labels; it is not merely an outdated comment.

| Variable | Original interpretation | Official Census meaning |
|---|---|---|
| `B25033_002E` | `pop_1unit_detached` | Owner occupied: |
| `B25033_004E` | `pop_2units` | Owner occupied:!!2 to 4 |
| `B25033_005E` | `pop_3to4units` | Owner occupied:!!5 or more |
| `B25033_010E` | `pop_5to9units` | Renter occupied:!!2 to 4 |
| `B25033_011E` | `pop_10plus_units` | Renter occupied:!!5 or more |
| `B25032_003E` | `occ_1unit_detached` | Owner-occupied housing units:!!1, detached |
| `B25032_004E` | `occ_2units` | Owner-occupied housing units:!!1, attached |
| `B25032_005E` | `occ_3to4units` | Owner-occupied housing units:!!2 |
| `B25032_008E` | `occ_5to9units` | Owner-occupied housing units:!!10 to 19 |
| `B25032_011E` | `occ_10plus_units` | Owner-occupied housing units:!!Mobile home |
| `B25002_001E` | `total_housing_units` | Estimate!!Total: |
| `B25002_002E` | `occupied_housing_units` | Occupied |

The diagnostic regrouping combines owner and renter numerators with matching occupied-unit denominators. B25033 supports **1 unit**, **2–4 units**, **5+ units**, mobile homes and other accommodation. It cannot provide separate detached-only, duplex-only, triplex-only, 5–9-only or 10+-only population ratios. The grouped counterfactual uses one ratio within each supported group; original caps, default bounds and non-residential density assumptions remain. Population accuracy remains unvalidated.

Official definitions: [B25033](https://api.census.gov/data/2022/acs/acs5/groups/B25033.html), [B25032](https://api.census.gov/data/2022/acs/acs5/groups/B25032.html). Values were downloaded from the [official table-based Summary File](https://www.census.gov/programs-surveys/acs/data/summary-file.2022.html), because direct data queries returned a missing-key page. Only Harris County block-group rows were retained; full downloaded response hashes and byte counts are recorded under `raw/`.

Evidence: [ACS summary](acs_summary.json), [mapping audit](acs_variable_mapping_audit.json), [regrouped ratios](acs_corrected_grouped_ratios.csv).

## 4. Frozen model impact

All seven packaged model artifacts passed their recorded checksums. Local CPU population inference reproduces all **21,888** returned LA population predictions, maximum absolute error **2.84e-14**. Holding type probabilities fixed and deliberately multiplying LA area by the historical median scale changes the median output from **2.344** to **98.358**. This proves sensitivity, not a valid correction. LA polygon areas are already metric; inflating them would reintroduce the training error.

The CNN intervention uses **300 training-package examples, 100 per archived active class**, all five frozen checkpoints, identical RGB/masks and identical image features. Replacing only numeric log-area changes **0 / 300** ensemble class decisions; mean probability L1 change is **0.0038**. Zeroing all geometry inputs changes **1 / 300** decisions. These are mechanistic diagnostics on a class-balanced package sample, not held-out accuracy estimates or a full LA CNN rerun. The sample can overlap original training and validation sets.

Evidence: [CNN sensitivity](cnn_area_sensitivity.json), [per-example outputs](cnn_area_sensitivity_predictions.csv), [population sensitivity](population_model_sensitivity.json), [full LA population interventions](la_population_model_sensitivity.csv). Local environment: Python 3.11, torch 2.0.0, torchvision 0.15.1, NumPy 1.26.4, scikit-learn 1.9.0; LA original CUDA run used torch 2.7.1. No backbone weights were downloaded.

## 5. Independent LA Assessor evaluation

Source: [LA County Assessor public feature service](https://services.arcgis.com/RmCCgQtiZLDCtblq/ArcGIS/rest/services/Parcel_Data_2021_Table/FeatureServer/0), roll **2024**, preceding the January 2025 fires. Retrieved **8,577 / 8,581** requested AINs; four were absent. The original LARIAC/DINS reference contains 14,369 outlines, with 11,403 existing GT-to-prediction pairs. These pairs include 1,810 rows where multiple GT buildings share one prediction; all such pairs are excluded from the evaluated cohorts.

Map use codes independently of the model: 01/02/03/07 → small residential (includes duplex/triplex, consistent with training); 04/05 → multi residential; unambiguous 1x/2x commercial → commercial. Exclude institutional/industrial/other uses, vacant/improvements-only records, condominiums, commercial combinations/parking, nonpositive main floor area, and residential records without a positive unit count.

The primary cohort requires a unique matched prediction, existing registration shifts, IoU ≥0.5, and exactly one LARIAC outline on the parcel within the reference region. A stricter sensitivity also requires one Assessor building data line. Multiple buildings on a parcel can have different functions; the broader cohorts are therefore lower-certainty parcel-use proxies. Assessor floor area and units are parcel attributes, not observations of residents per building. Assessor uses are independent of the Stage 2A training labels and predictions.

| Cohort | Buildings | Accuracy | Fixed three-class macro-F1 | Always-small baseline accuracy |
|---|---:|---:|---:|---:|
| broad_parcel_labels | 9,036 | 30.75% | 0.1700 | 97.08% |
| iou50 | 7,601 | 27.60% | 0.1575 | 97.00% |
| primary_single_outline_iou50 | 3,332 | 7.02% | 0.0484 | 98.32% |
| strict_single_assessor_building | 3,315 | 6.91% | 0.0475 | 98.40% |

Primary-cohort confusion matrix (rows reference; columns predictions; order small / multi / commercial):

```
[[ 202  847 2227]
 [   1    1   12]
 [   4    7   31]]
```

Primary reference counts: small **3,276**, multi **14**, commercial **42**. Multi and commercial estimates have limited support; the principal evidenced failure is low recall of the dominant small-residential class (**6.17%**). The majority-small baseline is descriptive, not a deployable classifier.

Supplementary input/match checks: removing all primary-cohort crops with any zero pixels leaves 2,932 buildings and accuracy 6.86%. An exploratory stricter IoU≥0.7 subset has 1,943 buildings and accuracy 5.15%. These post-evaluation diagnostics do not replace the fixed primary threshold or choose a best-performing subset. See [robustness metrics](assessor_robustness.json).

The primary set covers **3,332 of 21,888 predictions (15.22%)**, in **92 cells**, and has median footprint **282.9 m²**. The broader one-to-one set has median footprint **193.3 m²**. Thus the approximately 7% and 31% accuracy values describe different selected cohorts, not competing estimates for all LA buildings. 400 primary-cohort crops contain some zero pixels and remain included, matching the original frozen run. Type evaluation does not condition on DINS damage category.

Per-fire primary results: {"Eaton": {"n": 2477, "accuracy": 0.09204683084376261, "macro_f1": 0.06270605595491108}, "Palisades": {"n": 855, "accuracy": 0.007017543859649123, "macro_f1": 0.004645760743321719}}. Palisades has zero multi-residential and only one commercial reference example in this cohort, so its three-class macro-F1 is not a reliable estimate of those minority-class capabilities.

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
