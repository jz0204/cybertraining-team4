# Stage2a: complete development, data, evaluation, and results review

**Review date: 31 October 2026 (America/Chicago).** This document consolidates the original implementation, the recorded July experiments, the October data repairs, and the experiments and LA preview completed through this date. It was checked against local source code, frozen manifests, saved predictions, machine-readable results, and experiment reports in both project repositories. It is a documentation and saved-artifact review; it does not introduce a new training run or new evaluation cohort.

The current research baseline for LA building-use classification is the **two-seed ConvNeXt C_positive_negative ensemble**, with **macro-F1 0.610370**, **98.2893% accuracy**, and **multifamily precision/recall/F1 of 16.67% / 21.43% / 0.1875** on the unchanged 3,332-building reference cohort. The most recent close-up experiment did not improve it. The complete LA preview now contains 22,024 use predictions and exposure proxies. Its population estimates remain substantially too high against the available Census comparison and are not validated building occupancy.

There are three different meanings of “current” in this project:

| Meaning | Artifact/status |
|---|---|
| Original packaged integration default | July native-three-class, five-seed model and residual-HGB exposure package; still present in the original handoff. Its training labels have subsequently identified defects. |
| Repaired Harris reference model | October 15 area-mask-pooling, five-seed model, evaluated on repaired Harris labels and later on LA. |
| Best observed LA development baseline | October 30 C_positive_negative, seeds 42/43, used in the October 31 full preview; not a formally promoted production replacement. |
| Current preview exposure calculation | C probabilities + the frozen October 26 B_semantics_mean 3 exposure head. This combination has aggregate consistency checks, but no independent per-building occupancy validation or LA-specific refit. |

Older files saying “latest model,” “SD unopened,” or “exposure not recomputed” describe their own dates. They must not override the later records summarized here.

## Contents

1. [Task, outputs, and terminology](#1-task-outputs-and-terminology)
2. [Original implementation and July experiments](#2-original-implementation-and-july-experiments)
3. [How Harris training data were constructed and repaired](#3-how-harris-training-data-were-constructed-and-repaired)
4. [Additional training data and the current C training set](#4-additional-training-data-and-the-current-c-training-set)
5. [Where the validation and test data are](#5-where-the-validation-and-test-data-are)
6. [Evaluation rules and what each score means](#6-evaluation-rules-and-what-each-score-means)
7. [October experiment ledger](#7-october-experiment-ledger)
8. [Current classifier and latest results](#8-current-classifier-and-latest-results)
9. [Exposure models and the newest Census comparison](#9-exposure-models-and-the-newest-census-comparison)
10. [What has actually been implemented](#10-what-has-actually-been-implemented)
11. [Current challenges and unresolved questions](#11-current-challenges-and-unresolved-questions)
12. [Recommended next work and success criteria](#12-recommended-next-work-and-success-criteria)
13. [Reproduction and artifact navigation](#13-reproduction-and-artifact-navigation)
14. [Operational lessons and review limitations](#14-operational-lessons-and-review-limitations)

## 1. Task, outputs, and terminology

Stage2a estimates a building's use and an exposure value from pre-event imagery and an identified building footprint. The intended downstream chain is:

```text
Stage 1 supplied building/roof footprint
  + pre-event RGB + target mask + metric footprint area
                         |
                  Stage2a use classifier
                         |
             soft building-use probabilities
                         |
          area + predicted-use exposure regression
                         |
              per-building exposure proxy
                         |
        Stage 3 aggregation with Stage2b damage outputs
```

The established three-class output is `residential_small`, `residential_multi`, and `commercial`. Small residential includes single-family/mobile homes and duplex/triplex uses. The intended building-level boundary is 1–3 dwellings versus at least 4, but historical Harris labels implement parcel-code mappings, not verified dwelling counts for each roof. The five-class experiments add `institutional` and `other`; the later transfer experiments separately predict primary/auxiliary role. `unknown` is an abstention or missing-evidence outcome, not an ordinary supervised use class.

Important distinctions:

- A **parcel** is a property/assessment unit; a **physical building or roof** can cross parcels or contain several legal units. One parcel can also contain several roofs with different functions.
- A **use reference** is the recorded assessor/parcel use assigned under the documented matching rules. It is not automatically independently verified actual use at the image date.
- `estimated_units` and `estimated_population` in the Harris data are constructed targets. They are not measured dwelling units or observed occupants.
- Classification here is conditional on a supplied footprint. These experiments do not measure building detection AP, missed-building rates, or end-to-end detection-and-use accuracy.
- The LA preview preserves existing Stage2b damage votes. A new Stage2a use prediction does not imply a new damage inference.

The original rule chain was **type → units → population**. It did not estimate units by reversing a neural population prediction.

## 2. Original implementation and July experiments

### 2.1 Original components

The starting implementation had a geospatial label/exposure notebook and a multitask EfficientNet-B0 vision model. The rule workflow joined footprints to parcel use and Census block groups, assigned type, approximated units, and calculated residential exposure as:

`estimated_population = estimated_units × people_per_unit_ratio × occupancy_rate`.

Nonresidential targets used assumed area densities. Caps, defaults, and quality flags were included. The initial CLI adapter converted Stage 1 shared-instance tables to crop/mask inputs; a 114-row integration sanity run produced 114 predictions. That checked execution and schema, not model accuracy.

Early code/configuration branches include RGB-only, geometry-only, RGB+mask+geometry, population-only, type-only, soft type conditioning, oracle-type diagnostics, hard-cascade/OOF variants, and EfficientNet/ResNet/ConvNeXt backbone configurations. Configuration or implementation presence alone does not prove a successful experiment. Detailed completed historical results available in the consolidated owning document begin with the M20 series below.

### 2.2 Historical experiment ledger: M20–M41

These scores belong to the **original, subsequently audited dataset**. They are retained as the development history, not current corrected-data performance. The principal July split was tile-blocked `m20_split`, approximately 15% validation and **no separate test partition**. A high July validation score therefore is neither a fresh test result nor comparable directly with the repaired Harris or LA scores.

| Experiment | What was tried/implemented | Recorded result and decision |
|---|---|---|
| M20 | Compare taxonomy policies: merge institutional into other, drop institutional, drop other, or drop both. | Four-class macro-F1 approximately 0.7218–0.7306; drop-both three-class 0.8628. Selected the closed-world three-class task, retaining 21,428/23,014 rows. |
| M23 | Population models using OOF predicted type: HGB, ExtraTrees, Ridge, residual HGB on Ridge geometry; compare direct pixel population regression. | Residual HGB + geometry + soft type log-MAE 0.0812; ConvNeXt population-only 0.2889; EfficientNet population-only 0.4201. Selected the tabular type/geometry route. |
| M24 | Ablate population geometry, hard/soft type, confidence, direct versus residual model. | Log-footprint + soft type residual HGB 0.0738; no-type direct HGB 0.1171. Oracle true-type diagnostic 0.0186 was non-deployable. |
| M25 | Strict deployable type inputs, five seeds, historical five-logit compatibility/projection head. | Validation macro-F1 0.8632 ± 0.0057. Historical, superseded by a native three-logit head. |
| M26 | Five-fold OOF features and a population head for each of the five M25 seeds. | Population log-MAE 0.0735 ± 0.0037. Same-split seed variation, not geographic robustness. |
| M28 | Five-seed compatibility-head probability ensemble. | Type macro-F1 0.8679; population log-MAE 0.0694. One ensemble result, not five independent ensemble replications. |
| M29 | Compatibility-head input/weighting ablations. | RGB+mask 0.8681; strict geometry recipe 0.8629; RGB-only 0.8165; removing weighting 0.8464. |
| M30 | Match the residual population architecture with and without predicted type. | Log-MAE 0.0738 with type versus 0.1148 without; direct no-type HGB 0.1171. |
| M31 | Replace projected compatibility output with native three-class ConvNeXt, five seeds. | Type macro-F1 0.8529 ± 0.0106. Became the packaged native classifier family. |
| M32 | Native-head OOF population models across five seeds. | Population log-MAE 0.0760 ± 0.0032; shifted factor-2 rate 0.9603 ± 0.0034. |
| M33 | Native five-seed probability ensemble and population package. | Type macro-F1 0.8579, ECE 0.0342; population log-MAE 0.073351. Original packaged integration default. |
| M34 | Native-head RGB/mask/geometry and imbalance ablations. | RGB+mask 0.8626; strict RGB+mask+geometry 0.8601; RGB-only 0.8048; no sampler/class weights 0.8262. |
| M35 | Native-head matched predicted-type contribution to population. | Log-MAE 0.0742 with type versus 0.1157 without type in the same residual architecture. Historical target-fidelity result. |
| M36–M40 | Retain institutional in a native four-class drop-other model; logit adjustment, seed sweeps, OOF population, ensemble and ablations. | Four-class ensemble macro-F1 0.7312; institutional F1 0.4000. Population five-seed log-MAE 0.0987 ± 0.0060; ensemble 0.0909. Did not replace the three-class route. |
| M41 | Alternative tile-blocked splits, temperature/vector calibration, ensemble uncertainty. | Alternative-split population log-MAE 0.1041 and 0.0613. Calibration improved type ECE but worsened downstream population log-MAE: raw 0.0734, temperature 0.0827, vector 0.0816. Kept raw probabilities as default. |

The M20–M41 record established the architecture and deployment interface, and showed that split composition mattered more than the reported fixed-split seed variability. It did not validate the label-construction assumptions later found to be faulty. Do not use the historical 0.1157→0.0742 population comparison as the newest repaired-data result.

Historical owning record: [II_package/docs/stage2a.md](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/docs/stage2a.md>). The shorter [pipeline integration notes](/Users/zhou/Documents/GitHub/cybertraining-team4/pipeline/docs/stage2a.md) describe an earlier EfficientNet integration stage and are not the latest model specification.

## 3. How Harris training data were constructed and repaired

### 3.1 Source data and original defects

The archived population-label dataset contains **23,014 xBD Harvey pre-disaster building instances across 300 tiles**. The source imagery is dated **16 February 2017**. Parcel attributes come from a supplied **2025 Harris County archive**, and demographic ratios from **2022 ACS**. Geographic and pixel polygons retain the original xBD UIDs.

The October 15 audit reproduced the archived labels and identified several concrete problems:

1. **Wrong effective image scale in area construction.** The original code used pixel polygon area × metadata GSD². Median metadata GSD was 2.201535 m/px, while geographic/pixel geometry implied 0.466859 m/px. Stored area was inflated by a median **22.237019×**. This also affected nominal metric crop context.
2. **Area-based use overrides.** Large stored areas could override residential parcel uses, and unknown uses could receive area-heuristic classes. Archived flags marked 18,460 rows as area heuristic, with an additional source-flag provenance limitation. The original distribution was 349 small, 3,687 multi, 17,392 commercial, 632 institutional, and 954 other.
3. **Incorrect ACS variable interpretation.** Several B25033 owner/renter and grouped structure variables were interpreted as narrower building types. B25032 denominators were not consistently matched to those numerators. The archived values reproduced the old code, confirming the problem entered targets.
4. **Parcel selection was not the documented deterministic best-overlap/smallest-parcel choice.** A fresh join was required; an area-only counterfactual could not repair historical attribution.

These defects were related but not interchangeable. Replacing only the CNN's numeric log-area changed **0/300** ensemble decisions in a diagnostic sample. That ruled out treating a scalar feature correction as a complete model repair. LA metric polygons were already metric and must not be multiplied by the historical Harris error factor.

Source audit: [October 15 audit report](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261015_gsd_assessor/REPORT.md).

### 3.2 Rebuilt areas, parcel assignments, crops, and labels

The repaired builder does the following:

- Computes full physical footprint area from geographic polygons in **EPSG:5070**, independently of visible crop area. Cross-checking all 23,014 footprints against WGS84 ellipsoidal area gave maximum relative difference approximately 0.001454%.
- Ranks parcel candidates by building coverage, then smaller metric parcel area, then a stable parcel key. Requires at least 50% coverage and excludes close competing assignments with conflicting use families; “close” candidates have coverage within 0.05 of the best and at least 0.5.
- Preserves recognized parcel uses. Unknown uses remain unresolved; area no longer creates or overrides the semantic class.
- Uses **fixed 256×256 native-pixel crops**, centroid anchored with a representative-point fallback when needed for a nonempty mask. Records truncation, visible mask fraction, and zero-pixel fraction. Full polygon area remains separate from the crop.
- Stores the source UID, polygons, parcel IDs/codes, Census GEOID, area/GSD provenance, derived labels/targets, crop/mask paths, quality fields, and frozen split.

The rebuilt all-instance distribution is 15,991 small, 1,448 multi, 1,256 commercial, 623 institutional, 817 other, and 2,879 unknown. **18,707 original class labels changed** after the full rebuild. This is distinct from the earlier area-only diagnostic, which changed 18,280 labels.

### 3.3 Exact exposure-target rules in the rebuilt code

The current repaired builder retains explicit approximations; repairing the data did not turn them into observed occupancy:

| Target component | Rebuilt implementation |
|---|---|
| Single-family / mobile home | 1 estimated unit. |
| Duplex / triplex | 2 / 3 estimated units. |
| Parcel-coded multifamily, area <2,000 m² | `min(40, max(1, int(area_m2 / 90)))` estimated units. |
| Parcel-coded multifamily, area ≥2,000 m² | `min(40, max(1, int(area_m2 / 80)))` estimated units. |
| Residential exposure | Units × matching ACS grouped people-per-occupied-unit ratio × occupancy rate, with retained residential caps; rounded to one decimal. |
| Commercial exposure | `round(0.04 × area_m2, 1)`; estimated residential units are zero. |
| Other nonresidential all-instance targets | Retained density assumptions include institutional 0.06, industrial 0.015, agricultural/oil-gas/utility 0.003 people-equivalent per m²; these uses are outside the selected three-class training task. |

For ACS, the repaired numerator sums owner and renter populations in the same supported structure group, and the denominator sums matching occupied units. B25033 supports **1 unit**, **2–4 units**, and **5+ units**, with separate mobile-home grouping. It does not supply exact detached-only, duplex-only, triplex-only, 5–9-only, or 10+-only population ratios. The code therefore shares the 2–4 ratio across duplex/triplex and the 5+ ratio across multifamily size groups. The single-family code uses the supported one-unit grouping; it does not create a detached-only numerator.

PPU ratios outside [0.5, 6] or missing/invalid numerator-denominator pairs use defaults; occupancy is validated separately in [0, 1], with default 0.91. The audited frozen residential targets include 76 PPU fallbacks. A later provenance-flag patch did not alter the frozen CSV bytes or numeric targets; consult `ACS_DEFAULT_PROVENANCE.csv` alongside the frozen labels.

The old notebook/README has different thresholds and incorrect ACS descriptions. For a reproduction of the repaired targets, use [rebuild_stage2a_dataset.py](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/scripts/stage2a/training/rebuild_stage2a_dataset.py>) and its frozen run records.

### 3.4 Rebuilt and cleaned split versions

From 18,695 recognized three-class candidates, the freeze excluded 237 rows with recorded construction later than the 2017 imagery and 422 rows on cross-partition parcels. The frozen October 15 dataset contains **18,036 rows**:

| Partition | Small | Multi | Commercial | Total |
|---|---:|---:|---:|---:|
| Train | 11,385 | 1,097 | 873 | 13,355 |
| Validation | 1,826 | 75 | 127 | 2,028 |
| Test | 2,410 | 111 | 132 | 2,653 |

The tile/parcel-disjoint freeze was later found to contain different fragments of the **same physical building** across splits: 24 physical groups, comprising 48 rows. The follow-up clean split quarantined 22 train and 2 validation rows, preserving all test IDs:

| Version | Train | Validation | Test | Use |
|---|---:|---:|---:|---|
| October 15 frozen rebuild/poolfix | 13,355 | 2,028 | 2,653 | Historical controlled repair comparisons. |
| October 24 physical-group-cleaned source | 13,333 | 2,026 | 2,653 | Later source/data experiments and October 26 OOF exposure comparison. |

All removed rows were small residential. The cleaned source totals are 15,597 small, 1,283 multi, and 1,132 commercial. “Clean” here refers to documented cross-role group isolation, not verified semantics or absence of spatial autocorrelation.

### 3.5 Physical-building reconstruction was a separate deliverable

The geometry-only v4 work mapped 23,014 old records to **22,621 physical IDs**, accepted 591 cross-tile merge groups plus one reviewed three-fragment same-tile group, and produced 592 new crop/mask groups. It used seam continuity, catalog/date agreement, and native pixel-grid evidence, not proximity alone. It also used 238 supplemental xBD test geometries only for reconstruction; affected groups were marked for future test protection.

There remain 2,469 physical outlines with unresolved source-edge/pixel-coverage problems. Precise 2017 units for the 45 reviewed Timber Run/Thicket roofs were not established. Project-level “Duplex,” account-level “Commercial,” improvement indices, legal dwelling units, and physical roof IDs proved non-equivalent. The physical-building dataset explicitly has **`training_ready=false`**; reconstructed roofs and parcel totals were not silently converted into replacement labels.

Evidence: [rebuild report](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/rebuild_20261015/REPORT.md>), [physical-building v4 report](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/physical_buildings_20261015_v4/REPORT.md>).

## 4. Additional training data and the current C training set

### 4.1 Harris physical-scale augmentation

LA crops are approximately 0.3052 m/px, versus approximately 0.4669 m/px for Harris. A 256 px window therefore covers about **78.1 m in LA versus 119.5 m in Harris**. Resizing both to 224 px does not harmonize physical field of view.

S1 generated an alternative Harris view using an approximately 167 px native window resampled to 256 px, selecting native versus LA-scale views with 50% probability per training draw. Vector masks and mask geometry were recomputed; physical polygon area and target labels stayed fixed. Of 13,355 train rows, 13,348 had valid alternative views; seven self-intersecting polygons retained explicit native fallbacks. Validation/test retained native views. This increased target scale without inventing wider source context.

### 4.2 NYC BuildingSense extension

The public BuildingSense release inspected here contains 1,000 NYC examples, not the full paper's reported dataset. Filtering retained **578: 445 small, 43 multi, 90 commercial**; 422 taxonomy-incompatible/ambiguous examples were excluded. NYC C0 three-family buildings were treated differently from C1–C3 apartment categories. Coordinates in survey feet were converted correctly, RGB bands were explicitly selected, and output crops used approximately LA sampling scale. NYC had no population target and was used for type learning only.

Important coverage limit: NYC small-residential roofs had median area approximately 93.5 m², maximum 237.3 m², and **zero above 400 m²**. Harris had 1,497 training small-residential roofs above 400 m², and the LA primary cohort had 549. Adding a city did not automatically add the large-residential hard negatives causing LA commercial false positives.

A1/A2 mixed NYC at 10%/25% of draw mass while retaining within-domain weighting. N2 used natural sampling from 13,333 Harris + 578 NYC rows with a fixed 13,333-draw epoch budget. N2 supplied the initial checkpoints later used for LA adaptation, although N1 remained preferable for the earlier high-commercial-recall development criterion.

### 4.3 Stronger five-use/role transfer data

An independent branch assembled official-reference data for matched DINO and Qwen QLoRA experiments:

| City / role | Five-use references | Auxiliary role-only | Total |
|---|---:|---:|---:|
| NYC train | 600, 120/use | 120 | 720 |
| NYC spatial validation | 149 | 25 | 174 |
| Cambridge external test | 250, 50/use | 50 | 300 |
| Seattle external test | 244 | 0 | 244 |

NYC combined PLUTO/footprint links and conservative dwelling/single-main-building filters; Cambridge combined official footprints with FY 2024/FY 2025 agreement; Seattle used assessor counts and unique footprints. Auxiliary labels had explicit structural evidence. Unverified auxiliary use was masked from use supervision rather than guessed. Target-near and real wide views were preserved, with parcel and requested-window isolation. These are stronger official references, not independent human or exact-image-date gold. This branch did **not** become the current three-class CNN's training set.

### 4.4 LA adaptation reference construction

The October 29 LA dataset was acquired outside all **126 original product cells**, previous test-wide windows, and a **2 km buffer**. It used fixed 2 km grid roles: train, validation, and reserved. Complete requested context windows had to stay within their assigned role and outside exclusions; parcel and context overlap checks were saved.

Candidates required a 2024 assessor single building-data record with supported use, a countywide 2014 single outline, and a unique matching 2020 BLD_ID with outline IoU ≥0.8, plus complete pre-fire Maxar input coverage. Recorded construction year was constrained to precede the historical outline. Geometry consistency corroborates location; it does not certify unchanged use or units across dates.

Small-residential acquisition sampled one in 20 OBJECTIDs. Consequently validation uses **weight 20 for small residential, 1 for multi/commercial** when choosing epochs. This adjusts for the acquisition design; it does not establish a representative countywide validation sample after all filtering.

The initial new LA cohort contained **1,186 buildings**:

| Role | Small | Multi | Commercial | Total |
|---|---:|---:|---:|---:|
| Train | 478 | 93 | 177 | 748 |
| Validation | 220 | 127 | 91 | 438 |

All complete near/wide input pairs passed coverage QA. The legacy fixed-crop CNN interface still clipped some large roofs: 30 legacy views retained less than 99% of their known outline, with minimum approximately 51.3%. This limitation was recorded rather than hidden by dropping the rows.

### 4.5 Two distinct small-multifamily supplement branches

The **October 29 fourplex trial** reviewed 31 two-outline parcels and 62 roofs. It added 28 AI-reviewed fourplex main roofs from one neighborhood, recorded 28 rear auxiliary structures separately, and retained six unresolved roofs from three parcels. The auxiliary roofs did not enter the three-class classifier. This trial increased LA train multi 93→121 and fourplex 4→32, but worsened overall macro-F1. It is preserved as a separate experiment.

The later **October 30 hard-example ablation** starts from the augmentation-fixed low-rise baseline, not from the 28-roof fourplex model. It screened 325 multifamily and 554 single-family parcel candidates, retained 48/103 after official-source checks, reviewed 48 multifamily plus six model-confusable single-family targets, and withheld three uncertain multifamily cases. The accepted data were:

- **45 multifamily primary roofs across 11 training grids:** 23 fourplex, 9 five-unit, 5 six-unit, 5 seven-unit, 3 eight-unit records; footprint approximately 188–587 m².
- **6 confusable single-family primary roofs across 6 grids.** One was an ensemble multifamily false positive; the others were near-boundary or seed-disagreement cases, not all prior ensemble errors.

These supplements use NAIP 2022 imagery at native 0.6 m, resampled to approximately 0.305 m for the existing CNN interface. Upsampling preserves physical field of view, not native detail. Official links plus AI visual review do not constitute independent human gold.

### 4.6 Exact current C dataset and sampling

The current `C_positive_negative/DATASET.json` contains **18,480 records**:

| Partition/component | Small | Multi | Commercial | Total |
|---|---:|---:|---:|---:|
| Source train: cleaned Harris + NYC | 11,808 | 1,140 | 963 | 13,911 |
| LA train: original 748 +45 multi +6 small | 484 | 138 | 177 | 799 |
| **All train** | **12,292** | **1,278** | **1,140** | **14,710** |
| LA validation | 220 | 127 | 91 | 438 |
| Original LA diagnostic/reference set | 3,276 | 14 | 42 | 3,332 |

The separate 28-roof fourplex supplement is **not cumulatively included** in C. The C dataset's fourplex train count is 4+23=27, not 32+23.

Training draws assign 50% total mass to source and 50% to LA. Source sampling is natural; LA class mass is held at the original 478/93/177 proportions, even after supplementation. Within LA multifamily, low-rise sampling keeps its predefined unit-stratum mass. New examples split existing probability mass rather than simply increasing the multifamily prior. Actual per-example draw counts vary.

Frozen current data: [C DATASET.json](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_hard_ablation_20261030/execution/C_positive_negative/DATASET.json>), [C protocol](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_hard_ablation_20261030/execution/C_positive_negative/PROTOCOL.json>), [supplement review](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261030_hard_ablation/REVIEW.csv).

## 5. Where the validation and test data are

There is no single universal Stage2a test folder. Each experiment has a frozen manifest specifying eligible IDs, labels, input paths, and roles. JSON/CSV paths can retain cluster `/stage3/...` locations; those are historical container paths, not local Mac paths.

| Dataset / purpose | Exact local manifest or reference | What to select / associated inputs |
|---|---|---|
| Rebuilt Harris train/val/test | [labels_locked.csv](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/rebuild_20261015/training_locked_v2/labels_locked.csv>) | `rebuild_split=train/val/test`; test n=2,653. Local PNGs are in the next row. |
| Rebuilt Harris RGB/masks | [dataset directory](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/rebuild_20261015/dataset>) | `crops/` and `masks/`; match saved basenames/UIDs, not an arbitrary image subset. |
| Clean Harris manifest | [labels_explore.csv](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_explore_20261023/data/labels_explore.csv>) | 13,333/2,026/2,653; physical-group quarantine applied. |
| OOF exposure outer split | [labels_clean.csv](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_data_20261024/exposure_oof_20261026/data/labels_clean.csv>) | Same clean outer split; `oof_fold0.csv`…`oof_fold4.csv` beside it define nested roles. |
| LA fixed assessor references, all four cohorts | [la_assessor_matched.csv](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261015_gsd_assessor/la_assessor_matched.csv) | Boolean `primary_single_outline_iou50`, `strict_single_assessor_building`, `iou50`, `broad_parcel_labels`; preserve their frozen flags. |
| LA fixed 3,332 transfer/evaluation records | [la_transfer dataset.json](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_transfer_20261027/dataset.json>) | Reference use and near/wide input availability; 175 unavailable near/wide cases remain in the full denominator. |
| Original full LA crop/mask package | [LA inputs](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_eval_20261023/inputs>) | 126 `cell_*` directories and original product CSV. 21,888 originally valid inputs. |
| Current C train/validation/test | [C DATASET.json](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_hard_ablation_20261030/execution/C_positive_negative/DATASET.json>) | `split=val` gives 438; `split=test` gives the original 3,332 development references. |
| Complete repaired 22,024 LA input inventory | [preview DATASET.json](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_best_preview_20261031/DATASET.json>) | `input_status=original_valid` n=21,888 or `recovered_pixel_geometry` n=136. The full inventory has no complete building-use truth. |
| San Diego candidate audit | [label_audit.csv](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/sandiego_pilot_20261015/label_audit.csv>) | 225 candidates: 13 official-corroborated A, 62 weak-reference B, 150 unresolved C. Subsequent experiments use explicitly frozen subsets. |
| Five-class Harris/NYC/SD pilot | [generalizable DATASET.json](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/generalizable_pilot_20261026/DATASET.json>) | 1,790 prepared buildings; city-held-out fit roles and NYC eligibility recorded separately. |
| NYC/Cambridge/Seattle matched transfer | [reliable-transfer dataset.json](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/reliable_transfer_20261027/dataset.json>) | NYC train/validation; Cambridge/Seattle external test at the time of the frozen experiment. Use/role scoring masks differ. |
| Unit-supervision review panel | [REVIEW.csv](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_unit_panel_20261031/REVIEW.csv>) | 51 development candidates; **zero verified per-building unit labels and zero independent unit test cases**. |
| LA population references | [Census package](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_eval_20261023/census/la_census>) | 2020 block population, ACS 2022 block-group Parquet and supplied boundaries. Aggregate references, not per-building labels. |

The underlying LA matching artifacts are [lariac_dins_in_region.gpkg](/Users/zhou/Documents/GitHub/cybertraining-team4/evaluation/la_fire_validation/data/lariac_dins_in_region.gpkg) and [matched_v2.csv](/Users/zhou/Documents/GitHub/cybertraining-team4/evaluation/la_fire_validation/data/matched_v2.csv). Census, assessor, footprint, and damage references serve different tasks; do not interchange them.

Reserved LA grids were not used for image acquisition, scoring, or model selection in the reviewed adaptation experiments. A reserved pool is **not yet an evaluated final test set**. Denver/San Francisco had source-readiness investigation, not a completed frozen evaluation cohort or model result in the reviewed records.

## 6. Evaluation rules and what each score means

### 6.1 LA reference construction

The October 15 assessor evaluation retrieved 8,577 of 8,581 requested AINs for roll year 2024, predating the January 2025 fires. It started with 14,369 LARIAC/DINS outlines and 11,403 existing GT-to-prediction matches. The 1,810 match rows where several reference buildings shared one prediction were excluded from scored cohorts.

Use-code mapping is independent of the model predictions:

- Prefixes 01/02/03/07 → small residential, including duplex/triplex/manufactured housing.
- Prefixes 04/05 → multifamily.
- Unambiguous 1x/2x commercial → commercial, with mixed/parking/combination exclusions.
- Ambiguous/missing, vacant, condominium/own-your-own, nonpositive main floor area, unsupported uses, and residential records without positive unit count are excluded.

Geometry is compared in EPSG:32611 after the existing label-independent registration shifts. The fixed primary threshold is IoU ≥0.5. Cohorts are:

| Cohort | n | Additional conditions / interpretation |
|---|---:|---|
| Broad parcel | 9,036 | Valid type record, valid prediction, unique matched prediction. Lower-certainty parcel-use inheritance. |
| IoU 50 | 7,601 | Broad + registered IoU ≥0.5. |
| **Primary** | **3,332** | IoU 50 + one reference outline on the parcel **within the evaluated reference region**. |
| Strict sensitivity | 3,315 | Primary + one assessor building-data line. |

The original “single outline” criterion was based on the regional reference inventory, **not a countywide guarantee that the parcel has only one physical roof**. Later audits found additional outlines on some parcels. The frozen primary labels are retained for fair regression comparisons; doubts are recorded as separate sensitivity/audit evidence.

The primary cohort contains **3,276 small, 14 multi, 42 commercial in 92 cells**. Multifamily references occur in only six cells and commercial in seven. It covers about 15.22% of the originally valid prediction inventory; its class/area composition is selected and does not represent all LA buildings.

Owning evaluator: [evaluate_assessor.py](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/evaluate_assessor.py). Later comparisons reuse fixed IDs/flags rather than selecting a favorable cohort from new predictions.

### 6.2 Classification metrics and denominators

Primary reporting uses macro-F1, per-class precision/recall/F1, confusion matrices, and FP/FN counts. Accuracy alone is misleading: predicting every primary record as small gives **98.3193% accuracy**, higher than current C, but macro-F1 only **0.330508** and zero minority recall.

Commercial AP and precision at recall ≥90% were used in the October 23–26 development campaign. Thresholds from those LA sweeps are descriptive development choices, not held-out validation or applied production rules. “Best precision at recall ≥90%” need not select the same exact recall for every model; seed 43 N1's optimum used 40/42 rather than 38/42 commercial positives. Common FP32 exports take priority over BF16 trainer snapshots when comparing small score differences.

The later near/wide transfer models had **3,157 available image pairs**. Their full-cohort score keeps all 175 unavailable cases as unknown/errors. The CNN's existing 256 px inputs were still available; missing new near/wide exports do not mean missing CNN inputs. Always label whether a table uses 3,157 or 3,332 records.

For DINO/Qwen transfer tables, a two-seed **metric mean** is an arithmetic average of two run scores. For the current CNN, the main result is a **probability ensemble**, scored after averaging probabilities. These are different estimators and should not be silently combined.

Unknown/invalid or outside-three-class predictions count as errors for supported reference classes. Role-only examples are excluded from use accuracy. Selective accuracy must be reported with coverage and full-denominator performance; the permission to output unknown is not calibrated uncertainty.

### 6.3 Exposure metrics and OOF isolation

Exposure scoring uses log1p MAE, raw MAE/RMSE/R², class-specific errors, bias, and partial-inventory tile sums. The legacy factor-2 metric is:

`abs(log1p(prediction) − log1p(target)) ≤ log(2)`.

It measures a factor of two in **1+exposure**, not an ordinary multiplicative population error near zero.

OOF type features are generated by fitting classifiers without the row being predicted. Each fold also has a separate inner validation partition for checkpoint selection. The October 26 clean rerun preserved existing fold exclusions and quarantined 33 fold-local physical-group conflicts; every one of the 13,333 outer-training IDs received one held-out prediction per seed. Physical, parcel, property-ID, and tile groups were checked across fold roles. The exposure heads fit only outer train; outer validation/test do not fit the head. Exposure predictions on the head's own training set are not themselves OOF exposure scores.

### 6.4 Uncertainty and test-set status

Paired resampling uses cells/spatial groups rather than treating adjacent roofs as independent. Current LA ablations generally use 1,000 cell-bootstrap draws; close-up uses 2,000; repaired exposure uses 5,000 connected-tile draws. These intervals condition on fitted models, selected cohorts, and targets. They do not cover label errors, repeated adaptive design, all training variation, or all possible cities.

At first use, SD and the Cambridge/Seattle transfer cohorts were excluded from their respective training/model-selection procedures. After their outputs informed further development, they became inspected evidence. The current LA 3,332 cohort is a **fixed development/regression benchmark**, even though its IDs remain spatially excluded from adaptation training. Harris test was also inspected and triggered repairs. Neither should now be described as an untouched final confirmation test.

## 7. October experiment ledger

### 7.1 Data repair, pooling repair, and source-only adaptation

| Date / experiment | What changed | Main result | Status/lesson |
|---|---|---|---|
| Oct 15 original-data/model audit | Reproduced 23,014 archived labels, GSD/area, ACS values, and old LA predictions; numeric-area interventions. | Original LA primary accuracy 7.02%, macro-F1 0.04837. Area-only CNN intervention changed 0/300 classes. | Confirmed real construction defects; did not establish one causal explanation for all LA errors. |
| Oct 15 full rebuild v2 | Geographic area, deterministic fresh parcel join, corrected ACS grouping, 256 px crops, frozen train/val/test; five main and five nested OOF fits plus exposure head. | Harris type macro-F1 0.3346 old →0.7181 rebuilt; accuracy 0.5040→0.9397; proxy MAE 3.3497→1.1314 on common repaired test. | Original model's old split unavailable, so independence from this new test is unknown. |
| Oct 15 mask-pooling diagnosis | Checked target masks after downsampling to the ConvNeXt 7×7 feature grid. | Nearest sampling erased 9,245/18,036 targets, including 1,304/2,653 test targets. | A concrete implementation defect, not merely a difficult visual taxonomy. |
| Oct 15 area-pooling retrain | Fractional mask area weights in FP32; same data/labels/seeds/training rules, new OOF exposure fit. | Harris macro-F1 0.718129→0.900154; accuracy 0.939691→0.979269; proxy MAE 1.1314→0.8981. All nonempty masks retained. | Controlled repair, but early stopping changed compute and test had already triggered diagnosis. Old checkpoints retain nearest behavior. |
| Oct 15 physical reconstruction | Roof identity/seam stitching and evidence review, v1–v4. | 22,621 physical IDs; discovered 24 active cross-split physical groups. | Geometry/evidence deliverable; not a ready replacement training set. |
| Oct 15 San Diego source pilot | Official footprints/parcels, NAIP imagery, land-use and permit corroboration. | 225 reviewed candidates; 13 A, 62 B, 150 unresolved. No A-grade multifamily. | Data feasibility work, initially no training/inference. Later frozen subsets used by VLM/pilot studies. |
| Oct 23 repaired model on LA | Five area-pooling models, no LA training. | Primary accuracy 7.02%→78.39%; macro-F1 0.04837→0.350259. Commercial TP 40/FP 572, precision 6.54%, recall 95.24%. | Large repair gain, still severe target-domain false positives. |
| Oct 23 mask-fill numeric diagnostic | Adjusted mask-fill numeric feature only for scale on 98 outcome-stratified cases. | No ensemble class changed; all 32 sampled commercial FPs remained. | Negative diagnostic; did not test image resampling. |
| Oct 23 W1 sampler-only | Removed inverse-sqrt CE weights, kept weighted sampling. | Commercial TP 38/FP 429; P 8.14%, R 90.48%; P@R≥90%=8.30%. | Modest source-only gain, not sufficient for promotion. |
| Oct 23 W2 loss-only | Natural sampling, retained inverse-sqrt CE weights. | TP 39/FP 609; P 6.02%, R 92.86%; P@R≥90%=7.52%. | Did not solve false positives. |
| Oct 23 S1 physical-scale views | Native/LA-scale Harris views, otherwise baseline recipe. | TP 38/FP 371; P 9.29%, R 90.48%; P@R≥90%=9.43% versus B0 7.38%. | Missed predeclared 50% FPR-reduction gate; broad-cohort recall fell. No five-seed expansion. |
| Oct 24 C0 clean split | S1 after physical-group quarantine. | LA macro-F1 0.3462; P@R≥90%=6.28%; Harris val 0.9340. | Split/draw changes make this more than a pure leakage ablation. |
| Oct 24 C1 20 m ring | Added context-ring pooling. | LA P@R≥90%=10.95%, but Harris val 0.9074 below 0.9240 guardrail. | Context could help LA and harm source; no promotion. |
| Oct 24 G1 | Removed explicit type geometry. | P@R≥90%=5.67%; source val 0.9086. | Failed source guardrail. |
| Oct 24 N1 natural CE | Removed both class-weighted sampling and weighted CE in the source recipe. | P@R≥90%=12.26%, FP 272; source val 0.9510. The then-current five-seed reference at its development threshold had P 11.11%, FP 304. | Gain over ensemble only 10.53% fewer FP, not the 42.98% comparison against single B0. |
| Oct 24 D1 | Stochastic draw transforms plus ColorJitter. | P@R≥90%=7.48%; macro-F1 0.3109. | Combined augmentation change; no convincing gain. Distinct from later isolated augmentation repair. |
| Oct 24–26 A1/A2 NYC mixtures | Add 578 NYC rows at 10%/25% draw mass. | P@R≥90%=7.50%/6.70%; source val 0.9106/0.9233 fail guardrail. | Some improvements versus weighted C0, but not a high-recall improvement over N1. |
| Oct 24–26 N2 NYC natural | Natural source mixture, fixed update budget. | LA macro-F1 0.4736 seed 42; P@R≥90%=6.15%; AP 0.5458 versus N1 0.5347. | Better some lower-recall behavior, worse high-recall tail. Later useful as adaptation initializer. |
| Oct 25–26 follow-up | N1/N2 seed 43; natural+ring R1 seeds 42/43. | N1 P@R≥90%=12.26%/4.61%; N2 6.15%/4.87%; ring 13.97%/8.24%. Ring seed 42 failed source guardrail. | Large seed sensitivity; no stable NYC benefit or promotion. |
| Oct 26 geometry HGB controls | Log-area-only and four-geometry classifiers on clean source train. | LA macro-F1 0.3972/0.3758; multi recall 1/14 and 2/14. | High accuracy alone does not replace image-use learning; these are not matched CNN ablations. |
| Oct 26 clean OOF exposure | Three N1 seeds, nested five folds, matched area-only versus semantic exposure heads. | Test log-MAE 0.212750→0.153028; raw MAE 1.686739→0.953649. | Better fidelity to repaired proxies; detailed limits in Section 9. |

Sources: [poolfix report](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/poolfix_20261015/SUMMARY_ZH.md>), [W1/W2/S1 metrics](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_adapt_20261023/PILOT_METRICS.csv>), [source/NYC campaign comparison](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_explore_20261023/MODEL_COMPARISON.csv>), [follow-up results](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_data_20261024/followup_20261025/FOLLOWUP_RESULTS.md>).

### 7.2 VLM, DINO, role supervision, and transfer

| Experiment | Design | Result and consequence |
|---|---|---|
| Qwen3-VL-8B zero-shot, 296 buildings | Fixed BF16 model; raw and target-marked 256 px imagery plus footprint area; Harris 90 and LA 206 selected cases. | Harris macro-F1 0.7348 versus CNN 0.9547; LA commercial 16/42, multi 3/14. Corrected 34 selected CNN commercial FPs but predicted every LA-random 120 case small; 97.5% accuracy equaled majority prediction. |
| Prompt/interpolation diagnosis | Original versus 512 px interpolation; original versus evidence-comparison prompt; matched 296 cases. | Prompt-only changed zero classes; interpolation+prompt net corrected 1/296. More pixels/tokens did not add source detail. |
| Real Harris context | Native 512 px source view on Harris 90. | Correct 67→77/90; multi 13→19/30; commercial 24→28/30. Benefits partly clustered spatially; some “corrections” may reflect parcel rather than roof use. |
| Grounding/role probe, 12 cases | Predict target box, role and use; permit other/unknown. | 11/12 predicted boxes IoU≥0.5, but semantic/role failures remained; four correct controls included two regressions. Not a trained role model or new benchmark. |
| Five-class/dual-view VLM, 164 inputs ×5 arms | True near and ≥300 m wide views, five uses/role/unknown; frozen SD 101 scored references. | SD five-class dual view 63/101, macro-F1 0.4676; original three-class subset 53/74 versus 58/74 for three-class dual view. Multi 0/14; every role output was primary; unknown ineffective. Wider context helped, label expansion did not universally help retained classes. |
| Generalizable supervised pilot, 48 fits | Harris 960/NYC 729/SD 101 prepared; NYC training restricted to 247 source-corroborated cases; held-out city; frozen CNN, frozen DINO, fine-tuned CNN; near/dual views; seeds 42/43. | Best city-mean two-seed metric macro-F1 0.2669 for frozen DINO near. NYC/SD multi recall 0. DINO wide views did not consistently help. No production/generalization-model promotion. |
| Reliable-reference DINO/QLoRA transfer | NYC 720 train/174 val; Cambridge 300/Seattle 244 external; matched near/wide views, five-use plus role; frozen/fine-tuned DINO and NF4 zero/QLoRA Qwen. | DINO tuning improved mean macro-F1 by ≈0.070 Cambridge/0.056 Seattle; QLoRA by ≈0.054/0.088 versus respective controls. QLoRA auxiliary recall greatly improved in Cambridge, but multi transfer remained variable. These were class-stratified official-reference cohorts. |
| QLoRA evaluation repair | Match nonquantized-parameter FP32 preparation at evaluation to PEFT training. | Corrected `execution_v3` replaces earlier evaluation outputs for final reporting; source 174 replay matched exactly. Same trained weights and source-only selection. |
| Frozen reliable models transferred to LA | Keep full 3,332 denominator and common 3,157 available-image cohort. | Common macro-F1: N2 CNN 0.5048, Qwen zero 0.5263, QLoRA mean 0.4065; DINO frozen 0.0891/tuned 0.1212. QLoRA severely overpredicted multi. Zero-shot multi 0/14. No stable zero-shot advantage over CNN. |
| Fixed-budget single-/multicity DINO | A: 720 NYC; B: 260 NYC +260 Cambridge +200 Seattle; same use/role totals, 174 NYC validation, 15 epochs/675 updates, seeds 42/43. | Common LA mean macro-F1 0.1194→0.4433; full 3,332: 0.1175→0.4347. Strong improvement over failed NYC-only transfer, still below prior CNN/zero-shot point estimates; source NYC validation worsened. |
| Denver/SF readiness | Read-only candidate schema/count and source linkage investigation. | Denver prioritized for future independent evaluation. No completed new-city predictions or final performance result in the reviewed record. |

These experiments showed that a larger pretrained encoder or VLM is not automatically more transferable. Training-data composition, field of view, imagery provenance, roof attribution, and the output taxonomy materially affect the result. Positive Cambridge/Seattle results and negative LA results must both remain in the record.

Sources: [VLM pilot](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261026_vlm/REPORT.md), [VLM diagnosis](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261026_vlm/DIAGNOSIS_REPORT.md), [five-class VLM](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261026_vlm/MULTICLASS_REPORT.md), [48-fit pilot](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261027_generalizable_pilot/RESULTS.md), [reliable-transfer methods](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261027_reliable_transfer/METHODS.md), [reliable-transfer results](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261027_reliable_transfer/RESULTS.md), [LA transfer](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261028_la_transfer/REPORT.md), [multicity comparison](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261028_multicity_transfer/REPORT.md).

### 7.3 LA adaptation and latest experiments

All following main scores use the **same 3,332 references** and two-seed probability ensembles. The first two rows provide context. These are development results, not independent successive final tests.

| Model / change | Accuracy | Macro-F1 | Multi TP | Multi FP | Multi FN |
|---|---:|---:|---:|---:|---:|
| Repaired Harris five-seed poolfix, no LA training | 78.39% | 0.350259 | 4 | — | 10 |
| N2 source/NYC CNN, adaptation initializer | 95.83% | 0.4833 | 2 | 25 | 12 |
| Oct 29 ordinary LA adaptation CNN | 97.72% | 0.5716 | 3 | 29 | 11 |
| Oct 29 same-data adapted DINO | 97.69% | 0.5724 | 3 | 40 | 11 |
| Oct 29 low-rise resampling CNN | 97.87% | 0.5804 | 3 | 27 | 11 |
| Oct 29 separate 28 fourplex supplement | 97.42% | 0.5679 | 4 | 35 | 10 |
| Oct 30 augmentation repair, arm A | 98.11% | 0.590907 | 3 | 19 | 11 |
| Oct 30 +45 positive examples, arm B | 98.11% | 0.595872 | 3 | 19 | 11 |
| **Oct 30 +45 positive/+6 negative, arm C** | **98.29%** | **0.610370** | **3** | **15** | **11** |
| Oct 31 same-pixel target close-up | 97.84% | 0.580358 | 3 | 21 | 11 |

The augmentation repair addressed a confirmed implementation issue: transforms seeded only by sample index generated the same flip/rotation every time that building was drawn. The isolated repair allowed per-draw variation while preserving other settings. It reduced multifamily false positives 27→19, leaving the same 11 misses; its macro-F1 interval crossed zero. This is distinct from October 24 D1, which also changed photometric augmentation.

Arm C's gain over A is **+0.019464 macro-F1**, fixed-model paired 95% interval **[0.000198,0.046111]**; C−B is +0.014499 [0.000384,0.033285]. Individual seed directions were mixed. The observed ensemble improvement reduces false positives and improves commercial recognition; it does not show a robust individual-model or multifamily-recall improvement.

The close-up run matched labels, initialization, draw sequences, training budget, and geometry features. It retained every visible target-mask pixel, used at least 40 m physical crop width, and left edge-touching targets in the original view. It enlarged the target while reducing surrounding context; it did not recover missing roof/source pixels. Macro-F1 fell 0.610370→0.580358; weighted validation macro-F1 fell 0.925022→0.902005. The paired macro-F1 interval crossed zero and multi recall did not change. **No improvement was demonstrated; the conditional context-extension run was not started.** This does not prove every dual-view architecture would fail.

Sources: [LA adaptation](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_adaptation/REPORT.md), [fourplex trial](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261029_la_fourplex/REPORT.md), [augmentation repair](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261030_augmentation/REPORT.md), [A/B/C ablation](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261030_hard_ablation/REPORT.md), [close-up](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261031_closeup/REPORT.md).

## 8. Current classifier and latest results

### 8.1 Model and training recipe

The current C classifier is ConvNeXt-Tiny with RGB + a fourth target-mask channel, area-weighted target pooling, and four deployed geometry inputs:

`log_footprint_m2, mask_fill_ratio, bbox_aspect_ratio, geometry_compactness`.

Images/masks are resized to 224 px for the model. The final ConvNeXt spatial grid is 7×7; fractional mask weights prevent a small target from vanishing at this grid. The feature extractor still receives context through its receptive field; masked pooling does not mean it sees only target pixels.

Each seed starts from its frozen N2 source/NYC checkpoint. Adaptation uses 6 epochs×8,192 draws, batch 32, **1,536 optimizer updates**, AdamW lr 1e−5, weight_decay 0.01, gradient clipping 1, BF16 training and FP32 inference. The population head is frozen/untrained for this type adaptation. Epoch selection includes epoch 0 and uses only the 438-row sampling-weighted LA validation score. C selects epoch 4 for seed 42 and 6 for seed 43. The main prediction averages the two three-class probability vectors and takes argmax; no LA-test threshold is applied.

The saved adapted `best.pt` files contain trainable state, excluding the frozen population head. Reconstruct the model with its matching N2 initializer, protocol configuration, and frozen `common.py`, then apply the adapted state as the inference runner does. An arbitrary handoff `common.py` is not interchangeable: the older standalone handoff runtime predates area-pooling support, and its working-tree augmentation edit does not make it the current frozen runtime.

### 8.2 Current per-class performance

| Reference class | n | Precision | Recall | F1 | TP / FP / FN |
|---|---:|---:|---:|---:|---|
| Small residential | 3,276 | 99.326% | 98.962% | 0.991437 | 3,242 /22 /34 |
| Multifamily | 14 | 16.667% | 21.429% | 0.187500 | 3 /15 /11 |
| Commercial | 42 | 60.000% | 71.429% | 0.652174 | 30 /20 /12 |

Confusion matrix, rows=reference and columns=prediction, order small/multi/commercial:

```text
                 small   multi   commercial
small             3242      15       19
multi               10       3        1
commercial          12       0       30
```

Balanced accuracy is 0.639398. There are 57 reference disagreements. Accuracy is slightly below the always-small baseline because minority recognition incurs false positives; macro-F1 and class metrics show the relevant improvement over that trivial baseline.

### 8.3 Complete LA inference and visual product

The October 30 full run used the then-best low-rise model and produced 20,968 small/281 multi/775 commercial predictions. **Those are not the newest C counts.** The October 31 fresh C run covers all 22,024 records and yields:

| Predicted use | Count |
|---|---:|
| Small residential | 20,993 |
| Multifamily | 187 |
| Commercial | 844 |
| Total | 22,024 |

Originally,136 records had empty masks caused by mixed pixel/UTM coordinate interpretation. They were recovered from same-cell coordinate anchors and aligned pre-fire imagery, with original records preserved and status flags retained. Full C inference reproduced saved classes on all 3,332 references, maximum probability difference **1.818×10⁻⁶**. The 136 repaired records did not acquire independent new use labels.

The preview includes Palisades/Eaton maps, selectable scenes, target crops, class probabilities, all 14 multifamily reference cases, filters, exposure values, and CSV export. Browser inspection in the earlier authorized task exposed and corrected the 136 inherited invalid map centroids using recovered EPSG:32611 geometry. The other 21,888 coordinates and all use/damage values stayed unchanged. Current point counts are 2,870 Palisades and 19,154 Eaton.

Preserved damage votes are 14,486 no damage,122 minor,370 major,5,748 destroyed,1,298 unknown. These are the saved v3 Stage2b outputs, not new predictions from C.

Open [current interactive preview](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_best_preview_20261031/viewer/index.html>) or [current predictions.csv](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_best_preview_20261031/viewer/predictions.csv>). Crops depend on the external drive and its linked datasets remaining available.

## 9. Exposure models and the newest Census comparison

### 9.1 Clean OOF exposure experiment

The October 26 experiment retained the 18,012-row clean Harris dataset and outer splits. It trained 15 nested OOF classifiers (5 folds ×3 seeds 42/43/44) plus the new full seed 44 classifier; full 42/43 were reused and re-exported. Five exposure heads were fit: area-only A, primary mean-three-seed semantics B, and three seed-specific B diagnostics.

Both A and B use the same Ridge log-area baseline and HGB residual configuration. B adds soft N1 predicted-use features from OOF training probabilities; source N1 itself consumes imagery and geometry. This isolates added **predicted-use information in the chosen estimator**, not a pure image-only effect. The target also contains type/area rules.

| Split / metric | A area-only | B semantics mean 3 | Change |
|---|---:|---:|---:|
| Validation log1p MAE | 0.221884 | 0.174887 | −21.18% |
| Test log1p MAE | 0.212750 | 0.153028 | −28.07% |
| Test raw MAE | 1.686739 | 0.953649 | −43.46% |
| Test RMSE | 8.561801 | 7.491664 | Lower |
| Test shifted factor 2 | 92.99% | 98.79% | Higher |
| Test total relative bias | −5.77% | −1.88% | Closer to zero |

Test log-MAE B−A paired 95% spatial interval is [−0.115472,−0.018818], across 28 connected components. Per-class log-MAE improves, but **multifamily RMSE worsens 24.062447→26.759615**. A retained outlier with target 40.8 rises from 267.70 to 297.73. Lower overall MAE must not hide this failure.

This is the strongest repaired-data result for the semantic contribution to the **constructed exposure target**. It does not measure actual residents, cross-city occupancy, or final fire impact accuracy. Source: [OOF results](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_data_20261024/exposure_oof_20261026/RESULTS.md>).

### 9.2 Current LA preview exposure

On October 31, the frozen `B_semantics_mean3.pkl` head was applied to **current C seed-42/43 mean probabilities + log1p footprint area**. Saved Harris test log predictions reproduced exactly in the matching runtime before LA application. No exposure retraining or LA Census calibration occurred.

- Complete 22,024-building exposure total: **69,023.955**.
- Sum on minor/major/destroyed buildings: **17,522.957**.
- Unknown-damage records are excluded from the damaged subtotal.

These are proxy sums, not observed residents, displaced persons, casualties, or event-time occupants. The head was learned using N1 features; feeding C probabilities introduces an additional classifier-feature distribution change. Successful loading and exact saved-test replay establish implementation parity, not validation of this new LA combination.

### 9.3 Original-versus-current Census validation

The comparator is the **original saved v3 `pred_population` field**, not a reconstructed approximation. On the common 21,888 buildings, original total is **75,591.025**, current **67,978.803**. Including 136 recovered inputs makes current total 69,023.955; that inventory addition must not be counted as a model-only difference.

The validation recovers the 126 image-cell extents, each 600 m wide, and uses geometric union coverage of Census polygons. Primary “fully covered” requires ≥99.9999% area coverage and excludes polygons containing any original-output-missing building. Zero-population and zero-prediction polygons are retained; Census counts are not allocated proportionally into partial polygons. All 22,024 points uniquely match one block and one block group.

| Cohort | Areas | Reference population | Original MAE | Current shared-input MAE | Original bias | Current bias |
|---|---:|---:|---:|---:|---:|---:|
| **2020 blocks: fully covered, no original missing inputs** | **323** | **20,568** | **51.029** | **43.757** | **+70.98%** | **+61.95%** |
| 2020 blocks: fully covered, shared inputs | 383 | 30,129 | 69.344 | 55.842 | +79.91% | +61.98% |
| 2020 blocks: ≥99% coverage sensitivity | 397 | 31,704 | 74.552 | 58.717 | +82.86% | +62.24% |
| ACS2022 groups: fully covered, no original missing inputs | 2 | 1,338 | 1,071.85 | 1,072.53 | +160.2% | +160.3% |
| ACS2022 groups: fully covered, shared inputs | 10 | 10,157 | 629.452 | 698.575 | +57.6% | +68.8% |
| ACS2022 groups: ≥99% coverage sensitivity | 11 | 11,555 | 578.67 | 660.61 | +51.2% | +62.9% |

Primary 323 blocks contain 11,132 buildings. Current total 33,309.016 versus reference 20,568; original total 35,166.425. Block MAE improves **14.25%**, while substantial overestimation remains. ACS results do not consistently improve. No spatially independent significance/generalization claim was established.

Both classifier and exposure estimator differ from the original pipeline. This is a **whole-pipeline comparison**, not a classifier-only ablation. The supplied 2020 population/2022 ACS values are joined to TIGER 2025 boundaries; the geographic-vintage crosswalk was not independently verified. Usual-residence population and commercial/daytime exposure have different meanings. ACS point-error metrics do not incorporate retained sampling margins of error.

Residential-only sensitivity selects each model's own predicted residential subset, so selection also changes. In 323 blocks, MAE 34.125→27.221 and bias−51.02%→+32.63%; this is not evaluation against independently verified residential roofs.

### 9.4 Why the current total is still too high

The saved arithmetic decomposition of the primary 323 blocks is:

| Component | Count / exposure |
|---|---|
| Net excess over Census | 33,309.016−20,568 = **12,741.016** |
| Predicted commercial | 358 footprints contribute **6,029.830** |
| Predicted residential | 10,774 footprints contribute **27,279.185**, still 6,711.185 above Census |
| Census housing units in the same blocks | 8,119; not the same object as a detected footprint |
| Predicted residential footprints below 50 m² | 1,213 contribute 3,064.211; median approximately 2.524 |
| Predicted residential footprints below 100 m² | 3,739 contribute 9,445.363 |

Small structures may be garages, sheds, ADUs, or real homes. The clean Harris label manifest assigns `estimated_units=1` to 15,571/15,597 small-residential records across its train/validation/test partitions. Many small LA roofs receive approximately one household's exposure. This suggests a target/inventory attribution problem, but does not establish a causal explanation or prove that every small roof is nonresidential. Automatic small-roof exclusion would remove legitimate homes and has not been validated.

Commercial contributions amount to 47.3% of the arithmetic net excess, but that percentage is **not a causal error attribution**. Removing commercial exposure still leaves a large residential excess. The necessary distinction is resident population versus commercial/daytime occupancy, together with building role and unit attribution.

Evidence: [Census report](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261031_exposure_validation/REPORT.md), [full metrics](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261031_exposure_validation/RESULTS.json), [excess breakdown](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261031_exposure_validation/EXCESS_BREAKDOWN.json).

## 10. What has actually been implemented

| Capability | Implementation / scope |
|---|---|
| Stage1→Stage2a adapters and original inference | `pipeline/scripts/infer/build_stage2a_infer_csv.py`, `infer_stage2a.py`; later native 3 ensemble runner and standalone LA package. Original generic adapter's learned mask-area scale is historical, not a valid replacement for per-scene metric area. |
| Standalone LA preparation/finalization | [run_la.py](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/run_la.py): preserves `cell_id+bldg_uid`, polygon area, exclusions and run metadata. |
| Data/source audit and rebuild | `audit_sources.py`, `audit_training.py`, `audit_acs.py`, `audit_model.py`, `evaluate_assessor.py`; external `rebuild_stage2a_dataset.py`, freeze and training scripts. |
| Area-mask pooling | Frozen poolfix `common.py` and checkpoint config compatibility; missing old config remains nearest. |
| Physical identity/reconstruction | External `build_stage2a_physical_buildings.py`, mapping/split-conflict evidence and v4 data; not a new semantic training set. |
| Source scale/context/weighting experiments | Isolated runtime copies, alternate-view manifests, rings, weighted/natural sampling and saved common FP32 prediction exports. |
| Multi-city preparation and model training | `prepare_generalizable.py`, `train_generalizable.py`, `prepare_reliable_transfer.py`, `train_reliable_dino.py`, `train_reliable_qwen.py`, `prepare_multicity_transfer.py`; separate frozen datasets. |
| VLM diagnosis and scoring | `vlm_pilot.py`, `vlm_diagnosis.py`, `vlm_grounding_probe.py`, `vlm_multiclass.py`, and matching analysis scripts. |
| LA adaptation | `prepare_la_adaptation.py`, `train_la_adaptation.py`, `analyze_la_adaptation.py`; isolated validation, initial CNN/DINO and low-rise variants. |
| Fourplex and hard-example branches | `prepare_la_fourplex.py`, candidate/expanded-pool preparation and score scripts, `freeze_la_hard_ablation.py`, corresponding analyzers; separate evidence/label versions. |
| Dynamic augmentation repair | Isolated augmented runtime plus tests; existing tracked handoff `common.py` also has a pre-existing working-tree edit. This review did not alter it. |
| Close-up experiment | `target_closeup.py`, `prepare_la_closeup.py`, `check_la_closeup.py`, `analyze_la_closeup.py`; tested matched views and negative result preserved. |
| OOF exposure ablation | `prepare_exposure_oof.py`, `run_exposure_oof.py`, `tests/test_exposure_oof.py`, frozen features and five heads. |
| Full LA inference and 136 input recovery | `prepare_la_full.py`, `infer_la_full.py`, `analyze_la_full.py`; status/provenance retained for recovered inputs. |
| Full preview and exposure validation | `build_la_best_preview.py`, `la_best_preview.html`, `recompute_la_preview_exposure.py`, `validate_la_exposure.py`; CSV, interactive maps, Census comparisons, aggregate checks. |
| Direct-unit supervision preparation | `prepare_la_unit_panel.py` and 51-case review package only. No verified unit training labels or trained direct-unit estimator. |

The later experiments live mainly in isolated outputs and analysis scripts. Their existence does not automatically update the original framework's default model paths, package, or manuscript results.

## 11. Current challenges and unresolved questions

### 11.1 Small multifamily remains the principal classification gap

C still misses 11/14 multifamily references; nine misses are fourplex references, with the other two a five-unit and eight-unit reference. The current three true positives have much larger roofs. Earlier audit medians were 2.68 area-equivalent 7×7 mask cells for misses versus 6.30 for hits. This supports studying target detail and morphology but is not proof that coarse pooling is the cause. The matched close-up result did not improve recognition.

Reweighting repeatedly samples the same evidence. Adding 28 same-neighborhood fourplexes produced one extra hit and more false positives; adding 45 across more grids plus six hard negatives mainly improved false positives. The boundary between 3 and 4 units can be visually indistinguishable from overhead imagery. Better labels and independent unit evidence are necessary to know whether a proposed model is solving the intended task.

### 11.2 Roof, property, and dwelling attribution remain unresolved

Single assessor record does not guarantee single roof. A property can contain one fourplex plus a garage, or two duplexes. Legal unit parcels can lie within one physical building. Assigning a parcel's full units to each roof duplicates population; using footprint area alone cannot determine role. Existing review panels contain unresolved attribution and temporal cases. Auxiliary appearance does not prove zero occupancy.

The original 3,332-building cohort is kept for continuity, not certified as an error-free building-level benchmark. Three priority doubts in the multifamily audit involve a multi-record fourplex property, a PUD record, and an office property/roof-area mismatch. They were not silently relabeled or removed.

### 11.3 Transfer and input scale are still unstable

Harris/NAIP/Maxar differ in GSD, field of view, viewing geometry, season, crop positioning, and official versus predicted footprints. Some inputs include courtyard/pool contamination or partial roofs. Wider context helped particular VLM experiments but not every CNN/DINO experiment. The model may use neighboring-apartment context incorrectly for an individual residence. Current evidence does not isolate all these mechanisms.

### 11.4 Confidence and role outputs are not dependable gates

Many mistakes are high confidence: in the current-best-at-the-time multifamily audit,9/11misses had pmax>0.9. Early Qwen outputs marked evidence sufficient on every case, including errors; five-class zero-shot role outputs were allprimary. Source-calibrated coverage did not transfer reliably. A low-confidence review queue misses confident errors; neither self-reported evidence nor unknown permission is a calibrated safety valve.

### 11.5 Evaluation support and independence are limited

Fourteen multi references mean one hit changes recall by 7.14 percentage points. Their spatial concentration reduces effective support further. Many recipes have been designed after inspecting LA/Harris/SD. Grouped splits prevent direct overlap but do not undo adaptive benchmark reuse. Cambridge/Seattle were valid held-out cities for their frozen transfer comparison, but later use makes them development sources. A future final claim needs a genuinely unused geography and independently supported roof-level references.

### 11.6 Exposure is not yet resident-count calibrated

The target blends housing-based residential exposure and density-based nonresidential occupancy constructs. Almost every small-residential training roof gets one unit, including potentially auxiliary structures. Multifamily units use area rather than verified floors/units. Current C probabilities feed an older N1-trained head. Aggregate block overestimation remains+61.95%, and ACS results can worsen. Classification improvements alone do not establish population or disaster-impact improvements.

### 11.7 Operational integration lags the research baseline

The original packaged weights and integration documents still describe earlier model/data versions. Multiple source snapshots deliberately preserve behavior. Loading current weights with old pooling code, treating partial state as a complete checkpoint, or using a historical mask-area conversion can invalidate a run. Full input/protocol/checkpoint provenance is part of reproducing the result, not an optional detail.

## 12. Recommended next work and success criteria

The following are recommendations, not experiments already completed or newly launched by this review.

1. **Freeze a new building-level reference panel before further scoring.** Independently verify use, primary/auxiliary role, exact dwelling count where available, and date-to-roof attribution. Prioritize 4–8 unit main buildings, similar 1–3 unit homes, small offices/clinics, and accessory roofs across several areas. The 51-case panel is a starting review package, not a dwelling-unit test set.
2. **Separate the resident target from nonresidential/daytime exposure.** Decide which quantity the application needs; audit duplicate/fragmented/accessory/missed buildings before population calibration. Do not fit to the same Census totals later reported as validation.
3. **Use the reserved geography or a newly frozen city for confirmation.** Lock candidate rules, exclusions, input availability, model hashes, score denominator and grouped uncertainty before predictions. Existing Denver readiness work is a possible route, not an existing test result.
4. **Keep C as the frozen research comparator.** Compare a specific new mechanism with matched initialization, training/update budget, class/domain mass, validation selection, labels, and input support. Preserve source tradeoffs. Current evidence does not justify assuming a larger backbone, more resampling, or another prompt will solve the problem.
5. **Require improvement in the failure that matters.** Report multi precision/recall/F1 and FP/FN together with commercial and overall macro-F1. An unchanged 3/14recall, selective-only accuracy, or higher overall accuracy from the majority class is insufficient evidence of solving multifamily recognition. For exposure, report aggregate bias and per-building unit/role errors when truth becomes available.
6. **Only then update the deployable package and case study consistently.** Include the matching source/config/checkpoints, current exposure head, input geometry rules, versioned output schema, throughput/memory measurements, and the exact supported claim. A preview is not production acceptance.

No numeric improvement target is promised here; the new protocol should fix practical acceptance thresholds before evaluating the new test.

## 13. Reproduction and artifact navigation

### 13.1 Current baseline assets

| Asset | Exact local location |
|---|---|
| C adapted seed 42 | [best.pt](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_hard_ablation_20261030/execution/collected/C_positive_negative/runs/cnn_lowrise_s42/best.pt>) |
| C adapted seed 43 | [best.pt](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_hard_ablation_20261030/execution/collected/C_positive_negative/runs/cnn_lowrise_s43/best.pt>) |
| C frozen runtime | [execution/common.py](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_hard_ablation_20261030/execution/common.py>) and [train_la_adaptation.py](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_hard_ablation_20261030/execution/train_la_adaptation.py>) |
| N2 seed 42 initializer | [stage2a_best_model.pt](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_data_20261024/runs/N2_nyc_natural_seed42/stage2a_best_model.pt>) |
| N2 seed 43 initializer | [stage2a_best_model.pt](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_data_20261024/followup_20261025/runs/N2_nyc_natural_seed43/stage2a_best_model.pt>) |
| Current preview exposure head | [B_semantics_mean3.pkl](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_data_20261024/exposure_oof_20261026/heads/B_semantics_mean3.pkl>) |
| Current exposure values and provenance | [EXPOSURE.json](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_best_preview_20261031/EXPOSURE.json>) |
| Current all-LA run protocol | [preview PROTOCOL.json](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_best_preview_20261031/PROTOCOL.json>) |
| Current full-use predictions | [seed42 predictions](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_best_preview_20261031/execution/collected/runs/seed42/predictions.json>), [seed43 predictions](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_best_preview_20261031/execution/collected/runs/seed43/predictions.json>) |
| Main C scores and saved benchmark predictions | [RESULTS.json](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261030_hard_ablation/RESULTS.json), [PREDICTIONS.json](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261030_hard_ablation/PREDICTIONS.json) |
| Latest preview/result verification | [preview RESULTS.json](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261031_best_preview/RESULTS.json), [VERIFICATION.json](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261031_best_preview/VERIFICATION.json) |

C checkpoint SHA256 values are `faf583335401e88d0e4359066a63854b1db325d4f4f37dd918e07044565f187f` (42) and `9780be14c633f67c255ccf25978e38dcb7ffc9864d1ad14377403a370f42e7b9` (43). The current exposure head SHA256 is `29c66aee57b73d6737ff8b5436502c12f4d231ba3128cbc27f24f859ed3adaac`.

### 13.2 How to reproduce without confusing versions

For score reproduction, use the existing analysis entry points and their frozen inputs. They rewrite their own derived reports; preserve a copy or use a new output location when retaining a review snapshot. This review itself recalculated key metrics directly from saved predictions without retraining.

| Question | Entry point / authoritative input |
|---|---|
| Recompute C/A/B benchmark comparisons | `analyze_la_hard_ablation.py`; C/B run predictions, augmentationA and fixed 3332 references. |
| Recompute close-up result | `analyze_la_closeup.py`; fixed C and close-up 42/43 outputs. |
| Rebuild a full inference run | Frozen `infer_la_full.py` + matching `common.py`, local/cluster path staging, `PROTOCOL.json`, dataset/input hashes, both N2 initializers and adapted states. The runner refuses an existing run output directory. |
| Recompute current exposure | `recompute_la_preview_exposure.py`; frozen B head, population helper source, saved test NPZ/CSV parity check, and current preview CSV. |
| Rebuild preview | `build_la_best_preview.py`; all-LA predictions, geometry/coordinate repair metadata, original damage/report source, and exposure values. |
| Recompute Census consistency | `validate_la_exposure.py`; original saved population CSV, current exposure, supplied Census files, and source-hashed ACS projection. |

The exposure pickle was trained with **NumPy 2.2.6/scikit-learn 1.9.0**. The prior task used an isolated temporary matching runtime because the normal GIS Python environment had NumPy 1.x dependencies. Temporary `/private/tmp/...` runtimes may expire; their presence is not guaranteed. Do not patch a pickle or upgrade the GIS environment globally merely to load it. Use the saved environment records and create a matching isolated environment when needed.

Original raw xBD source inputs remain under [source_data/xBD](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage3_harvey/source_data/xBD>). The old label-generation archive/notebooks remain under [stage2a_old_training_data_codes](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/stage2a_old_training_data_codes>). Historical receipts may point to the old `/Users/zhou/Documents/GitHub/Instance-Impact-IGUIDE-SpatialAIChallenge2026` checkout; the inspected data repository now resides on Crucial X 9.

## 14. Operational lessons and review limitations

### 14.1 Measured resource costs

| Workload | Recorded cost / meaning |
|---|---|
| October 15 rebuild v2 campaign | 83.13 min on one A30. |
| October 15 area-pooling campaign | 152.73 min on one A30; five main fits, nested OOF, exposure and comparison. |
| October 23 full five-seed LA inference | 21,888 valid records in 227 s, including CUDA smoke check. |
| LA adaptation/close-up fit | Approximately 175–190 s per six-epoch fit; CNN peak PyTorch reserved 2.52 GiB. Not a general hardware benchmark. |
| Current full C CPU inference | Approximately 790 s per seed, 22,024 records, eight Torch threads; includes per-run hash/witness work. Two-seed deployment cost must include both passes. |
| Qwen 8B BF16 zero-shot | Initial pilot≈2.50 s/building average; peak reserved≈16.50 GiB, about 16.33 GiB weight bytes. Later image/output variants differ. |
| Matched QLoRA | Approximately 35 min/seed; training-loop peak reserved 11.06 GiB, cold-load/witness peak 23.13 GiB. Fits on the tested 24 GiB A30; not established for an RTX 3080. |
| DINO reliable-reference tuning | Approximately 2.5–2.8 min/seed excluding initialization/image loading. |

### 14.2 Pitfalls already encountered

- **Storage pressure interrupted actual training and LLM restoration.** Nominal PVC capacity or nodeReady is insufficient. Record nodefs/imagefs headroom and allow space for model images, checkpoints, source data, and service restart. Root/home mounts and host symlinks differ inside containers; use explicit verified mounts.
- **GPU requests alone did not expose CUDA** in one VLM job without the NVIDIA runtime class. Require a real forward/backward or inference witness inside the actual job.
- **Large tar streams truncated checkpoints.** A full-looking file size was insufficient; later successful shell commands masked pipe failures. Load/hash collected weights and preserve failed attempts.
- **MPS smoke success did not establish sustained Mac throughput.** Full local inference caused heavy swap; the completed cluster result is distinct from the interrupted local attempt.
- **Shared-memory defaults blocked full CPU inference.** The successful rerun used the existing 2 Gi shared-memory configuration; failed outputs were not counted as results.
- **BF16 training scores and FP32 exports can differ by hard decisions.** Use a common export path for comparisons; retain trainer results as such.
- **NumPy/scikit-learn pickle and pandas ABI mismatches require matched isolated runtimes.** Do not silently transform random-generator state or global environments.
- **A desired LLM replica is not service recovery.** Earlier GPU jobs verified actual ready/available counts, HTTP health/model listing and endpoints after restoring the recorded prior state. These are historical checks; this documentation review did not inspect live cluster health.
- **Old checkpoint pooling must stay old.** A repair default must not reinterpret an old checkpoint when measuring its baseline.
- **Hash-frozen datasets and auto-generated reports should not be hand-edited.** Fix generating code, create a new dataset/run version, and retain prior labels/results and all exclusions.

### 14.3 What this review verified and did not verify

The accompanying [review verification record](/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la/analysis/20261031_comprehensive_review/VERIFICATION.json) records checks performed for this documentation: frozen dataset counts, current C score reconstruction, full-preview probability/class agreement, current exposure inventory/totals, Census summary arithmetic, important artifact existence, current checkpoint/head hashes, and document links. Those checks are separate from older experiments' recorded image/GPU/browser checks.

No models, source labels, previous results, default model pointers, or manuscript claims were changed. No new city or reserved-set predictions were made; no browser or cluster was controlled; no experiments were retrained. The original July individual run folders and all historical source URLs were not re-fetched; July results are attributed to the owning consolidated historical record. This is not an independent fresh-agent experiment audit or a full per-image reinspection of every dataset.

The central remaining limitation is evidential: the project now has functioning, auditable research implementations and several real improvements, but **reliable small-multifamily recognition, independently verified roof-level units/roles, unseen-geography confirmation, and calibrated resident population remain unfinished**.
