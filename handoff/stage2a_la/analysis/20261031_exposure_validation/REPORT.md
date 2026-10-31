# LA exposure validation and original-model comparison

The current pipeline improves agreement with 2020 Census blocks, but still substantially overestimates population. The 2022 ACS block-group check does not consistently improve. These are aggregate consistency checks, not validated building occupancy or fire-time exposure.

## Same-building comparison

The original saved v3 output contains 21,888 buildings. Their original total is 75,591.0, versus 67,978.8 for the current pipeline. The current complete inventory adds 136 repaired inputs and totals 69,024.0. Do not compare these different inventories as model-only gains.

The original comparator is the saved `pred_population` field, not a reconstructed approximation. The current pipeline uses C42/43 mean probabilities and the frozen B_semantics_mean3 exposure head. Both classifier and exposure estimator changed; this comparison cannot attribute improvement to the classifier alone.

## Spatial comparison

Coverage is computed from the union of 126 recovered 600 m image cells in EPSG:32611. Fully covered means at least 99.9999% polygon area coverage. The primary cohort additionally excludes areas containing any of the 136 records absent from the original output. Zero-prediction and zero-population areas are retained. No Census population is allocated by area to partial polygons.

| Reference and cohort | Areas | Census population | Original total | Current shared total | Original MAE | Current MAE | Original bias | Current bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 Census blocks: complete, no missing old inputs | 323 | 20,568 | 35,166 | 33,309 | 51.03 | 43.76 | +71.0% | +61.9% |
| 2020 Census blocks: complete, shared inputs | 383 | 30,129 | 54,205 | 48,802 | 69.34 | 55.84 | +79.9% | +62.0% |
| 2020 Census blocks: 99% coverage sensitivity | 397 | 31,704 | 57,974 | 51,437 | 74.55 | 58.72 | +82.9% | +62.2% |
| 2022 ACS block groups: complete, no missing old inputs | 2 | 1,338 | 3,482 | 3,483 | 1071.85 | 1072.53 | +160.2% | +160.3% |
| 2022 ACS block groups: complete, shared inputs | 10 | 10,157 | 16,005 | 17,143 | 629.45 | 698.57 | +57.6% | +68.8% |
| 2022 ACS block groups: 99% coverage sensitivity | 11 | 11,555 | 17,474 | 18,822 | 578.67 | 660.61 | +51.2% | +62.9% |

Primary block cohort: 11,132 buildings; current MAE reduction 14.3%. MAE is people per area; bias is (predicted total − reference total)/reference total. This is descriptive; no spatially independent significance or confidence claim is made.

## Residential-only sensitivity

Census population measures usual residence, not commercial/daytime occupants. Restricting each model to its own predicted residential buildings changes both the population sum and the included building set; it is a sensitivity analysis, not verified residential selection.

| Cohort | Original MAE | Current MAE | Original total bias | Current total bias |
|---|---:|---:|---:|---:|
| 323 primary Census blocks | 34.12 | 27.22 | -51.0% | +32.6% |
| 10 fully covered ACS groups | 545.68 | 391.13 | -53.7% | +31.4% |

## Interpretation and next step

The preview should continue to label exposure as a proxy. Define resident population separately from commercial/daytime occupancy, audit accessory/duplicate/missed buildings, and validate any population calibration on geographically held-out blocks. Do not rescale to these Census totals and then call the same totals validation. The existing 17,523 damaged-building proxy has no observed displaced-person or casualty reference here.

## Limits

- 2020 population/2022 ACS versus pre-2025 fire imagery; not fire-time occupancy.
- Provided boundaries are TIGER2025 with older population joined by GEOID; vintage crosswalk not independently verified.
- Full tile coverage does not establish complete building detection or populated pixels.
- Primary comparison excludes blocks with 136 previously unavailable original-model inputs; zero-prediction blocks retained.
- Current-versus-original compares whole pipelines, including a different exposure head; no classifier-only causal claim.
- Commercial/daytime occupancy differs from Census usual residence; residential-only sums are sensitivity checks.
- No population rescaling, fitting or tuning to LA Census values.

ACS sampling margins of error are retained per group in block_groups.csv. Point-estimate errors do not incorporate ACS sampling uncertainty; the 10-group sensitivity cohort is small and the two-group complete-input cohort is insufficient for a strong generalization claim.

## Provenance and checks

Used the already-downloaded LA County Census package, not a new population model. Its README identifies 2020 Decennial counts, 2018–2022 ACS estimates, and TIGER2025 boundaries. All 22,024 buildings matched exactly one block and one block group. The old/current shared join has 21,888 records. Grid alignment is checked against available original polygon vertices; joins, valid population values, counts and source SHA256 are asserted or recorded in RESULTS.json. No calibration or model changes were made.

[Census usual-residence definition](https://www.census.gov/programs-surveys/decennial-census/decade/2020/about/residence-rule.html) · [ACS five-year geography and data](https://www.census.gov/data/developers/data-sets/acs-5year.html)

Files: RESULTS.json; blocks_2020.csv; block_groups.csv (with ACS 90% margins of error); building_comparison.csv; ACS_REFERENCE.json (source-hashed projection of geoid/B01003_001E/B01003_001M).

Reproduce: run validate_la_exposure.py with the existing py311 GIS environment. ACS_REFERENCE.json was extracted from the supplied Parquet using pyarrow; its original Parquet hash is checked at every run. Temporary reader packages are isolated in /private/tmp/la-census-parquet, and NumPy2 in /private/tmp/la-exposure-runtime; normal GIS runtime packages were not changed.
