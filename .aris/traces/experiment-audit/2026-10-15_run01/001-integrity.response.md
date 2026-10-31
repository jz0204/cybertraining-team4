**Overall verdict: WARN.** The saved artifacts support the central findings: a large training-area inconsistency, propagation into rule-derived labels, incorrect original ACS variable meanings, and poor frozen-model agreement with independent LA parcel-use records. Current scoring and reported numbers reproduce. The remaining warnings concern historical parcel provenance and the limited scope of the reference cohorts.

Review metadata: `review_independence: same-family`; `acceptance_status: provisional`; read-only review. No expensive CNN inference or network downloads were rerun.

**A. Ground-truth provenance — WARN**

- The LA reference is independently sourced. `audit_sources.py:41–64` requests Assessor records using LARIAC AINs and restricts responses to roll year 2024. `evaluate_assessor.py:50–53` loads those records, checks unique AINs and the year, and maps use without consulting predictions. The snapshot contains 8,577 distinct returned AINs from 8,581 requested.
- This remains a **parcel-use reference inherited by buildings**. The primary filter requires one outline **within the retained reference region**, not proof of one building on the complete parcel. The stricter cohort additionally checks one Assessor building data line. This limitation is explicit in `evaluate_assessor.py:87–95` and the report.
- Training-area calculations use archived geographic polygons rather than predictions: `audit_training.py:60–85`. I independently recalculated all 23,014 areas. EPSG:5070 calculations matched the saved CSV to floating-point precision; maximum disagreement with ellipsoidal area was `1.4540268481688656e-05` relative, or **0.001454%**.
- Original-label reproduction is real but conditional on reconstructed codes. `audit_training.py:44–51` reconstructs codes from archived classification outcomes, and `:89–98` replays the rules. My independent replay found **zero baseline type/unit/population mismatches and zero counterfactual mismatches**.
- Historical parcel attribution is not independently reproduced. The original notebook says overlapping parcels are resolved by smallest area, but performs an unsorted `drop_duplicates(..., keep="first")` at `building_population_pipeline.ipynb:1122,1149`. Its classifier also overrides oversized duplex/triplex codes at `:1324–1325`, while its source flag only recognizes single-family/mobile overrides at `:1354–1363`. Consequently, archived source flags cannot establish the full history of parcel-derived versus heuristic labels.
- The area and ACS counterfactuals are explicitly identified as synthetic diagnostics, not corrected ground truth: `audit_training.py:105`; `audit_acs.py:66`. No model-derived target was presented as independent truth.

**Resolved provenance defect:** The original Assessor mapper admitted condominium variants `010E`/`010H`, mixed-use `1720`, and records with blank use type and missing descriptions. Current exclusions at `evaluate_assessor.py:24–35` repair this. Regression cases appear at `tests/test_audit_eval.py:19–23`. The primary cohort changed from **3,340 to 3,332**, and the broad cohort from **9,127 to 9,036**; the principal finding remains.

**B. Score normalization — PASS**

- `evaluate_assessor.py:41–46` computes ordinary accuracy, an explicit three-class macro-F1, confusion matrices and per-class support. No performance denominator comes from the model’s own maximum, mean or score distribution.
- The figure normalizes confusion rows by reference support (`summarize_audit.py:40–44`) and displays raw counts. This is an appropriate visualization.
- Population clipping and `expm1` in `audit_model.py:32–44` are inference transformations, not normalization of an evaluation score.
- The population model’s stored training mean and standard deviation are used for its ridge features (`infer_stage2a_native3_ensemble.py:217–222`); they are not fitted from evaluation predictions.

**C. Result existence and numerical agreement — PASS**

All currently reported result files exist. The original LA manifest records `status: complete`. Its input, raw-prediction and final-prediction CSVs each contain **21,888 rows**, with **136 excluded inputs**.

Independent recomputation from the final matched CSV reproduced:

| Cohort | n | Accuracy | Three-class macro-F1 |
|---|---:|---:|---:|
| Broad parcel labels | 9,036 | 0.3075475874 | 0.1699943329 |
| IoU ≥ 0.5 | 7,601 | 0.2760163136 | 0.1575253740 |
| Primary single-outline | 3,332 | 0.0702280912 | 0.0483700220 |
| Strict Assessor-building | 3,315 | 0.0690799397 | 0.0474595274 |

The final primary confusion matrix is:

```text
                      Predicted small   multi   commercial
Reference small                  202      847         2227
Reference multi                    1        1           12
Reference commercial               4        7           31
```

Additional checks:

- Saved median training area inflation: **22.2370191248×**, independently reproduced.
- CNN CSV: **300 unique buildings, 100 per archived active class**.
- Numeric-area replacement: **0/300** class changes; mean probability L1 change approximately **0.00384613**, matching JSON within CSV/float precision.
- Zero geometry: **1/300** class changes; mean probability L1 change approximately **0.00746932**.
- All three retained ACS-table hashes match their provenance files.
- All **40 entries** in the final artifact hash manifest match current files.
- The narrow evaluator test module passed both tests, including the added mapping regression cases.

The completed CNN files replaced an earlier in-progress state; the final report does not cite nonexistent CNN results.

**D. Dead code and execution path — PASS**

Relevant metric and diagnostic functions have reachable callers and corresponding outputs:

- `score()` is called for cohorts, fire groups and area groups: `evaluate_assessor.py:95–98`.
- Archived rules are executed by `audit_training.py:89–98`.
- ACS comparisons and counterfactual populations execute in `audit_acs.py:25–66`.
- Both model diagnostics are invoked under the script entry point: `audit_model.py:121–124`.
- CNN intervention output is written at `audit_model.py:107–117`. The hook captures image features before geometry concatenation, consistent with `common.py:827–838`; the original-logit equality assertion at `audit_model.py:96–97` checks that the intervention path reconstructs the baseline.

No unused metric was found being represented as an executed result.

**E. Evaluation scope — WARN**

- The primary LA result covers **3,332/21,888 predictions (15.22%)**, in **92 cells**, across Eaton and Palisades. Selection depends on reference availability, record validity, unique matching, overlap and parcel-outline count (`evaluate_assessor.py:81–94`).
- Its reference support is **3,276 small residential, 14 multi residential and 42 commercial**. Small-residential recall is **202/3,276 = 6.17%**. This is the strongest supported classification failure. Minority-class results are descriptive and have little support; Palisades has no multi-residential and only one commercial reference.
- I independently verified the geometric comparison in EPSG:32611. Saved IoUs reproduce with maximum absolute discrepancy `1.11e-16`. No duplicate GT index or duplicated geometry was found. All **1,810 match rows sharing a prediction with another GT building** are excluded from scored cohorts.
- Matching is inherited from representative-point containment and median registration shifts (`match_table.py:10–24`). It is not a globally optimized one-to-one polygon assignment. The IoU filter and duplicate exclusions reduce ambiguity but do not establish perfect correspondence.
- The primary and broad sets have different footprint distributions. Their approximately 7% and 31% agreement figures must retain their cohort names; neither is an estimate for every LA building.
- The CNN experiment is a single deterministic, class-balanced diagnostic sample using five frozen checkpoints (`audit_model.py:56,72`). Five ensemble members do not constitute five independent evaluation replications. Its 96% agreement with archived proxy labels is not held-out semantic accuracy.
- The area-only result of **0/300 changed decisions** limits any claim that the numeric area input directly explains the LA type failure. The report correctly preserves this result and distinguishes it from training-label construction.
- Historical imagery, ACS and parcel vintages differ; the report discloses this. Population labels remain mixed exposure proxies without observed building occupancy validation.

**F. Evaluation-type classification — PASS**

| Component | Classification | Meaning |
|---|---|---|
| Geographic-area and ACS-definition checks | `real_gt` reference checks | Independent source geometry and Census definitions; deterministic provenance checks |
| Assessor comparison | `real_gt`, at parcel-use level | Independent administrative labels mapped to selected buildings; imperfect building-level reference |
| Archived-label replay | `synthetic_proxy` | Reproduction of existing rule-generated targets |
| Area/ACS counterfactual labels | `synthetic_proxy` | Rule-generated diagnostic targets, not observed truth |
| CNN and population perturbations | `self_supervised_proxy` / no-GT sensitivity | Controlled output comparisons; proxy agreement fields remain synthetic-proxy metrics |

Here `synthetic_proxy` includes rule-generated references; none of these targets was fabricated from the evaluated model’s predictions.

**Concrete repairs and claim impact**

1. **Completed:** repair Assessor ambiguity exclusions, add regression cases, and regenerate metrics and report.
2. **Preserve:** describe the 18,460 source flags as the archived `area_heuristic` count. The legacy flag omission means this is not an independently established exhaustive count of heuristic-derived labels.
3. **Before rebuilding training labels:** preserve raw parcel identifiers/codes, explicitly implement the intended overlap-resolution rule, and derive the classification-source flag from the same branch that returns the class. Do not overwrite frozen historical labels.
4. **Supported:** archived areas are inconsistent with their own geographic polygons; this affects the reconstructed rules substantially; original ACS variable interpretation is wrong; frozen LA predictions agree poorly with the specified independent parcel-use cohorts.
5. **Needs qualification:** the counterfactual flip count of **18,280** applies to the documented reconstructed-code diagnostic. It is not the measured outcome of rerunning the original raw parcel pipeline.
6. **Unsupported:** corrected semantic labels, accurate resident counts, full-LA accuracy, reliable minority-class performance, or a claim that numeric area input alone causes the LA failure.

Skipped: full CNN rerun, network source reacquisition, raw Harris parcel-join reconstruction and observed-occupancy validation. Remaining risk: parcel-to-building attribution and selected-cohort support limit interpretation beyond the audited diagnostic claims.
