# Frozen LA transfer results

All seven new runs completed; downloaded Qwen artifacts passed individual SHA256
checks. `RESULTS.json` and `SUMMARY.csv` are produced by
`../../analyze_la_transfer.py`. The analysis verifies the old CNN score parity,
sample ordering, available-input flags, and Qwen raw-output parsing.

## Same 3,157 available native-image pairs

| Pipeline | Accuracy | Macro F1 |
|---|---:|---:|
| CNN N2, two-seed probability ensemble | 96.36% | 0.5048 |
| DINO frozen, two-seed metric mean | 1.96% | 0.0891 |
| DINO fine-tuned, two-seed metric mean | 1.65% | 0.1212 |
| Qwen NF4 zero-shot | 98.64% | 0.5263 |
| Qwen QLoRA seed42 | 45.87% | 0.4276 |
| Qwen QLoRA seed43 | 60.53% | 0.3853 |
| Qwen QLoRA, two-seed metric mean | 53.20% | 0.4065 |

Macro F1 covers the three supported LA reference uses. New models still emit
five uses plus unknown; predictions outside the three reference uses count as
errors. These are supplied-footprint classification metrics, not detection AP.

## Complete 3,332-reference denominator

| Pipeline | Accuracy | Macro F1 |
|---|---:|---:|
| CNN N2 | 95.83% | 0.4833 |
| Qwen zero-shot | 93.46% | 0.5173 |
| Qwen QLoRA metric mean | 50.41% | 0.3983 |

All 175 unavailable native-context inputs remain unknown/errors for the new
pipelines. The common-image table isolates their effect; it does not replace
the full denominator.

## Interpretation

The current NYC-only QLoRA recipe degrades LA transfer relative to its base
model in both seeds. Its common-image macro-F1 difference is -0.1199, with
95% paired spatial-cell bootstrap interval [-0.1576, -0.0935] (1,000 draws,
91 cells). Full-cohort difference is -0.1190 [-0.1547, -0.0895] across92 cells.
These intervals concern this reference cohort, not label uncertainty or all cities.

Zero-shot's high accuracy masks zero recall on all14 multifamily references.
QLoRA recalls6/14 and5/14, but multifamily precision is only0.41% and0.65%.
Seed42 alone misclassifies1,464 small-residential references as multifamily.
CNN N2 recalls2/14 multifamily and31/42 commercial references on the common
cohort. No method here demonstrates reliable multifamily recognition.

Zero-shot's common-image macro-F1 advantage over CNN N2 is only0.0215;
the separate frozen paired comparison interval [-0.0728,0.1353] crosses zero
(`QWEN_ZERO_PAIRED.json`). Do not claim a stable VLM improvement over CNN.

The input-loader audit found valid RGB/mask tensors in its checked cases.
A separate diagnostic of four fixed-index zero-correct/QLoRA42-error cases
found valid near/wide views, with a suspected pool/patio inclusion in one target
outline and a possible partial-roof outline in another. This is AI visual
diagnosis, not human gold or an estimate of defect prevalence. Predicted polygon
area must not be treated as verified physical building area. Inputs and labels
were not changed following inspection.

LA was already inspected and is development evidence. It has only14 multi and
42 commercial references, no institutional/other reference panel, and no role
ground truth. NAIP/official-outline training versus Maxar/predicted-outline LA
also changes imagery and geometry provenance. CNN fields of view differ from
the new pipelines. These results cannot isolate a backbone effect or establish
generalization to an untouched city. The completed fixed-budget single-/multicity
DINO experiment is [reported separately](../20261028_multicity_transfer/REPORT.md).
