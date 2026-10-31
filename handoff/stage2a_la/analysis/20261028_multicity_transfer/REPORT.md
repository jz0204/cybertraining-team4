# Fixed-budget single-city versus multicity DINO

All four fits and four LA evaluations completed. Both arms passed an independent
documented GPU witness before admission. All30 result artifacts, including four
57,516,367-byte checkpoints, were downloaded and SHA256-verified. The independent
analysis passed sample-ID, source-validation, checkpoint-selection, epoch-budget,
prediction-hash and unavailable-input checks.

## Controlled comparison

- A:720 NYC training buildings.
- B:260 NYC +260 Cambridge +200 Seattle training buildings.
- Both:120 records per primary use plus120 auxiliary records; identical174 NYC
  validation records, seeds42/43,15 epochs and675 updates per fit.
- Same DINOv3 ViT-B/16, final-two-block fine-tuning, near target-mask pooling plus
  wide-view global pooling, five-use/two-role heads, optimization and source-only
  checkpoint selection. No LA checkpoint selection or threshold tuning.
- Primary use subtype totals match. Auxiliary subtype composition and municipal
  reference/image provenance are not identical; this tests data composition,
  not geography isolated from every other factor.

## LA development results

| Available3,157 images | Accuracy | Macro F1 |
|---|---:|---:|
| A NYC seed42 | 5.00% | 0.1334 |
| A NYC seed43 | 7.48% | 0.1053 |
| A NYC metric mean | **6.24%** | **0.1194** |
| B multicity seed42 | 60.63% | 0.4344 |
| B multicity seed43 | 64.24% | 0.4523 |
| B multicity metric mean | **62.43%** | **0.4433** |

B-minus-A macro F1 is **+0.3240**, paired spatial-cell95% interval
**[0.2120,0.4301]**,1,000 draws over91 cells. Both seeds improve. On all3,332
references, including175 unavailable-input unknown/errors, accuracy is
5.91%→59.15% and macro F1 is0.1175→0.4347; difference+0.3172
[0.1900,0.4188],92 cells. The stricter single-assessor-building sensitivity
also improves: macro F1 difference+0.3246 [0.1906,0.4294].

The result supports a narrow conclusion: replacing part of the NYC training
set with these Cambridge/Seattle examples substantially reduces the LA
development failure under equal data and optimization budgets. It does not yet
establish performance on an untouched final city.

## Remaining weaknesses and tradeoffs

Multicity small-residential recall rises to61.04%/64.85%, but multifamily recall
is6/14 and3/14 with precision only2.47%/5.66%. Commercial recall is15/42 and14/42,
with precision83.33%/87.50%. Multifamily recognition remains unreliable.

The source-selected confidence thresholds retain81.03%/88.98% of available LA
inputs, with accepted-case accuracy65.72%/66.54%. The source80%-coverage rule
does not provide a target-domain accuracy guarantee or a reliable unknown gate.

NYC validation use macro F1 falls from0.5947/0.5873 (A) to0.4548/0.4095 (B).
Role macro F1 is0.8736/0.8736 versus0.8685/0.8736. Better LA transfer comes with
a source-domain use-classification tradeoff. There is no LA role ground truth.

For context, the prior CNN N2 probability ensemble achieves96.36%/0.5048
accuracy/macro F1 on the same available cohort; Qwen zero-shot achieves
98.64%/0.5263, and NYC-only Qwen QLoRA metric mean53.20%/0.4065.
Multicity DINO has not surpassed those CNN/zero-shot macro-F1 point estimates.
CNN uses a probability ensemble; DINO/Qwen fine-tuning summaries average seed
metrics. These comparisons concern complete pipelines, not a controlled
backbone-only ablation. See [the complete LA report](../20261028_la_transfer/REPORT.md).

## Evidence boundary and next step

LA was previously inspected, has only three reference uses (including14 multi
and42 commercial cases), and uses Maxar imagery/predicted outlines rather than
the NAIP/official-outline training inputs. Official assessor references are
not independently adjudicated actual-use or occupancy truth. No institutional,
other, auxiliary-role or detection-AP generalization result is established here.

Keep these results and checkpoints frozen. Next, prepare and freeze the proposed
Denver reference/image cohort before predictions, retaining all exclusions and
missing-input cases, then evaluate both locked arms. Do not tune on that final
city after observing results. No Denver data or predictions were used in this run.

Both original services passed final independent health checks at20:23:31 UTC;
DiskPressure=False. No production classifier was replaced.
