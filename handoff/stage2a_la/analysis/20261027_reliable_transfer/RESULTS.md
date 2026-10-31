# Matched DINO / QLoRA transfer experiment

Training and model selection use NYC only. Cambridge and Seattle are frozen external cities.
Numbers below are building-level official-reference classification results, not independent human gold or detection AP.

| Model | City | Accuracy | Macro F1 | Multi recall | Institutional recall | Auxiliary recall |
|---|---|---:|---:|---:|---:|---:|
| dino_frozen_s42 | cambridge | 0.408 | 0.377 | 0.720 | 0.500 | 0.840 |
| dino_frozen_s42 | seattle | 0.373 | 0.326 | 0.240 | 0.740 | not measured |
| dino_frozen_s43 | cambridge | 0.432 | 0.418 | 0.580 | 0.580 | 0.920 |
| dino_frozen_s43 | seattle | 0.414 | 0.362 | 0.360 | 0.760 | not measured |
| dino_finetune_s42 | cambridge | 0.488 | 0.472 | 0.900 | 0.380 | 0.920 |
| dino_finetune_s42 | seattle | 0.410 | 0.364 | 0.340 | 0.760 | not measured |
| dino_finetune_s43 | cambridge | 0.476 | 0.463 | 0.860 | 0.260 | 0.900 |
| dino_finetune_s43 | seattle | 0.467 | 0.436 | 0.480 | 0.700 | not measured |
| qwen_nf4_zero | cambridge | 0.428 | 0.374 | 0.040 | 0.160 | 0.020 |
| qwen_nf4_zero | seattle | 0.525 | 0.437 | 0.000 | 0.160 | not measured |
| qwen_qlora_s42 | cambridge | 0.448 | 0.450 | 0.260 | 0.500 | 0.960 |
| qwen_qlora_s42 | seattle | 0.541 | 0.540 | 0.240 | 0.500 | not measured |
| qwen_qlora_s43 | cambridge | 0.428 | 0.406 | 0.120 | 0.520 | 0.980 |
| qwen_qlora_s43 | seattle | 0.557 | 0.509 | 0.040 | 0.660 | not measured |

Paired changes in five-use macro F1 (mean of two seeds; 500 draws of fixed 2km spatial blocks):

- cambridge:dino_finetune: +0.070, 95% interval [+0.028, +0.116].
- cambridge:qwen_qlora: +0.054, 95% interval [+0.004, +0.129].
- seattle:dino_finetune: +0.056, 95% interval [+0.029, +0.084].
- seattle:qwen_qlora: +0.088, 95% interval [+0.041, +0.146].

- Official-source references, not independent human or exact-imagery-date gold.
- Purposive class-stratified samples; accuracy is not city prevalence-weighted accuracy.
- Seattle has primary-role references only; no Seattle auxiliary recall estimate.
- Conditional building-use classification on supplied footprints; no detector AP measured.

Checks: prediction IDs and train/validation IDs match the frozen dataset; all city metrics agree with independent sklearn scoring; no held-out city entered training or checkpoint selection.
Bootstrap intervals describe this selected reference cohort, not variability over all cities.
