# Preliminary DINO-only LA transfer

The four DINO runs are complete and independently reproduced. Qwen and the full LA analysis remain unfinished.

Eight remote prediction/evaluation files (6,341,821 bytes) match their local SHA-256 hashes. Dataset, protocol and execution code hashes also match. All four remote evaluations and the historical CNN N2 scores reproduce.

`CNN_N2mean2` is the existing two-seed probability ensemble. DINO means are arithmetic means of seed metrics. Accuracy and recall are percentages; macro-F1 is on the 0–1 scale.

## full3332: 3332 buildings, 92 LA cells

| Model | Accuracy % | Macro-F1 | Balanced acc. % | Multi recall % | Commercial recall % |
|---|---:|---:|---:|---:|---:|
| CNN_N2mean2 | 95.8283 | 0.483279 | 61.5181 | 14.2857 | 73.8095 |
| dino_frozen_s42 | 0.7203 | 0.074954 | 34.9817 | 92.8571 | 11.9048 |
| dino_frozen_s43 | 3.0012 | 0.102334 | 34.1677 | 85.7143 | 14.2857 |
| dino_frozen_seed_mean | 1.8607 | 0.088644 | 34.5747 | 89.2857 | 13.0952 |
| dino_finetune_s42 | 1.2305 | 0.123407 | 31.9597 | 71.4286 | 23.8095 |
| dino_finetune_s43 | 1.8908 | 0.118325 | 38.5124 | 92.8571 | 21.4286 |
| dino_finetune_seed_mean | 1.5606 | 0.120866 | 35.2361 | 82.1429 | 22.6190 |

## common3157: 3157 buildings, 91 LA cells

| Model | Accuracy % | Macro-F1 | Balanced acc. % | Multi recall % | Commercial recall % |
|---|---:|---:|---:|---:|---:|
| CNN_N2mean2 | 96.3573 | 0.504806 | 61.7095 | 14.2857 | 73.8095 |
| dino_frozen_s42 | 0.7602 | 0.075023 | 34.9851 | 92.8571 | 11.9048 |
| dino_frozen_s43 | 3.1676 | 0.103227 | 34.2148 | 85.7143 | 14.2857 |
| dino_frozen_seed_mean | 1.9639 | 0.089125 | 34.5999 | 89.2857 | 13.0952 |
| dino_finetune_s42 | 1.2987 | 0.123644 | 31.9718 | 71.4286 | 23.8095 |
| dino_finetune_s43 | 1.9956 | 0.118783 | 38.5360 | 92.8571 | 21.4286 |
| dino_finetune_seed_mean | 1.6471 | 0.121214 | 35.2539 | 82.1429 | 22.6190 |

## Paired LA-cell bootstrap

Macro-F1 differences below use 1,000 paired cell draws, RNG seed 20261028 and percentile 95% intervals. Each family averages its two seed macro-F1 values within each draw.

| Cohort | Candidate minus control | Delta macro-F1 | 95% interval |
|---|---|---:|---|
| full3332 | dino_frozen_s42-CNN_N2mean2 | -0.408325 | [-0.493867, -0.318475] |
| full3332 | dino_frozen_s43-CNN_N2mean2 | -0.380946 | [-0.459636, -0.294278] |
| full3332 | dino_finetune_s42-CNN_N2mean2 | -0.359873 | [-0.445504, -0.278479] |
| full3332 | dino_finetune_s43-CNN_N2mean2 | -0.364955 | [-0.441913, -0.290705] |
| full3332 | dino_frozen-CNN_N2mean2 | -0.394635 | [-0.476337, -0.307600] |
| full3332 | dino_finetune-CNN_N2mean2 | -0.362414 | [-0.443918, -0.291448] |
| full3332 | dino_finetune-dino_frozen | +0.032222 | [-0.006298, +0.054113] |
| common3157 | dino_frozen_s42-CNN_N2mean2 | -0.429783 | [-0.509580, -0.331085] |
| common3157 | dino_frozen_s43-CNN_N2mean2 | -0.401579 | [-0.476084, -0.301366] |
| common3157 | dino_finetune_s42-CNN_N2mean2 | -0.381162 | [-0.462175, -0.282342] |
| common3157 | dino_finetune_s43-CNN_N2mean2 | -0.386023 | [-0.454792, -0.305270] |
| common3157 | dino_frozen-CNN_N2mean2 | -0.415681 | [-0.492952, -0.318052] |
| common3157 | dino_finetune-CNN_N2mean2 | -0.383592 | [-0.455559, -0.293740] |
| common3157 | dino_finetune-dino_frozen | +0.032089 | [-0.004553, +0.053115] |

## Prediction behavior and scope

- dino_frozen_s42: 3066 multi-residential predictions; multi precision 0.4240%; small-residential recall 0.1832%.
- dino_frozen_s43: 2916 multi-residential predictions; multi precision 0.4115%; small-residential recall 2.5031%.
- dino_finetune_s42: 3011 multi-residential predictions; multi precision 0.3321%; small-residential recall 0.6410%.
- dino_finetune_s43: 2930 multi-residential predictions; multi precision 0.4437%; small-residential recall 1.2515%.

The high multi-residential recall coincides with extensive false positives on small-residential references. All DINO runs retain 175 input-unavailable unknowns; none emits an additional unknown or invalid use. Source-threshold selective coverage/accuracy, per-class metrics and confusion matrices are in DINO_PRELIMINARY_RESULTS.json.

- Only the four completed DINO runs are collected and scored; no Qwen or complete LA result is claimed.
- LA is previously inspected development evidence, not an untouched final city.
- Only three assessor reference uses are evaluated; there are no LA role or five-class accuracy labels.
- Field of view and resampling differ from CNN; this compares complete pipelines, not a backbone-only effect.
- The 175 unavailable small-residential inputs stay unknown/errors in full3332; common3157 is scored separately.
- Only 14 multi-residential and 42 commercial references; high multi recall must be read with precision and prediction counts.
- Bootstrap intervals are conditional on these fixed predictions and two seeds; they do not measure training-seed uncertainty.
- Qwen, strict single-assessor sensitivity and the complete final analysis remain unfinished.

No remote writes, job/admission changes, training, frozen-runner edits or multi-city runner authorship were performed.
