# Reliable-reference transfer experiment: methods and audit trail

This experiment predicts the actual use of an individual supplied building footprint,
plus its primary/auxiliary role. Five uses are small residential (1–3 dwellings),
multi-residential (at least 4 dwellings), commercial, institutional, and other
(known industrial, warehouse, utility, agricultural or supporting storage/parking use).
Unknown is abstention, not a sixth supervised use class. Auxiliary references without
a verified use contribute role supervision only; they are excluded from use accuracy.

## Frozen cohort

| City / split | Five-use references | Auxiliary role-only references | Total |
|---|---:|---:|---:|
| NYC train | 600 (120 each use) | 120 | 720 |
| NYC spatial validation | 149 (30 each except other 29) | 25 | 174 |
| Cambridge external test | 250 (50 each use) | 50 | 300 |
| Seattle external test | 244 (50/50/45/50/49) | 0 | 244 |

Training and checkpoint selection use NYC only. Both external cities were withheld
from training and selection. There are zero cross-split parcel duplicates and zero
intersections between the actual requested wide image windows across splits.
Images are NAIP, with target-near (minimum 40 m) and wide (minimum 300 m) crops.
Native resolution is 0.3/0.6 m in NYC, 0.3 m in Cambridge, and 0.6 m in Seattle;
input resizing does not create new observed detail.

NYC uses official PLUTO/footprint BBL links and conservative single-main-building
and dwelling-count filters. NYC auxiliary footprints use official codes 1004 and
5110, with association/size/distance checks for detached garages. NYC parcel-polygon
overlap was not measured. Cambridge combines official BLDG/OUTBLDG footprints with
agreement between FY2024 and FY2025 use and dwelling records. Seattle uses assessor
building/unit counts and unique 2023 footprints. Cambridge/Seattle main targets
require at least 95% parcel overlap. Sources and raw acquisition receipts are
preserved under the X9 experiment root. These are stronger official references,
not an independent human audit of actual occupancy at the imagery acquisition date.
Twenty-three visual checks verified image alignment; they do not certify all labels.

## Compared models

All methods use the same buildings, labels and paired physical views. This matches
data exposure, not model capacity, optimization compute or tensor resolution.

- DINOv3-B: 448 px views; target-mask area pooling on the near view and global pooling
  on the wide view; a 1536→128→7 head. Frozen encoder versus last two blocks and final
  normalization fine-tuned. Head LR 3e-4, encoder LR 1e-5, batch 8 × accumulation 2.
  Maximum 40/15 epochs, source-validation patience 7/3. Seeds 42 and 43.
- Qwen3-VL-8B-Instruct: NF4 double quantization, BF16 quantized computation, 512 px
  marked near/wide views. Zero-shot control versus language attention/MLP QLoRA,
  rank 8, alpha 16, dropout .05; vision parameters frozen. LR 1e-4, batch 1 ×
  accumulation 16, two epochs (90 optimizer updates), seeds 42 and 43. Sparse
  token supervision masks missing use values and balances role weights.
- All trained checkpoints are selected by the mean of NYC validation use macro-F1
  and role macro-F1. No external-city feedback enters selection. Reported seed
  means are separate-run means, not a prediction ensemble or best-seed selection.
- DINO abstention uses a threshold targeting 80% coverage on known NYC validation
  uses; external coverage and accuracy among answered cases are reported separately.
  Qwen can generate unknown, but prompt permission alone is not calibrated rejection.

## Evaluation repair and independent checks

An independent reload exposed a precision mismatch: training-time PEFT preparation
casts non-quantized parameters to float32, but the original standalone evaluation
omitted this step. Version 3 applies the identical preparation to both the zero-shot
control and saved adapters. Training data, prompts, weights, hyperparameters and
source-selected checkpoints are unchanged. Old outputs remain in execution_v2;
corrected outputs belong to execution_v3. The repair is explicitly recorded as
occurring after the earlier outputs, not retrospectively presented as preregistration.
A fresh GPU witness replayed all 174 source-validation cases from seed 42 with zero
use or role differences. Full final replay checks and collection receipts accompany
the corrected results.

The primary scorer independently recomputes accuracy, macro-F1, per-class recall
and role metrics with sklearn, checks prediction IDs and source-only selection,
and compares against metrics saved by inference. Paired macro-F1 intervals use
500 draws of fixed 2 km spatial blocks (9 Cambridge, 59 Seattle), averaging the
two seed-specific differences in each draw. This block rule was amended before
any held-out predictions were observed. Intervals concern this selected cohort;
they do not estimate uncertainty over all possible cities or training seeds.

## What the results can support

The evaluation is conditional building-use classification on supplied footprints,
not detection AP or an end-to-end Stage 1→Stage 2a evaluation. The cohort is
purposively class-stratified, so its accuracy is not city-prevalence-weighted.
Seattle contains no reliable auxiliary test references: primary false positives
can be assessed, but Seattle auxiliary recall cannot. The same official field
used for a reference label must not later be supplied as an input and described
as independent multimodal generalization. No model or threshold was retuned after
reading these external-city results. Subsequent adaptive development requires a
new untouched final test region.

## Reproduction and artifact locations

X9 root:
`/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/reliable_transfer_20261027`

- Dataset SHA256: `0a9fe81ed273067a7a01f962409ce779f9055a501211a2aa33faa6af0161411f`.
- Original training protocol SHA256: `6718bd87b79607188d7156f320145c9c7fbe4089241d18f3b7f5f85be4e6ce50`.
- Corrected evaluation protocol SHA256: `4f2ad6f1a04e7261a0cdbd8ff3e4245778f1689c8c9e1adc0afb7ebb0d0ee817`.
- `protocol/SOURCES.md`, `raw/`, `RAW_SHA256.json`: official source provenance.
- `DATASET_AUDIT.json`, `INPUT_SHA256.json`: sampling, split and image integrity.
- `execution_v3/PROTOCOL.json`, `ENVIRONMENT.json`: complete model and environment pins.
- `execution_v3/remote_v3`: corrected predictions, metrics, selected adapters/checkpoints,
  logs and remote/local hash verification after collection.
- `execution_v3/LOCAL_INPUT_AND_TRAINING_VERIFIED.json`: all 8,629 local input hashes,
  selected epochs and measured per-run training resources.

The actual training and inference were run on one A30 24 GiB. QLoRA takes roughly
35 minutes per seed in this setup; the training-loop reserved-memory peak is
11.06 GiB, but cold-load/witness peak reservation is 23.13 GiB. These measurements
do not validate this exact procedure on an RTX 3080. DINO fine-tuning takes about
2.5–2.8 minutes per seed (excluding initialization and image loading) here.
