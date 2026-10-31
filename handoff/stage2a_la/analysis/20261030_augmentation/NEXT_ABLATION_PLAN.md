# Isolated small-multifamily data ablation

Keep the frozen three-class definition: small residential includes 1–3 units;
multifamily starts at 4 units. Do not move duplex/triplex records into the positive
class. The preceding recommendation's “2–8 units” wording was inconsistent with
this definition; this experiment uses 4–8-unit positives.

## Data gates

- Exclude the original complete LA product region plus its existing 2km buffer.
- Preserve the existing hash-based 2km train/validation/reserved grid assignment;
  acquire only training-grid imagery. Full image context must fit its grid.
- Require one Assessor building-data line, one countywide historical outline,
  compatible floor/roof areas, and matching newer building ID with outline IoU≥0.8.
- Visually review the target, surrounding buildings and crop before admission.
  These are official records corroborated by AI image review, not human gold labels.
- The original imagery pool is exhausted for diverse compact positives. The wider
  batch uses dated public NAIP2022 RGB, with native resolution recorded separately
  from export resolution. Match the previous CNN crop's physical extent. Original
  training images and frozen validation/test inputs remain unchanged.
- Score negatives only with the frozen augmentation ensemble. Consider a candidate
  confusing if ensemble multifamily probability≥0.05 or either seed≥0.10; admit
  only visually/officially supported single-family buildings. Cap at 4 per grid,
  40 total, ranked by ensemble multifamily probability. Report true false positives
  separately from near-boundary candidates; do not call easy correct examples hard.

## Comparisons

| Arm | Data |
|---|---|
| A | Existing augmentation-fixed baseline, seeds42/43, reused without retraining |
| B | A data plus admitted isolated 4–8-unit buildings |
| C | B data plus admitted confusing single-family negatives |

Use the same N2 initialization, ConvNeXt-tiny architecture, loss, optimizer, 6 epochs,
8,192 draws/epoch, batch32, stochastic flip/rotation and weighted438-validation
checkpoint selection. Keep source/LA mass at 0.5/0.5; keep LA class masses at
478:93:177. Within multifamily preserve equal mass for 4-unit, 5–8-unit, and 9+-unit
groups. These are expected sampling probabilities; realized random draw counts can
vary. Sample counts change, not these group masses. Verify normalization before
launch. Do not select seeds, epochs, or thresholds using the LA test outcomes.

Report both seeds and their probability ensemble on the same frozen3332 cohort:
accuracy, macro-F1, all class P/R/F1, multifamily TP/FP/FN and paired spatial
bootstrap. Report new-source and geographic diversity together: this is a complete
data-addition comparison, not an isolated causal estimate of label quantity.

The LA cohort is development evidence with only14 multifamily references. Keep
reserved grids untouched; a new-region/city evaluation is required for a later
generalization claim. If candidate gates cannot be met, record the unmet gate rather
than quietly lowering label quality or using test-neighborhood images.
