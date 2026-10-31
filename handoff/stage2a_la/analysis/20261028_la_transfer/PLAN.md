The next decision is whether the existing NYC-trained DINOv3 and Qwen QLoRA
models improve the same LA building-use regression cohort used for the CNNs.
This is a retrospective transfer diagnostic. LA has already informed previous
development and cannot become a fresh final test by renaming its split.

Freeze all 3,332 primary assessor references: 3,276 small residential,
14 multi-residential, 42 commercial. Preserve original five-CNN, N1 and N2
probability ensembles and their original inputs. Report their unchanged scores
alongside DINO frozen/fine-tuned seeds42/43, Qwen NF4 zero-shot, and QLoRA
seeds42/43, with exactly the already-selected checkpoints, prompts and precision.
No LA training, checkpoint selection, threshold fitting or label edits.

Recover pre-fire Maxar context from the public event catalog. All cohort
centroids are covered. Select scene identity by agreement with original crops,
without model predictions or class-specific optimization. Require three probes
per cell where available, each correlation >=.97; preserve alternatives and
receipts. Derive near >=40m and wide >=300m using the existing target-extent
rule; retain native source windows, dates, crop georeferences, missing-pixel
checks and image hashes. Review representative paired views before inference.
No enlarged 256px crop counts as new context. Resampling and field of view differ
from the CNN protocol, so this compares complete pipelines, not an isolated
backbone effect. If the full cohort cannot pass image checks, record the problem
before any predictions rather than silently dropping buildings.

Pre-inference image-QA amendment: the original cell_00046 crop contains40.7%
nodata; scene matching therefore requires >50% jointly observed pixels and
reports that fraction. Cell_00349 is visibly the same scene but has raw-pixel
correlation .93-.96 from interpolation differences, including when replaying
the original mosaic code. A4x4 box average (about1.2m) gives .983-.989. Accept
raw or4x4-averaged correlation >=.97, retaining both values. This matches scene
identity only; native target/context missing-pixel gates remain unchanged.

Input-coverage rule fixed before inference: preserve every one of the3,332
reference buildings. A building failing native target/context checks gets an
explicit input-unavailable record and an unknown/error in the full-cohort
pipeline score. Also compare every model on the same eligible subset. Report
the missing-input counts by class separately from model-emitted unknowns.
No synthetic fill or replacement date is used to complete a missing window.

Score accuracy, macro-F1 across the three supported reference uses, balanced
accuracy, per-class precision/recall, confusion, predictions outside the three
reference uses, and unknown/invalid output rates. Five-use models retain all
five choices; institutional, other, unknown and invalid are errors on this
three-use cohort. Show the all-small baseline and stricter single-assessor-
building sensitivity. Role predictions have no LA reference and receive no
accuracy claim. Frozen DINO confidence thresholds retain their NYC definition;
report coverage and selective accuracy separately. Report each seed and the
arithmetic mean, not a fabricated Qwen probability ensemble. Use paired spatial
bootstrap over LA cells for changes, acknowledging only 14 multifamily labels.

Use the existing pinned A30 environment and independent automatic restoration
watcher. Separate bounded inference batches so each original-service pause
stays below90minutes. Verify source-validation replay in the actual new runner
before admission. Keep all artifacts and interrupted attempts. Do not replace
production models or change downstream exposure calculations.

In parallel, prepare an equal-count NYC-only versus NYC+Cambridge+Seattle DINO
training comparison. The existing external cities are now development data.
Keep validation and context/parcel separation explicit, expand minority-use
subtypes, and reserve genuinely new cities for a later locked final test.
