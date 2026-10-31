# Locked comparison and interpretation

The controlled question is whether replacing 460 of 720 NYC training buildings
with Cambridge/Seattle buildings improves geographic transfer under the same
model, coarse primary subtype counts, validation panel and optimization budget.
This tests training-data composition. It does not isolate geography from every
municipal labeling convention, auxiliary subtype or image-acquisition difference.

Fit DINO A/B with seeds42/43 for15 epochs and675 updates each. Select checkpoints
only with the unchanged174 NYC validation records. Preserve each seed and their
arithmetic metric mean; do not select a winning seed on LA or the final city.
Keep the source-derived80%-coverage confidence rule unchanged at evaluation.

LA is a development regression: score the unchanged3332 cohort and separately
the3157 eligible image pairs. It supports only three reference uses and no role
accuracy. Cambridge/Seattle are retired tests and cannot supply new heldout
evidence after entering the training pool. All metrics on them are development.

Denver is the proposed untouched final city. Freeze its reference rules,
selection queues, image-quality exclusions, actual cohort and hashes before
any predictions. Conservative official records are not independent human or
exact-imagery-date truth. Report unavailable inputs and every class/role count;
never relax references or replace failures based on a model's prediction.
Stop acquisition with a readiness report if a required class cannot meet the
declared sample count. Keep San Francisco as a separate later candidate.

For a ready Denver cohort, evaluate all four locked A/B checkpoints once on
identical inputs. Primary endpoint is B-minus-A five-use macro-F1, averaged
across seeds. Report each seed, per-class precision/recall/F1, use accuracy,
role macro-F1/auxiliary recall, and source-threshold coverage/selective accuracy.
Use1000 paired bootstrap draws of the frozen2km spatial groups; bootstrap
intervals describe this cohort, not uncertainty over all cities/training seeds.
Missing-use auxiliary rows contribute role evaluation only. Unknown/invalid
predictions count as errors for known labels and are reported separately.

A positive point estimate alone is tentative. A paired95% interval above zero
and the same direction in both seeds support a narrow positive transfer claim
for this city and reference protocol. Class failures, role tradeoffs and source
validation regressions remain part of the result. A negative result must not
trigger unreported retuning on Denver; any later adaptation requires another
untouched test. One new city cannot establish universal generalization.

These experiments classify supplied footprints. They do not measure detection
AP, city-prevalence accuracy, actual occupancy/population, downstream exposure
benefit, or production readiness. No reference use/units field is a model input.
