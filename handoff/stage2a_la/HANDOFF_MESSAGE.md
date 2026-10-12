Hi Xihan,

Here is my fork with the updated Stage 2A inference package for the LA v3 buildings:
https://github.com/jz0204/cybertraining-team4

Start with handoff/stage2a_la/RUN.md. Download the archives and SHA256SUMS from
https://github.com/jz0204/cybertraining-team4/releases/tag/stage2a-la-v1

I have
included the standalone code, RUN.md and model checksums in the code archive,
and the five type checkpoints plus population regression model in a separate
weights archive. Extract both into the same directory.

Please follow RUN.md to run the first 100 rows as a smoke test, then the full
v3 product on one A6000. The runner reuses your existing pre-fire crops and masks,
computes area from the product's UTM footprints and preserves cell_id/bldg_uid
for joining the results. It writes to a new output folder.

Please return the full-run output folder, including predictions, the input audit,
exclusions, environment/run manifest and inference log. No assessor labels or
Census inputs are needed for this inference run. Building-type validation can
follow after the assessor use-code labels are available.

The full five-model path passed a local two-building CPU execution check using
LA demo imagery. The remote v3 inputs and GPU environment still need the smoke
check described in RUN.md.

Thanks,
Jinyu
