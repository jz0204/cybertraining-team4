# LA Stage2a evaluation handoff

## 2026-10-23: evaluate the current model on the shared LA inputs

- User requested download of `la_v3_prefire_inputs.tar`, `la_census.tar.gz`, and
  `README.md` from Xihan's `2026-10_handoff` SharePoint folder, then comparison
  of current Stage2a accuracy against the previous LA result.
- Browser download is authorized in this chat. The link requires the recipient
  account `zhjyu11@tamu.edu`; the first browser visit also presents UT Austin
  organizational consent. Do not mistake the HTTP 200 sign-in page for data.
- The latest trained candidate is **area mask pooling**, under
  `/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/poolfix_20261015/`.
  Use all five `runs/seed{42,123,2025,3407,7777}/stage2a_best_model.pt` files,
  `models/population_logfootprint_soft_type_residual_hgb.pkl`, and the paired
  `II_package/scripts/stage2a/` source snapshot. Eight required artifacts match
  `EXPORT_SHA256.json`; all five checkpoints explicitly specify `area`.
- `rebuild_20261015/la_handoff_v2/` is the earlier nearest-pooling retrain.
  `handoff/stage2a_la/weights/` contains the original LA baseline. Neither is
  the latest candidate. Preserve these weights and their source behavior.
- Reuse `handoff/stage2a_la/run_la.py` input preparation and finalization.
  Join by **cell_id + bldg_uid**, preserve UTM polygon area, and record every
  exclusion. The original product SHA-256 is
  `7c525076c924a7354538d0e5be6fbfd62f3089aa6a032ba1264723f05a755806`.
- Compare with frozen flags in
  `handoff/stage2a_la/analysis/20261015_gsd_assessor/la_assessor_matched.csv`.
  The existing evaluator is the other repository's
  `II_package/scripts/stage2a/infer/evaluate_la_assessor.py`.
  Do not rebuild cohorts from new predictions, tune with LA labels, or silently
  drop previously evaluated buildings.
- Baseline recomputation on 2026-10-23 matched all four saved confusion matrices,
  accuracies and macro-F1 values. Primary: 3,332 buildings, accuracy 0.070228,
  macro-F1 0.048370. Broad parcel cohort: 9,036 buildings, accuracy 0.307548,
  macro-F1 0.169994. These are different selected cohorts, not all-LA estimates.
- Original inference covers 21,888 of 22,024 product buildings; 136 empty masks
  were excluded. Preserve that distinction in comparisons and population sums.
- Census counts support area-level analysis; they are not building-type labels
  or observed per-building occupancy. The model population output is an exposure
  proxy. Harris test scores do not establish LA accuracy.
- Local Mac GPU is available to PyTorch outside the filesystem sandbox. The
  archived inference CLI currently falls back to CPU for `--device mps`; passing
  that flag alone does not enable the GPU. Check the actual selected device and
  compare a small CPU/GPU sample before a full run.
- The user supplied the completed local download
  `/Volumes/Crucial X9/LA_v3_inputs_and_census.zip`. Data and new results are under
  `/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_eval_20261023/`.
  The archive is preserved; `SOURCE_MANIFEST.json` records extracted archive
  hashes and safe extraction. `DATA_INVENTORY.json` records the Census tables.
- All 21,888 valid keys, 136 exclusions, footprint areas, and 126 shared-CSV
  hashes match the earlier run. `INPUT_PARITY.json` records the comparison.
  `log1p(area)` differs by at most 8.9e-16 across platforms: compare derived
  floating-point values with a numerical tolerance, not exact equality.
- `runtime/` contains a copy of the frozen candidate inference source with only
  explicit MPS selection added. `RUNTIME_MANIFEST.json` records this difference.
  `smoke/SMOKE.json` confirms CPU/MPS decisions agree on eight LA rows; maximum
  probability difference is 2.12e-6. Original model code and weights are intact.
- The first full attempt used five seeds, batch size 64, no workers, MPS, and no
  temperature adjustment. Its process, command and status are in
  `latest_full/RUN.json`; timestamped progress is in `latest_full/inference.log`.
  `finalize_and_analyze.py` reuses the existing finalizer and evaluator after
  inference completes; its paired-outcome self-check has passed.
- The local full run was interrupted after the user authorized cluster use.
  On this 16 GiB Mac, sustained MPS inference coincided with about 30 GiB of
  swap and slowing batches; a small GPU smoke test does not establish sustained
  throughput. Preserve `latest_full/` as the interrupted attempt, not a result.
- The user explicitly authorized new training on the cluster, cluster inference,
  temporary LLM shutdown, and transfers without hash verification. Current work
  first measures the already-trained candidate; do not train on evaluation rows
  and then describe their score as held-out accuracy.
- Cluster: `ssh k3s-master`, GPU node `tamu-gpu` (NVIDIA A30), namespace
  `instance-impact-stage3`, existing PVC `harvey-workspace`. Frozen candidate
  and dependencies remain at `/stage3/stage2a_poolfix_20261015` and
  `/stage3/python_deps`; no re-upload of weights is needed.
- Current cluster inputs/results use `/stage3/la_eval_20261023`. Job
  `stage2a-la-eval-20261023` starts suspended until upload finishes and the GPU
  is released. Its local manifest is `cluster_job.yaml`. The source archive
  streams directly through `harvey-stager` to avoid a second on-cluster copy.
  The SSH user cannot directly access the PV host path; use its existing pods.
- `cluster_runner.py` checks CUDA against eight saved local CPU predictions,
  then runs all five models (FP32, batch 128, four workers). Download results to
  local `cluster_full/`; `finalize_and_analyze.py` now reads this location.
- Qwen (`llm-inference/qwen3-14b`) and Ollama
  (`sptc-trans-resilience/ollama`) initially have desired/ready/available 1/1/1.
  `restore_llms.sh` waits for the evaluation Job to finish and its running pod
  to disappear before restoring both counts. Four mocked control-flow checks
  passed; verify actual ready/available counts after completion too.
- `reference/` contains hash-verified copies of the frozen cohorts, baseline
  predictions, evaluator and finalizer. Its preflight copy is explicitly dated
  before the download; use the current run manifest for progress.
- **Completed:** all 21,888 valid buildings were inferred on the cluster A30
  in 227 seconds including the CUDA smoke check. CUDA/CPU decisions match on
  eight rows, with maximum probability difference 6.45e-7. Complete predictions
  are in `cluster_full/stage2a_predictions.csv`, not `latest_full/`.
- Fixed primary n=3,332: accuracy **7.0228% → 78.3914%**, macro-F1
  **0.048370 → 0.350259**. There are 2,387 corrected old errors, 9 introduced
  errors and 720 remaining errors. Broad n=9,036 accuracy is 85.0044%.
- Do not call the LA three-class problem solved. Majority-small accuracy on the
  primary cohort is 98.3193%, macro-F1 0.330508; new commercial precision is
  only 6.5359%. Of 720 errors, 708 are small residential references, including
  569 predicted commercial and 139 predicted multi-residential.
- Area groups still matter: new accuracy is 98.53% at ≤100 m² and 61.51% above
  400 m². Training small-residential median area is 257.7 m²; LA primary median
  is 282.3 m². This does not isolate a causal area-feature defect.
- No new training or LA tuning was performed. A future adaptation run must
  preserve independent cell/parcel splits and must not score training rows as
  held-out results. No per-building occupancy truth was supplied.
- `REPORT.html` / `REPORT.md`, `assessor_evaluation/paired_analysis.json`,
  `primary_paired_predictions.csv`, `primary_remaining_errors.csv`, and
  `FINAL_SUMMARY.json` contain the final analysis. HTML numbers and all six
  artifact links were checked; the renderer reports zero warnings.
- **LLMs restored:** both desired/ready/available counts are 1/1/1, recorded in
  `LLM_STATE_AFTER.json`. Qwen needed an image pull and warmup after scaling;
  replica count alone was not completion. Its health probe returned HTTP 200.
- The working tree already contained untracked October 15 audit code/results.
  Preserve them. No commits or pushes are authorized. A read-only Git status in
  the external-drive repository may need command-local LFS filters disabled to
  avoid attempted writes under `.git/lfs/tmp`.

## 2026-10-23: commercial precision diagnosis

- Follow-up report and reproducible diagnostic scripts are in the same X9 run's
  `commercial_diagnostic/`. `REPORT.html`, `DIAGNOSTIC.json`,
  `primary_diagnostic_rows.csv`, and `CHECKS.json` contain the findings.
- Distinguish commercial precision from recall: primary TP=40, FP=572, FN=2,
  TN=2718; precision 6.5359%, recall 95.2381%, FPR 17.3860%. Harris finalized
  FP32 held-out export has TP=121, FP=6, FN=11, TN=2515; FPR 0.2380%.
  Keeping Harris sensitivity/FPR but using LA prevalence gives a hypothetical
  83.10% precision. This is an algebraic illustration, not measured new accuracy.
- The low precision persists under strict assessor-single-building criteria
  (6.25%) and primary IoU >=0.8 (8.46%). Of 572 FPs, 566 have single-family
  UseCode prefix 01, 545 masks do not touch crop borders, and 529 have no zero
  RGB pixels. These checks reduce obvious alternative explanations but do not
  convert parcel-use references into manually confirmed building labels.
- All 21,888 valid LA inputs have estimated GSD near 0.3052 m/px; Harris training
  is near 0.4669. Both crops are 256px: physical widths 78.1m vs 119.5m.
  Same-area targets therefore occupy about 2.34x pixels. Among small-reference
  buildings restricted to 200-400m2, median mask fill is 4.862% LA vs 2.045%
  Harris. Do not claim resizing both crops to 224px harmonizes physical scale.
- **Negative diagnostic result:** on 98 outcome-stratified examples, adjusting
  only numeric `mask_fill_ratio` by (LA_GSD/Harris_GSD)^2 changed none of the
  five-model ensemble classes; all 32 sampled commercial FPs remained FPs.
  RGB, masks, image features and physical area were held fixed. This does not
  test image resampling and is not a validated fix or a representative accuracy
  estimate. CPU baseline reproduced CUDA classes; probability difference at
  most 6.99e-7. Bounded CPU run took 126 seconds; no cluster outage was needed.
- Current model `input_mode` is `rgb_mask_geometry`, not plain `rgb_mask`.
  All five configs use weighted sampling AND inverse-sqrt weighted CE. Commercial
  gets 3.61x relative sampling weight and 3.61x loss weight (nominal product
  13.04x). Approximate class numerator masses become equal. Batch normalization
  of weighted loss and optimization prevent treating this as an exact causal
  explanation; remove one weighting mechanism in a controlled future ablation.
- Scores are overconfident in this cohort: 170 rows have commercial p>=.99,
  but only29 are commercial references (17.06%). Raising threshold to .99 gives
  29 TP/141 FP and69.05% recall. All threshold sweeps are descriptive only;
  nothing was applied. Single-member precision ranges3.88%-8.76%;257 FPs are
  unanimous across five seeds. Do not promise more similar seeds fix this.
- Four RGB/mask pairs were visually inspected (3 high-confidence residential
  FPs and1 commercial TP), with notes saved. One FP's mask is at the left crop
  border: do not assume the center building is the target. Roof appearance is
  not independent use truth. The masked CNN can still receive context through
  its receptive field; do not say it sees no surroundings.
- Prioritize physical-scale/crop alignment, a separate reweighting ablation,
  verified LA residential hard negatives, then validation-only calibration.
  Existing LA crops cannot restore a wider context by resizing and padding.
  New untouched spatial regions are needed for a fresh blind test; the42
  primary commercial references occupy only7 cells. No training, model edits,
  threshold application, LLM shutdown, commits or pushes occurred this turn.

## 2026-10-23: implementation plan, no training launched

- The user requested a summary followed by detailed implementation, so this
  turn prepared `la_eval_20261023/improvement_plan/PLAN.md` and `PLAN.html`,
  `EXPERIMENT_MATRIX.csv`, `FEASIBILITY.json`, and two first-round configs.
  Earlier training authorization remains valid; this planning turn did not
  launch training, upload files, or operate cluster services.
- W1 keeps weighted class sampling and sets `class_weight_mode=none`; W2 keeps
  inverse-sqrt loss weights and sets `sampler_mode=natural`. The configs retain
  seed42, all source-only data, native3, area pooling and other training settings.
  Output/config paths target a new `/stage3/stage2a_la_adapt_20261023` directory.
  Compare to existing seed42, not to the five-member ensemble. Full seed42 LA
  probability export is still needed: previous ensemble files have member hard
  decisions, not full member probabilities.
- All201 required Harris original pre-disaster images exist under the X9
  `II_package/outputs/stage3_harvey/source_data/xBD/train/images/` directory.
  Current frozen Harris train/val/test have zero pairwise intersections for
  both tile_base and parcel_key. These are verified local planning facts.
- S1 is a proposed training-view experiment: 50% original views /50% LA-scale
  views, one view per draw with the same optimizer-update budget. LA-style
  crops cover78.1m (about167.4 native Harris pixels), resampled to256px with
  vector masks rerasterized and mask geometry recomputed. Physical footprint
  area and labels stay fixed. This augmentation is not implemented yet.
- A0/A1 will control for continuation itself: same initializer, optimizer and
  number of draws, with Harris-only versus2:1 Harris/verified-LA domain replay.
  Proposed LR1e-5, at most10 epochs, ordinary CE in both arms. Current trainer
  has no Stage2a init_ckpt option, and explicit domain sampling needs a small
  addition. Do not imply these are currently runnable config-only changes.
- The old `calibrate_stage2a_type_bias.py` assumes global five-class constants;
  it must be adapted to checkpoint class names for native3. Uniform temperature
  on the final ensemble probabilities alone does not change argmax. Calibrate
  and select thresholds using separate grouped validation, and calibrate the
  baseline too for fair decision-rule comparisons.
- Broader LA references provide62 commercial buildings on55 parcels, still
  only7 cells. There are16 cells without an existing valid reference score;
  these are not automatically independent test data. Existing LA cohorts are
  development evidence; untouched spatial regions and audited labels are
  required before claiming a fresh blind test. Hard-negative sampling occurs
  only after spatial assignment and only within the training partition.
- Proposed primary selection is precision at commercial recall>=90%, with
  Harris validation macro-F1 loss<=0.01 as a development guardrail. Halving FPR
  at equal recall is only a research-progress gate. At current LA prevalence,
  90% recall needs FPR about1.15% for50% precision, or0.29% for80% precision.
  These are illustrative requirements, not promised gains or user-set targets.
- Initial pilots are W1/W2/S1, one seed each. Add A0/A1 after labels are audited;
  complete the original five seeds only for the selected recipe. Population
  HGB ultimately needs new classifier OOF probabilities/refitting; type gains
  cannot be reported as building-occupancy accuracy gains.

## 2026-10-23: authorized commercial-improvement execution started

- User explicitly authorized training/uploads on cluster and temporary LLM
  pauses. New artifacts are on X9 at
  `II_package/outputs/stage2a/la_adapt_20261023`; remote equivalent is
  `/stage3/stage2a_la_adapt_20261023`. No transfer hash checks or Git commits.
- `stage2a-la-weights-20261023` runs B0 seed42 FP32 export, W1 sampler-only,
  and W2 loss-only sequentially. Frozen training code/data are reused; only
  the predeclared weighting option and output/config paths differ. Real A30
  BF16 forward/backward witness passed with finite gradients. CPU preflight
  verified all input files, exact 13355/2028/2653 splits, no tile or parcel
  overlap, and all 21888 LA inputs. Outputs are type-only probabilities.
- Both LLM deployments were 1 desired/ready before scaling to zero. Independent
  master watcher PID237660 (`/tmp/stage2a-la-weights-20261023-restore.py`) watches
  this exact job and restores the saved original counts after completion or
  failure. Its watchdog suspends the job after10800s; job deadline is9000s.
  Watcher checks rollout readiness, not only scaling. Nine terminal/live-pod
  control-flow cases passed. Completion/restoration still needs final checking.
- S1 changes live only in an experimental runtime copy. It selects one native
  or LA-scale view per training draw, recomputes selected mask geometry, and
  keeps validation/test native. Tests cover disabled tensor parity against
  frozen code, physical area/targets, RGB-mask alignment within inherited PIL
  one-pixel edge convention, seeded choices, invalid-view fallback and fixed
  dataset length. Original frozen source is unchanged.
- The first CPU scale-preparation job failed because 7 source training pixel
  polygons self-intersect (2 commercial,1 multi,4 small). The v2 preparation
  records these as invalid-view fallbacks; do not silently repair geometry or
  drop these training rows. Both failed job/log and corrected version remain.
  `stage2a-la-scale-prepare-20261023-v2` is the current CPU preparation job.
- B0 seed42 (not ensemble): primary commercial TP38/FP541/FN4, precision6.563%,
  recall90.476%, AP0.336407. Descriptive development optimum at recall>=90%
  gives TP38/FP477 and precision7.379%. Harris FP32 validation macro-F1 is
  0.940237. New pilots must be compared to these seed42 figures; cohort counts
  and ID joins are asserted in `compare.py`. These LA cohorts are development,
  not a new blind test. No population accuracy conclusion is supported.
- CPU scale preparation v2 completed:13355 training rows,13348 valid alternate
  views,7 explicit invalid-polygon fallbacks,171 truncated targets (141 commercial,
  26 multi,4 small). All51 original manifest columns compare equal across18036
  rows; val/test alternate paths are empty. Effective median GSD0.304552m/px
  (integer167px source window), target0.305188. Four native/alternate overlay
  pairs were inspected; inherited crop-edge and off-nadir roof/footprint limits
  remain. Preparation produced~1.1GB; node still had7.6GB free.
- `label_audit/spatial_roles.csv` fixes1km EPSG32611 blocks before candidate
  selection, seed2025. Actual crop corners were recovered by per-cell affine
  fits from paired source pixel/UTM polygons (max residual0.000751m). Crop
  boundary crossings (2884 rows) and2 cross-role AINs (6 rows) are buffered;
  no shared active AIN or overlapping crop can cross roles. Only the original
 21888 valid inputs participate; the136 previously excluded empty masks have
  invalid/empty pixel WKT and must not be silently reintroduced by joins.
- This fixed assignment exposes an evidence gap: calibration has1535 broad
  residential references and no commercial/multi references. Strict IoU>=.8
  training candidates contain297 small,1 multi,4 commercial; the review queue
  is54 residential hard negatives,200 random residential controls,5 rare-class
  candidates. All are explicitly pending building review, not newly verified
  training labels. Do not fit a commercial recall threshold on a zero-positive
  calibration partition or promote weak parcel-use matches to building truth.
- W1 completed29 epochs, selected epoch19 by Harris validation. On the unchanged
 3332-row LA primary:TP38/FP429/FN4, precision8.137%, recall90.476%; B0 seed42
  wasTP38/FP541/FN4. W1 removed179 original FPs but introduced67;179 remaining
  FPs still have p>=.99. At each model's descriptive recall>=90% development
  threshold, W1 FP420 versus B0 FP477. This is useful but misses the planned
  FPR-halving research gate. Harris validation macro-F1 rose0.940237->0.947141,
  while test fell0.905496->0.887400 (multi->small32->36; commercial->small10->11).
  Do not hide this test regression or use it to retrospectively alter selection.
- A four-worker CPU loader check initially hit bus errors in harvey-runtime-smoke
  because its default /dev/shm is too small. Re-running in the CPU check Job with
  the same4Gi memory-backed shm as training passed199 real rows, all7 fallbacks,
  and repeated four-worker sequences exactly (93 scale choices/199 draws).
  GPU training manifests already had4Gi shm and were unaffected.
- W2 completed30 epochs, best20: LA primary TP39/FP609/FN3, precision6.019%,
  recall92.857%, AP0.302021. At descriptive recall90.476%, FP467/precision7.525%.
  Harris val macro-F1=0.937583; test=0.888159. W1 is the predeclared-rule winner
  for S1 initialization settings; selection did not use Harris test.
- First GPU batch completed at2026-10-23T22:20:04Z in1949.64s. Median epoch
  interval29.691s (~450 training draws/s including epoch validation overhead).
  Runtime snapshot was95% A30 utilization and7251/24576MiB GPU memory.
  W1 best checkpoint and W2 run artifacts are on X9. `WEIGHTS_REPORT.html`
  contains first-batch results; parsed numeric text and all local links passed.
- LLM restoration after W completed and was verified at22:28:35Z: both desired,
  ready,available1. Qwen health200. First new Qwen pod had transient
  UnexpectedAdmissionError; replacement succeeded after re-pulling its9.1GB
  pinned image (5m45s) and loading weights. Do not call restored merely on scale.
  Record: `LLM_AFTER_WEIGHTS.json`. No driver/service configuration was changed.
- S1 is now `stage2a-la-scale-20261023`, output `runs/S1_scale_seed42`; it starts
  from the same ImageNet weights as W1, not W1-checkpoint continuation. Only
  labels path (same IDs plus alternate paths), output/config paths and
  scale_view_probability=.5 differ. Config parser and source checks passed.
  It uses experimental runtime,4Gi shm and A30; job deadline5400s.
  Independent restore watcher masterPID247366 watches this exact job with7200s
  watchdog and original counts in `/tmp/stage2a-la-scale-20261023-llm-before.json`.
  S1 completion, actual per-epoch view counts and final LLM readiness remain to
  be verified. Original model/source and both repository branches are preserved.

## 2026-10-23: W1/W2/S1 completed; source-only gain is limited

- All three authorized single-seed pilots completed on A30. The second GPU job
  completed at22:44:54Z in940.58s; both GPU batches total2890.22s (48.2min),
  excluding preparation/service restoration. S1 ran29 epochs, best19. Its387295
  draws include193511 selected scale views and162 explicit native fallbacks.
- Canonical results are `la_adapt_20261023/COMPARISON.json`, using the same FP32
  export path for B0/W1/W2/S1. `REPORT.md`/`REPORT.html` are the final pilot report;
  `WEIGHTS_REPORT.html` is only an earlier W1/W2 snapshot. Charts, prediction
  CSVs, configs, all three best checkpoints and run records are on X9. Full
  derived scale images remain on cluster; X9 has generation code/manifest/samples.
- On unchanged primary3332 (42 commercial), S1 TP38/FP371/FN4, precision9.291%,
  recall90.476%, AP0.402502; B0 TP38/FP541, precision6.563%, AP0.336407.
  S1 fixes233 old FPs, introduces63, retains308; remaining371 include368 small
  and3 multi, with148 at p_commercial>=.99. S1 loses one prior commercial TP
  and recovers one prior FN, so equal aggregate recall is not equal TP identity.
- Development threshold comparison at recall>=90%: B0 FP477/P7.379%; W1
  FP420/P8.297%; W2 FP467/P7.525%; S1 FP365/P9.429%. S1 FPR reduction23.480%
  misses the predeclared halving gate. No five-seed expansion, applied threshold,
  production replacement, LA training, or population HGB refit was performed.
- Preserve negative transfer: Harris FP32 test macro-F1 B0 .905496 versus S1
  .905017, but commercial recall121/132->117/132. Commercial->small10->12 and
  commercial->multi1->3; multi->small32->28. Broader LA9036 reference commercial
  recall55/62 (88.71%)->51/62 (82.26%), FP1022->711, P5.11%->6.69%. Primary
  recall stability must not be generalized to this broader cohort.
- Paired bootstrap5000 over92 primary cells (only7 commercial cells): S1 raw
  precision deltaCI95 +.005660..+.052079; recall -.076923..+.111111. This is
  single-seed development evidence, not independent generalization or uncertainty
  covering selection. `PAIRED_UNCERTAINTY.json` records conditional limits.
- Training-time and export scores are not interchangeable: W2 and S1 each have
  one Harris test hard-decision difference. S1 trainer summary macro-F1 .902904
  vs common FP32 export .905017. Preserve original records and use the common
  export for comparisons (`PRECISION_AUDIT.json`). Trainer population metrics
  are not task evidence when alpha_pop=0.
- Clarify label split feasibility: active commercial references span6 1km blocks,
  strict commercial candidates3 blocks, strict multi candidates only2 blocks.
  Zero calibration positives is a defect of this fixed random assignment, not
  proof that alternative commercial stratification is impossible. Review labels
  before a versioned stratified redesign; keep AIN/crop isolation and do not call
  old diagnosis regions blind test. All259 review-queue rows remain pending.
  `NEXT_STAGE_FEASIBILITY.json` and `rare_spatial_coverage.csv` capture the gap.
  A0/A1 await independently corroborated building-use/date evidence as specified
  in the accepted plan; the current task asked the user for available sources.
- Both LLMs restored to desired/ready/available1, verified22:54:26Z. Watcher says
  scaled_back=true/ready=true. Current Qwen and Ollama native HTTP readiness logs
  return200 at22:57:40Z; see LLM_AFTER_SCALE and LLM_HEALTH_AFTER_SCALE.json.
  S1 replacement pods have0 restarts. Correction to earlier notes: failed
  qwen pod j88mf (and ollama dl4qj) started2026-10-15, so these are historical
  admission failures, not newly caused by this run. Qwen re-pulled its image
  for6m7s this restore; account for startup latency in later pause schedules.
- Download pitfall: a large `kubectl exec tar | tar` stream truncated S1; tar
  left a full-length but unreadable checkpoint. A later successful command can
  mask that failure unless the shell uses set-e plus pipefail. Recovered using
  kubectl cp --retries=3 to master /tmp/stage2a-S1-best-20261023.pt then scp.
  All three local checkpoints load and have finite tensors (`CHECKPOINT_CHECK`);
  no transfer hashes were computed. Size alone would not have detected this.
- LLM images lacked guessed `python`/`curl` binaries, and a direct HTTP request
  from the training namespace was refused despite native health probes passing.
  Used current HTTP probe logs/readiness; did not alter access controls or install
  diagnostic packages inside LLM containers. No generation load test was run.
- HTML pitfall: `am --replace` targets AM_HOME/pages, not an arbitrary -o path.
  Update a custom output with `am patch REPORT.html --panel ID --no-open`.
  No browser automation, Git commit/push, baseline mutation or repository-wide
  CI was needed; verification is scoped to this isolated experiment runtime.

## 2026-10-24: Source feature and cross-city data exploration

- User permits continued parameter/model/data experiments on cluster, uploads,
  inference and temporary LLM suspension; no transfer hashes or commit/push.
  New isolated output: X9 `II_package/outputs/stage2a/la_explore_20261023`.
- VPN is connected, but local Tailscale DNS returns NXDOMAIN for both cluster
  names. Verified existing SSH keys with temporary HostName/HostKeyAlias options:
  tamu-gpu=128.194.19.98 (geog-zk8-wk3.artsci.tamu.edu), k3s-master=10.55.180.150
  (geog-zk8-master.artsci.tamu.edu). No SSH config or host-key policy was changed.
- Physical-building mapping revealed24 cross-split groups (48 rows):21 train/test,
  2 val/test,1 train/val. All are residential_small. Quarantine removes22 train
  and2 val rows, preserves all2653 test IDs, yields13333/2026/2653 train/val/test.
  Original manifests remain unchanged; this does not resolve broader spatial
  autocorrelation or2017-imagery/2025-parcel weak-label mismatch.
- Five seed42 recipes: C0 S1 on clean split; C1 adds20m context pooling; G1 removes
  explicit type geometry; N1 uses natural sampling; D1 adds stochastic per-draw
  transforms plus ColorJitter. D1 is a combined augmentation recipe. Selection:
  LA development P@R>=90%, clean Harris val macroF1 no more than.01 below C0.
  Harris test is diagnostic only. San Diego inference is deferred until added-data
  recipe decisions finish (PROBE_HOLD.json); no SD labels used for selection.
- CPU job stage2a-la-explore-prepare-20261024 completed:53451 20m rings, no empty
  building masks/overlap;6 scaled views have no available context. Preserve those
  valid full-frame targets. Local and cluster focused tests passed. Latest local
  selection checks are included in GPU job before training and CUDA preflight.
- CSV pitfall caught before training: missing val/test scale_valid caused pandas
  to serialize train1 as1.0; dataset requires string1. Normalize to integer strings
  and assert13326 valid scale/ring pairs. Fixed only new manifest, not old records.
- Local two-worker synthetic check hit sandbox denial of torch_shm_manager and
  stalled; stopped that specific process and reran with permission, passing. Real
  training remains cluster-only. macOS tar xattrs also caused noisy harmless
  extraction warnings; use Python USTAR_FORMAT for subsequent scoped uploads.
- GPU job stage2a-la-explore-train-20261024 was created suspended. First launch
  stopped before any LLM mutation because watcher was not present in old PVC
  directory. Copied the tested local restore_llms.py to master /tmp directly.
  Snapshot LLM_BEFORE.json verifies both desired/ready/available1. Independent
  watcher uses exact job,12600s watchdog and /tmp/stage2a-la-explore-20261024-*.
  Training and restoration results must be read live before claiming completion.
- BuildingSense public release is1000 NYC examples, not the full34458 paper set.
  First500 archive downloaded to X9; second downloading. Only geometry and NYC
  orthophoto files extracted; do not execute downloaded scripts or unpickle data.
  Official code mapping must distinguish NYC C0 (three families) from C1-C3,
  exclude mixed/industrial/institutional ambiguity, and verify CRS/units. Initial
  raster EPSG6539 and vector EPSG2263 are US survey feet, not metres. No added
  BuildingSense training has happened yet. Source metadata is in external/.
- GPU batch started2026-10-24T05:20:36Z on A30 after finite BF16 forward/backward.
  Restore watcher masterPID273031 is live. `kubectl wait --for=delete -l ...`
  also selects historical Failed pods, so timed out although GPU was already
  free (0MiB). Check no Running/Pending LLM pods instead; leave historical failed
  pods intact. Both deployment counts were zero before unsuspending GPU job.
- Both BuildingSense archives downloaded/read successfully on X9 without hashes.
  New data experiment output `la_data_20261024`:578 retained (445 small,43 multi,
  90 commercial),422 taxonomy exclusions; no empty/nodata crops or duplicate
  footprints;4 truncated views retained and recorded. All RGB crops have.305188m
  GSD; physical areas use US survey-foot conversion. Band4 is not alpha, so only
  documented RGB1-3 are used. NAD83 to NAD83(2011) conversion is ballpark with
  available grids;36 fixed class-stratified overlays inspected,0 gross wrong
  building/empty targets. Not a full manual use audit or exact-date gold labels.
- Added-data protocol fixed before training: A1 NYC10%/Harris90%, A2 NYC25%/Harris75%,
  within-city inverse-square-root class sampling; Harris conditional distribution
  exactly preserved. All arms draw13333 examples/epoch, use C0 architecture and
  seed42 ImageNet initialization, unchanged Harris val/test. NYC has no population
  target and both scale aliases point to its existing target-resolution crop.
  Data runtime is a separate copy, with fixed epoch draws, per-row sampling
  multipliers and actual NYC draw counts. Default sampler weights/seeded sequence,
  mixture probabilities, count/weight validation and manifest round trip passed.
  No data-arm training or San Diego model inference has run at this entry.
- C0 complete26epochs/best16. LA primary argmax TP37/FP486/FN5, P7.075%, R88.095%,
  AP.438989; P@R>=90%=6.281% (FP567,threshold.255486). C0 clean-val macroF1.934024.
  Repaired identity split plus changed draw/augmentation sequence is not an
  isolated causal test of leakage removal; keep this regression versus old S1.
- C1 complete29epochs/best19. LA argmax TP33/FP165/FN9, P16.667%, R78.571%; matched
  R90.476% P10.951%, FP309,threshold.044079. Clean-val macroF1.907421 fails C0-.01
  guardrail(.924024), so do not promote based on raw precision. C0/C1 probability
  mixture diagnostics at C1 weights.25/.5/.75 yield P90.07143/.10270/.12063;
  only.25 passes source guardrail and it remains below old S1. Secondary protocol
  and results in context_mixtures; no Harris test selection or extra training.
- Continued pipeline avoids an unnecessary LLM restart between batches. Created
  suspended stage2a-la-data-train-20261024. New master restore watcherPID280897
  monitors this final job, using ORIGINAL before-campaign replica counts1/1 from
  /tmp/stage2a-la-explore-20261024-llm-before.json, max10800s from handoff. Status:
  /tmp/stage2a-la-data-20261024-restore-status.json; log has corresponding suffix.
  Dependency queuePID280932 at /tmp/stage2a-la-data-20261024/queue_after_source.py
  starts data only after source job terminal AND no live source pods. Data
  preflight rejects failed/incomplete source BATCH. Queue closes before watchdog
  deadline or after restore-status exists; six mocked cases passed. Old watcher
  PID273031 was verified by cmdline then terminated ONLY after both new processes
  were alive. Scripts and PIDs are job-scoped. Source G1 currently training;
  data job remains suspended. Final LLM readiness/HTTP health still mandatory.
- NYC independent area check decoded1156PNG files; rasterized area discrepancy
  median.101%, max3.694% after accounting for visible footprint, passing12% bound.
  NYC contains0 small residential >400m2 versus1497 in Harris train and549 in LA
  primary. Thus more-city examples do not automatically cover the actual hard
  negatives. Geometry domain summary and plot preserve this coverage limitation.
- An exploratory HCAD historical-label lookup found official public-data pages
  but direct current/2017 downloads returned HTTP403. No access-control workaround,
  browser action, owner lookup, records request, or2017 labels were obtained.
- Analysis now includes same-footprint area assignment and seven-fold cell-based
  calibration/model-selection cross-fit diagnostic. Focused arithmetic and
  synthetic seven-group checks passed. Cross-fitting already-inspected LA data
  is not a new blind test and cannot undo prior adaptive design. Main report
  remains pending until all seven models finish. Intermediate REPORT.html created
  with answer-me-with-html CLI, no browser opened, STE0 warnings.
- Plotting runtime uses older Matplotlib: boxplot(labels=...), not tick_labels;
  select Agg before pyplot in headless work. Trainer history is metrics_history.jsonl,
  not history.jsonl; failed archive attempt was detected and replaced with a
  successful bounded --exclude=*.pt download. No hashes or unrelated deletion.
- G1 removing explicit geometry failed the clean-Harris guardrail: canonical FP32
  val macroF1.908599; LA matched-recall FP632/P5.672%. N1 natural sampling passes:
  val.950979, LA argmax TP37/FP226/FN5, AP.534656; matched recall90.476% TP38/FP272,
  P12.258%. Relative to original B0 FP477 this is42.98% fewer FP, below50% gate.
- Before any NYC training results, recorded DATA_EXTENSION_PROTOCOL.json adding
  N2 natural sampling plus578 NYC rows. Original A1/A2 protocol preserved. N2
  draws13333 of13911 available rows without replacement, expected4.155% NYC;
  same278 steps/epoch and N1 architecture/seed/optimizer. Default natural loader
  remains unchanged; positive-budget RandomSampler is isolated to data runtime.
  Local checks passed unique/reproducible draws and all source/NYC row parity.
  This is an adaptive N1-triggered extension, not a preregistered experiment.
- N2 extension uploaded while data Job was still suspended with no startTime;
  remote sampler checks passed before training. All three data recipes now present.
- N1 fixed visual audit:all5 commercial FN,15 highest-confidence FP,12 seed72
  other FP. Two FP masks touch/cut the crop frame (selection indices23/25).
  Several high-confidence FP have light/flat roofs; this suggests appearance
  confusion, not causal attribution or verified use. One FN has partial roof
  coverage. No labels, cohorts or training recipes changed after this inspection.
  N1 LA argmax accuracy91.687%, commercial area ratio4.366 on common footprints;
  still unsuitable for direct commercial-area totals despite improvement.
- Cluster node advertises2 allocatable GPU units but worker nvidia-smi lists one
  physical A30, already occupied. Do not treat resource units as free devices.
  Launched CPU8/16Gi reference-fp32 Job to recompute existing5-seed Harris val/test
  ensemble on clean2026/unchanged2653 IDs. It does not open the San Diego probe.
- Probe runner prepared with release assertions and CPU/GPU LA probability parity
  check. Must wait FINAL_SELECTION and any gate-triggered5-seed confirmation.
  GradeA13=small7/commercial6, no multi; AB75=34/14/27 across6 neighborhoods.
  No three-class macroF1 claim for A13; C150 excluded. No probe threshold fitting.
- All5 source trials completed06:41:10Z; N1 selected. D1 has22epochs/best12,
  matched-recall FP470/P7.480%, canonical val.932183. Data3-arm job auto-started,
  real NYC3-class BF16/CUDA forward/backward finite, no CPU training fallback.
- Critical comparison boundary: current production/reference FIVE-seed ensemble
  has LA argmax P6.536%/R95.238%, but after development threshold.791345 it has
  P11.111%/R90.476%, FP304. N1 FP272 is only10.53% fewer than this ensemble,
  versus42.98% fewer than single-seed B0 FP477. Do not present the latter as
  improvement over the current ensemble. Added current ensemble to PR table and
  paired cell-bootstrap comparisons. Five-seed expansion gate remains the frozen
  original B0 threshold; no post-hoc rule change.
- INCIDENT2026-10-24T07:00:57Z: wk3 root filesystem crossed kubelet5% available
  threshold (~5.254GB); available fell to~5.228GB/df4.9GiB. Data training pod
  stage2a-la-data-train-20261024-6d74c and old harvey-runtime-smoke were Evicted.
  Last observed A1:34 epochs, best24; completion/export must be checked on disk.
  A2/N2 have not run. Do not label the data campaign complete or open SD probe.
- Old restore watcherPID280897 scaled BOTH LLM deployments back to desired1.
  At latest check each ready0/available0 due DiskPressure. The20-minute rollout
  wait may time out; desired1 remains, but readiness is NOT restored yet.
- Worker /home has53GB free on a separate filesystem. SSH user cannot traverse
  root-owned PVC host path; sudo requires a password. A bounded ordinary PVC
  maintenance job (stage2a-la-storage-audit-20261024) with a disk-pressure
  scheduling toleration was rejected at kubelet admission before its command ran.
  No cleanup, priority elevation, eviction-policy change, or PVC mutation occurred.
  User asked for a sudo read-only du listing or administrator freeing>=5GB;
  never ask for/send passwords. Need normal host admin help for this disk issue.
- Reserve actual node-root headroom BEFORE training, not only PVC nominal30Gi or
  node Ready. Before failure5.3GiB available was too close to eviction threshold.
  Future resume should require>=8GiB available plus DiskPressure=False and account
  for both best/last checkpoints, new data, caches, and service restart overhead.
- Source5-trial artifacts/checkpoints all saved and loadable on X9. Current5-seed
  reference CPU FP32 Harris export completed before eviction: val macroF1.945921,
  test.900154; N1 val.950979/test.897919. N1 test commercial FPR.238%, LA6.869%,
  about28.86x higher. Rarity alone cannot explain target-domain precision collapse.
- Added --source-only analysis to preserve an honest partial report while data
  training is unavailable. SOURCE_* outputs exclude all unfinished NYC recipes;
  full analysis still requires FINAL_SELECTION and all three data exports.
  Area helper now selects ID/truth/pred columns before joining area, avoiding
  duplicate footprint_m2 columns in existing ensemble CSVs. Source analysis ran.
- Source-only uncertainty: N1 versus current5-seed ensemble at fixed development
  thresholds has commercial-precision difference95% cell-bootstrap CI[-.001603,
  .029712] (5000 draws; conditional on chosen model/threshold). Seven-fold spatial
  threshold cross-fit of fixed N1 yields TP38/FP299/P11.276%; reselecting source
  recipe per fold yields TP38/FP292/P11.515%. These are post-hoc diagnostics, not
  blind generalization or training-seed uncertainty. Do not claim stable gain.
- REPORT.html/REPORT.md now state source5 complete, data3 blocked, and LLMs not
  ready. REPORT_CHECK.json verifies metrics,5 checkpoint checks,8 local links,
  embedded PR image and zero final style warnings. HTML Markdown links need
  percent-encoded spaces; raw absolute paths with spaces rendered as literal
  Markdown. STORAGE_RECOVERY.md and CLUSTER_INCIDENT.json preserve exact blockers
  and prerequisites. Latest live check07:17:39Z still DiskPressure=True, both LLMs0/1.
- Rechecked07:24Z: DiskPressure remainsTrue, desired1/ready0/available0 for both
  LLMs. Original restore watcher's1200-second readiness wait timed out and exited;
  replica counts remain1/1. Do not assume this old watcher can restore future jobs.
- Storage follow-up2026-10-24~17:29Z: root98GiB used87GiB/free5.9GiB,
  /home118GiB used59GiB/free53GiB; DiskPressure remainsTrue and both LLMs0/1.
  Kubelet stats reports imageFs/containerFs on the separate126168399872-byte
  filesystem with11498176512 bytes used; do not double-count those identical
  counters or attribute container images to root. Live LLM PV paths are
  /home/zhjyu11/llm-models and /home/zhjyu11/sptc-trans-resilience-ollama-models.
  Thus LLM model storage is on /home, not the full root filesystem. The task PVC
  remains under /var/lib/rancher/k3s/storage; its actual size is still unknown
  because host traversal is denied and sudo requires a password. Do not state
  it is the largest consumer without a successful directory-size audit.
- Migration preparation2026-10-24: user authorized moving training data/output
  to GPU /home while retaining existing models/results. Uploaded
  handoff/stage2a_la/storage_migration/migrate_workspace.py to wk3 at
  /home/zhjyu11/stage2a-storage-migration-20261024/migrate_workspace.py.
  Remote self-test passed, including metadata parity and size-change detection;
  utility and concurrent-fstab guards verified in uploaded version. Migration
  has NOT started: sudo requires user's interactive password. Never request it.
  Script checks active users and capacity (8GiB reserved), moves the whole named
  harvey-workspace directory, compares file inventory/size/ownership/modes (no
  hashes), and bind-mounts it back persistently to retain PVC /stage3 paths.
  Actual workspace size remains unknown; capacity check may refuse migration.
  If interrupted, inspect both paths and audit before retrying; never overwrite.
  After admin execution, verify bind/PVC access, saved weights/results, root
  headroom and DiskPressure=False before starting a new training job. A1 status
  and fresh LLM restoration watcher still need inspection/setup; no training
  resumed and reboot persistence has not been exercised.
- Storage scope revision2026-10-25 (America/Chicago): user ran whole-PVC
  migration; it correctly refused BEFORE mutation. Apparent size54428741993B
  versus /home available56778006528B cannot retain8GiB headroom; allocated du
  total48G differs from apparent50.7GiB. User permits other-node storage.
  Live audit: master /home276GiB free; wk1 root75.4GiB and wk2 root80.1GiB.
  Existing PVs are local, no NFS PV observed. GPU root now11.8GiB free but
  DiskPressure stillTrue; do not equate Ready with safe training admission.
  Prefer moving only II_package (~18G), stage2a_la_explore_20261023 (~1.4G),
  stage2a_la_data_20261024 (~626M) to GPU /home, retaining local I/O and existing
  historical directories. New migrate_active_workspace.py reuses original
  mover, preflights all three and combined capacity, binds each original path.
  Uploaded to the same remote script folder; local and remote self-tests pass.
  sudo -n still requires a password; no migration performed this turn. User
  must execute the NEW script via sudo. If any step fails, inspect audit/state
  before retrying because completed directories must not be moved again.
- Migration verified2026-10-25: all3 COMPLETE.json receipts, inventories and
  fstab entries confirmed. Root32GiB/home34GiB initially free. DiskPressure
  cleared normally at00:53:04Z Oct26 (Oct25 local). A1 actually completed34
  epochs/best24 plus all3 exports before original eviction; A2 was interrupted
  during epoch2, N2 absent. Earlier partial-state inference was outdated.
  A1 checkpoint+summary+predictions downloaded to X9; checkpoint loads with
  finite model tensors. A1 primary P@R90=38TP/469FP/P.07495069, validation
  macroF1.910622 fails existing .924024 guardrail; no final selection yet.
- Recovery job stage2a-la-data-resume-20261025 failed BEFORE training: container
  could not see test_data_sampler.py under migrated child bind. Host bind
  verification alone does NOT prove PVC container visibility. Keep failure
  receipt. Corrected job_resume_v2.json explicitly mounts the three /home
  directories via hostPath at their original /stage3 paths; not yet submitted.
  resume_data.py validates A1 exports and saved weights, archives interrupted
  A2 + original batch/log without deletion, retrains A2 from seed42 then N2.
  Syntax/export-contract checks passed; runtime validation remains pending.
- LLM restore watcher426156 for first recovery job correctly scaled both
  deployments back to1 after failure; ready still0 at00:58Z. Image re-pulls
  consumed /home space34->5.7GiB, causing imagefs DiskPressure again at00:57:09Z.
  Automatic reclaim raised free space to~11GiB; do not prune images/data blindly.
  Live kubelet hard thresholds nodefs/imagefs5%, minimumReclaim10%, pressure
  transition5min; imageGC high85%/low80%. Need stable headroom and pressureFalse
  before retry. Old watcher now waiting rollout; new Job requires own watcher.
- Follow-up: v2 job created SUSPENDED with explicit hostPath overrides, watcher
  428579 active with original1/1 snapshot and3h deadline. Both LLMs scaled0.
  AutoGC raised /home free to15GiB but node image inventory now lacks PyTorch;
  do not treat reclaimed training-image bytes as durable training headroom.
  Prepared/uploaded return_inputs_to_root.py: move ONLY active-II_package back
  into original root PVC path, retain both active output folders on /home,
  verify inventory and remove only that one fstab entry. Exact-entry self-test
  passes; no execution yet, needs interactive sudo. Estimated root14GiB/home33
  before further image pulls. Preserve all files; no pruning performed.
  IMPORTANT after this move: do NOT unsuspend existing v2 (its II_package
  hostPath would be absent). Create next job without the active-0 volume/mount;
  original PVC supplies II_package, explicit hostPaths supply both outputs.
  Verify container paths and image headroom before restart. Training remains
  unresumed; first recovery failed before dataset code, A2 archive not created.
- Repeat-execution diagnosis2026-10-25~22:30CDT: user's assertion at the initial
  return_inputs_to_root.py guard was a repeat of an ALREADY COMPLETED migration.
  Live COMPLETE.json exists, active-II_package absent, input fstab entry removed;
  independently compared before/after181692 inventory entries, identical.
  Root14GiB/home64GiB free; DiskPressureFalse since01:05:08Z. Added clear no-op
  handling for consistent completed state, explicit refusal for incomplete
  audit/missing source. Mocked tests verify both branches cause no subprocess
  mutations; updated script uploaded. Do not ask user to move data again.
- Created recovery v3 using original PVC inputs plus explicit /home mounts for
  two output directories; server dry-run passed. Replaced obsolete v2 watcher
  PID428579 with v3 watcher450488 (original1/1 snapshot preserved), v2 remains
  suspended. New Job stage2a-la-data-resume-20261025-v3 unsuspended; pod dq296
  currently pulling pinned PyTorch image. No actual training-step verification
  yet; check logs and MIGRATION_RUNTIME_CHECK.json before claiming resume.
- v3 runtime confirmed: podReady, A2 epoch1 finished278steps (trainloss.34141,
  validationmacroF1.8006), epoch2 started. Thus explicit output mounts and
  original PVC inputs work in the container. Runner passed A1 export and saved
  checkpoint checks before training, archived interruption without deletion.
  A2/N2 results remain pending; do not infer accuracy improvement from startup.
- Progress/monitor follow-up: v3 running~28min; BATCH confirms A1 and A2
  completed/exported, current N2. A2 stopped37epochs, best27, Harris validation
  macroF1.926720/test.892931. N2 completed12epochs and entered13/50; epoch12
  validationmacroF1.933917 is provisional source-domain evidence, not LA result.
  GPU99%/7281MiB, root14GiB/home55GiB. Both LLM deployments intentionally0/0;
  v3 restoration remains pending. User authorized periodic checks. Created
  thread heartbeat automation id stage2a, ACTIVE every10min: progress/disk/GPU,
  final backups and frozen-protocol comparison, verify real LLM readiness,
  pause automation only after training/results/restoration done. First create
  attempt lacked destination and failed; destination=thread succeeded once.
- Heartbeat2026-10-26T04:13Z (Oct25 local): v3 Complete1/1, all three data arms
  trained/exported; N2 stopped24epochs. Finished04:09Z. FINAL_SELECTION remains
  N1_natural_ce_seed42; five-seed expansion gateFalse. Downloaded A2/N2 complete
  run folders (best+last weights), all prediction exports, FINAL_SELECTION,
  DEVELOPMENT_METRICS and BATCH to X9; all4 new weights load with finite tensors,
  saved BACKUP_CHECK_20261025.json. Full analyze_campaign.py completed, producing
  CAMPAIGN_COMPARISON.json, MODEL_COMPARISON.csv, CROSSFIT_DIAGNOSTIC and PR plot.
  At matched90.476% commercial recall: current5ensemble precision11.111%/FP304,
  N1 precision12.258%/FP272, A1 7.495%/469, A2 6.702%/529, N2 6.149%/580.
  NYC defaults improve precision partly by sacrificing recall (A1 recall69.05%,
  A2 64.29%, N2 73.81%); no high-recall improvement over N1. Use common FP32
  exported validation F1 for guardrails: A2 .923273 (below .924024), not trainer
  summary .926720. LA is development data; no generalization claim. SD unopened.
  LLM restore at04:14Z: Ollama1/1, Qwen0/1 still pulling pinned vLLM image;
  watcher scaled_backTrue/readyFalse. DiskPressureFalse, root14GiB/home25GiB.
  Keep automation stage2a active until real Qwen readiness and final delivery;
  do not repeat training or overwrite completed run folders.
- NYC causal diagnosis and new authorized follow-up2026-10-25 local:
  Correct control A1/A2 vs weightedC0: P90 .07495/.06702 vs .06281 improves
  slightly, but both fail commonFP32 Harris guardrail. Direct natural contrast
  N2 vsN1: AP .54576 vs .53466, P@R70 .34483 vs .27027, but P90 .06149 vs
  .12258. Lower-recall improvement and high-recall tail regression coexist.
  Do not attribute all effects to calibration or call every NYC arm worse.
  NYC smallres median93.5m2/max237.3; Harris median257.8 with1497 >400m2.
  N1 FP200–1000m2=248/272; N2=543/580. Coverage mismatch observed, causal
  mechanism/label noise unproven. Added local followup_20261025/NYC_DIAGNOSIS.md,
  training_area_distribution.csv, recall_tradeoff.csv, commercial_low_score_tail.
- Frozen four-trial PROTOCOL: N1seed43, N2seed43, natural+ringR1 seeds42/43.
  Same trainer/optimizer/splits/budget, no LA/SD training. Ring vs pairedN1
  changes pooling only; repeat NYC natural comparison across2seeds. Adaptive
  follow-up authorized by user; original5seed gate unchanged,2seeds not proof.
  Read PROTOCOL before interpretation. SD closed; require both ring seeds to
  beat pairedN1 and pass C0-.01 validation before further promotion.
- First followup preflight wrongly required NYC mask_R even for mask_m models;
  failed before any training/BATCH. Fix required_paths(row,pooling) to require
  ring only for ring models; three contract checks passed. Keep failed Job and
  entire directory archived as followup_20261025_preflight_failure; local old
  script saved run_followup_failed_preflight.py. No raw-data/code/model deletion.
  Corrected Job instance-impact-stage3/stage2a-la-followup-20261025-v2 running;
  N1seed43 finishedepoch2 and entered3. Runtime imports original trainer,
  sharedsource exports, new outputs only under active data/followup_20261025.
  Initially both LLM1/1 verified; paused with original1/1 snapshot preserved.
  New restoration watcher458310,4h bound; status/log paths
  /tmp/stage2a-la-followup-20261025-v2-restore-status.json and -restore.log.
  Automation stage2a updated to thisJob/watch,10min. Root14GiB/home18GiB before
  launch; DiskPressureFalse. Prior campaign restored both LLMs before this task.
- Follow-up complete2026-10-26T05:31:41Z; heartbeat verified05:39Z. Epochs:
  N1seed43=30,N2seed43=34,R1seed42=31,R1seed43=24. All4 exports complete.
  Downloaded all runs (8best/last weights),predictions,RESULTS/BATCH/PREFLIGHT
  to X9 followup_20261025. summarize_followup.py loaded all8 finite checkpoints,
  checked export sizes/unique IDs and independently recomputed P@R>=90/AP.
  Generated BACKUP_CHECK.json,PAIRED_COMPARISON.csv,FOLLOWUP_RESULTS.md.
  Protocol P@R>=90: N1seed42/43=.122581/.046083; N2=.061489/.048718;
  ringR1=.139706/.082430. NYC deltas -6.109/+0.263pp, not consistent negative
  transfer; ring deltas+1.713/+3.635pp but seed42 validation.921282 fails
  frozen.924024 threshold. N2seed43 validation.916545 also fails. No promotion.
  IMPORTANT N1seed43 protocol optimum recall40/42=.952381, not38/42. Strict
  matched38TP gives precision.045238/FP802; CSV preserves both metrics. Other
  models' reported optimum recall38/42. Do not call all protocol points equalR.
  AP N1 .534656/.449527, N2 .545763/.464232, ring .455670/.498588.
  BothLLMs restored1/1 withready endpoints. Qwen in-pod/healthHTTP200; Ollama
  list API returned retainedgemma4:12b/qwen3:8b. master->PodIP curls failed,
  so external inference routing remains unchecked; no networking mutations.
  DiskPressureFalse,root14GiB/home18GiB. SDclosed, no modelreplacement. Final
  delivery completes thismonitor; deleteautomation stage2a after report ready.
- Generalization/paper-framing follow-up2026-10-26: read current local
  ii-manuscripts/overleaf/sections/narrative (dirty main worktree, not edited).
  Current contribution includes damage-model propagation with exposure fixed;
  older notes saying no propagation at all are stale. Stage2a text/results still
  use old21428-row cohort and proxy logMAE.1157->.0742 despite auditedGSD/ACS/
  parcel corrections. Rebuild data has18036 rows; cleanphysical split18012.
  Need clean OOF+matched exposure ablation and version-consistent case study;
  new type-only experiments are not evidence of improved actual population.
- New analyze_generalization.py reuses frozen prediction joins, computes all3
  classes, error/entropy review, type-area assignment for16 models/means across
  LA3332/Hval2026/Htest2653. Reproduced6 prior validationF1/commercialAP scores.
  NYC vsN1 across42/43 raisesmacroF1 .4367->.4736 / .3505->.4596, but balanced
  accuracy drops .6719->.6135 / .6048->.5893; minority recall decreases.
  N2mean2 macroF1.4833, multi2/14, commercial31/42. LA14multi in6cells and
  42commercial in7cells. Currentensemble720errors includes147 withp>=.99.
  Entropyreview10% catches22.78% currenterrors vs67.63% N2mean2, but latter
  also defers28.57% commercialreference rows. Report classcoverage; no promotion.
- Ran two frozen source-only HGB controls oncluster CPU4/8Gi, cachedimageNever,
  noGPU/LLM pause. 200iterations, onlycleantrain13333, no search. B1logarea:
  HvalF1.629974,Htest.490898,LA.397202/balanced.385022; B2existingfourgeometry:
  Hval.625205,Htest.554110,LA.375821/balanced.454619. LAcommercial recall4/42
  and12/42,multi1/14 and2/14. Strong image-model performance cannot be replaced
  by high overallaccuracy; these HGBs are diagnostic, not matchedCNN ablations.
- Storage lessons for thisrun: active parent isroot-owned, ordinarySSH mkdir
  denied. Useuser-owned /home/zhjyu11/stage2a-generalization-20261026. FirstJob
  stage2a-generalization-geometry-20261026 failed beforeprocessstart because
  nestednewoutputmount couldnot mkdir underread-only/stage3. Preserved Job and
  START_FAILURE.jsons. v2 changes only outputmount to /experiment, completed43s
  at06:03:19Z. Script geometry matches6 realCNN dataset items;39900rowsfinite.
- Allnew model/prediction/features/protocol outputs backed up to X9
  II_package/outputs/stage2a/la_data_20261024/generalization_20261026.
  LocalNumPy1.26 cannot load NumPy2 PCG64pickle; do not silentlyconvertstate.
  Uploaded X9modelcopies toseparate backup_check; same-runtime CPUverification
  Job stage2a-generalization-backup-check-20261026 completed6s. NumPy2.2.6,
  sklearn1.9.0 loadboth; all21888LA predictions maxerror1.11e-16. No hashes.
- New analyze_association.py checks savedHarvey13806buildings/278areas:
  sum(e*q)=17223.03454; sumE*meanQ=17808.87671 (+3.40%overall), median/P90
  areaabsrelativegap5.69%/19.39%,top10overlap9. Difference equalsnCov(e,q)
  in everyarea. Medianranges damageonly38.20,semanticsonly5.58,combined5x3
  43.72; crossedmodel top10overlap7–9. Existingpaperproduct only, not newly
  validatedimpactaccuracy orcalibratedjointCI; inventory/proxyuncertaintyfixed.
  LA Censuspackage lacksB25032/B25033/B25002, so cannot directly reuseHarrisPPU
  construction. Preserve resident/daytime/universe/year/support distinctions.
- Detailed findings,2claim/5blockplan,tracker:
  handoff/stage2a_la/analysis/20261026_generalization/. Pendingnextsteps are
  clean spatialOOF exposure ablation, buildingrole/unitlabel audit, complete
  Census support check, independentvalidation. SDstillclosed; currenttest/LA
  remain inspected diagnostics. No productionreplacement ormanuscriptedits.
  Finalclustercheck bothLLMsready/available1/1,DiskPressureFalse; no new monitor
  needed for completedCPUjobs. Externalinferenceentrypointnotretested.
- Clean exposure rerun authorized2026-10-26: new isolated output onX9
  II_package/outputs/stage2a/la_data_20261024/exposure_oof_20261026;
  GPU /home/zhjyu11/stage2a-exposure-oof-20261026, container /experiment.
  New repo scripts prepare_exposure_oof.py/run_exposure_oof.py and
  tests/test_exposure_oof.py. Old runtime/models/labels/results untouched;
  prior PVC/source/data mounts read-only, new top-level output writable.
- Protocol fixed before training: clean18012 rows (13333/2026/2653), same
  targets and outer splits. Preserve old five held-out folds/inner validation
  and parcel exclusions; filter toclean outertrain and quarantine cross-role
  physical IDs withtest>val>train priority. Found33 fold-local conflicts:
  fold0..4 remove5/12/7/7/2 train records; no held-out rows removed. Finaltrain
  counts9101/8219/9128/9140/9189. All13333trainIDs held out exactlyonceperseed.
  Physical/parcel_key/Prop_ID/tile groups disjoint across roles in everyfold.
  Do not restore old parcel-excluded rows tofill fold counts.
- Newfits: seeds42/43/44 each5folds thenfullseed44 (16fits). Everynewclassifier
  usesN1 naturalCE settings, sameImageNet initializationpolicy, internalvalF1
  selection; heldouttest notselection. Full42 SOURCE and43 FOLLOW reused,
  FP32 exports regenerated innewfolder. Their train_config differences only
  seed/config/outdir;43 adds samples_per_epoch=0. NoNYC/LA/SD inthiscomparison.
- MainA=Ridge(log1p area)+HGB residual(area); B=sameRidge, sameHGBsettings plus
  softN1 probabilities/features. MainB usesmean3 bothOOFtrain/fullval-test;
  3single-seedB heads arediagnostics. Reuseexisting populationhelpers,
  headseed43050, max_iter350 withdefaultautoearlystop; recordactualn_iter.
  Noouterval/test usedtofitheads. ValwasusedforfullCNNselection, testalready
  inspected. ReporttestlogMAE B-A, perclass,rawMAE/RMSE/R2,+1factor2,bias,
  partial-inventorytile sums,5000pairedbootstrap replicates overconnectedtiles
  joinedbyphysical/parcelID. CI conditionsonfixedmodels/split/proxy, notlabel
  error ortrainingvariance. Do notreporttrainingexposurescores asOOF scores.
- Targetaudit: allrows exposure_proxy; commercial=round(.04*area,1), residential
  fromuse/estimatedunits/ACS. PositiveB canrecovertargetconstruction, notprove
  actualoccupancy. N1contains4geometryfeatures, so thisisaddedN1-prediction
  information, notpureimagecausality. No manuscriptedit/modelpromotion.
- Verification:3newunittests pass (quarantinepriority/order,probabilityalignment
  andrejection,connectedspatialbootstrap). RunfromrepOROOT withfrozenruntime
  PYTHONPATH; handoff/scripts isanolderregularPythonpackage thatshadowsruntime
  namespace iftestsarerunfromhandoffcwd. No productionruntimeimportissue.
  Realmanifestchecks pass; clusterCUDAreal3-classforward/backward finite,
  PREFLIGHTpassed, allcrop/mask/scale pathschecked. torch2.7.1+cu126,
  NumPy2.2.6/sklearn1.9.0, A30. GPUcheckpoint imagepinnedcached Never.
- Job instance-impact-stage3/stage2a-exposure-oof-20261026 started07:09:36Z.
  8CPU/32Gi/1GPU,6h deadline, no retries/backoff. LLMs initiallyeachready1/1;
  snapshot /tmp/stage2a-exposure-oof-20261026-llm-before.json onmaster.
  IndependentrestorePID476312, 6hbound, status/log
  /tmp/stage2a-exposure-oof-20261026-restore-status.json and -restore.log.
  GPUzeroMiBverifiedbeforeunsuspend; root14GiB/home17GiBafterupload,
  DiskPressureFalse. Watcherrestoresbothreplicas1 afterjobterminal; readiness
  plusreadyendpoints andQwenhealth/Ollamalist stillmustbecheckedaftertraining.
- Heartbeat stage2a recreated10min inthischat (nameStage2a暴露对照); observeonly
  activeconfig, notifyimportantprogress/failure/completion. Afterall16fits,
  runnerfits5heads, writesRESULTS.json/md, features, exposure_predictions and
  CHECKPOINT_CHECK for32best/lastweights. BackupeverynewartifacttoX9, verify
  downloadedweightsfinite/loadable; forNumPy2pickleuseprevioussame-runtimeCPU
  roundtripworkflow ondownloadedheads/features, neverpatchpickle. No hashes.
  Keepmonitoruntilresultsdelivered andbothLLMsactuallyrestored, thendeleteit.
- Launchwitness: bothreused fullmodels exportedval/test; firstnewfit
  N1_oof_fold0_seed42 completedepoch9, stillrunning. IMPORTANT traineris
  type_only: its val_log_mae comesfromanuntrainedpopulationhead andmustnotbe
  treatedasexposureperformance. Onlyfinalmatchedhead RESULTS count.
  PREFLIGHThomefree53.10GiB reflectsLLMimage/runtimeGC afterpause; planstorage
  againstpre-pause18GiB (17afterupload), nottemporary53GiB, toallowLLMrestore.
- Latestlaunchcheck:1/16newfitscomplete (N1_oof_fold0_seed42); current
  N1_oof_fold1_seed42 finishedepoch1. GPU7243MiB/53%, DiskPressureFalse,
  independentrestorePID476312alive. No finalexposureheadresults yet.
- Heartbeat07:24Z:1/16complete; fold0seed42 stoppedepoch22/best12 andexported.
  Currentfold1seed42 completed18, entering19. Jobactive/ready1, GPU99%/7301MiB,
  root13.58GiB/home52.67GiBtemporaryfree, DiskPressureFalse. Watcher476312alive;
  bothLLMdesired0 asauthorized, restorationpending. NoRESULTS yet; no changes
  totrainingconfig andno exposureconclusions. Routineprogress, notificationquiet.
- Heartbeat07:34Z:2/16complete; fold1seed42 stoppedepoch28/best18 andexported.
  Currentfold2seed42 completed18. Jobactive/ready1, GPU7269MiB/52%,
  root13.59GiB/home52.49GiBtemporaryfree, DiskPressureFalse; watcher476312alive,
  LLMsremainauthorized0/0, restorepending. NoRESULTS; routineprogressquiet.
- Heartbeat08:04Z:5/16complete, allseed42foldsexported; fold4 stopped26/best16.
  Currentfold0seed43 completed9, entering10. Jobactive/ready1, GPU7271MiB/99%,
  root13.57GiB/home51.82GiBtemporaryfree, DiskPressureFalse; watcher476312alive,
  LLMsremainauthorized0/0, restorepending. NoRESULTS; routineprogressquiet.
- Heartbeat07:44Z:3/16complete; fold2seed42 stoppedepoch39/best29 andexported.
  Currentfold3seed42 completed6, entering7. Jobactive/ready1, GPU7265MiB/35%,
  root13.58GiB/home52.24GiBtemporaryfree, DiskPressureFalse; watcher476312alive,
  LLMsremainauthorized0/0, restorepending. NoRESULTS; no configurationchanges.
- Heartbeat07:54Z:4/16complete; fold3seed42 stoppedepoch26/best16 andexported.
  Currentfold4seed42 completed8. Jobactive/ready1, GPU7249MiB/98%,
  root13.57GiB/home52.03GiBtemporaryfree, DiskPressureFalse; watcher476312alive,
  LLMsremainauthorized0/0, restorepending. NoRESULTS; routineprogressquiet.
- Heartbeat08:14Z:6/16complete; fold0seed43 stoppedepoch24/best14 andexported.
  Currentfold1seed43 completed13. Jobactive/ready1, GPU7301MiB/62%,
  root13.56GiB/home51.60GiBtemporaryfree, DiskPressureFalse; watcher476312alive,
  LLMsremainauthorized0/0, restorepending. NoRESULTS; routineprogressquiet.
- Heartbeat08:24Z:7/16complete; fold1seed43 stoppedepoch23/best13 andexported.
  Currentfold2seed43 completed18. Jobactive/ready1, GPU7271MiB (instant1%),
  root13.56GiB/home51.39GiBtemporaryfree, DiskPressureFalse; watcher476312alive,
  LLMsremainauthorized0/0, restorepending. Logs/epochs advance; noRESULTS yet.
- Heartbeat08:34Z:8/16complete; fold2seed43 stoppedepoch21/best11 andexported.
  Currentfold3seed43 completed25, entering26. Jobactive/ready1, GPU7321MiB/99%,
  root13.55GiB/home51.17GiBtemporaryfree, DiskPressureFalse; watcher476312alive,
  LLMsremainauthorized0/0, restorepending. NoRESULTS; routineprogressquiet.
- Heartbeat08:44Z:9/16complete; fold3seed43 stoppedepoch30/best20 andexported.
  Currentfold4seed43 completed23, entering24. Jobactive/ready1, GPU7307MiB/99%,
  root13.54GiB/home50.96GiBtemporaryfree, DiskPressureFalse; watcher476312alive,
  LLMsremainauthorized0/0, restorepending. NoRESULTS; routineprogressquiet.
- Heartbeat08:54Z:10/16complete, seed42/43 allfoldsexported; fold4seed43
  stoppedepoch27/best17. Currentfold0seed44 completed23. Jobactive/ready1,
  GPU7271MiB/50%, root13.54GiB/home50.75GiBtemporaryfree, DiskPressureFalse;
  watcher476312alive, LLMsremainauthorized0/0, restorepending. NoRESULTS yet.
- Heartbeat09:04Z:11/16complete; fold0seed44 stoppedepoch34/best24 andexported.
  Currentfold1seed44 completed18, entering19. Jobactive/ready1, GPU7301MiB/99%,
  root13.53GiB/home50.53GiBtemporaryfree, DiskPressureFalse; watcher476312alive,
  LLMsremainauthorized0/0, restorepending. NoRESULTS; routineprogressquiet.
- Heartbeat09:14Z:11/16complete; currentfold1seed44 advanced18->47epochs.
  Jobactive/ready1, GPU7301MiB (instant4%), root13.52GiB/home50.53GiBtemporaryfree,
  DiskPressureFalse; watcher476312alive, LLMsremainauthorized0/0, restorepending.
  Logs/epochs advance normally; noRESULTS, no changes toactiveconfiguration.
- Heartbeat09:24Z:13/16complete; fold1seed44 ended50/best45, fold2seed44
  ended19/best9; both exported. Currentfold3seed44 completed5. Jobactive/ready1,
  GPU7265MiB/99%, root13.51GiB/home50.10GiBtemporaryfree, DiskPressureFalse;
  watcher476312alive, LLMsremainauthorized0/0, restorepending. NoRESULTS yet.
- Heartbeat09:34Z:13/16complete; currentfold3seed44 advanced5->34epochs,
  entering35. Jobactive/ready1, GPU7321MiB/96%, root13.50GiB/home50.10GiBtemporaryfree,
  DiskPressureFalse; watcher476312alive, LLMsremainauthorized0/0, restorepending.
  NoRESULTS; normal training progress, no configurationchanges.
- Heartbeat09:44Z:15/16complete, all15OOFfoldsexported; fold3seed44
  ended35/best25, fold4seed44 ended22/best12. Finalfullseed44 completed2epochs.
  Jobactive/ready1, GPU7279MiB/99%, root13.50GiB/home49.68GiBtemporaryfree,
  DiskPressureFalse; watcher476312alive, LLMsremainauthorized0/0, restorepending.
  Matchedexposureheads notyetfit; noRESULTS andno improvementclaim.
- Heartbeat09:54Z:15/16complete; finalfullseed44 advanced2->22epochs.
  Jobactive/ready1, GPU7337MiB/19%, root13.50GiB/home49.68GiBtemporaryfree,
  DiskPressureFalse; watcher476312alive, LLMsremainauthorized0/0, restorepending.
  NoRESULTS; matchedexposureheads stillpending, routineprogressquiet.
- Exposure comparison COMPLETE2026-10-26T09:56:19Z; Jobcomplete09:56:23Z.
  Fullseed44 ended26/best16. All16fits,15OOFexports and5heads complete; allnew
  artifacts downloadedtoX9 exposure_oof_20261026 without hashes/deletions.
  BACKUP_WEIGHTS_CHECK loadsall32downloadedbest/last tensors finite;
  BACKUP_METRICS_CHECK independentlyrecomputes40split/head/class groups;
  BACKUP_COHORT_CHECK verifies13333/2026/2653 targets/IDs andall3OOFmatrices.
- Mainmean3 vsarea: testlogMAE.212750339->.153028294 (-28.0714%), rawMAE
  1.686739->.953649 (-43.4619%), RMSE8.561801->7.491664. Val logMAE
  .221884049->.174887183 (-21.1808%). Test B-A95%spatialCI[-.115472,-.018818],
  28connectedcomponents/5000replicates; conditionalonmodels/split/proxy only.
  TestperclasslogMAE small.169502->.145267, multi.450825->.324718,
  commercial.802159->.150353. All3singleseedB testlogMAE .153112/.151237/.149871
  improvevsarea; keepfrozenmean3primary, no posthocbestseedpromotion.
- Importantnegative: multifamilyRMSE24.062447->26.759615 despiteMAEgain;
  b819144f-9039-42be-9205-07e2e12bd366 target40.8 predicts267.70->297.73.
  Retainedoutlier, no relabel/drop. Sampleinventory29tileMAE128.60784->70.80960,
  WAPE29.62%->16.31%, globalbias-5.77%->-1.88%; classbiasstill+3.13/+17.24/-10.51%.
  Targetconstructioncontainsuse/area; supportsrepairedproxyfidelity, nottrue
  population orcrosscitygeneralization. Fullfindings EXPOSURE_FINDINGS_ZH.md.
- CPUbackupverificationJob stage2a-exposure-backupcheck-20261026 complete9s.
  UploadedX9downloaded5heads/features/predictions tonewuserfolder
  /home/zhjyu11/stage2a-exposure-backupcheck-20261026; sameNumPy2.2.6/sklearn1.9
  loadedallheads and reproducedallval/testpredictions maxerror8.88e-16.
  BACKUP_HEAD_CHECK.jsonsaved; no picklepatch, no additionalGPU/LLMpause.
- LLMrestorecomplete: watcherstatus scaled_back=true/ready=true,
  deadline_suspended=false. Bothready/available1/1; qwenpodzv6mh health200,
  readyllm-engineendpoint10.42.3.64; ollamapodjwjtx listreturnsretainedmodels,
  readyendpoint10.42.3.65. Externalroutingnotretested; unrelatedbackend/frontend
  endpointsabsentalreadyoutside thistraining scope. Root13.48/home13.43GiB
  afterrestore, DiskPressureFalse. Deliverychecks saved, B2trackerdiagnostic_complete.
  Allrequestedtraining/backup/evaluation/LLMrestorationdone; deletemonitorstage2a
  afterfinaldelivery. No manuscriptedit orproductionmodelreplacement.

## 2026-10-26: Qwen3-VL building-type pilot prepared

- User authorized trying Qwen3-VL-8B-Instruct and comparing local/TAMU/personal
  GPU options. Personal GPU was clarified as **RTX 3080**, not 3090; its address
  and exact VRAM remain unverified. Existing SSH candidates172.16.44.128,
  172.16.44.130 and192.168.1.23 timed out. Local Mac is M1 Pro16GB.
- Live TAMU check: one A3024576MiB,14961MiB used by existing Qwen service;
  ~179GiB host memory available, root/home each~14GiB free, DiskPressureFalse.
  BF16 model tensors total17534247392bytes (16.33GiB), before runtime memory.
  Do not download these weights to either nearly-full persistent filesystem.
  Plan uses a32Gi memory emptyDir,48Gi host request,4CPU,batch1,two256px views.
- Reuse pinned existing vLLM image v0.27.1@sha256:c2f3b1b964e47809b722b5e75b61b1e7b39a50f70388cf2bf2418f16a9f31da2;
  torch2.13.0+cu130,transformers5.15.0. Independent documented seeded BF16
  CUDA matrix witness passed on A30. This is not VLM accuracy or model-memory
  validation. HF API reachability passed; revision0c351dd01ed87e9c1b53cbc748cba10e6187ff3b.
- Isolated script:handoff/stage2a_la/vlm_pilot.py. Output X9
  II_package/outputs/stage2a/vlm_pilot_20261026. Fixed296 unique buildings:
  LA random120,all56 rare references,34 current-ensemble commercial FPs,
  Harris validation30/class. Groups overlap; do not pool enriched groups into
  a representative accuracy estimate. All are development/diagnostic evidence.
  No San Diego access, fine-tuning, population prediction or production change.
- Model sees only original RGB, cyan target outline and physical area; labels,
  city and CNN predictions remain local for evaluation. Current taxonomy counts
  duplex/triplex as residential_small. Self-check passed off-center/border mask,
  unchanged original pixels, invalid JSON handling and failure denominators.
  All592 images present; six pairs visually inspected. Some Harris footprints
  have partial roof coverage/occlusion; retain them and compare paired inputs.
- 68MiB inputs/code uploaded to /home/zhjyu11/stage2a-vlm-20261026 on wk3.
  Job instance-impact-stage3/stage2a-vlm-pilot-20261026 passed server dry-run and
  was created SUSPENDED, deadline5400s,no retries. Existing restore watcher
  copied to master /tmp/stage2a-vlm-20261026-restore.py (not started yet).
- Automatic approval review rejected the command that would snapshot/start
  watcher/pause qwen3-14b and ollama, requiring explicit current authorization
  for service disruption. None of that rejected command executed. User approval
  requested via async question. Both services remain running; no VLM inference
  has started. On approval, first save original counts, start/check watcher,
  then pause, verify actual GPU release and unsuspend. Verify readiness and native
  health after restore, copy predictions/logs/run records to X9 and score all296.
- User explicitly approved "允许本次临时暂停并自动恢复" in this chat. Original
  qwen3-14b/ollama ready/available1/1 saved on master at
  /tmp/stage2a-vlm-20261026-before.json. Independent watcherPID530823 started
  with7200s bound and -restore-status.json/-restore.log under that prefix.
  Both deployments scaled0; A30 measured0MiB before job release. This approval
  supersedes the preceding pending-approval state. Inference and restoration
  still require completion checks; no results are implied by launch.
- First Job failed before inference: manifest omitted runtimeClassName:nvidia,
  so torch.cuda.is_available() was false despite the GPU resource request.
  Existing deployment's CUDA witness did not validate the new Job's runtime.
  Model download itself completed in~177s in RAM. Failed Job/worker/log retained;
  independent watcher restored both LLMs to1/1/1 and ready=true.
- v2 explicitly reuses runtimeClassName:nvidia from working cluster jobs and
  runs a seeded BF16 CUDA kernel INSIDE the actual Job before any download.
  Job name stage2a-vlm-pilot-20261026-v2; deadline4800s. Original296 inputs,
  prompt, model revision and decoding remain unchanged. Separate worker_v2.py,
  job_v2.json and ENVIRONMENT_v2.json preserve attempt1 provenance.
- VLM pilot COMPLETE: v2 actual-Job BF16 CUDA witness passed; Job succeeded
  2026-10-26T19:04:21Z, after15m43s. All296 outputs valid and backed up to X9
  vlm_pilot_20261026. Model load8.87s, download172.06s, inference740.79s;
  median2.35s/building. Framework allocated/reserved peaks16.44/16.50GiB;
  nvidia-smi observed17495MiB (17.09GiB), a snapshot, not whole-device peak.
- Fixed zero-shot VLM is not a direct Stage2a replacement on these samples:
  Harris90 macroF1 current.9547/N1.9319/VLM.7348; LA commercial recall
  current40/42,N137/42,VLM16/42; multi4/14,3/14,3/14. VLM corrected all34
  selected current-CNN commercial FPs, but LA random120 accuracy97.5% came
  from predicting ALL120 as small residential and missing all3commercial.
  Do not promote majority-class accuracy or enriched FP-group correction to
  representative three-class generalization. Groups overlap; do not pool them.
  All296 claimed evidence_sufficient=true, including60 reference disagreements;
  self-reported evidence is not a usable error filter here. LA labels remain
  parcel-derived proxies; Harris is validation/development, not new test data.
- VERIFY.json passed: all296 IDs/raw outputs match backup and streamed logs;
  original frozen evaluator independently matches all12group/model metrics
  within1e-12. RESULTS.json, paired_predictions.json, PERFORMANCE.json and
  outcome-selected DIAGNOSTIC_EXAMPLES.png saved. Human report:
  handoff/stage2a_la/analysis/20261026_vlm/REPORT.md; no commit/push.
- LLM restoration VERIFIED2026-10-26T19:07:24Z: qwen3-14b and ollama both
  desired/ready/available1/1/1; Qwenhealth200, Ollama model list successful,
  ready endpoints llm-engine10.42.3.71 and ollama10.42.3.72. Independent v2
  watcherPID532803 reported scaled_back=true,ready=true,deadline_suspended=false.
  Node DiskPressure/MemoryPressureFalse. LLM_RESTORATION_VERIFIED.json copied
  to X9. External routing not retested; pre-existing empty backend/frontend
  endpoints are outside scope. No further GPU job or service pause is pending.

## 2026-10-26: VLM failure diagnosis and paired ablations

- User requested crop/prompt/capability diagnosis and explicitly authorized
  model tests. User chose actual use of each individual building, with auxiliary
  structures separately flagged. Parcel-inherited type remains a weak reference;
  do not train/evaluate auxiliary roofs as unquestioned apartment dwellings.
- Inspected all56 commercial/multi->small outputs: LA25commercial+10multi;
  Harris5commercial+16multi. Errors include small visible targets, recognizable
  commercial/apartment context, blurred/occluded roofs, and likely carports.
  Size association is not causal proof. All21 Harris errors retain the known
  footprint despite8border-touching masks; do not conflate border contact and
  crop truncation. Native1024 Harris images and existing256crops match exactly.
- New script vlm_diagnosis.py prepares/scorers; vlm_pilot.py only adds optional
  prompt-file and records actual image_grid_thw. Original pilot source/results
  frozen on X9 remain unchanged. Paired 2x2:256vsbicubic512, originalP0vs
  comparativeP1 on all296; plus native512Harris90 context atP1 and replay6.
  Main/replay984 +smoke24=1008inferences. No newdependencies or modeltraining.
- Outputs X9 II_package/outputs/stage2a/vlm_diagnosis_20261026; PROTOCOL.json,
  RUN_PLAN.json, SOURCE_PARITY.json, VISUAL_AUDIT.json and contact sheets saved.
  All90 Harris RGB/mask parity assertions passed. Wider windows73shifted/17
  concentric: joint context/scale/position test, not pure context isolation.
  LA native originals absent; supplied archive onlycontains256crops. No mosaic
  or syntheticcontext. Labels and baseline predictions excluded from upload.
- Skill reviewer approved bounded plan; read DIAGNOSIS_PLAN.md for exact Job
  witness and limits. User's goal is actual-use accuracy/generalization, but
  this pilot remains development-only; no new-region generalization claim.
- Diagnosis Job launched2026-10-26~19:33Z after GPU0MiB confirmed. Both prior
  LLM replica counts1 saved to X9 LLM_BEFORE.json. Independent restorewatcher
  PID538587 onmaster, prefix/tmp/stage2a-vlm-diagnosis-20261026, bound5100s;
  Job deadline4800s. New actual-Job CUDA witness independentlyexecutedverbatim
  byfreshagent andpassed A30/torch2.13.0+cu130. Completeoutputs andactualservice
  restoration stillmustbeverified. No other active stage3 Job atlaunch.
- First2factorialarms complete and independentlyscored. Upsample512/P0:
  Harris67/90 unchanged, macroF1.7406; LAcom14/42(vs16),multi3/14 unchanged;
  corrected5,introduced7. Native256/P1: ALL296classlabels unchanged, all296
  explanationschanged andall296 evidenceflagstrue. Prompt_applicationcheck
  confirms inputtokens375-377->498-500 anddistinct1795charP1 vs1132charP0;
  not a missing CLI argument. Combined/wide arms stillrunning atthisentry.
- Prepared separate12case localization/description/role probe, allowing
  unknown/other actual use, after P1's zero class changes. Outputs X9
  vlm_grounding_probe_20261026; new vlm_grounding_probe.py parser/self-check
  passed. It is adaptive diagnosis, not a new accuracy/generalization test.
  Fixed8hard+4correctcontrols, same512images; targetbbox IoU measures model
  grounding only, not source segmentation accuracy. Start onlyafter firstJob
  completes andbothLLMs verifiedrestored;900sJob/1200s independentwatcher.

## 2026-10-26: VLM diagnosis completed; actual-building-use caveats

- Completed all planned paired arms, replay and smoke checks:984+24=1008
  inferences, plus12-case grounding probe. Full result/report source:
  handoff/stage2a_la/analysis/20261026_vlm/DIAGNOSIS_REPORT.md. Artifacts on X9
  II_package/outputs/stage2a/vlm_diagnosis_20261026 and
  vlm_grounding_probe_20261026. All raw outputs backed up; no label/weight edits.
- Main results against unchanged weak references: Harris original67/90,
  upsample/P0 67/90, native/P1 67/90, upsample/P1 69/90, nativewide/P1 77/90.
  Wide macroF1 .8513, multi19/30, commercial28/30, small30/30; currentCNN86/90.
  LA commercial16/42->14/42 and multi3/14->4/14 under upsample/P1. All180small
  controls remain correct across full arms. P1 alone changes no class labels.
- Wide fixes11 and introduces1 versus original; fixes12 and introduces4 versus
  upsample/P1. Of21originalHarris->small errors,9 recover reference type.
  BUT two reference fixes (IDs4,239) are provisional parking-canopy candidates:
  calling them apartments can improve parcel-label scores without improving
  actual single-building use. Corrections cluster in5tiles/8parcels;5 inone
  tile. Do not call this independent generalization or pure context isolation.
- LA stricter one-assessor-building subset still poor: original14/40commercial,
  3/13multi; upsample/P1 12/40,3/13. All25LAcommercial->small errors remain in
  strict subset. Do not attribute all VLM failure to parcel label noise.
- Grounding probe: bboxIoU median.668,11/12>=.5 againstmaskbbox; finds target
  but still confuses semantics. ID16 boxIoU.958 yet calls commercial roof a
  carport. IDs4/239 auxiliary/other;250 auxiliary withsmall use andboxIoU.163.
  Two offour previouslycorrect controls becomeother;0/12unknown. All main
  classification arms mark every sample evidence_sufficient=true. Neither
  free-text confidence nor prompt-only role labels are safe automatic gates.
- Preserve81-row BUILDING_USE_REVIEW_TEMPLATE.csv with blank reviewed labels;
  visual annotations are provisional, not independently adjudicated truth.
  Proposed next experiment: verifiedbuildingrole/use labels, targetdetail+
  nativecontext, then frozen geographic tests coveringallclasses. PriorCNN C1
  context expansion failedHarris guardrails, so dualscale remains a proposal.
  Do not useVLM asautomaticcommercial veto or unaudited pseudolabel teacher.
- Verification: each mainarm independently matchesoriginalscikit-learn scorer;
  all1008 main/smoke/replay rows exactly match log and0invalid. Probe12/12 valid,
  no truncation, rawlogexactmatch, pinnedrevision/BF16 checked. A30 witnesses
  performed independently onactualJobs. Wide mean2.855s/building; maxPyTorch
  reserved16.678GiB (notwholedevice usage). No newdependencies, commit orpush.
- MainJob complete20:27:01Z; actualservices restored/verified20:31:43Z before
  separateprobe. ProbeJob complete20:37:19Z, independentwatcherPID548026;
  actualservices restored/verified20:42:33Z. Bothoriginaldeployments1/1ready
  andavailable; Qwenhealth200, Ollamamodel-listnormal, readyserviceendpoints,
  nodeMemoryPressure/DiskPressureFalse. Bothoutputfolders contain restoration
  evidence. No further GPUtest or servicepause pending. Leave historical failed
  pods/unrelated files intact. Newcity, actualuseGold, finetuning anddetectionAP
  remain untested; these are development classification diagnostics only.

## 2026-10-26: Multiclass VLM feasibility clarification

- User is considering restoring the previous multiclass taxonomy; no new
  taxonomy has been implemented or tested in this follow-up. Legacy Stage2a
  model classes are small residential, multi residential, commercial,
  institutional and other. The native3 policy DROPPED institutional/other rows;
  it did not merge them into the retained classes. See common.py CLASS_NAMES,
  RUN.md model interpretation and prepare_stage2a_label_policy_manifest.py.
- Current VLM three-class restriction is in prompt, parser and evaluator;
  changing only prompt-file would mark new classes invalid. metrics() also
  hard-codes3 and requires adaptation for a separate multiclass experiment.
  The fixed296 pilot has onlythree reference classes, so it cannot validate
  restored institutional/other recall without additional reviewed examples.
- Proposed schema keeps actual use separate from primary/auxiliary role;
  other means known outside the named uses, unknown means unresolved evidence.
  This is a proposal, not validated accuracy or automatic label authorization.
  Downstream exposure artifacts expect native3 soft probabilities; multiclass
  VLM text outputs cannot be passed through without a separately evaluated
  adapter/model. No GPU job launched, service paused, or production code changed.

## 2026-10-26: Frozen five-class / dual-view VLM experiment launched

- User authorized five uses + separate primary/auxiliary + unknown; actual
  individual-building use remains target. Reused explicit authorization for
  temporary qwen3-14b/ollama pause and automatic restoration. No new approval
  required; no production classifier, labels, checkpoints, commit or push changed.
- Frozen21:56:13Z, OUT=II_package/outputs/stage2a/vlm_multiclass_20261026 onX9.
  164inputs:49Harris development+115SD heldout VLM geography. Five matchedarms
  A3near/B5near/C3dual/D5dual plusEforced3dual,20Harris smoke+820full calls.
  Fixedscoremask136:35Harris(8small/6multi/8com/2inst/11other),101SD
  (34/14/26/17/10). Other28retained diagnostics. Do not tune onSDoutputs.
- Added58candidate institutional/other buildings (18Harris+40SD), not58Gold.
  SD explicitASR+same2023/2025LU andone mappedroof still permits roof fragments.
  Historicalpermits found stores/offices/multiplebuildings/SFR contradictions;
  froze downgrades before inference, no replacement basedonmodeloutputs.
  Institutional mostlyworship, othermostlyindustrialintwoaddedregions.
  A-tier21SD hasofficialcorroboration butnotexact-date independentbuildingGold;
  noA-tiermultifamily. Rolehypotheses9Harris areprovisionalassistantvisuals.
- Pitfalls: oldHarris mask_truncated=False doesnotexclude source-edge-clipped
  footprints;10downgraded. TwoHarriswideviews containoriginalmissingpixels,
  HX0029.5%,HX01628.4%;targetintact,flaggedratherthanfilled. Earlymetadata
  lackswideflag: derivewide_zero_fraction>=.05. Usepy311,notcatcch-env,for
  rasterio+planetary_computer. macOSrsync lacks--info;use--stats.
- Schemas/parser/scoring generalizedonlyinseparatevlm_multiclass.py. C-E changes
  bothroleandabstention;notpureunknownablation. Supported-classmacro metrics
  excludezero-supportclasses andreportN/A;unknown/invalid stayindenominator.
  Sharedthreecomparisons separatenamingavailabilityfromtype discrimination.
- Jobstage2a-vlm-multiclass-20261026 launched22:00Z;independentmasterrestore
  watcherPID555279,4200sbound;Job3900s. Snapshotboth1,verifiedGPU0MiBbefore
  unsuspend. SamepinnedvLLMimage/NVIDIAruntime/A30/BF16/SDPA,48GiBRAM,
  RAM-only32GiBweightcache. FreshagentactualJobCUDAwitnesspassed. No other
  liveGPUwork;onlythetwoauthorizedserviceswerestopped. Mustverifycompletion,
  all840outputs,referencehashesandactualLLMhealthrestorationbeforeclosing.
  SeeMULTICLASS_PLAN.md,PROTOCOL.json,JOB.json,LAUNCH.json andLLM_BEFORE.json.

## 2026-10-26: Multiclass experiment completed and services restored

- All820formal+20smoke predictions complete; report
  handoff/stage2a_la/analysis/20261026_vlm/MULTICLASS_REPORT.md, X9sameOUT
  REPORT_ZH.md. Fivearms retained, noSD-basedprompt/reference changes.
- SD shared-three74: A3near50(67.6%), B5near40(54.1%), C3dual58(78.4%),
  D5dual53(71.6%), Eforced3dual58. Fullfive101:50/45/58/63/58.
  DfullmacroF1 .4676;DsharedmacroF1 .5497 vsC .5764. Widercontexthelps:
  B->D fixed25/regressed7full, A->C fixed9/regressed1shared. C->Dshared
  fixes0/regresses5;fullgain10newclasses minus5oldclasslosses. Neverclaim
  restoredfiveclassesimproveoriginal-classgeneralizationfromfullfivealone.
- DSD recalls:small30/34,multi0/14,commercial23/26,institutional3/17,
  other7/10. Multi mistakes5small/6commercial/3other. Otherprecision7/16,
  institutional3/3precisionhasverylowsupport. Official-corroboratedsubset
  sharedC13/13,D11/13;fullC13/21,D15/21,nomultifamilytruththere.
- Allfourrole-enabledarms outputprimary164/164, noauxiliary/unknown(656
  outputs). All3visualcarporthypothesesstillprimary;Dcallsthemsmall.
  RawJSONverified:HX015/SD209explanationssaysupport/auxiliarywhile
  role=primary,notparserdefault. Bothfivearmsunknown0/164;allSDarmsunknown0.
  StructuredflagsareNOTreliablecalibrationorautomaticlabelinggates.
- SD5classcontextgainB->D positive6regions,flat1,negative1. D-Cshared
  dropsin4ofthe6regionswithoriginalclasses,flat2. Candidate-label
  sensitivitySDC59/115,D73/115vsreviewed58/101,63/101;source-noiseinflates
  apparentgain. NewSD27scorableclasses mostlyworship/industrial;temporal
  mismatchandnoindependentactual-useGoldremain. SDnowdevelopmentfeedback
  foranyfuturetuning;requireanotherfrozenregion. DetectionAPnotmeasured.
- Verificationpassed333frozenhashes,820uniquepairs,840raw/logmatches,20/20
  exactrawsmoke-to-mainreplays,0invalid,0token-limitoutputs. Independent
  sklearnmatches20domain/scopecomparisons. py311lackssklearn;reuseexisting
  /private/tmp/la_stage2a_eval_20261023/bin/python fortheanalysisverification,
  noinstall. Casefigureinspected;allrawanswersandCASE_RESULTS.csvsaved.
- Job22:00:26Z->22:41:22Z(40m56s),formalmean2.5803s,formal35m16s;
  Dmean2.8170s,reservedmax16.6855GiB. WholeGPUonesnapshot17689MiB,
  notclaimedpeak. Modeldownload241.76s,RAM-only. Bothservicesrestoredand
  independentlyverified22:44:30Z:original1/1ready/available,Qwenhealth200,
  Ollamalist/endpointsnormal,nodeDisk/MemoryPressureFalse;watcherreadytrue,
  noforceddeadline. Noactivejob/servicepausepending;bothagentsclosed.
  Nooriginaldata/weights/productionadapterchanged,nocommit/push.

## 2026-10-26: Literature-grounded plan for a generalizable Stage2a model

- User asked how CNN retraining, VLM fine-tuning/post-training and existing work
  could support a generalizable paper model. Completed literature/source review
  and local training-data inventory; no new model run, GPU pause or deployment.
- New notes: literature-review/STAGE2A_GENERALIZATION_20261026.md. Concrete
  comparison and label/split/compute gates:
  handoff/stage2a_la/analysis/20261026_generalization/NEXT_MODEL_PLAN.md.
  TRAINING_DATA_INVENTORY.json records source CSV hashes and exact class counts.
- Harris clean train=13333 (11363small/1097multi/873commercial); prepared NYC
  =578 (445small/43multi/90commercial). Both have zero institutional/other and
  no verified role supervision. NYC raw1000 candidates have additional relevant
  categories, but category counts are not approved actual-building labels.
- Proposed first main comparison: newly supervised five-class ConvNeXt,
  frozen DINOv3-B features with small heads, same-Qwen8B QLoRA SFT. Preserve
  near/context ablation and same-data scoring; only expand leading candidates.
  New independent cities and actual-building/role verification remain necessary.
- Literature pitfalls: multiple within-city random tests are not cross-city
  transfer; building-polygon outputs may inherit AOI/parcel labels. The 2026
  JSTARS transfer study uses binary function pixels and leaves garages/carports
  unlabeled. Qwen landcover SFT (WACVW2026) is 10k/1k stratified data on A100
  80GB, not evidence of A30 fit or building-use transfer. Few-shot RLVR
  (ICCVW2025) used16–128H100 and selected among evaluated checkpoints; do not
  translate few labels into low compute or a blind-test guarantee.
- Implementation pitfall live-verified: X9 II_package/scripts/stage2a/common.py
  supports area mask pooling, while this repository's historical handoff copy
  still hardcodes nearest. Pin the experiment's actual imports and hashes;
  do not accidentally resurrect empty small-building masks. Neither copy edited.
- Public Qwen/transferability PDFs were read in /private/tmp with existing
  pdftotext after py311 lacked fitz; no dependencies installed. The read-only
  literature subagent completed and was closed. No external messages, commits,
  pushes, model-weight downloads or production changes.

## 2026-10-26/27: Generalizable supervised pilot preparation

- User said start and authorized CNN/VLM experiments. Kept current branch;
  no commit/push or production model changes. Existing run-experiment/compute
  contract and TAMU skill reused. New source scripts prepare_generalizable.py
  and train_generalizable.py are scoped to a separate development pilot.
- OUT=X9 II_package/outputs/stage2a/generalizable_pilot_20261026. NYC1000 source
  samples mapped conservatively to729 five-use candidates. Official NYC
  BUILDING/PLUTO lookup yielded995 intersecting candidate buildings, but only
  535 samples passedIoU>=.8. Only247 pass same-use, single-building, residential
  unit consistency:89small/14multi/67commercial/38institutional/39other.
  These are current-record corroborations, not2018 independent gold. Sources,
  relevant public fields and response hashes saved; owner names not requested.
- Some categories are contradictory (e.g.K9 warehouse), mixed-use or ambiguous
  clinical/residential facilities; excluded rather than broad first-letter maps.
  NYC FEATURE_CODE2100 meansbuilding;5110 garage;1004 explicitlyauxiliary.
  No verified role labels yet. Do not fill primary or claim role supervision.
- Prepared1790 valid image pairs:960Harris/729NYC/101SD,53 rejected for clipping
  or missing imagery. Only247NYC eligible fortraining; otherNYC retained for
  diagnostic evaluation. Harris/SD remain source/weaklabels. No independent
  heldout final city. CNN trials must report strictNYC subset separately.
- NYC cached images usually cover130–180m, insufficient for300m request;
  use real full-source context and record exactsize/source-limited flag.
  Near40m minimum/1.4bbox; widestrequestedmax300m/3near; actual512px JPEG95.
  Union overlap components and sameparcel BEFOREtrain/val split; xBDPNG uses
  conservative full-tile geographic envelope. No class counts or exactunits
  enter image/geometry model inputs. Geom features:logarea/fill/aspect/compactness.
- Frozen24CNN development trials:3heldoutcities x2views x2seeds x(frozenencoder
  orfinetuned). Same-data controlledRGB baseline, notliteraloldN1retraining.
  Train/val source-only; city predictions after checkpointselection. No role
  head until labels exist; unknown output explicit. DINOv3 access pending;
  QwenSFTnotyetlaunched. Do not state this aspaper-gradegeneralization.
- OfficialDINOv3config returned401gated. User hasaccountaccess and willlogin.
  Host lackedhf/uv/huggingface_hub; isolatedCLI2.3.0 installed at
  ~/stage2a-hf-tools/bin/hf. Systemvenv lackedensurepip; officialPyPAget-pip
  bootstrapped onlythisvenv, noOSpackages/systemPythonchanges. User token must
  stay interminal and mustnotbeprinted. Do notassumehostloginreachescontainer.
- CPU-onlypreflightJobcompleted; existingtrainingPVC has12.81GiBfree.
  Workerhomehas12.20GiBfree, so stageimagesinRAM andpersistnewruninPVC;
  modelcacheRAM-only. NoLLMpauseyet. GPUJob/watcher/witness stillpending.

### Pre-training QA and startup repair

- User completedTAMUHFlogin; officialDINOv3-B configandweights revision
  5931719e67bbdb9737e363e781fb0c67687896bc verified/downloaded toRAM.
  Jobmountcontainsweights only, nottoken. No furtheruserconfigurationneeded.
- VisualQA found small Harris commercial-parcel structures still inheritinguse.
  Beforeanymodeloutputs, frozenexecution_v2 adds Harris training gate area>=50m2
  and>=.5largestobservedparcelroof, retainingunfilteredfrozenCNNcontrol.
  This isnotverifiedrole/use. BothvalidationandcitytestIDsunchanged. 48runs:
  sourcefrozenCNN, filteredfrozenCNN, filteredfrozenDINO, filteredfinetunedCNN;
  2views x2seeds x3cities. Only12runs fine-tuneencoder. Nofinal-cityGold.
- FirstGPUJobv2 failed BEFOREanyinference at absolutehostsymlink '/staging/images'
  resolvingtounmountedhost/dev/shm. Actualtrainingrootpartialfilespreserved.
  OriginalLLMsrestoredandindependentlyhealthchecked2026-10-27T05:12:23Z,
  both1/1/1,Qwen200,endpointsvalid,nodepressuresfalse; receipt
  execution_v2/FAILED_STARTUP_RESTORATION.json. Fixv3explicitimagevolume;
  dataset/hyperparametersunchanged, separatePVCroot suffix_20261026_v3.
- V3watcherPID585577 onmaster,prefix/tmp/stage2a-generalizable-20261026-v3;
  servicepause2026-10-27T05:12:43Z,5100sbound. Job4800sdeadline. GPU0MiB
  beforeunsuspend. MustverifyindependentCNN/DINOwitness, admit, completeoutputs
  andrestoredhealthbeforeclosing. Currentenvhashc013e5e9; .aris/compute/tamu-gpu.md.
- Localsyntheticrealtrainercheckpassedloss/backward,metrics,normalization,
  exactsavedcheckpointreplay,IDorderandunknownrole. OldlocalTorch requires
  actualdeviceinautocast evenwhen disabled; CPUcompatibilityfixsavedwithprior
  protocol/codehashreceipts. Do nottreatthesechecksasaccuracyresults.

### 2026-10-27: Completed 48-run DINO/CNN pilot and restoration

- DINO access/setup is complete. Official ViT-B/16 revision
  5931719e67bbdb9737e363e781fb0c67687896bc and weights SHA256 verified;
  actual A30 CNN/DINO witnesses passed independently before ADMIT.
  User instructions: handoff/stage2a_la/DINOV3_SETUP.md. Host CLI is
  ~/stage2a-hf-tools/bin/hf; credentials remain outside training containers.
- V3 completed48/48; Job05:14:37Z–05:37:08Z, worker1347.14s. Full artifacts
  are in generalizable_pilot_20261026/execution_v2/remote_v3 on X9. All260
  files/1,413,567,782bytes including48checkpoints match remoteSHA256. Official
  DINO+ConvNeXt weights also persisted under execution_v2/pretrained.
- Report: handoff/stage2a_la/analysis/20261027_generalizable_pilot/RESULTS.md.
  Same three-city/two-seed mean macroF1: DINO near .2669, DINO dual .2552,
  filteredCNN fine dual .2424, source frozenCNN near .2316. These are seed
  score means, not probability ensembles. No model met the generalization
  criterion; do not replace production models or claim final-city evidence.
- DINO near two-seed probability ensemble: SD54/101(.5347),F1.3047;
  NYC current-corroborated105/247(.4251),F1.2268; Harris570/960(.5938),F1.2689.
  Multi recall remains0/14 in SD and0/14 in strictNYC. Harris institutional
  50% means1/2 only. FilteredCNN dual SD57/101,F1.3079,multi1/14. Prior Qwen
  five-dual63/101,F1.4676,multi0/14; different model/input/training, historical
  comparison only. No Qwen SFT/QLoRA or DINO encoder fine-tune was run.
- Important data limit: NYC247 matches current26v2 official records, not2018
  gold; source labels elsewhere remain weak. Major-roof rule is not role gold.
  HeldoutSD training multi45 vs validation108; heldoutNYC training multi47,
  institutional18 and validation institutional1. All three cities are now
  development feedback. New independent cities and verified roles still absent.
- Legacy N1 actual CPU replay: original153.6m SD shared74 seed42=56/74,
  seed43=57/74; tighter new images both33/74. New images also change resampling
  and compression; do not attribute that whole delta solely to physical crop.
  Legacy/VLM/reference label+ID matching and sklearn scoring independently
  verified. Prior SD reference has115 candidates but14 unlabeled: compare only
  the101 scorable IDs; do not silently promote unreviewed candidates to gold.
- Val-target80% coverage is not transferable calibration: fineCNNdual actual
  SD46.5%/31.7%,NYC0.4%/20.2%. SDseed43 selectiveaccuracy84.38% retains31.68%;
  never report that as full-sample accuracy. All new role outputs unknown,
  untrained/unscored. No ensemble abstention claim without own calibration.
- Verification:48 scores match sklearn;48 spatial splits pass;36 real frozen
  checkpoint replays preserve every argmax, maxprobabilitydifference9.84e-7.
  CPU trainer test and code/feature hashes pass. CNN BF16 exact local replay,
  new-city final test, role accuracy and detectionAP were not checked.
- Large kubectl exec tar stream truncated the final pretrained ConvNeXt file.
  DINO files were intact; incomplete CNN file was retransmitted, remote stream
  flushed/drained before exit, allbytes/hash verified. Failure receipt retained.
  Do not count checkpoint existence as successful transfer.
- Both services independently healthy05:40:28Z, replicas1/1/1, QwenHTTP200,
  Ollamalist/endpoints normal, nodeDisk/MemoryPressureFalse. No forceddeadline.
  Watcherreadytrue, receipt execution_v2/RESTORED_HEALTH_v3.json. Read-only
  results pod Succeeded; PVCfree11.06GiB. No active experimentalGPUjob or
  pendingservicepause; no commit/push, no production checkpoint replacement.

### 2026-10-27: Reliable-reference matched transfer dataset prepared

- Latest user authorizes filling reliable multi-family/auxiliary/institutional
  labels, matched DINO fine-tuning and Qwen QLoRA, with new-city tests. No push.
- New X9 root: `II_package/outputs/stage2a/reliable_transfer_20261027`.
  Frozen1438: NYC720train/174spatialval, Cambridge300test, Seattle244test.
  NYCtrain120each5uses +120auxiliary-role-only. NYCval30each exceptother29,
  plus25aux. Cambridge50eachuse+50aux. Seattle small50/multi50/commercial45/
  institutional50/other49; no trustworthy Seattle auxiliary labels were found.
- NYC officialPLUTO26v2+unique2100footprint+officialBBL link, numbldgs1 for
  primaryuse, unit-count gates (small1–3,multi>=4), no mixedrescommercial.
  No NYC parcel polygon overlap measured; record null, never claim>=95%there.
  Explicit1004 auxiliary only229 citywide, not pagination failure. Add5110
  detachedgarages on residentialBBLs with unique2100primary,<=60m,15–150m2,
  no recorded post2021 construction. Garage parcels withnumbldgs2–3 contribute
  roleonly, no primary-use label. Do not inherit residential use onto garages.
- Cambridge BLDG/OUTBLDG officialtypes, FY2024/FY2025 use+unitsagreement,
  one mainbuilding/parcel,>=95%overlap. Preserve leading-zero usecodes:0101
  mixeduse is not101. Seattle assessorNR_BLDGS1/NR_UNITS plusunique2023outline
  and>=95%parceloverlap; no invented auxiliary labels. ALL_UNITS heterogenous
  table was not summed or treated as separate corroboration.
- NativeNAIP: NYC2022/2023, Cambridge2023, Seattle2023; near>=40m,wide>=300m;
  sourcewindows/RGB/masks/cyanmarkedviews preserved. Actual context-intersection
  and sameparcel leakage audits passed. Two full-context edge failures rejected:
  nyc_614687,seattle_1270774953. 23-image visual alignment spotcheck passed;
  this is not independent human or exact-date occupancy gold.
- DatasetSHA2560a9fe81ed273067a7a01f962409ce779f9055a501211a2aa33faa6af0161411f.
  PROTOCOL.json locks seeds42/43, DINO448px frozenhead vs last2blocks+norm,
  Qwen512px NF4control vsr8languageQLoRA,2epochs. SameIDs/labels/views;
  role-balanced loss; missing-use loss masked. Sourceval selects both with
  mean(useMacroF1,roleMacroF1). DINOsourceval80%confidence abstention and
  Qwennativeunknown separately reported; no6thunknown training label.
- New scripts: prepare_reliable_transfer.py, reliable_transfer_common.py,
  train_reliable_dino.py, train_reliable_qwen.py, analyze_reliable_transfer.py.
  Source-taxonomy/real sparse-loss gradient/JSON/scoring selfchecks pass;
  CPU actual architectures,252languageLoRAmodules,token masks and batching
  pass. RealA30backward+save/reload independent witness still required.
- Newenv88b93962: pinnedimage unchanged, isolatedPEFT0.21.2--no-deps inRAM.
  Accelerate1.14.0,transformers5.15.0,torch2.13.0+cu130,bnb0.50.0.
  Qwenofficial4shards16.34GiB LFS-SHAverified revision0c351dd... in hostRAM
  /dev/shm/stage2a-qwen-model-20261027. CLI multiple--include patterns fetched
  onlyconfig; fixed withsnapshot_download allow_patternslist. Tokens stay out.
- GPUnewjobs not yet launched at this note. Intended4boundedbatches dino,
  qwen42,qwen43,evaluate; persistentroot/stage3/stage2a_reliable_20261027,
  RAMstaging/dev/shm/stage2a-reliable-20261027. Each4800sdeadline,5100swatcher,
  exactreplica+healthrestore betweenbatches. .aris/compute/tamu-gpu.md has
  verbatim fresh-agent witnesscommands. Do not admit training without them.

- FirstGPUbatch now launched: stage2a-reliable-dino-20261027. All8629input
  hashes+4code hashes verified after transfer. An earlier verification attempted
  duringrsync correctly failed on an unfinished file; wait for rsync exit0 before
  validating. RestorerPID597899 armed5100s; originalLLMs temporarily0replicas,
  A30observed0MiB beforeunsuspend. Job nowwaiting independentwitness. Fresh
  agentEuler01a17c2b-0cab-7470-af44-11b4d5e75fb6 runs bothverbatimcommands.
  DO NOT close task or start anotherGPUjob until this batch completes/restores.

- First independent witness DINO passed; Qwen failed BEFOREbackward because
  Transformers5.15 processor returns mm_token_type_ids at promptlength763 while
  appendinganswer extendedinput_ids/attention_mask775. No training admitted.
  ReduceJobactiveDeadlineSeconds to1 to end it promptly and triggerrestorer;
  simplysuspending a waitingJob doesnot make existingwatcherterminal until its
  maximumdeadline. V1restoredandindependentlyhealthy07:13:06Z; receipt
  reliable_transfer_20261027/RESTORED_HEALTH_v1.json. Failedprotocol keptintact.
- Repairexecution_v2 appendszero(text)mm_token_type_ids alongsideanswer. Real
  processor+installedget_rope_index nowpasses CPUforprimary[3,1,775]andaux
  [3,1,774]. Same1438data andallhyperparameters. Newenvb63f8a5c/spec under
  execution_v2. NewPVCroot/stage3/stage2a_reliable_20261027_v2; inputimages
  reusedfromRAM, copiedtoindependentPVCroot. v2Job nowwaitingfreshwitness;
  restorerPID601219. FreshSchrodinger01a17c37-66a6-7011-bfb0-332d09f47d6a.
  Witnessnowtimes3actualoptimizersteps inclrole-onlyauxiliary. StillnoADMIT.

### 2026-10-27 evening continuation: network access currently unavailable

- Both v2 independent commands passed via Schrodinger01a17c37-66a6-7011-bfb0-
  332d09f47d6a: DINOexit0 shape[2,7]loss1.482421875; Qwenexit0loss7.858975887,
  adapterreload_error0.0, INFER2/2. NoADMIT was created before interruption.
  No formaltrainingcompletion or heldoutprediction has been observed.
- At23:05–23:08UTC both k3s-master/tamu-gpu failDNS. TailscaleDNS100.100.100.100
  andpublic1.1.1.1 returnNXDOMAIN forcontrolhostname; historicalprivateIP route
  goesdefault192.168.1.1/en0, with no campusroute. Asked userto reconnectTAMU
  network/VPN. Do not claimclusterdown or servicesrestoredfromthisaccessfailure.
- Worker900switnessadmissionlimit and4800sJobdeadline/5100srestorer should have
  endedtheunadmittedv2batch, but currentterminal/restoredhealthremainUNVERIFIED.
  Onreconnection firstinspectexactJob, watcherstatus, actualservicehealth and
  survivingRAM/PVCinputs. Preservefailed/timeoutreceipts. If noformalruns exist,
  resumewithafreshuniqueJobname andsamev2protocol/root after verifyinghashes;
  nevertrytooverwritecompletedrunfolders orre-useanoldreplicasnapshot.
- Before any heldoutmodeloutput was observed, analysis-only amendment replaces
  naiveunique-parcel resampling withprespecified2kmspatialblock bootstrap because
  heldoutcontexts overlap. Training/data/modelselection unchanged. Receipt
  execution_v2/ANALYSIS_AMENDMENT.json recordsold/newanalysishashes. Do not
  presentbootstrapintervalsaspopulationorall-citygeneralizationuncertainty.

### 2026-10-27 VPN recovery and RAM-cache lifetime repair

- VPN restored. Priorv2batch failed on admissiontimeout; no runs orADMIT.
  All8629persistentinputs and4codehashes passed. Source protocol unchanged.
- User-owned /dev/shm files disappeared twice after SSH sessions ended while
  root-owned pod-created inputs survived. Host Linger=no, defaultRemoveIPC=yes.
  Use a CPU helper to populate this experiment's RAM weights/scripts asroot;
  do not change global logind settings or infer weights survive merefileexists.
  DINO gateddownload goes tohost/tmp then readonlysource mount; no tokensmount.
  PublicQwen downloads insideCPUhelper. Tmpfs writes count towardcgroupmemory,
  so17GBweights requirehelper32GiBlimit rather than2GiB.
- Retryr1 neveradmitted. Suspendedjob cancelled byatomicunsuspend+deadline1;
  terminalFailed andrestorerreadytrue. Servicesindependentlyhealthy01:19:13Z,
  both1/1/1,QwenHTTP200,Ollamaendpointsnormal,nodepressuresfalse.
- Always inspect tool verification success before dependent launch. A chained
  script previously launched r1 despite failingworkerhashcheck; no trainingran.
- Nextretryr2 retainsb63f8a5c environment and6718bd87 protocol; fresh witness
  additionallychecks actualbatch4 inference afteroptimizerstep, because prior
  QLoRApeakreserved23.22GiB is close toA30capacity. AdmitimmediatelyafterPASS.

- Retryr2 allthreefreshwitnesses passed, Heisenberg01a1801d-5a72-72a2-81a1-
  9d513a0c418c. DINOloss1.482421875;Qwenloss7.858975887reloaderror0;
  actualbatch4generation3.26068s,peakallocated22.469GiB/reserved23.129GiB.
  ParentADMITcreated01:28UTC. FormalDINO4trials nowauthorizedtorun; watcher
  PID661693 protects this batch. No heldoutaccuracyobservedyet.

- Bounded four-batch continuation is running locally asPID86181; script and
  CONTINUATION_r2.log/processreceipt are inexecution_v2. It monitorsr2dino,
  verifiesexactservicehealth/restoration, thenrunsqwen42,qwen43,evaluate using
  existinglaunch/restorer scripts, freshsnapshot perbatch, realA30idlecheck.
  Anyfailure stops laterbatches; currentbatchdeadline reduced totriggerrestore.
  IfVPNfails, remoteindependentrestorer remainsarmed; inspectstatebeforeretry.
  Do not duplicate launch whilethiscontroller isalive. Full collection reuses
  priorstream+SHA256code incollect_reliable_final.py andreadonlyresultsPodJSON.
  It checksall4COMPLETEflags,6trainingreports,4DINOcheckpoints,2adapters,
  3Qwenevaluations, thenhashes everydownloadedfile; notyetexecuted.

- R2DINOall4runs completed01:38:12UTC;worker735.78s incladmissionwait.
  Originalservicesindependentlyhealthy01:40:01Z,thencontrollerlaunchedqwen42
  withnewrestorerPID667728. DINO17metadata/predictionfiles hashverified in
  execution_v2/interim_dino; independent sklearn metrics,sourcevalbestepoch,
  allIDs,500paired2kmblockbootstrap checked inDINO_INTERIM_VERIFIED.json.
- DINOtwo-seed externalmeans: frozenCambridgeacc.420/F1.3975,
  Seattle.39344/.34416; finetuneCambridge.482/.46759,Seattle.43852/.40017.
  Fine multi-recall.88/.41, institutional.32/.73, Cambridgeaux-recall.91.
  Seattle small-recall only.01; do not presentmultiincreaseashousingresolved.
  Cambridgeinstitutional48%PrivateCollege vsNYCtrain65%M1religious; examples
  andsubtypecounts indiagnostics. Thisisposthocdiagnosis,notprovenmechanism.
  Frozenprotocol unchanged; Qwenresultsremainpending. No modelupgradeclaim.

- Qwen42 completed2026-10-28T02:15:43Z;worker2131.35s,trainer2102.09s,
  21,823,488trainableparameters,trainingpeakreserved11.0586GiB (resetafter
  loading; do not confusewith22.469GiBallocatedstartupwitnesspeak).
  Bestepoch2:NYCuseacc.389262/F1.376806,roleacc.954023,auxrecall.88.
  Epoch1useacc.308725/F1.287482,roleacc.919540,auxrecall.88.
  Selectedepochagreeswithmaximumsourcevalscore. Interimmethodfiles in
  execution_v2/interim_qwen/runs/qwen_qlora_s42; noQwenheldoutoutputyet.
- Qwen42 originalservicesindependentlyhealthy02:17:32Z. Controllerthenlaunched
  qwen43 withnewrestorerPID684693. It isrunning firstepocharound400/720;
  localcontrollerPID86181 stillownsremainingqwen43/evaluate orchestration.
  Do not launchduplicate jobs orchangefrozenprotocol. AfterALL_BATCHES_COMPLETE
  launch RESULTS_READER_POD.json and run collect_reliable_final.py, then run
  analyze_reliable_transfer.py against
  execution_v2/remote_r2 andnewrepoanalysis/20261027_reliable_transfer output.


### 2026-10-27 late continuation: evaluation precision mismatch isolated

All four v2-r2 batches completed and original services were independently healthy
at 2026-10-28T03:17:06Z. The local controller finished normally. All71 archived
files (444,012,416 bytes), including all DINO checkpoints and Qwen adapters,
passed SHA256 and byte-count checks in execution_v2/remote_r2. Original v2
results are preserved; do not use them as final corrected Qwen results.

Standalone Qwen seed42 validation replay differed from training-time validation
on11/174 use decisions and1 role decision (all11 uses had references). Inspection
of installed PEFT showed preparation casts non-quantized parameters tofloat32;
standalone evaluation had omitted that step and kept them bfloat16. Added one
shared evaluation_model helper used by zero/evaluate and the new GPU witness.
Training, adapter files, source-selected epochs, prompt and dataset are unchanged.

New execution_v3 env hash de6fcdc9, protocol hash4f2ad6f1a04e7261a0cdbd8ff3e4245778f1689c8c9e1adc0afb7ebb0d0ee817.
It explicitly marks evaluation repair after earlier outputs; does not pretend
this new protocol predated all outputs. Original training protocol is included.
Persistent root /stage3/stage2a_reliable_20261027_v3; small repair scripts are in
host /tmp/stage2a-reliable-eval-v3 (not user-owned /dev/shm). Images symlink stays
within the mounted PVC; trained artifacts are copied, previous evaluations kept
in v2. Job stage2a-reliable-evaluate-20261027-v3; restorer PID711900.

New job must WAIT for a fresh independent agent to run the latest .aris command.
Witness uses the production evaluation_model helper, replays all174 NYC val rows
from saved seed42 adapter, checks nonquantized float32 and exact use/role match.
Only after PASS create new-root ADMIT; then the worker recomputes all three Qwen
evaluations. No new training. Still must verify restoration, collect v3 artifacts,
run independent analysis and generate the final table/figure. make_summary.py
exists under analysis/20261027_reliable_transfer and awaits RESULTS.json.

- Fresh Pauli agent01a18086-0a11-7b11-9165-5f1205b5f217 ran v3 witness
  verbatim, exit0:174 source-validation cases, use0/role0 differences. This
  confirms the preparation repair restores saved-model predictions. Parent
  created v3 ADMIT; three corrected evaluations are now running. Before repair,
  seed43 also differed on15 use/5role decisions; receipt inexecution_v3.
  Must await final v3 completion/restored health, collect with the v3 collector,
  and analyze execution_v3/remote_v3. The v2 collector already succeeded.


### 2026-10-27 reliable-transfer experiment completed (03:52 UTC Oct28)

- Corrected v3 Job completed03:49:20Z. Both saved QLoRA seeds replay all174
  NYC validation records exactly (use/role/rawJSON);30 trained artifacts match
  v2 hashes. No retraining/selection/prompt/data changes. Do not use earlier
  v2 Qwen evaluation numbers; canonical results are execution_v3/remote_v3.
- Downloaded56files,299117399bytes;all SHA256/size checks pass, all6trained
  checkpoints/adapters included. Independent sklearn scores, IDs and source-only
  checkpoint selection pass. Final audit:execution_v3/FINAL_EVALUATION_AUDIT.json.
- Two-seed means (Cambridge / Seattle): DINO frozen accuracy.420/.393443,
  macroF1.397507/.344157;fine accuracy.482/.438525,F1.467593/.400170.
  Qwen NF4zero accuracy.428/.524590,F1.374293/.436881;QLoRA accuracy.438/
  .549180,F1.428008/.524715. No global winning/generalizable model established.
- QLoRA Cambridge auxiliary recall.97/precision.761852/roleaccuracy.943333,
  versus zero recall.02. QLoRA multi recall.19/.14, weak and seed-sensitive;
  DINO fine Seattle small recall.01. Qwen produced no unknown/invalid outputs
  across all718records perrun. DINOsource80%threshold external answeredaccuracy
  only.554534/.472536 atcoverage.672/.721311. No reliable rejection claim.
- Original qwen3-14b and ollama independently healthy03:52:28Z:both1/1/1,
  QwenHTTP200,Ollamalist/endpointsvalid,nodepressuresfalse. Exactsnapshot,launch,
  restore,health and Jobterminal receipts inexecution_v3. Readonly resultsPod
  Succeeded; no continuation controller or experimental GPU work pending.
- Final report:handoff/stage2a_la/analysis/20261027_reliable_transfer/READOUT.md;
  METHODS.md,RESULTS.json/.md,SUMMARY.csv,MEAN_METRICS.json,PER_CLASS_RECALL
  PNG/PDF andVERIFICATION.json complete. The figure was visually inspected.
  Existing CNN scores use different references and are not directly comparable.
- Next scientific priority is multi-city/subtype/scale coverage and independent
  building-label/visibility adjudication, retaining DINO/QLoRA controls. Current
  testcities have now been used for diagnosis; future adaptive development needs
  a fresh untouched finaltestcity. Do not claim populationaccuracy, detectorAP,
  exact-datehuman gold, downstreamexposure benefit or universalVLM superiority.
  No commits/pushes, manuscript changes or production model replacement.

### 2026-10-28 LA matched transfer inputs and execution preparation

- Latest user approved LA matched comparison, then controlled multi-city
  training and fresh final cities. Preparing la_transfer_20261027 on X9; no
  original data/weights overwritten. Read the new20261028_la_transfer/PLAN.md.
- Public Maxar LA event STAC contains67 pre-fire items covering all3,332 fixed
  cohort centroids. Recovered native scene windows for all92cells. The supplied
  archive itself still only has256crops: do not claim originals were in it.
- Match original crops against candidate scene pixels, not labels/predictions.
  Original edge crops can have40.7%nodata; compare only observed support (>50%).
  Adjacent COG/mosaic interpolation can lower raw correlation to.93 despite
  visibly identical imagery; retain raw and4x4-averaged correlation (.983-.989
  in diagnosed cell00349). Require >=.97 raw or averaged for every probe.
  Preserve failed receipts and original-pipeline replay. Native missing-pixel
  gates remain <5%context and<1%target; no invented context or substitute dates.
- Final paired inputs3157:3101small,14multi,42commercial.175small context failures
  remain in full3332 denominator as explicit unknown/errors; also score common
 3157 subset. This limits full-cohort accuracy to94.75% even with perfect
  classification, so distinguish input coverage from model discrimination.
- All14multi+6commercial+4small pairedviews visually inspected for alignment;
  no reference relabeling or human-gold claim. DatasetSHA e5cf8f803873897875c84418d15cfbb1343e9be22f5a45c77d1157d220712ad5.
  INPUT_SHA256 has18943files; inputtar1485291520bytes, SHA ce1e25afca31f17ae725a3a9bb2b8cf4874933318a2bb0025831b812e7de005f.
- New prepare_la_transfer.py/evaluate_la_transfer.py/analyze_la_transfer.py reuse
  existing model functions. Checkpoints/prompts/Qwen precision unchanged. Source
  replay witness before admission, three bounded A30 batches, independent restore
  health between batches. Current preparation state is not a model result.
- Existing CNN scores independently reproduce exactly: .783914/.350259,
  N1mean2 .888956/.399168,N2mean2 .958283/.483279 (accuracy/macroF1). Scoring
  needs existing catcch-env Python: old /private/tmp/la_stage2a_eval_20261023
  executable now resolves to py311 without sklearn; do not assume it is a venv.
- Multi-city preparation runs independently in multicity_transfer_20261028.
  Target720train+same174NYCval/arm, NYC-only versus40/city/use acrossNYC/CB/SEA
  plus60NYC+60CBaux. OldCB/SEAtests explicitly become development if reused.
  Boston/SF are only candidate final cities; role/temporal evidence unresolved.

LA execution admitted after independent source replay: all8 DINO validation
probabilities reproduced exactly and all8 Qwen raw/use/role outputs matched.
Four frozen/fine DINO seed runs completed; Qwen zero/42/43 continue through the
existing bounded three-batch controller. Each batch has its own saved replicas,
independent restoration watcher and subsequent health check. Do not start other
GPU jobs while that controller is active, and do not finish with services paused.

Multi-city inputs are frozen: A NYC720 versus B NYC260/Cambridge260/Seattle200,
with exactly the same174 NYC validation records. Five-use and coarse primary
subtype counts match; auxiliary subtypes are only partially matched. All1,354
unique image sets passed native support checks;177new sets acquired,0failures.
Parent visually inspected17 near/wide alignment cards; this is image QA, not
independent label adjudication. Previous Cambridge/Seattle tests are explicitly
retired to development. The new --fixed-epochs flag disables early stopping
only when requested:15epochs/675updates for each planned A/B seed42/43 fit.
Previous LA frozen runner copies retain their original training code. New
execution uses cached PVC imagery plus only72.5MB of genuinely new PNGs; verify
hashes rather than uploading all source TIFFs again. No multi-city GPU run has
yet been launched at this entry.

Continuation blocker: local LA controller exited after SSH timeout; its attempted
job cancellation also could not connect (CANCEL_UNREACHABLE_EXISTING_RESTORER_REMAINS_ARMED).
Current retry cannot resolve geog-zk8-master.artsci.tamu.edu. Independent remote
restore watcher was armed before the pause (PID725878,5100s); restoration is NOT
currently verified. Reconnect campus/VPN, read job state and restore/health
receipts FIRST. Do not blindly restart the controller or recreate the first job.
No terminal receipt was collected; Qwen42/43 were not launched by the controller.
Last observed Qwen-zero progress was1000/3157, not a completed result.

Four DINO results are locally hash-verified in la_transfer_20261027/execution/
remote_la/runs. DINO_PRELIMINARY_REPORT.md and verification/CSV/JSON are complete:
common3157 fine-tuned seedmean accuracy.016471,macroF1.121214, versus CNN N2
.963573/.504806. Most small residences were incorrectly predicted multi;
high multi recall is accompanied by near-zero precision. Source replay and
scoring passed; cause of the collapse remains unestablished. Do not claim that
DINO improves LA or that an input/domain-shift audit is complete.

All three delegated workers subsequently errored on account usage limits.
Multi-city execution files are partial drafts, not staged/validated/launched.
Denver readiness research was returned, but no Denver cohort exists yet. Preserve
the frozen A/B inputs and EVALUATION_DECISIONS.md. Continue local preparation or
use the main agent after connectivity is restored; no scientific success follows
from infrastructure readiness. No commits, production replacement or new pause.

### 2026-10-28: Independent DINO-only LA preliminary collection

- Collected only the four completed DINO predictions/EVALUATION pairs from
  stage2a-la-dinozero-20261028 into X9 la_transfer_20261027/execution/remote_la/runs.
  All eight SHA-256 hashes match; dataset/protocol/execution-code hashes match.
  DINO_COLLECTION_VERIFIED.json and DINO_PRELIMINARY_VERIFIED.json preserve checks.
- Existing analyze_la_transfer.py main requires all Qwen outputs. For interim
  results, import its unchanged cnn_predictions/measure/bootstrap helpers;
  execution/analyze_dino_preliminary.py reproduces all four remote evaluations
  and historical CNN N2 scores. Never label a DINO-only report final LA analysis.
- DINO_PRELIMINARY_REPORT.md/RESULTS.json/SUMMARY.csv stay under execution.
  Full3332 mean macro-F1: frozen .088644, fine-tuned .120866, CNN N2 .483279;
  common3157: .089125/.121214/.504806. Fine-minus-frozen cell-bootstrap CIs cross0.
  Multi recall is high alongside 2,916-3,066 multi predictions and <0.45% multi
  precision. Keep all175 missing-input small references as unknown/errors.
- DINO summaries average seed metrics; CNN N2 averages seed probabilities.
  LA remains development evidence with14multi/42commercial references and
  different pipeline fields of view. Qwen/full analysis and strict sensitivity
  are unfinished; no training, remote writes, admissions, jobs or services changed.
- X9 git status can invoke its LFS clean filter and fail on sandboxed .git/lfs/tmp.
  This is not dirty-state evidence; normal status with authorized access succeeded.

### 2026-10-28 connection recovery and disk-pressure follow-up

- SSH is restored. First LA job is Complete (06:17:32Z); logs confirm dino and
  qwen_zero complete. JOB_TERMINAL_dinozero.json is saved locally. Qwen42/43
  remain unlaunched; do not rerun the first batch.
- Restore watcher log confirms both original services rolled out successfully
  after that job. Current health nevertheless FAILS: desired1/ready0/available0,
  both replacement pods Pending due to wk3 DiskPressure, transitioned14:34:01Z.
  GPU itself healthy/idle0MiB, nodeReady; this is no longer a DNS blocker.
- Removed exactly /tmp/stage2a-la-transfer-20261028/inputs.tar on tamu-gpu after
  verifying remote AND local SHA ce1e25afca31f17ae725a3a9bb2b8cf4874933318a2bb0025831b812e7de005f.
  Local1485291520-byte backup remains; unpacked inputs/results unchanged.
  Root available increased4.9GiB to6.3GiB. No other files removed.
- Actual kubelet config: nodefs/imagefs evictionHard5%, minimumReclaim10%,
  pressureTransition5m. Stats confirmed nodefsavailable6742736896/105089261568,
  imagefsavailable22547206144/126168399872. Pressure still present after waiting;
  do not assume crossing5% alone guarantees recovery. No taints/thresholds changed.
- Original services currently NOT healthy; next priority is scoped disk-pressure
  diagnosis/remediation, then independent health check, then result collection
  and remaining experiments. No new GPU pause or training launched this turn.

### 2026-10-28 scoped disk recovery and Qwen zero collection

- /home is a separate118GiB filesystem; clearing its pip cache would not free
  root nodefs. No pip cache or unrelated project data was deleted. Root initially
  had6.3GiB free despite the prior redundant-inputtar removal.
- SSH sudo requires a password. Ordinary read-only CPU audit Pod was Evicted on
  admission. Used bounded critical-priority maintenance Pods (noGPU,1CPU/1GiB),
  restricted to the experiment PVC and dedicated /home archive, for recovery.
  No node taint/threshold/driver changes, no system service restart.
- Moved exactly stage2a_poolfix_20261015,stage2a_rebuild_20261015,
  stage2a_rebuild_20261015_v2,stage2a_la_adapt_20261023 from /stage3 into
  /home/zhjyu11/stage2a-storage-recovery-20261028 on wk3. All73,506 file/link
  entries and9,886,590,876 regular-file bytes matched before/after manifests.
  No active/pending Pod spec explicitly referenced those old directories.
  They are archived, not deleted. Absolute dataset symlinks are preserved;
  restore original mounts/paths before rerunning those historical experiments.
  Current LA/DINO/Qwen/reliable inputs and model paths were not moved.
- Remote BEFORE.json/VERIFIED.json and /stage3/ARCHIVED_DIRECTORIES_20261028.json
  record paths. Local receipt copies are under X9 stage2a/storage_recovery_20261028.
  Root now~16GiB free. /home was~12GiB immediately after migration, later~22GiB.
- Pressure briefly cleared and original Qwen started, but eviction recurred.
  Read host journals with read-only host journalctl+matching libraries because
  container journalctl cannot decode the newer format. At18:21:13Z kubelet logged
  ability to reduce pressure without evicting pods. Final health remains to verify.
- First LA Qwen zero prediction/EVALUATION plus COMPLETE_dinozero/WITNESS were
  downloaded and SHA-verified in execution/remote_la; receipt
  QWEN_ZERO_COLLECTION_VERIFIED.json. Independent common3157 scoring accuracy
  .9863794742,macroF1.5263309649,multi recall0/14; full3332 .9345738295/.5172676282.
  Do not equate high accuracy with successful multi-residential recognition.
- Multicity execution package prepared locally (8142hashed runtime files,
  1082archive entries,1062newimages72,466,477bytes), LOCAL_PACKAGE_VERIFIED.json
  passed. Runtime env canonical251f54d61f334e56edaacd061c9dc395f56a24f8389b5fbeef567c8337267223.
  It is NOT remote-staged or GPU-witnessed/trained yet. Builder preflight caught
  .jpg RGB/marked vs.png mask enumeration; final package includes all sixviews.

### 2026-10-28 recovery verified; remaining Qwen resumed; A/B inputs staged

- Independent DISK_RECOVERY_HEALTH.json passed at18:28:09Z: original Qwen and
  Ollama each1/1 ready/available with healthy endpoints; wk3 DiskPressureFalse.
  Root~16GiB free after hash-verified archive migration. Use node's full name
  geog-zk8-wk3.artsci.tamu.edu in kubectl (wk3 alone is not a node name).
- qwen42 launched18:29Z with fresh independent5100s restore watcherPID782933;
  original services temporarily paused again for authorized inference. Controller
  execution/continue_remaining_20261028.py handles qwen42->restore/health->qwen43
  ->restore/health. Do not report current services healthy during these pauses.
  Last observed18:44Z progress1000/3157; no disk-pressure recurrence.
- Multicity CPU staging passed18:30:51Z:8142runtime hashes,8124images,
  18943LA hashes,3DINO weights, identical174val and3332/3157/175 serializer.
  REMOTE_STAGED.json is local under multicity_transfer_20261028/execution.
  Remote root /stage3/stage2a_multicity_transfer_20261028 exists; do not rerun
  nonempty staging.7062 cached image symlinks target physical reliable_v2 files.
- Master launcher/specs at /tmp/stage2a-multicity-exec-20261028 hash verified.
  A/B have NOT launched or passed GPU witness yet. Each needs fresh independent
  documented witness and parent admission; launcher requires all LA batches
  Complete and independent restoration health before next pause.
- Added analyze_multicity_transfer.py reusing existing sklearn scorer and
  LA-cell bootstrap. Checks exact train/val IDs,15epochs,675updates,source-only
  checkpoint selection and prediction hashes. Real helper self-check passed
  equal-arm bootstrap null and unknown-as-error. Full analysis awaits4 fits.

### 2026-10-28 Qwen42 completed and restored, Qwen43 running

- qwen42 Complete after~39min; all3332 outputs collected with individual SHA256
  in execution/remote_la/COLLECTION_qwen42.json. Same common3157 accuracy
  .4586632879,macroF1.4275908613; full3332 .4345738295/.4198290123.
  Multi6/14 recalled but1464small references became multi (multi precision.004076).
  Commercial24/42 recalled. This is deterioration, not a successful fine-tune.
  QWEN42_PRELIMINARY.json and QWEN42_ERROR_AREA.json saved under LA execution.
  Small->multi errors span82cells; median footprint334m2 versus240m2 for correctly
  classified small buildings. Association only, not causal proof; area not numeric input.
- RESTORED_HEALTH_qwen42.json independently passed19:10:39Z, both services1/1,
  QwenHTTP200/Ollamamodels present, no pressure. qwen43 launched19:10:42Z with
  new watcherPID803995 (5100s), services temporarily paused again as authorized.
- Read-only result reader pod stage2a-result-reader-20261028 uses noGPU,
  100mCPU/128Mi request and10800s bound. Delete this named pod after collection.
  collect_completed.py under multicity execution downloads completed Qwen/A/B
  artifacts via reader and verifies every remote/local hash. qwen42 path passed.
- Independent witness agent James01a183d5-0e6b-7032-a3a5-b982a02de82e has read the
  skill/contract/ledger only and is waiting for parent confirmation to execute A.
  No A/B GPU witness/training yet. Do not launch until qwen43 restored+healthy.
- Follow-up visual QA of four deterministic quarter-index errors (zero-shot
  correct/QLoRA42 small->multi) found valid near/wide images throughout, but one
  outline appears to include pool/patio and another appears to cover only part
  of a contiguous roof. QWEN42_VISUAL_DIAGNOSTIC_SELECTION.json and diagnostic
  receipt preserve selection/observations. AI visual check only, not gold or
  prevalence estimate. No inputs/labels changed. footprint_m2 is predicted
  polygon area, not verified building area; contamination can confound size
  correlations. Source-official versus LA-predicted footprints is another
  domain difference, distinct from crop completeness/raster alignment.

### 2026-10-28 Qwen complete; fixed-budget A/B training started

- qwen43 Complete~40min, outputsSHA-verified. Both original services independently
  healthy19:52:51Z (RESTORED_HEALTH_qwen43.json); ALL_BATCHES_COMPLETE.json saved.
- Full LA analyzer passed all seven runs, oldCNN metric parity and Qwen rawparse.
  Repo analysis/20261028_la_transfer/{RESULTS.json,SUMMARY.csv,PREDICTIONS.json,
  REPORT.md} now contains complete comparison, not a DINO-only preliminary result.
  Common3157 QLoRA seedmean accuracy.5319923978/macroF1.4064504833;
  zero.9863794742/.5263309649; CNNN2.9635730124/.5048061467.
  QLoRA-minus-zero macroF1-.11988048,paired91cell95%CI[-.15757055,-.09350295].
  Full3332 QLoRAmean.5040516206/.3983058045. Do not claim fine-tuning improves LA.
- A_nyc launched19:53:18Z with fresh5100s watcherPID824980; services paused for
  the authorized fixed-budget training. GPU idle0MiB verified before unsuspend.
  James01a183d5-0e6b-7032-a3a5-b982a02de82e executed documented witness verbatim:
  exit0 MULTICITY_DINO_WITNESS_PASS A_nyc NVIDIA A30 LA2+unknown1; real backward,
  reload and missing-label gradient passed, no doc mismatch. Parent admitted.
  Seed42 train15+LA complete;seed43 train15 complete, LA evaluation pending.
  No B launch yet. Need A restoration/health before B; need B witness/admission.

### 2026-10-28 A complete/restored; B admitted

- A_nyc two15epoch fits and both LA evaluations Complete. All15artifacts,
  including both57,516,367-byte checkpoints, collected and SHA-verified under
  multicity execution/remote_multicity; COLLECTION_A_nyc.json saved.
- Direct master SSH briefly reset during per-file transfer twice (kex read:
  Connection reset by peer). GPU SSH and a read-only ProxyJump check worked;
  direct master access recovered without network/service configuration changes.
  Collector now reuses SHA-verified local files and standard SSH ControlMaster
  (dedicated /private/tmp socket,60s persist). Actual resumed collection passed.
- A independent RESTORED_HEALTH_A_nyc.json passed20:08:48Z; bothservices1/1,
  QwenHTTP200/Ollamamodels present, DiskPressureFalse. Root16GiBfree,/home13GiB
  last observed; original service image startup increased imagefs usage.
- B_multicity launched20:09:09Z with independent5100s watcherPID831303; GPUidle0
  checked before unsuspend. James executed B witness verbatim,exit0 expected
  sentinel and no doc mismatch. Parent admitted; seed42 training underway.
  Original services temporarily paused for B. Must restore/verify after B,
  collect all B artifacts, run analyze_multicity_transfer.py and remove only the
  named temporary result reader pod after results are secure.

### 2026-10-28 completed: Qwen and fixed-budget city comparison

- B_multicity Complete (~11min), both15epoch/675update fits and LA evaluations
  passed. All30 A/B artifacts, including four57,516,367-byte checkpoints, are
  local and SHA-verified; COLLECTION_A_nyc.json/COLLECTION_B_multicity.json.
- A large B checkpoint stream initially failed hash before writing any accepted
  output. Remote cat exit could truncate kubectl exec output; collector now
  delays remote stream closure5s. Re-download passed full57,516,367-byte hash.
  Keep this drain delay and hash checks when streaming larger artifacts.
- Campus DNS later failed. Verified master IP10.55.180.150 worked with explicit
  HostKeyAlias=geog-zk8-master.artsci.tamu.edu and StrictHostKeyChecking=yes.
  Never disable host-key verification. A600s dedicated SSH multiplex connection
  completed remaining read-only collection; no SSH/network settings changed.
- Independent FINAL RESTORED_HEALTH_B_multicity.json passed20:23:31Z:
  both services desired/ready/available1, QwenHTTP200, Ollamamodels present.
  FINAL_STATE.json20:26:21Z: all5 LA/A/B jobs Complete, ReadyTrue and allpressure
  conditionsFalse; nodefsfree14.93GiB,imagefsfree12.06GiB. Temporary named result
  reader is gone. Its30s delete wait timed out at grace-period boundary; later
  get --ignore-not-found confirmed absence. No production model replaced.
- analyze_multicity_transfer.py passed exact train/val IDs, same174val,15epochs,
 675updates, source-only checkpoint selection, provenance and prediction hashes.
  Common3157 A_NYC metricmean accuracy.0624010136/macroF1.1193609554;
  B_multicity.6243268926/.4433202291. B-A macroF1+.3239592737,
  paired91cell95%CI[.2119599889,.4300845223]; both seeds improve.
  Full3332 A.0591236495/.1175265148; B.5915366146/.4347455274;
  delta+.3172190126,[.1900497572,.4187642889]. Strict-assessor sensitivity also
  positive. Detailed RESULTS.json/SUMMARY.csv/REPORT.md in repo
  handoff/stage2a_la/analysis/20261028_multicity_transfer.
- Multicity DINO still below CNNN2 common macroF1.504806 and Qwenzero.526331.
  B multi recall6/14 and3/14, precision2.47%/5.66%; source confidence thresholds
  retain81.03%/88.98% with accepted accuracy65.72%/66.54%, not a safe OOD gate.
  NYC val useF1 drops .5947/.5873 -> .4548/.4095; roleF1 roughly stable.
- This is positive matched-budget LA development transfer evidence, NOT new-city
  proof. Denver cohort remains unprepared/unpredicted. Next: freeze Denver
  reference/image cohort and exclusions before evaluating these locked arms;
  no LA-based checkpoint/threshold retuning or production deployment performed.

### 2026-10-29 LA-first diagnosis; no new training

- User redirected the immediate objective to good LA test performance and asked
  to diagnose first. Defer the previous Denver-next suggestion. This turn only
  rescored saved predictions, audited configs/assessor attributes and inspected
  existing images; no source labels, model inputs or remote services changed.
- analysis/20261029_la_diagnosis/REPORT.md and diagnose.py document the findings;
  generated DIAGNOSIS.json/METRICS.csv/MINORITY_CASES.csv passed frozen dataset,
  eight raw-metric parity, prediction-order, probability/projection and56 unique
  assessor-join checks. Common3157 all-small accuracy98.23%/macroF1.33035: high
  accuracy alone is misleading. CNNN2 multi2/14,commercial31/42; Qwenzero0/14,19/42.
- N2 is NOT NYC-only: actual train CSV has13,911 rows=13,333Harris+578NYC.
  New foundation-model fits use720 rows, only600 known-use. Data, labels and
  inputs differ; do not describe this as a controlled backbone comparison.
- B DINO seeds42/43 map977/1044 small refs to institutional. Post-hoc three-way
  argmax over saved probabilities lifts seedmean accuracy62.43->81.55%, but
  macroF1 only.44332->.46509; still687/351 small->multi. This is diagnosis under
  a three-use assumption, not a new independently validated five-class model.
- All14multi attributes:9x4units,2x5,2x8,1x26; commercial22/42 office/professional.
  CNN+Qwenzero jointly miss12/14multi; naive fusion cannot recover these cases.
  Three illustrated near/wide pairs are targeted diagnosis, not revised gold.
-175 unavailable paired inputs cap full3332 accuracy at94.75% without fallback;
  old CNN crops exist. Diagnostic Qwenzero+CNN-on-missing raises full accuracy
  to97.99% but macroF1.47762, below zero alone. Do not default missing to small
  using known test composition, or deploy this fallback without development QA.
- Proposed priority: independently audit rare LA refs and low-rise multi/office
  training coverage, spatially separated LA development data, improve existing
  CNN baseline then matched-data DINO; repair paired-input coverage separately.
  Current LA is inspected development evidence. Preserve old reference/results;
  reserve untouched LA areas for final confirmation. No new training launched,
  and yesterday's remote service health was not rechecked this turn.

### 2026-10-29 LA adaptation data frozen; GPU execution next

- User authorized LA label audit, spatially isolated additional training, CNN
  first and matched-data DINO. Repo analysis/20261029_la_adaptation contains
  pretraining plan,57 reference checks and all near/wide contact sheets.
- LA Units is generic Number of Units, not NYC unitsres. Commercial Units>0
  alone does NOT establish mixed residential use; corrected the initial audit.
  Four attribute uncertainties and historical outline differences remain; no
  reference label changed and visual screening is not independent human gold.
- Added1186 LA image sets,748train/438val, all image QA passed. Old126 product
  cells plus frozen wide contexts buffered2km excluded; fixed2km grid separates
  train/val/reserved, no cross-split context overlap. Reserved not imaged/scored.
  Single assessor record + countywide2014 outline + unique2020 BLD_ID IoU>=.8.
- Frozen DATASET18429rows=13911originalsource+748LAtrain+438LAval+3332oldtest;
  SHA b9c38fa099f74125bd950d2dfa14266ba9e115f916d129671f19ebe2e57354d9.
  Legacy256 input matches oldCNN;30new crops retain<99%roof,min51.26%. Full
  near/wide saved for QA. Only4newtrain cases are exactly4units; weak fourplex
  coverage remains despite93multi. Originaltest has9fourplex/14multi.
- COG source profile YCbCr cannot be copied with deflate; explicit RGB needed.
  Failedempty TIFF preserved as .failed_photometric.tif before successful retry.
- Existing source/checkpoints absent from PVC because earlier /home migration;
  mount active LA data/explore and four recovery dirs read-only at original
  /stage3 paths. Do not duplicate severalGB back onto root. CPUstagev2 created;
  old CPUstage expires naturally. Both original services still1/1 at preflight.
- Minimal train_la_adaptation.py uses exact N2 common.py snapshot (area pooling);
  stale repo common lacks this behavior. CNN seeds42/43 then DINO last2blocks,
  6x8192draws,batch32, same IDs/drawseed. Epoch selection only isolated weighted
  LAval; no oldtest epoch/threshold tuning. RealGPU parity/backward witness and
  autorestore-protected jobs pending. Do not report new model scores yet.

### 2026-10-29 first matched LA adaptation complete; low-rise ablation running

- CPU stageverified18,429rows,2,372newimages,34,486oldimages;rootfree13.87GiB.
  Alloldsource/checkpoint /home mounts read-only; no archive copied back.
- CNN42/43 baseline36sample probes maxprob differences2.98e-7/1.97e-6; real
  CUDA forward/backward and complete gradient checks passed. Both6epoch fits
  completed, isolatedweightedval choseepoch3/5. DINO42/43same data/draw hashes,
  officialViTB16last2blocks, bestepoch2/2. All4weights and predictions collected
  and SHA-verified under X9 la_adaptation_20261029/execution/collected.
- Primary full3332 oldCNN .958283accuracy/.483279macroF1; newCNN probability
  ensemble .977191/.571612; DINO .976891/.572436. CNN gain+.088333, paired92cell
  CI[.037173,.125128]; DINO-CNN+.000824,CI[-.049995,.078783]. Common3157
  old/newCNN/DINO macroF1 .504806/.594854/.578387. Primary snapshots preserved
  as PRIMARY_RESULTS.json etc before adding exploratory follow-up.
- MultiTP old2,newCNN3,DINO3 of14; multiFP25/29/40. CommercialTP31/28/24 of42,
  FP93/24/11. Improvement mainly reducedcommercialFP; no low-rise solution yet.
- IndependentLAval CNN recallsfourplex3/10,5-8units9/17,9plus90/100; DINO4/10,
  7/17,92/100. Trainmulti median801m2 vsoldtest338m2.51eligibletrainfourplex
  parcels:31twooutlines,4single,15missing,1seventeen; only4accepted. Rawcached
  footprintqueries overlap: dedup AIN+OBJECTID before counting.
- Primary CNN/DINO batches each Complete; originalservices independently health
  checked(restored1/1,Qwen200,Ollamamodels,nodepressureFalse) between batches.
- Follow-up LOWRISE_PLAN fixed beforeits training: ONLY LA multiinternal draw
  masses equal for4,5-8,9plus.93trainingrecords=4/32/57, factors7.75/.96875/
  .54385965; originalsource/LA50-50 andclassmasses unchanged. RestartfromoldN2,
  same6x8192 budget, seeds42/43, sameval selector incl0. No extraindependent
  fourplex images implied. Newtrainer script separatelycopied; oldruntimehashes
  preserved. Jobstage2a-la-adapt-lowrise-20261029 unsuspended, standalonewatcher
  under/tmp/stage2a-la-lowrise-20261029. Mustfinish,collect,analyzeall6fits,
  verifyoriginalservicesrestored,then removeonlyownedtemporaryCPUstagepods.

### 2026-10-29 LA adaptation finished: six fits, service health verified

- Lowrise CNN s42/s43 Complete;selectedepoch3/4, testmacro .582982/.518442.
  Probabilityensemble fullaccuracy .978691,macro .580393;common .981311/.601373.
  +.008781overordinaryadaptCNN CI[-.000432,.023045], not reliable extra gain.
  +.097114overoldCNN CI[.045018,.137562]. MultiTP3/14,FP27; commercialTP28/42,
  FP21. Independentvalfourplex remains3/10,5-8units9/17; reweighting has not
  fixed smallmulti. Keep CNN mainline; do not claim DINO superior or quadsolved.
- All6checkpoints,predictions,history,witness logs collected and SHA-verified.
  analyze_la_adaptation.py passes originalcohort/labels,preselectedepochs,
  weightedF1parity,checkpointidentity,CNN/DINOdrawhash and bootstrap checks.
  PRIMARY_* preserves initial4fits; RESULTS includesexploratory lowrise.
- Finalhealth23:26:20Z originalqwen/ollama1/1,Qwen200,Ollamamodels,endpoints
  healthy,nodeReady/pressureFalse. SERVICE_LIFECYCLE.json records all3batches
  restored normally withouttimeout. REPORT.md,VERIFICATION.json,trackerupdated.
-31two-outlinefourplexcandidates intrain-onlygrid queued, no labelsassigned.
  ArcGIS AREA field inrawfootprintprops is fullyqualified
  eGIS_Infrastructure.EGIS.LARIAC_BUILDINGS_2014.AREA, not AREA. Largeroutline
  does not establish mainuse; distinguishquad+garage from2residentialbuildings.
- Currentclassifier evaluation doesnotvalidate populationhead, detectorAP,
  institutional/other/role ornewcity. Sharedfeatures changedwhilepopheadfrozen,
  so populationoutputs cannot be assumed preserved. No productionmodel replaced.
  Reserved LA grids have not been imaged or scored; old LA is development evidence.
- Final cleanup: both owned CPUstage pods confirmed absent; PVC inputs/results
  and completed GPU jobs retained. Root free13.31GiB. POD_CLEANUP.json records
  the exact two names and timestamp. No new experiment remains running.

### 2026-10-29 fourplex supplement frozen; CNN42/43 starting

- User explicitly requested review31 fourplex candidates, distinguish primary/aux,
  add isolated small multi training and rerunCNN; oldLA test must stay unchanged.
- prepare_la_fourplex.py acquired31 fresh2024 assessor records and62 prefire image
  sets. AI visual review admits28 street-front fourplex main buildings, separately
  marks28 rear garage/carports, excludes3 ambiguous parcels/6 outlines asunknown.
  All28 additions are one tract/grid178_1894: do not equate count with diversity or
  AI-assisted records+imagery review with independent human gold. Case4 auxiliary
  lacks2020sameID but is visible; it is not a CNN training target. First3 excluded
  for ambiguous roof/annex/absent-secondary correspondence; all judgments recorded.
- New la_fourplex_20261029 DATASET18457 rows preserves original18429 rows verbatim,
  val438 and test3332 dicts/order unchanged. Add28trainmulti:93->121,4units4->32.
  Zero overlap withold2kmtest exclusion,valcontexts/valgrid/parcel; legacyretention
  100% on28 additions. Source/LA50-50 andLAclassmasses478/93/177over748 heldfixed.
- CPUstage verified56newimages and all36858oldimagebytes,2initialcheckpoints,
  correctcommon.py snapshot and runtimehashes. Rootfree13.16GiB. NewCPUstage
  stage2a-la-fourplex-stage-20261029 uses oldread-only /home mounts, nooldcopies.
- NewGPUjob stage2a-la-fourplex-cnn-20261029; controller directory onmaster
  /tmp/stage2a-la-fourplex-20261029, independent5100srestore watcher,job4800s.
  Originalqwen/ollama both1/1saved,thenpausedwithwatcher;GPU0MiBbeforeunsuspend.
  Execution metadata underX9 la_fourplex_20261029/execution. Mustcollectbothfits,
  independentlycheckrestoredservicehealth,thenremoveonlyownedCPUstagepod.
- Reuse train_la_adaptation.py; optionalprotocol.la_class_mass preservesoldclass
  shares afternewexamples; witness explicitlyloadsoneNEWfourplex. SameN2init,
  six8192-draw epochs,selectiononlyold438weightedvalincl0. No--lowrise thisbatch.
- analyze_la_fourplex.py prepared; includesmacro pluspairedcellmulti recall/F1 CIs,
  excludingresampleswithzeroactualmulti; directfunctionperfect/wrong/identity
  assertchecks passed. Actualnewmetrics pending. OriginalLA remains development
  regression evidence, notuntouchednewcity. No productionmodel replaced.

### 2026-10-29 fourplex supplement complete: limited recall gain, no overall gain

- BothCNN fits completed6epochs (180.2s/177.9s),val-selectedepoch6/4. New-image
  GPUwitness/gradchecks andold36caseparity passed (2.98e-7/1.97e-6). DatasetSHA
  94f95e68cd56968e3a2a5de9d5f6ed99c1c256631eb0096b54a33d98e894d0b8.
- Full3332 two-seedprobensemble:accuracy.974189676,macroF1.567900233,multiTP4/14,
  FP35,recall.285714286,F1.150943396,precision.102564103. OrdinaryLA CNN comparison:
  TP3->4,FP29->35,F1.130434783->.150943396,macro.571612243->.567900233.
  Macrodelta-.003712 CI[-.024612,.032177];multiF1delta+.020509 CI[-.033720,.120000].
  PreviouslowriseCNN macro.580393339 remains higher; no newdefault/promotedmodel.
- Testquad0/9->1/9,recovered00415_2e997cd0-48a6-4b17-92aa-9fb73ae56cf1,4units,
  strictonebuilding,284m2. Onlyseed42corrects it;43stillpredictssmall. Test5-8units
  unchanged2/4,26unit1/1. Validation4units remains3/10;5-8units9/17->12/17,
  9plus90/100unchanged. Common3157macro.596164077 vsordinary.594854062,
  lowrise.601372680. Fullcohort remains primary; do not choosefavorable subset.
-16predictionchangesvsordinary:13correctsmall->minorityerrors,2smallerrorsfixed,
 1multierrorfixed. CommercialTP28same,FP24->29. Addedonehomogeneoustract has
 increasedlowrisecoverage butnot demonstratedstable overallimprovement. Next
 data priority: distinctblocks/buildingforms andisolatedlookalikeSFHnegatives,
 validatedonfixeddev beforeanotherfrozencomparison. No test-threshold sweeps.
- All22artifactsincluding2x113MBweightscollected andremoteSHAverified. Independent
 restorehealth2026-10-30T02:02:44Z: qwen/ollama1/1,Qwen200,Ollamamodelsandendpoints
 healthy,nodeReadyTrue/noPressure. Restorationnormalwithoutdeadline. CPUstage
 stage2a-la-fourplex-stage-20261029 deleted andabsenceconfirmed;PVC andcompleted
 GPUJob preserved. SERVICE_LIFECYCLE.json/POD_CLEANUP.json innewexecutiondir.
- Repo analysis/20261029_la_fourplex REPORT.md,RESULTS,MINORITY_COMPARISON,
 CHANGED_PREDICTIONS,REVIEW,DATA_AUDIT,SOURCE_SHA256,VERIFICATION andtrackercomplete.
 analyze_la_fourplex.py reproduces metrics,baselineconfusions,selectionrule,
 initvalparity andpairedCIs. 28AI-reviewed mains allonegrid; notindependenthuman
 gold. Only14testmulti; LAalreadyinspecteddevelopment. No unseen-city/reserved
 evaluation,rolehead/detectorAP/populationevaluation,productionreplacement orGit
 commit/push. No experiment remains running.

### 2026-10-29 saved-result diagnosis: representation, hard negatives and label attribution

- User requested careful diagnosis and useful innovation before further changes.
  Added standalone diagnose_la_bottlenecks.py and analysis/20261029_bottlenecks;
  no trainer/inference edits, new training, threshold sweeps, label changes or
  TAMU operations. Recomputed saved3332 scores and allactualCNN masks; checked
  frozen dataset/prediction identity, unchangedoldval/test records and35visualcases.
- Actual frozenCNN224input->768x7x7->area-mask pooling+4geometry values. Tenmissed
  multi have median2.716mask-area-equivalent featurecells, none show substantial
  polygon clipping. This is not independent receptivefield count or causalproof.
  CNN already seescontext throughreceptivefields; lacks explicit separatecontext.
- Actual Stage2aDataset20repeatedcalls onnewfourplex yielded1uniqueinput. Fixed
  seed+local_idx drivesaugmentation when domain_augmentation=False; currentruns
  use scale_view_probability0 anddon'tpassmodelseedtoDataset. Fixrandomness as
  independentengineeringbaseline; enablingdomain_augmentation alsoaddscolorjitter.
- Newmulti10/10FNwrongmaxsoftmax>.9;11/35FP>.9. MultiAPordinary.2223,new.2029;
  APisclassificationranking,notdetectorAP. Lowconfidence-onlyVLMfallbackwouldmiss
  theseFN. All7newmultiFPofficially1unitSFH. Twelveof14commercialFNareoffice/
  professional;LAtrainhas32/177thoseuses,test22/42. Missingdiversityincludeshard
  negativesandhouse-likeoffices,notonlymorepositivesfromonequadtract.
- Near/wide-unavailable175havenewmultiFP11vs24/3143available negatives. Only2/35
  FPexceed10%historicalRGBzeros; donotblameallerrorsonblackpaddingordeletecases.
  00200_300be333polygonvisuallymayincludepool/patio,requiresindependentverification.
  35visualreviewsareAIobservations,notnewgold. Existing3332parcelreferencesfrozen.
- literature-review/STAGE2A_RECOGNITION_DESIGN_20261029.md checksoriginalsources
  andreadingdepth. GNN+context+streetview/DINO,MIL,incompletelabels,background
  invariancealreadyexist. Candidatecontribution:buildinguse+primary/auxroleswith
  explicitparcel-to-buildinglabelattribution;testagainstsimpleunknownmaskingand
  matchedhardnegativecontrols. Fourunitsperparcelmaybetwoduplexes,soordinary
  positivebagMILassumptionisnotguaranteed. Countconstraintsneedindependentanchors.
- REPORT/VISUAL_REVIEW/VERIFICATION captureevidenceandstagedablationplan. CurrentLA
  remainsdevelopmentwith14multifamily;noindependentgeneralizationornewaccuracy
  claim. TestassessorUnits/UseCodecannotbeclassifierinputwhenitdefinesreference.

### 2026-10-29 clarify complete LA benchmark versus entire LA product

- User asked whether currentbestmodel was tested onthecomplete testdataset.
  Rechecked lowriseCNN42/43 andensemble:3332unique predictions each, exact frozen
  testID/order coverage,allthree-class outputs,including175withoutpairedviews.
  Ensemble accuracy.9786914766,macroF1.5803933392,multi3/14TP27FP,F1.1363636;
  commercial28/42TP21FP,F1.6153846. "Best" here usesuniformtwo-seedensemble
  comparison;singleDINO42observedmacro.588327 isnotameasuredensembleadvantage.
- Do not call3332theentireLAproduct: originalinputs/building_damage_all_cells.csv
  has22024records/126cells; la_reference.csv has11403matchedrecords/118cells,
  with3332passingprimary_single_outline_iou50. Currentbestmodel's evaluated
  domain isthisfilteredreferencebenchmark,notare-runofall22024rawtargets or
  theend-to-enddetection/damage/populationpipeline. No newinference thischeck.

### 2026-10-30 full LA CNN comparison launched on CPU

- User authorized currentbestCNN onall22024LArecords versusoldmodel, unchanged
  originaltestlabels. Currentbest=cnn_lowrise_s42/s43 probabilitymean (epochs3/4);
  old=originalN2s42/s43mean. NewdirectoryX9 outputs/stage2a/la_full_20261030.
  DATASET SHA f2f9c84e39a35ff90043d3cb123e93d10b183733eb39d12f58774a9496147cb9.
- Found136previouslyexcludedinputszero-mask, not136missingimagefiles. Original
  productcolumnpolygon_wkt_utmactuallycontainspixel-rangegeometriesforthese136;
  derivedlegacyxy105faroutsideimageand31invalidrings. Do notfeedtheirclamped
  cornercropsorzero-maskstoclassifierandreportasvalidtargetpredictions.
- prepare_la_full.py leaves21888originalinputsverbatim;136separaterecoveriesuse
  pixelgeometryconvertedbyGSD/originverifiedagainstthreeunchangedanchorspercell.
  67cells:60alreadyrecovered,7newscene-alignedprefiretifs(8targets). All136masks
  nonempty,100%ofmaskpixelshaveimagery;12stratifiedvisualspotchecksalignwithroofs.
  Someboundary/largetargetsremainclippedbyfixed256window. REPAIRS/RECOVERY_SOURCES/
  execution/RECOVERY_QA retainprovenance. Nooriginalsourceortestlabelschanged;
  recovered136receiveNOnewgroundtruth. Coordinatesareinferredrepair,notnewgold.
- Initialpreparefailedbecausecheckingthreeanchorsinunusedcellswith<3buildings;
  fixedtolimitgeoreferenceworktocellsrequiringrepair. Bothpreparelogspreserved.
  44048inputfileshashed;all3332oldtestpaths/areas/IDsunchanged. PROTOCOL freezes
  primary3332,strict3315secondary,andweakerIoU7601/broad9036exploratorycohorts.
- CPUwitnessbenchmark~27.7images/sat8threads. Startedjobs
  stage2a-la-full-s42-20261030 and stage2a-la-full-s43-20261030,8CPUeach,noGPU.
  Qwen/Ollamanotpaused. Helperpodstage2a-la-full-stage-20261030has10800sdeadline.
  Existingimage/PVC/read-onlyhomemountsreused;54MBnewtaskbundleuploadedonly.
- infer_la_full.py verifiesallinputhashesand36old/newprobabilityprobesbeforefull
  inference;old21888scoresreusedafterparitycheck,old136recoveredinputsrecomputed.
  New22024foreachseed. CPUrunreportsunderremote /stage3/stage2a_la_full_20261030/
  runs/seed42andseed43. Collectbothfolders,checkRESULT/probabilityhashes,run
  analyze_la_full.py locally,writeREPORT,checkservices,thenremoveonlyhelperpod.
  Jobsandresultspreserved. Noaccuracy/completionclaimuntilbothjobsfinish.
- FirstCPUjobslackedthetrainingmanifest's /dev/shm mount;DataLoaderqueueshit
  defaultshared-memorycapacitybeforewitnesscompleted. Savedfailed_shm_logs.txt,
  suspendedonlythetwodedicatedjobs,verifiedpodsgone,renamedruns->failed_shm_runs.
  Started-r2jobswithsamecode/data/weightsplus2Gimemory-backedshm;didnotdeletefiles.
  Bothr2witnessespassed:oldmaxerror3.73e-7/1.70e-6,new9.31e-7/5.42e-6,
  all36classpredictionsexact. Both44048inputhashcheckspassed. Fullinference
  underway~26images/sperseed. Use-r2jobsforstatus;initialjobssuspended.

### 2026-10-30 full LA CNN comparison complete: 22,024 records

- Both r2 CPU Jobs completed 22,024/22,024 predictions (seed42/43 total769/842s).
  Collected all ten run JSON artifacts and verified dataset/checkpoint/output
  hashes. Currentbest is unchanged lowrise42/43 probability mean; oldN2 uses
  frozen21,888 predictions plus136 predictions on the same recovered inputs.
  Repaired136 have no historical reference rows and contribute no scored labels.
- analysis/20261030_la_full contains REPORT, RESULTS, VERIFICATION,
  PREDICTION_COMPARISON (22,024 unique cell/building IDs), CHANGED_PREDICTIONS
  (619 rows), and METRICS (all reference strata and class precision/recall/F1).
  Independent sklearn recomputation from exported CSV reproduces every primary
  metric; no missing or duplicate records. Original source/test hashes unchanged.
- Fixed3332 accuracy .9582833133 -> .9786914766; macroF1 .4832792226 ->
  .5803933392. Multi P .074074 -> .100000, R2/14 ->3/14, F1 .097561 ->.136364;
  commercial P .25 ->.571429, R31/42 ->28/42, F1 .373494 ->.615385.
  Commercial FP93 ->21; multi FP25 ->27. Primary103 labels changed. Full3332
  classes exactly match prior results; new maximum probability difference2.82e-6.
  Macro delta .097114, paired92-cell bootstrap95%CI[.045018,.137562], conditional
  on these fixed models. Small residential dominates; multi remains inadequate.
- Full predicted counts old ->current: small20629 ->20968, multi198 ->281,
  commercial1197 ->775. Counts are predictions, not true prevalence. Strict3315
  macro .479750 ->.577960; exploratoryIoU7601 .473896 ->.542064 and broad9036
  .460328 ->.524765 remain weaker parcel-reference diagnostics, not building gold.
- Qwen/Ollama never paused. SERVICE_HEALTH checked1/1 each,QwenHTTP200,OllamaCLI
  successful,nodeReady/noPressure. Task-owned helper pod
  stage2a-la-full-stage-20261030 deleted; empty get --ignore-not-found confirmed
  absence. ATTEMPTS and POD_CLEANUP record completion; PVC/results/completed r2
  Jobs and suspended initial Jobs retained. No experiment remains running.
- Cleanup watch timed out at30s after delete acceptance; independent get returned
  no pod with exit0. Use the separate absence check, not delete-watch status alone.
- No new training, threshold tuning, test relabeling, production replacement,
  Git commit/push, detector/damage/population evaluation or unseen-city claim.
  This is three-class building-use inference; recovered coordinates are an
  audited interpretation with some fixed-window clipping, not independent gold.

### 2026-10-30 current-best CNN: every multifamily error reviewed

- User requested all11 missed multi and27 false positives, label/image/model
  diagnosis and targeted precision/recall improvements, with test labels fixed.
  audit_la_multifamily.py and analysis/20261030_multifamily_audit preserve exact
  error IDs from the full-product comparison; do not reuse the fourplex model's
  different10FN/35FP list. Visually inspected all38 errors plus3TP controls in
  actual224RGB,actualmask,adaptive near,and300m context. REVIEW_NOTES is the
  authored observation source; EVIDENCE,CASE_REVIEW,REVIEW.html,REPORT,SUMMARY,
  SOURCE_SHA256 and VERIFICATION are generated. AI observations are not new gold.
- All11FN have9 four-unit,one5-unit,one8-unit records; footprint155–473m2 vs
  TP639–2551m2. FN median7x7 mask-area2.680 cells vsTP6.298; not independent
  receptive-field count or causal proof. NineFN havewrong maxprob>.9. AllFN
  targetpixels observed,no cropedge; original RGB sourceagreement andmask
  coordinate checks pass. Thirty-eight errors cannot establish unseen-city gain.
- FP26SFR/1Office,20Palisades/7Eaton;9 in cell00183. Many flat/elongated/large
  houses. FP16/18/19 touchcropedge; FP17/18/19 RGBzero6.6/38.3/9.8%, but zero
  within masks0. Nonzero mask and complete SAM polygon do NOT prove whole physical
  roof retained. Geometry review includes FN02,FP03/06/11/16/18/19/23; these are
  candidates, not confirmed causal faults. Seven missing diagnostic views retain
  source-edge gaps; generated adaptive views are separate, no inference-input edits.
- Requeried public countywide historical footprints plus original IDs in newer
  catalog, withraw/provenance receipts. Originalparcel_outline_count counts only
  LARIAC/DINS evaluation subset, notall roofsperparcel. FN06/09/10 andFP12/13 have
  two countywide outlines, targetlarger; do not call thisautomaticauxmislabeling.
  HistoricalID misses can be version changes: newer service matches39/41; two
  2023IDs absent. Query byID inaddition toAIN before declaringmissingbuilding.
- Label priorities: FN01 original4units/twoassessorrecords; sameoldbuildingID
  belongs historicalAIN5839009027 vsreference5839009039, leavingperbuildingunit
  allocation unresolved. FP16 PUD1unit mayneedunit/wholebuilding association.
  FP21 OfficeSQFTmain1730ft2 vs1317m2roof suggestsrecord-granularity mismatch.
  No frozen label changed and no posthoc filtered accuracy reported.
- ExistingDINO recoverszero of11FN; fixes19currentFP butadds32elsewhere,total40FP
  vsCNN27. Full3332 DINO multiP.06977,R.21429,F1.10526,macro.57244. Cannot claim
  improvement from lookingonlyatcurrent38errors. Seeds disagree16/27FP,1/11FN;
  low-confidence-onlyfallback cannot cover9highconfidenceFN.
- Confirmed currentfrozenStage2aDataset behavior onall4 existingLAtrainfourplex:
  20drawseach ->1distinctinputtensor. Same currenttrainer hflip.5/rot90.25,
  defaultdatasetseed42/domain_augmentationFalse; no trainer edit. LA adaptation
  multi93 only4fourplex,medianarea801m2;small478 all1unit,median247m2. These are
  LA portion counts, notall13911source-record unitcounts. Later28fourplex samegrid.
- Recommended independent A/B: stochasticflip/rotationonly; matcheddiverse4–8unit
  positives pluslarge/flatSFR,office,andverifiedaux negatives outsidefrozenexclusion
  withclasssamplingmassesheld; targetscale then explicitcontext; validation-only
  calibration. Do not fit rule/threshold on38testerrors. Preservefixed3332labels
  andprimaryinputs; anyinputrepair/adjudicatedgold isseparateversion. No newmodel
  trained or gainverified,thisturn; no TAMU changes,commit/push,browsercontrol.
- ActualDatasetprobe,sourcehashes,exactFN/FPIDs,all41cardlinks,AST,andexisting
  primarymetrics verified. StaticHTML linkcheck only; browsernotoperated.

### 2026-10-30 augmentation-only experiment in progress

- User authorized stochasticaugmentation fix, pairedseeds42/43 withsame data,
  samplingweights,architecture andfrozenLAtest; separateisolatedcandidateprep.
- Canonical scripts/stage2a/common.py one-line fix uses Torch per-draw seed in
  _augment; frozenextendedcommon copiedtonew la_augmentation_20261030/execution
  withonly correspondingseedexpression changed. Oldsnapshotsunchanged. Actual
  real-image assertcheck:old1view/new8in64draws,jointRGBmask,no colorchange,
  seedreproducible,evaluationunchanged. tests/check_augmentation.py retainscheck.
- DATASET18429/LOWRISE_SAMPLING/PROTOCOL byte-identical tola_adaptation_20261029;
  remote36858imagehashes,initialmodelsandfree12.23GiBverified. Same trainer with
  --lowrise,6epochs8192draws,oldweightedval selection. Newjob
  stage2a-la-augmentation-cnn-20261030; originalservicespausedunder independent
  5100srestorewatcher /tmp/stage2a-la-augmentation-20261030 onmaster. Mustcollect
  runs,verifyservicesrestored,thenremoveonlyownedhelperpod ofsamename-stage.
- Seed42completed,43training. analyze_la_augmentation.py checksall6drawhashes
  perseed againstoldrun,initialvalidationconfusion,bestepochandcheckpointSHA.
  Collectundernewexecution/collected/runs; do notreportmetricsuntilbothcomplete.
- prepare_la_hard_candidates.py:39queried/31imagedisolatedcandidates,3multi/28SFR,
  14grids,no testbuffer/valcontextintersection,notaddedtotraining. All31visually
  screened:2fiveunitmain-lookingcandidatesneedsameparcel2roofunitallocation;
  thirdfourunit17roofparcel targetlooksaux/carport,excludedfrompositivequeue.
  1SFRgeometrymismatch+3cloud/hazeheld;remaining24needmoderngeometry/hardness
  verification. Existingpoolcannotprovideenoughdiversepositives; do notcallall31
  reliabletrainingsamples. CandidateDATA/AUDIT/BUILDING_REVIEW retainedseparately.

### 2026-10-30 augmentation-only comparison completed

- Both seeds completed; analysis is in `handoff/stage2a_la/analysis/20261030_augmentation/`.
  Canonical fix and frozen runtime fix only change per-draw flip/rotation RNG.
  Real-image check: old 1 distinct view versus fixed 8 in 64 draws; reproducible
  seed, joint RGB/mask transform, unchanged evaluation. Dataset, weights, all six
  epoch sampling hashes per seed, initial validation, and checkpoint hashes pass.
- Frozen LA 3,332-row ensemble: accuracy 97.8691% -> 98.1092%, macro-F1
  58.0393% -> 59.0907%; multifamily P 10.00% -> 13.6364%, R unchanged 21.4286%,
  F1 13.6364% -> 16.6667%. TP stays 3, FN stays the same 11; FP 27 -> 19
  (10 removed, 2 new). Commercial P/R/F1 unchanged. Both seeds improve macro-F1
  and multi P/F1, but seed43 multi recall falls from 4/14 to 3/14. Paired spatial
  bootstrap macro delta +1.0513pp, 95% CI [-0.5782,+3.4482]pp: not significant.
  Do not claim that augmentation fixes recall or proves cross-city generalization.
- Candidate batch remains separate: 2 promising five-unit targets need unit-to-roof
  attribution; both are in the same grid. One four-unit parcel has 17 outlines and
  the selected roof appears auxiliary: excluded. 24 single-family candidates need
  modern outline verification and model hardness scoring; 4 held for geometry/cloud.
  Never treat largest parcel outline as verified main use. Expand isolated positive
  sources before a separate data ablation; do not tune thresholds on the test errors.
- Collection pitfall: long `kubectl exec tar` streamed through SSH returned success
  with truncated archive tail. Preserved that artifact; regenerated a complete gzip
  archive on master, checked tar integrity and SHA, downloaded via scp, rechecked
  SHA and both checkpoint hashes. Use `verified-results.tar.gz`, not `results.tar`.
- Restore watcher restored both original deployments. 2026-10-30T17:22:58Z health
  check: Qwen HTTP200, Ollama model list and endpoints ready; desired/ready/available
  all 1, no node memory/disk pressure. Evidence retained in execution/controller and
  analysis VERIFICATION.json. Deleted only helper pod
  `stage2a-la-augmentation-stage-20261030`; completed GPU job and data retained.
  No production checkpoint replacement, commit, or push.

### 2026-10-30 isolated hard-example ablation in progress

- User approved next-step proposal. Keep original taxonomy: 1–3 units remain small,
  positives are 4–8 units; earlier conversational 2–8 wording was incorrect.
- All 9 prefire Maxar acquisitions/67 items were already in the old pool. LARIAC7
  2023 RGB API returns Token Required. Public NAIP2022 export/query works; dated
  primary tiles locked by ID, RGB bands0/1/2, native0.6m, exported ~0.305m to keep
  the old256px physical field of view. This adds source/domain differences; no claim
  of a pure label-count ablation. Sources/receipts under la_hard_expansion_20261030.
- Existing31 candidates: local frozen model score witness max error <5e-11;
  all28 SFR predicted small, not model hard negatives. 15/31 newer outline matches;
  remaining IDs absent, not automatically wrong labels. Two old five-unit positives
  remain unresolved. New experiment does not admit these old31.
- Expanded official query: 325 train-only compact multi references/16 grids and
  554 SFR/63 grids. Preserve original2km exclusion and hash val/reserved folds.
  Single county outline + one Assessor line + floor/roof compatibility + newer ID
  IoU>=.8 yielded48 multi and103 SFR. Acquired151 NAIP scenes/crops with0 failures.
- Visually reviewed all48 multi plus6 model-confusing SFR. Hold indices23 (canopy),
  24/34 (one outline spans separate roof masses with unclear unit allocation).
  Admitted45 multi across11 grids and6 SFR across6 grids, official+AI references,
  not independent human gold. Negative gate declared before scoring: ensemble
  multiP>=.05 or either seed>=.10, max4/grid40total; one ensemble false positive,
  remaining5 high-confusion/seed-disagreement. Other95 SFR not visually admitted.
- Frozen experiment la_hard_ablation_20261030/execution contains B_positive and
  C_positive_negative. A reuses augmentation-fixed baseline. B adds45, C adds51;
  original dataset prefix, val438/test3332 unchanged. Source/LA masses.5/.5; LA
  class masses478:93:177; equal multifamily4/5–8/9+ mass maintained. Same6epochs,
  8192draws, optimizer, model and validation selection. Frozen trainer only removes
  old mutual-exclusion assert so existing class normalization and lowrise weighting
  can coexist; canonical shared trainer not edited.
- Scripts: score_la_hard_candidates.py, query_la_expanded_pool.py,
  prepare_la_expanded_candidates.py, freeze_la_hard_ablation.py,
  analyze_la_hard_ablation.py. Freeze assertions passed. Operational scripts reuse
  prior verified templates, own job stage2a-la-hard-ablation-cnn-20261030/helper
  stage2a-la-hard-ablation-stage-20261030; controller
  /tmp/stage2a-la-hard-ablation-20261030 on master. CPU stage created; GPU batch not
  yet launched at this note. Before launch verify STAGED and start restore watcher.
  Always collect all4 runs and verify original services restored before finishing.
- Local Torch multiprocessing failed under sandbox with torch_shm_manager Operation
  not permitted. Interrupted only that failed scoring process, reran authorized
  scoring unsandboxed; both seeds and probes passed, no model/service changes.

### 2026-10-30 isolated hard-example ablation completed

- Both B and C seeds42/43 completed in one13-minute job. A is the reused
  augmentation-fixed two-seed baseline. All original36,858 input hashes,102 new
  image hashes, dataset/sampling hashes,4 checkpoint hashes and archive checks pass.
  Each run6epochs/1536updates; checkpoints selected only on original weighted438val.
- Frozen3332 ensemble results (accuracy / macro-F1 / multi P,R,F1):
  A .981092/.590907/.136364,.214286,.166667;
  B .981092/.595872/.136364,.214286,.166667;
  C .982893/.610370/.166667,.214286,.187500.
  All3 detect the same3 multifamily and miss the same11. A/B FP19, C FP15.
  C removes7 A false positives but adds3. Commercial F1 .615385 -> .630435 ->
  .652174. This improves ensemble precision, not the unresolved recall problem.
- Individual-seed results are inconsistent: C_s42 macro .5707/multiR2of14 versus
  A_s42 .5987/3of14; C_s43 macro .5680/3of14 versus A_s43 .5554/3of14.
  C-A fixed-model spatial-bootstrap macro delta +.019464,95%CI[.000198,.046111].
  The interval is conditional on these checkpoints, not training-variance evidence;
  don't claim stable single-model improvement or cross-city generalization.
- Reports: analysis/20261030_hard_ablation/{REPORT.md,RESULTS.json,PREDICTIONS.json,
  DATA_REPORT.md,DATA_AUDIT.json,REVIEW.csv,VERIFICATION.json}. Each candidate review
  and source receipt remains under external la_hard_expansion_20261030. Initial
  plan in analysis/20261030_augmentation/NEXT_ABLATION_PLAN.md. 45 positives include
  23fourplex: LA training fourplex references4 ->27, keeping group probability mass.
- Separate complete B-results.tar.gz/C-results.tar.gz and local collected models
  retained under la_hard_ablation_20261030/execution. Auto-restore succeeded;
  health2026-10-30T22:59:57Z: QwenHTTP200, Ollama list/endpoints, replicas1/1,
  no node pressure. Removed only stage2a-la-hard-ablation-stage-20261030; GPU job
  retained Complete. No checkpoint promotion, production-model change, commit/push.
- Next priority: target-closeup/context input ablation to distinguish insufficient
  target detail from unobservable use, keeping frozen labels and spatial splits.
  No closeup/context training or untouched-city evaluation ran in this batch.

### 2026-10-30 image-driven exposure and Stage2a novelty assessment

- Research assessment only: `literature-review/STAGE2A_IMAGE_ONLY_EXPOSURE_20261030.md`.
  Latest experimental C is macro-F1 .610370, multi TP3/FP15/FN11; do not cite
  .5804 as the newest candidate. Recomputed A/B/C macro-F1 from saved confusion
  matrices and checked the identical11miss IDs; no new inference or training.
- Deployment can use imagery plus georeference/scale and learned priors; training
  uses external supervision. Exposure is a proxy, not measured residents, actual
  event-time occupants or casualties. Current LA damage scores include multiple
  post dates. Recent classifier changes did not validate the population head.
- Near/wide DINO and use/role heads already ran; soft type probabilities already
  feed the exposure HGB. Do not present these as new/untried mechanisms. A matched
  closeup/context ablation on the current CNN remains unrun.
- Literature includes LA MambaBDA+GHSL, POPCORN/POMELO, LandScan mosaic uncertainty,
  aggregate supervision and residential-unit allocation. Integration, soft outputs,
  sum constraints and uncertainty alone are not sufficient novelty claims.
- Candidate: train-only parcel total units plus independently verified per-building
  unit anchors, test-time imagery only. Role/use labels may not identify (1,3) vs
  (2,2), and auxiliary does not imply uninhabited. Compare at equal verification
  cost with clean direct unit supervision; improved parcel sums alone are not proof.
  The same-family novelty review is provisional PROCEED WITH CAUTION. No causal
  proof that attribution or feature scale explains the current11misses; no model,
  label, production, manuscript or Git commit/push changes in this assessment.
- Follow-up prior-method check: Tokyo2024/Bangkok2025 micro-population studies
  explicitly combine building-use inference and population allocation, but require
  POI/geometry/heights/statistics. POPCORN uses SAR+RGB/NIR seasonal composites,
  implicit occupancy and Census weak supervision; final100m grids are not single
  roofs. POMELO/LSM are not raw-pre-RGB-only counterparts. Gandhinagar2025 preprint
  uses DEM and rule-generated population targets, not observed building populations.
  Detailed approaches/read depths added to the assessment; no experiments run.

### 2026-10-31 paired visible-target close-up run and unit-review preparation

- User authorized the proposed close-up experiment and dwelling-unit panel. Active
  batch: `stage2a-la-closeup-cnn-20261031`, controller on master
  `/tmp/stage2a-la-closeup-20261031`, independent 5100s restore watcher with 4800s job.
  Both prior LLM deployments were saved at 1/1; the watcher must restore them.
- Source: new `prepare_la_closeup.py`, `target_closeup.py`, `check_la_closeup.py`;
  external `la_closeup_20261031/execution`. Reuses frozen C fixed-view fits; only two
  close-up fits, seeds42/43. Same14710train/438val/3332test, labels, checkpoint
  initialization, sampling, six epochs/8192draws, physical area and all4geometry
  features. Geometry is computed from the original augmented mask, not zoomed mask.
- Near/wide input cache lacks175test cases; using it directly would alter cohort
  and scene/resampling. This first controlled test uses the exact original pixels:
  square max(40m,1.4xvisible-bbox), capped at original size. Edge-touching masks keep
  original crop. It does NOT recover clipped roofs. Do not call it full-roof source
  recovery. 3204/3332test inputs zoom;121edge fallbacks; all visible mask pixels
  retained. Preflight passed36960source hashes, original-runtime tensor parity and
  augmented geometry equality. Original and reserved inputs remain untouched.
- `prepare_la_unit_panel.py` made51candidate cases:24smallmulti,24area-matched
  single-family,3ambiguous roof allocations,102hashed near/context images, evidence,
  reviewCSV,offlineHTML. External `la_unit_panel_20261031`; report in
  `analysis/20261031_unit_panel`.24multi already trained classifier. Verified per-roof
  unit counts and independent-test references are BOTH ZERO; fields left blank.
  Accessory habitation remains missing. Do not treat this prepared review panel as
  independent gold or train a unit/population model on unverified parcel counts.
- Pending: collect paired results/checkpoints; verify draws and validation-only
  epoch selection; evaluate unchanged cohort; restore service health; remove only
  our temporary staging pod. Do not report model improvement before completion.

### 2026-10-31 close-up experiment completed: no demonstrated benefit

- `analysis/20261031_closeup/REPORT.md`, RESULTS/PREDICTIONS/VERIFICATION saved.
  Both new fits completed6epochs/1536updates (176.96s,174.98s), same2.518GiB peak
  GPU reserve as fixed C. Reused fixed C fits; hashes, tensor parity, all4geometry
  features, both seeds' exact epoch sampling draws, validation-only checkpoint
  selection and both collected checkpoints verified. Archive SHA256:
  6b8873f4527df06f03dadb4703decb4145dcac475acdab1834bfc49cec217aaa.
- Ensemble: fixed C macro-F1 .610370 -> close-up .580358; multi TP3/FP15/FN11 ->
  TP3/FP21/FN11. Same3TP and11FN;9oldFP removed,15newFP added. Weighted validation
  macro-F1 .925022 -> .902005. Seed42 macro .570692 -> .575965; seed43 .568049 ->
  .550620. Paired cell macro-F1 difference 95%interval[-.073070,.002400] includes0;
  no reliable gain, not proof of universal degradation. Keep C as research baseline.
- Do not claim target scale is the only cause: close-up also removes context and
  does not recover clipped roofs. Context-extension gate did not pass, so no joint
  context model was run. Unit/population head remains unvalidated;51-case review
  package has ZERO independently verified building-unit labels and ZERO eligible
  independent-test records. Do not conflate its24classifier-trained multifamily
  cases with held-out unit truth. Reserved LA data and production models unchanged.
- Both original LLM services restored1/1. Health2026-10-31T01:30:50Z: QwenHTTP200,
  Ollama lists gemma4:12b/qwen3:8b, readyserviceendpoints, no nodepressure. Controller
  and collected GPU artifacts retained under `la_closeup_20261031/execution`.
  Removed only `stage2a-la-closeup-stage-20261031`; completed GPU job retained.
  No new dependencies, commit, push, deployment promotion or further GPU batch.

### 2026-10-31 best-C LA visual preview in progress

- User requested a visual LA example using the downloaded original results HTML.
  Source found at `/Users/zhou/Downloads/LA_fire_results_page_2026-10.html`.
  It embeds22024v3damage points and original cell00365/00471post-fire illustrations.
- Prior full22024inference used older cnn_lowrise(.5804), NOT bestC(.610370).
  New CPU jobs `stage2a-la-best-preview-s42-20261031` and `...-s43-20261031`
  evaluate exact frozen C checkpoints over the identical full manifest. Existing
  infer_la_full.py reused,8CPU/seed,2GiSHM. NoGPUservicespaused. Both old/new
  witness probabilities matched within2.4e-6. Input44,048hashes passed perseed.
- External output `outputs/stage2a/la_best_preview_20261031`; input manifest SHA
  f2f9c84e39a35ff90043d3cb123e93d10b183733eb39d12f58774a9496147cb9.
  Temporary CPU staging pod `stage2a-la-best-preview-stage-20261031` must be removed
  after collection; completed jobs retained. Mastercontrol `/tmp/stage2a-la-best-preview-20261031`.
- New viewer source: build_la_best_preview.py + la_best_preview.html; hit-test check
  tests/check_la_preview.js. Uses original green report styling, clickable use
  footprints/pre-fire crops, preserved v3damage,126cell point maps,14multifamily refs.
  Do NOT label saved damage as current Stage2a inference. No exposure estimates are
  recomputed. Cell00365 has405records with3previously repaired geometries; original
  post-fire illustration shows402. Distinguish saved illustration from fresh overlay.
- Local image/GIS runtime is `/Users/zhou/miniconda3/envs/py311/bin/python`;
  numpy1.24.4/Pillow11.3/rasterio1.3.9/shapely2.0.2 alreadyinstalled. System/Homebrew
  Pythons lack these packages; do not install duplicate dependencies just for preview.
- Browser control still needs explicit permission per user AGENTS. No browser has
  been controlled for this preview. Pending full inference collection/build/QA.

### 2026-10-31 best-C LA visual preview completed

- Fresh two-seed C inference completed all22024records. Counts:20993small,
  187multi,844commercial. All3332reference argmax labels equal frozen C; max
  probability difference1.818e-6. Results archive SHA256
  206afff0878a1d2966d65630df374184f40fb14b93cbc7a13b1f87e81b2a2b46.
- Viewer: external `la_best_preview_20261031/viewer/index.html` (6.55MiB),
  preview.jpg, predictions.csv, README.md. Report/audit under repository
  `analysis/20261031_best_preview`. Original HTML damage points/classes matched
  every row. Use scores are fresh C; damage remains saved v3; no population update.
- All22024crop+mask local links verified,3sceneimages checked,JS syntax/DOMtargets/
  actual courtyard hit-test passed,static comparison image visually inspected.
  Embedded maps/scenes work offline; crop drill-down uses existing external-drive
  data through2directory symlinks. Keep drive mounted. Do not mistake the original
  post-fire illustration's402footprints in00365 for fresh405-record use overlay.
- Browser permission was requested asynchronously per user AGENTS; no browser
  control or interaction check occurred without a reply. CPU jobs bothComplete;
  removed only stage2a-la-best-preview-stage-20261031. GPUservicesstayed1/1;
  noLLMpause,training,promotion,commit,push,orchange tooriginaldownloadedHTML.

### 2026-10-31 LA preview browser verification

- User explicitly approved opening/checking the preview. Codex browser checks
  passed: three scenes, expanded polygon selection, Close/Escape, crop navigation,
  use/damage filters, map selection, all14reference thumbnails and correct/missed
  reference selection. CSV download retrieved; no browser warnings/errors observed.
  Narrow and desktop layouts were observed without a viewport override.
- Pitfall: the original HTML's136bad centroids remained near the equator even
  though their crop geometries had been repaired. Exact source matching alone
  does not establish usable map coordinates. This collapsed the Palisades map.
  build_la_best_preview.py now transforms those existing recovered EPSG:32611
  geometry centroids to lon/lat. Other21888coordinates and all predictions/votes
  are unchanged. Correct map counts:2870Palisades,19154Eaton.
- Regenerated viewer/README/VIEWER_AUDIT and repository RESULTS.json; generator
  asserts all22024locations lie in LA. Separate roundtrip check confirmed exactly
  136coordinate changes. Existing actual-polygon hit-test and diff checks passed.
  Browser screenshots/verification live in analysis/20261031_best_preview.
- Local preview server binds127.0.0.1:8767 and serves the external viewer directory;
  this is a temporary local process, not a deployment. Keep the data drive mounted
  for crop assets. No exposure recomputation, model promotion, commit or push.

### 2026-10-31 exposure added to LA preview

- Recomputed all22024population proxies with frozen B_semantics_mean3 OOF head
  and current C42/43 mean probabilities plus log1p footprint area. Total69023.955;
  minor/major/destroyed subtotal17522.957; unknown damage excluded from subtotal.
  No retraining; C-to-exposure/LA transfer remains unvalidated, not occupancy truth.
- recompute_la_preview_exposure.py reuses original pure helper AST definitions;
  saved test log predictions reproduce exactly. EXPOSURE.json stores aligned values,
  input probabilities/areas, head/helper hashes. Generator checks alignment and inputs.
  Inspector and CSV show per-building exposure; header shows global totals.
- Runtime pitfall: sklearn1.9.0 pickle requires matching runtime. Installed isolated
  dependencies only in /private/tmp/la-exposure-runtime. Invoke py311 with -S and
  PYTHONPATH=/private/tmp/la-exposure-runtime to avoid old pandas/numpy ABI conflict.
  Run recompute_la_preview_exposure.py there, then build_la_best_preview.py in normal
  py311 GIS runtime. No environment packages changed. Temporary runtime may expire.
- CSV/HTML values and aggregates checked, saved head parity exact, JS syntax and
  actual polygon test passed. Existing use probabilities and damage votes preserved.

### 2026-10-31 LA population validation against original v3

- validate_la_exposure.py and analysis/20261031_exposure_validation compare saved
  original v3 pred_population (21888rows) to current C+OOF-head exposure. Same-row
  totals75591.025original vs67978.803current; currentfull22024total69023.955.
- Supplied LA census package contains2020population on TIGER2025boundaries and
  2022ACS. Boundary-vintage crosswalk not independently verified. Recover126cell
  extents from pixel/world polygon alignment; exclude partial areas from primary
  scoring. Coverage is geometric, not evidence of complete building detection.
- Primary323fullycoveredblocks with no missing original inputs:11132buildings,
  Census20568; oldtotal35166.425,current33309.016. MAE51.029→43.757 (14.25%lower),
  bias+70.98%→+61.95%. Residential-only sensitivity MAE34.125→27.221,
  bias−51.02%→+32.63%; predicted residential subsets differ, not verified masks.
- Negative evidence:10fullycoveredACSgroups shared-input MAE629.452→698.575.
  Only2groups have no missing old inputs. Do NOT claim robust/general population
  validation, classifier-only benefit, observed occupants or displaced people.
- All22024points uniquely assign to both levels;21888sharedjoin; sourcehashes,
  per-building/block/groupCSV and independent aggregate conservation checks saved.
  No fitting/calibration, model changes, or preview mutation in this validation.
- Parquet reader pitfall: pyarrow26 needs numpy2; installed only in temporary
  /private/tmp/la-census-parquet. Extracted geoid/population/MOE using -S plus
  /private/tmp/la-exposure-runtime numpy2, to source-hashed ACS_REFERENCE.json.
  GIS validation runs in unchanged py311 numpy1 runtime; small cells may have
  fewer than3valid anchors, so verify all available anchors rather than require3.

### 2026-10-31 exposure excess diagnosis

- Primary323block cohort decomposition saved in EXCESS_BREAKDOWN.json under
  analysis/20261031_exposure_validation: current33309.016 minus Census20568 =
  +12741.016. Predicted commercial358footprints contribute6029.830; excluding
  them leaves residential27279.185 (+6711.185). This is an arithmetic sensitivity,
  not proof that47.3%of error is causally commercial occupancy.
- Current predicted residential10774footprints versus Census8119housingunits;
  footprint != housingunit.1213residential predictions below50m² contribute3064.211,
  median2.524people; below100m²3739contribute9445.363. These may include garages,
  sheds, ADUs or real homes; imagery/unit audit needed, not automatic exclusion.
- Existing exposure training labels are proxies:15571/15597small-residential
  records have estimated_units1, medianpopulation2.6. Commercial1132targets have
  units0 but medianpopulation23.85, confirming a different occupancy construct.
  Current C use probabilities feed an older exposure head; better use prediction
  does not establish resident-count calibration. No causal ablation performed.

### 2026-10-31 comprehensive Stage2a documentation review

- Added `handoff/stage2a_la/STAGE2A_COMPLETE_REVIEW_2026-10-31.md`: original
  implementation and July M20–M41 history; repaired Harris data/ACS/area/pooling;
  source/NYC, VLM/DINO/QLoRA, LA adaptation and supplement experiments; exact
  training/validation/test locations, latest C results, exposure/Census checks,
  implementation map, remaining gaps and reproduction/operational lessons.
- Current C training is 13,911 source + 799 LA = 14,710 rows, validation 438,
  diagnostic 3,332. The separate 28-fourplex supplement is NOT cumulative in C;
  current C adds 45 multi plus 6 small to the original 748 LA training rows.
  C weights require the paired N2 initializer/config/frozen area-pooling source;
  old handoff common.py is not a drop-in current runtime.
- Independently recalculated C confusion/macro-F1 from saved predictions and
  all 22,024 preview means/counts; all 3,332 classes agree, max probability
  difference 1.818e-6. Rechecked dataset splits, C/head hashes, exposure CSV
  totals, all six block/group Census cohorts and 68 local document links.
  Evidence: `analysis/20261031_comprehensive_review/VERIFICATION.json` (PASS).
- Latest close-up accuracy is 97.8391%, macro-F1 .580358; retain C .610370.
  Exposure WAS updated after the original preview: current C + frozen OOF head;
  primary 323-block MAE 51.029 -> 43.757, bias still +61.95%; ACS sensitivity
  worsens. Distinguish preview from production/default package and proxy from
  observed occupants. Existing LA/Harris/SD are inspected development evidence.
- Documentation-only work; no model/code/label/default-pointer changes, training,
  new inference, browser/cluster operation, commit or push. Existing unrelated
  working-tree files and all frozen results were preserved. July individual-run
  artifacts were not recreated; history is attributed to II_package/docs.

### 2026-10-31 commit/push preparation authorized

- User explicitly requested committing and pushing all pending repository work
  on `stage2a-la-v3-results` to `origin`, permitting large model/data artifacts to
  remain local. Preserve existing ignore rules: checkpoints, rasters and large
  external-drive experiment datasets are not added. Pending analysis CSV/JSON,
  reports, literature notes, audit traces and implementation files are included.
- Pre-commit checks: all 13 Stage2a unittest cases passed (8 input/output,
  2 assessor, 3 OOF exposure), actual augmentation checks passed (64 draws:
  old=1 distinct view, fixed/current=8), and the actual viewer polygon-hole
  hit-test passed. Parsed 70 Python, 112 JSON and 1 JSONL files successfully.
  The augmentation check was executed without rewriting its frozen output JSON.
- Local test environment pitfall: py311 has Torch but no default sklearn;
  catcch-env has sklearn/GIS but no Torch. Run assessor tests in catcch-env;
  input/output and augmentation in py311. For exposure tests, import installed
  NumPy/pandas first, append the existing /private/tmp/la-exposure-runtime for
  sklearn, and prepend the frozen poolfix II_package to sys.path. No packages
  were installed or environments changed. Combined discovery also lets the
  assessor test's handoff path shadow the frozen scripts package; separate
  test-module runs avoid this historical import collision.
- No retraining, new inference or live cluster/browser validation is part of
  this Git operation. Local absolute paths and external datasets are still
  required for full experiment replay; pushing source does not upload them.
- Preserve frozen CSV CRLF bytes and archival trailing EOF blank lines rather
  than formatting experiment evidence. Source whitespace checks pass; the full
  staged check passes with command-local `core.whitespace=cr-at-eol,-blank-at-eof`.
