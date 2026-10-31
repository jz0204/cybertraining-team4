# TAMU Stage2a generalizable pilot

## env: stage2a-fiveuse-cnn@70251ffb

Spec: X9 `II_package/outputs/stage2a/generalizable_pilot_20261026/ENVIRONMENT.json`.
Canonical sorted compact JSON SHA256 prefix: `70251ffb`.
Pinned existing vLLM image, torch2.13.0+cu130, torchvision0.28.0+cu130,
transformers5.15.0; no package installs in the training image. One A30 24GiB,
4 CPUs, 48GiB RAM. `runtimeClassName: nvidia` is mandatory.

Inputs staged at `/dev/shm/stage2a-generalizable-20261026` on worker3;
Job copies them to existing PVC `/stage3/stage2a_generalizable_20261026`.
Persistent input/model growth is expected below2GiB. Weights downloaded to
memory-backed `/ram/torch_cache`; no language-model weights in this batch.
Root/PVC free12.81GiB and home free12.20GiB before staging. Do not put models
on home or blindly clean old caches. Original LLMs have not yet been paused.

The Job copies the exact X9 `common.py` with repaired area mask pooling.
Training scripts, data and hashes are frozen in `PROTOCOL.json`.
Validation is pending until an independent agent follows this exact command
against the actual running Job, while its main worker waits for admission:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-generalizable-cnn-20261026 -- python3 /stage3/stage2a_generalizable_20261026/train_generalizable.py witness'
```

Expected: `WITNESS_PASS NVIDIA A30 [2, 5]`, finite loss, exit0. This runs the
actual dual-view RGB ConvNeXt plus geometry and five-class head with a backward
pass, including a single-pixel mask test. Do not repair or change the command.
Report a mismatch; parent then decides whether to admit training.

Auto-restoration uses the existing master watcher with5100s bound and exact
pre-run replica snapshot. Job deadline4800s. Required post-run checks: actual
terminal Job/pods, full output count and hashes, original services ready,
health endpoints, and node disk/memory pressure.

## env: stage2a-fiveuse-cnn-dino@e43fc9d0 (v2; prior batch never launched)

Spec is `generalizable_pilot_20261026/execution_v2/ENVIRONMENT.json` on X9.
Official DINOv3-B revision5931719e67bbdb9737e363e781fb0c67687896bc downloaded
using user-configured host authentication. Read-only RAM weight mount
`/dinov3-model`; no token or login directory enters the training container.
Persistent root now `/stage3/stage2a_generalizable_20261026_v2`.
Dataset hashes unchanged; v1 protocol retained. Before any model outputs, image
QA identified small Harris parcel-support structures. V2 adds a conservative
major-roof training filter and keeps the unfiltered frozen-CNN control. This is
not a verified primary-role label. 48 predeclared runs:36 small frozen-head
fits and12 fullCNN fine-tunes; paired near/dual,2seeds,3heldout developmentcities.

Fresh agent must execute BOTH commands exactly and report any divergence:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-generalizable-cnn-20261026-v2 -- python3 /stage3/stage2a_generalizable_20261026_v2/train_generalizable.py witness'
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-generalizable-cnn-20261026-v2 -- python3 /stage3/stage2a_generalizable_20261026_v2/dinov3_witness.py'
```

Expect first `WITNESS_PASS NVIDIA A30 [2, 5]`, finite loss andexit0;
second `DINO_WITNESS_PASS NVIDIA A30 [2, 201, 768]`, finite tokens andexit0.
Do not improvise repairs, change the job, or admit it; report results to parent.
Parent is responsible for bounded restoration and admission after your result.

Prelaunch CPU compatibility fix changes current env hash to `dd82673e`.
`ENVIRONMENT_before_cpu_fix.json` and prior source/protocol retained. Same
commands and workloads; no model experiments ran before this correction.
Real CPU trainer test passed loss/backward/metrics, exact checkpoint reload,
normalization, prediction IDs and unknown-role contract.

## env: stage2a-fiveuse-cnn-dino@c013e5e9 (startup repair v3)

V2 failed before any inference/training: absolute host symlink inside /staging
pointed to an unmounted /dev/shm location. Preserve v2 Job and receipts. V3
explicitly mounts the existing image directory read-only at /input-images.
No data/model/hyperparameter/split changes. Spec ENVIRONMENT_v3.json and
JOB_v3.json under execution_v2. Persistentroot is now suffix `_20261026_v3`.
This supersedes earlier witness commands; run both EXACTLY:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-generalizable-cnn-20261026-v3 -- python3 /stage3/stage2a_generalizable_20261026_v3/train_generalizable.py witness'
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-generalizable-cnn-20261026-v3 -- python3 /stage3/stage2a_generalizable_20261026_v3/dinov3_witness.py'
```

Expectedsentinels/shapes unchanged. Parent must await completed service
restoration from v2 before arming the new snapshot/watcher/pause for v3.

V3 independent witness completed by fresh agent
01a17bc8-b51e-7ac1-97d2-f17038d2e013. Both commands verbatim, exit0:
`WITNESS_PASS NVIDIA A30 [2, 5] 1.59765625`;
`DINO_WITNESS_PASS NVIDIA A30 [2, 201, 768]`.
The DINO script asserts torch.isfinite(y).all() before printing its sentinel;
no mismatch. Parent created ADMIT. This is kernel/environment validation,
not classification accuracy or evidence of generalization.

V3 completed all 48 trials, Job 2026-10-27T05:14:37Z to 05:37:08Z.
Worker wall time 1347.14s. All 260 archived files, including 48 checkpoints,
matched remote SHA256. Local replay of all 36 frozen heads preserved every
classification; maximum probability difference 9.84e-7. All 48 scores were
independently checked with sklearn. This pilot did not meet the transfer goal.

Both original LLMs restored and independently checked at 05:40:28Z:
desired/ready/available 1/1/1, Qwen health HTTP200, Ollama list/endpoints valid,
Disk/MemoryPressure=False. Watcher ready=true; no forced deadline suspension.
Read-only results pod stage2a-generalizable-results-20261027 ended Succeeded.
PVC free11.06GiB. No service pause or active experimental GPU job remains.
Receipts: X9 execution_v2/RESTORED_HEALTH_v3.json and JOB_TERMINAL_v3.json.

DINO and ConvNeXt pretrained weights are now also persisted on X9 under
execution_v2/pretrained, with verified hashes. The TAMU /dev/shm copies are
volatile; the login and pinned download instructions are in
handoff/stage2a_la/DINOV3_SETUP.md. A large kubectl exec stream initially lost
its tail: preserve the failed receipt, flush and allow the remote stream to
drain before exit, then require byte counts/SHA256. Never accept file existence
as proof of a complete checkpoint.

## env: stage2a-reliable-transfer@88b93962 (new matched reference experiment)

Spec: X9 `II_package/outputs/stage2a/reliable_transfer_20261027/ENVIRONMENT.json`;
canonical sorted compact JSON hash88b93962. Same pinned vLLM image; isolated
RAM `/dev/shm/stage2a-reliable-20261027/python` adds onlyPEFT0.21.2 with--no-deps.
Resolved accelerate1.14.0/safetensors0.8.0/hub1.27.0; torch/transformers unchanged.
CPU imports, real config architecture (12DINO blocks,252language LoRA modules),
real tokenizer unknown-label masking and two-image batching passed. GPU readiness
is pending. Qwen pinned revision0c351dd01ed87e9c1b53cbc748cba10e6187ff3b is in
read-only `/qwen-model` from hostRAM; DINO mount unchanged. No credentials mount.

New persistent root `/stage3/stage2a_reliable_20261027`; shared inputs in RAM
`/dev/shm/stage2a-reliable-20261027`. Four bounded batches dino/qwen42/qwen43/evaluate,
with4800sJobdeadline and5100swatcher, exactreplica+healthrestore between batches.
First dino batch waits for ADMIT. Fresh independent agent must execute BOTH
commands verbatim, preserve every output, and report mismatches without fixes:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-reliable-dino-20261027 -- python3 /stage3/stage2a_reliable_20261027/train_reliable_dino.py witness'
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-reliable-dino-20261027 -- python3 /stage3/stage2a_reliable_20261027/train_reliable_qwen.py witness'
```

Expected first `DINO_TRANSFER_WITNESS_PASS NVIDIA A30 [2, 7]`, finite loss,
encoder gradient, missing-use zero gradient, saved-state reload equality.
Second `QWEN_QLORA_WITNESS_PASS NVIDIA A30`, finite NF4 forward/backward,
language-adapter gradient, missing-use token mask, adapter save/unload/reload
max error<.05, and real paired-image generation. These are hardware/interface
witnesses, not task quality results. Parent alone admits the matched training.

Gotcha: hf CLI2.3.0 interprets multiple bare --include patterns as filenames;
initial call fetched config only. Use snapshot_download with allow_patterns
list, then verify all4shardSHA256 against official LFS metadata. Do not mistake
successful CLI completion for complete8Bweights. Do not expose/login tokens.

## env: stage2a-reliable-transfer@b63f8a5c (pre-training input repair v2)

V1 independentEuler01a17c2b-0cab-7470-af44-11b4d5e75fb6: DINO exit0,
`DINO_TRANSFER_WITNESS_PASS NVIDIA A30 [2, 7] 1.482421875`; Qwen exit1
beforebackward, mm_token_type_ids remained763 while appended answer made
input_ids/attention_mask775. No training admitted. FirstJob deadline reduced
for immediate guarded restoration; originalrun/artifacts preserved.

V2 uses unchanged dataset, labels, model settings and packages; append text-zero
mm_token_type_ids with assistant answer. Real CPU training_input now also feeds
installed Qwen get_rope_index to check sequence alignment before GPU retry.
Spec under reliable_transfer_20261027/execution_v2/ENVIRONMENT.json,
canonicalhashb63f8a5c. NewPVCroot /stage3/stage2a_reliable_20261027_v2.
The firstv2job is stage2a-reliable-dino-20261027-v2 and againwaits forADMIT.
These commands supersedev1; fresh agent runs BOTH verbatim, nofixes/admission:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-reliable-dino-20261027-v2 -- python3 /stage3/stage2a_reliable_20261027_v2/train_reliable_dino.py witness'
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-reliable-dino-20261027-v2 -- python3 /stage3/stage2a_reliable_20261027_v2/train_reliable_qwen.py witness'
```

Expected sentinels/gradient/missing-label/save-reload contracts remain above.
Parent must verify oldbatchrestoredhealthy before newpause, then await fresh
independent result beforeADMIT. Witness is not task accuracy evidence.

## env: stage2a-reliable-transfer@b63f8a5c — restored cache, resumed batch r1

Same pinned image, packages, code/data protocol6718bd87, and environmentb63f8a5c.
At2026-10-28T01:00:57Z both originalservices were independentlyhealthy1/1/1,
QwenHTTP200, Ollamalist/endpointsvalid, nodepressuresfalse. Priorv2job failed
only on900sADMITtimeout; noADMIT and no runsfolder. All8629persistentinputhashes
and4codehashes passed; RAMinputs restoredfromread-onlyPVC. Volatilemodelweights
were missing and re-downloaded at the same pinned revisions, verified against
all official LFS hashes. CPUrecovery initially had a syntaxerror; fixedbefore
anyservicepause and successful. Originalreceipts retained.

Root remains /stage3/stage2a_reliable_20261027_v2; first resumedJob is now
stage2a-reliable-dino-20261027-v2-r1. Same4800sjob/5100srestorationbound.
Previous independentv2witnesses passed; Qwenstep times5.23/.994/.990s,
reloaderror0, peakreserved23.22GiB. Because declaredvalidationbatch4 was not
exercised by the two-examplewitness, add a narrowbatch4memoryprobe using actual
productionhelpers and a realoptimizerstep. No trainingrecipe ordata changes.

Fresh agent must run ALL THREE commands exactly and sequentially; no fixes,
training admission orservicechanges. Use network-enabled execution for SSH.

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-reliable-dino-20261027-v2-r1 -- python3 /stage3/stage2a_reliable_20261027_v2/train_reliable_dino.py witness'
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-reliable-dino-20261027-v2-r1 -- python3 /stage3/stage2a_reliable_20261027_v2/train_reliable_qwen.py witness'
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-reliable-dino-20261027-v2-r1 -- python3 /staging/qwen_batch4_witness.py'
```

Expected exit0 for each. DINO prints DINO_TRANSFER_WITNESS_PASS NVIDIA A30
[2,7] and finite loss after assertions for encoder gradients, missing-use zero
loss-gradient and exact saved-state reload. Individual gradient numbers/reload
values are not printed. Qwen prints QWEN_QLORA_WITNESS_PASS NVIDIA A30 plus
finite loss and reload_error<.05 after3optimizersteps(includingrole-onlydata),
missing-value tokenmask checks and saved-adapterreload. Details/timings are in
QWEN_WITNESS.json; not allassertionresults are printed. Third prints
QWEN_BATCH4_WITNESS_PASS NVIDIA A30 and JSON(n4,seconds,finite loss,peakallocated/
reservedmemory). These are interface/kernel/memory checks, never accuracy claims.
Report command failures or differences; parent alone admits training.

## env: stage2a-reliable-transfer@b63f8a5c — root-owned cache, retry r2

The r1 batch never entered training: user-owned host /dev/shm weights and worker
vanished after SSH logout, consistent with logind RemoveIPC=yes and Linger=no.
Root-owned files survived. Do not alter global logind settings. CPU helper now
owns only this experiment's RAM files and verifies frozen input/model hashes.
R1 cancelled by unsuspend + activeDeadlineSeconds=1; both original services
independently healthy at2026-10-28T01:19:13Z. No model/data/protocol change.

Fresh agent must run these THREE commands verbatim, sequentially, using
network-enabled execution. Do not edit files, repair commands or admit training:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-reliable-dino-20261027-v2-r2 -- python3 /stage3/stage2a_reliable_20261027_v2/train_reliable_dino.py witness'
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-reliable-dino-20261027-v2-r2 -- python3 /stage3/stage2a_reliable_20261027_v2/train_reliable_qwen.py witness'
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-reliable-dino-20261027-v2-r2 -- python3 /staging/qwen_batch4_witness.py'
```

Expected all exit0: DINO_TRANSFER_WITNESS_PASS NVIDIA A30 [2,7], finite loss;
assertions check encoder gradient, missing-use zero gradient, saved-state reload.
QWEN_QLORA_WITNESS_PASS NVIDIA A30, finite loss, reload_error<.05; assertions
check NF4 backward, language-adapter gradient, role-only unknown token masking,
and real paired-image generation. QWEN_BATCH4_WITNESS_PASS NVIDIA A30, n4,
finite loss and memory statistics. Gradient values need not be printed; report
what is observed, do not infer unseen values. Parent alone handles ADMIT and
bounded service restoration. These checks do not establish model accuracy.

## env: stage2a-reliable-transfer@de6fcdc9 — evaluation precision repair v3

All v2 training and evaluation completed. Services were independently healthy
at2026-10-28T03:17:06Z. A source-validation replay found11/174 use decisions and
1 role decision changed after standalone reload. Installed PEFT prepares
non-quantized parameters as float32 for training, whereas v2 standalone eval
left them bfloat16. V3 uses one shared evaluation_model helper to apply the same
preparation to both zero-shot and adapter evaluation. NF4 compute remains bf16.
Training, adapter weights, data, prompts and selection are unchanged. All v2
outputs remain intact. New results root: /stage3/stage2a_reliable_20261027_v3.
Spec/hash under X9 reliable_transfer_20261027/execution_v3/ENVIRONMENT.json.

Fresh agent must run this command exactly, using network-enabled execution:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-reliable-evaluate-20261027-v3 -- python3 /repair/precision_witness.py'
```

Expected exit0 and EVAL_PRECISION_WITNESS_PASS NVIDIA A30 174, with use0/role0
differences. The script calls the production evaluation_model helper, asserts
all floating non-4bit parameters are float32, generates all174 source-validation
predictions from saved seed42 adapter and requires exact use/role agreement with
training-time validation. No gradient training changed, so this witness targets
only the changed evaluation path. Report observed results without fixes or
inferences about unseen outputs. Do not create ADMIT or change jobs/services.
Parent admits all three corrected evaluations only after PASS and handles the
existing4800sjob/5100srestore bounds. This is repeatability, not task accuracy.


### Corrected reliable-transfer evaluation completed

Job stage2a-reliable-evaluate-20261027-v3 completed2026-10-28T03:49:20Z.
Fresh witness passed; final full evaluation also exactly replays all174 source
validation raw/use/role outputs for BOTH saved seeds. No training artifacts
changed:30 hashes matchv2. All56 collectedfiles(299117399bytes) passSHA256.
Independent sklearn scoring/IDs/source-onlyselection pass. Corrected canonical
outputs:X9 reliable_transfer_20261027/execution_v3/remote_v3.
Original qwen3-14b/ollama restored1/1/1 and independentlyhealthy03:52:28Z;
QwenHTTP200,Ollamalist/endpointsvalid,nodepressuresfalse;resultsreaderSucceeded.
No taskGPUwork pending. Final user report and verification are in
handoff/stage2a_la/analysis/20261027_reliable_transfer/. Do not rerun the
completed-job witness commands above. Current outcomes show partial improvement,
not a robust generalizablefive-usemodel; preserve the source/testboundary.

## env: stage2a-la-frozen-transfer@63a047fb

New inputs and runner, same pinned image/packages/model revisions and corrected
float32 non-quantized Qwen evaluation. Spec is X9 la_transfer_20261027/execution/
ENVIRONMENT.json. Frozen dataset e5cf8f803873897875c84418d15cfbb1343e9be22f5a45c77d1157d220712ad5:
3,332 unchanged LA references;3,157 eligible native Maxar near/wide pairs.
175 small-residential inputs fail context completeness and stay unknown/errors
on the full cohort. Source-trained model/checkpoint/prompt/threshold unchanged.

Staging CPU Pod verifies18,943 input files and the1,485,291,520-byte archive.
Three bounded4800s GPU Jobs: stage2a-la-dinozero-20261028, then qwen42 and qwen43
with the same prefix/date. Each pause has its own5100s restore watcher/snapshot,
and independent service health must pass before the next pause. Dataset/runs
live at /stage3/stage2a_la_transfer_20261027; sources stay read-only logically at
/stage3/stage2a_reliable_20261027_v3, with all trained artifact hashes checked.

Fresh agent: execute this command verbatim using network-enabled execution:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=15 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-la-dinozero-20261028 -- python3 /stage3/stage2a_la_transfer_20261027/evaluate_la_transfer.py witness'
```

Expected exit0: LA_TRANSFER_WITNESS_PASS NVIDIA A30 source_n8. The actual runner
must reproduce saved DINO seed42 validation probabilities within1e-5 and all8
Qwen seed42 validation raw JSON/use/role outputs exactly. It checks frozen
checkpoint hashes and uses corrected evaluation_model. No LA predictions enter
admission. Do not repair files, admit work or change jobs/services. Parent alone
handles ADMIT and restoration. Later identical-code batches use this witnessed
environment; their launcher independently checks health and GPU availability.

## env: stage2a-multicity-fixed15@251f54d6

Canonical env251f54d61f334e56edaacd061c9dc395f56a24f8389b5fbeef567c8337267223.
Same pinned vLLM image, torch2.13.0+cu130/transformers5.15.0, DINO revision and
cached weights. Runtime protocol e5a1d53585d1d7e1f3a07efb23641becb9ba8584443139e226f9a17ba798349b.
CPU staging passed8142runtime hashes,8124images,18943LA hashes and3DINO files.
Task root /stage3/stage2a_multicity_transfer_20261028; A_nyc/B_multicity each
720train plus identical174NYCval. Four last-two-block fits:15epochs/675updates,
seeds42/43, same source-validation selection. LA is development only.

Parent must finish both remaining LA Qwen batches and verify service health
before launching A, then restore/verify health before B. Master launcher and
SPEC_SHA256 live at /tmp/stage2a-multicity-exec-20261028. Each GPU job waits for
an independent witness and parent admission;4800s job/5100s restore watcher.
Do not self-certify the new runner based only on CPU staging.

Fresh independent agent, after parent confirms A is running awaiting admission,
execute exactly:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=15 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-multicity-a-20261028 -- python3 /stage3/stage2a_multicity_transfer_20261028/evaluate_multicity_la.py witness --arm A_nyc'
```

Expected: exit0 MULTICITY_DINO_WITNESS_PASS A_nyc NVIDIA A30 LA2+unknown1.
This checks real training backward/update/reload, missing-use gradient masking,
all frozen files, two real LA predictions and explicit unavailable-input unknown.
No full fit/LA-based selection. Do not modify code, admit work, or change services.
Report any doc-vs-execution mismatch. B has the same code but its own data witness:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=15 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-multicity-b-20261028 -- python3 /stage3/stage2a_multicity_transfer_20261028/evaluate_multicity_la.py witness --arm B_multicity'
```

Run B only when parent confirms its job is waiting. Parent admission uses
worker.py admit A_nyc/B_multicity --reviewer <independent-agent-id>.
