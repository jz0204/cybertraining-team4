# Frozen VLM diagnosis, 2026-10-26

The user authorized model tests and clarified that the target is each building's actual use, with auxiliary structures separately flagged. Keep the original parcel-derived reference for paired diagnostics; do not equate agreement with these proxies to actual-use accuracy or relabel the pilot based on predictions.

All 56 commercial/multifamily references predicted as small residential were visually inspected before the new outputs. Smaller target footprints, recognizable commercial/apartment context, ambiguous roofs, occlusion, and likely auxiliary structures all occur. The original 256px Harris crops match native imagery exactly; all 21 Harris errors retain the entire known footprint. A border-touching mask is not evidence that the crop cut the known building. Original image boundaries can still hide unobserved structure.

| Arm | Rows | What changes |
| --- | ---: | --- |
| Original256 / P0 | cached296 + replay6 | Frozen baseline; six-row execution check |
| Upsampled512 / P0 | 296 | Same pixels and prompt; bicubic interpolation and larger visual encoding budget |
| Original256 / P1 | 296 | Compare visible cues across classes; preserve building-level use definitions |
| Upsampled512 / P1 | 296 | Completes the resolution-by-prompt factorial |
| Native512 context / P1 | Harris90 | Wider native field of view at matched image dimensions |

There are 984 main/replay inferences and 24 additional smoke inferences, 1008 total. Each non-replay arm first runs six rows with its actual input and prompt. Same model revision, BF16, batch1, SDPA, greedy decoding and 160-token limit. No prompt search, demonstrations, training or confidence threshold. P1 does not require readable business signs or countable housing units and does not automatically classify auxiliary roofs as dwellings. Full prompt strings are frozen in `PROTOCOL.json`.

Upsampling cannot add source detail. An improvement identifies sensitivity to visual presentation/encoding, not a newly resolved source feature. The wide-context arm jointly changes context, target scale and target position: 73/90 windows shift when constrained to the native image, and 17 stay concentric. It cannot isolate context alone. LA native images are unavailable in the supplied crop archive, so do not fabricate wider LA views. A separate tight-zoom arm was omitted because it adds another context/presentation tradeoff.

Report every group and class, baseline-error recovery, new errors on the 180 originally correct small-residential samples, and changes grouped by cell/tile and shared parcel. Compare P1 at both resolutions and upsampling under both prompts. Evidence flags are outputs, not a calibrated filter. All 296 rows are development data; any final candidate needs a frozen comparison on genuinely new spatial regions/cities. Repartitioning these inspected rows does not create an independent test.

The reviewer agent approved the bounded factorial and the explicitly joint context comparison. The initial compute estimate is 45–60 minutes; update it from 512px smoke measurements. Job deadline is 4800s, with a separate 5100s restoration watchdog and the original 90-minute disruption bound. The established automatic watcher restores exact original replica counts. No persistent model weight download, package installation or existing model replacement.

Environment: reuse the previous pinned vLLM image, `runtimeClassName: nvidia`, A30 24GiB, 4CPU/48Gi host RAM, 32Gi RAM weight cache. Canonical `ENVIRONMENT.json`, `job.json`, `worker.py` and label-free inputs are in the X9 diagnosis directory. Remote output is `/home/zhjyu11/stage2a-vlm-diagnosis-20261026` on `tamu-gpu`.

Actual-Job witness, to execute verbatim after the new Job starts:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-vlm-diagnosis-20261026 -- python3 -c "import torch; torch.manual_seed(20261026); x=torch.randn((64,64),device=\"cuda\",dtype=torch.bfloat16); y=x@x; torch.cuda.synchronize(); assert torch.isfinite(y).all(); print(\"WITNESS\", y.shape, torch.cuda.get_device_name(), torch.__version__)"'
```

Expected: `WITNESS torch.Size([64, 64]) NVIDIA A30 2.13.0+cu130`, exit0. This checks actual-Job CUDA visibility; completion, output coverage, paired metric verification and service restoration still require separate checks.

Local checks already passed: `vlm_pilot.py self-check`, `vlm_diagnosis.py self-check`, 90/90 native RGB and rendered-mask parity assertions, all56 visual-audit IDs covered, and uploaded archive excludes references. Model inference still needs runtime validation.
