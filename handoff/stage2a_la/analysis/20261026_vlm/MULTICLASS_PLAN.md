# Frozen multiclass / context experiment, 2026-10-26

Actual individual-building use is the target. Five named uses plus unknown, and a separate primary/auxiliary/unknown role. This is a classification experiment on supplied footprints, not a detector/AP experiment.

Artifacts: `/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/vlm_multiclass_20261026`.

Frozen at 21:56:13Z before any model output: 164 inputs, 49 Harris development +115 San Diego transfer. Score eligibility fixed identically across arms: Harris35 (small8/multi6/commercial8/institutional2/other11), SD101 (34/14/26/17/10). Remaining28 are diagnostics. Original candidate labels remain available for sensitivity analysis; no replacement after QC. Permit corroboration is available for21 SD references, none for multifamily. All roles lack independent ground truth; nine Harris visual hypotheses are provisional only.

SD additions were sampled from explicit assessor and agreeing2023/2025 land-use codes, one mapped roof per parcel; official historical permit descriptions and visual geometry checks downgraded conflicting records. These are proxy references, not actual-use Gold. Single-roof spatial selection can still select roof fragments. Harris's old mask_truncated flag misses source-edge clipping. Missing wide-context source pixels are retained and flagged, not invented. Institutional additions are mostly worship; other additions largely industry/warehouses from two new industrial areas. SD2022 image dates differ from source dates; Harris2017 differs from later parcels.

| Arm | Vocabulary | Images | Role + unknown |
|---|---|---|---|
| A | 3 | near | yes |
| B | 5 | near | yes |
| C | 3 | near+wide | yes |
| D | 5 | near+wide | yes |
| E | 3 | near+wide | no, forced choice |

Near=min40m,1.4x target extent; wide=min300m,3x near. Both use native imagery resampled to512px, cyan target outline; footprint area is provided. Harris source availability caps context. Prompts, images, ordering and reference hashes are in PROTOCOL.json. SD never appears in smoke, examples or tuning. Smoke is four Harris buildings across all five arms(20); full matrix820 calls. No subsequent prompt/label adjustment from SD predictions.

Compare D-C vocabulary at fixed views; D-B context at fixed vocabulary; A/B/C/D factorial. E differs from C in role and abstention together, so C-E cannot isolate either. Shared-three subset is primary for vocabulary comparisons. Full-five scores additionally measure naming capability unavailable to three-class arms. Report correct/all, supported-class macroF1/recall, coverage, selective accuracy, and invalid/unknown denominators. Classes with no reference support have N/A recall/F1. Report A official-corroborated separately, spatial groups, and all164 operational counts. No independent-building confidence intervals. Geographic results are a frozen single-city transfer test, not proof of universal generalization or unseen model pretraining.

Execution: A30 24GiB; pinned vLLM image SHA from JOB.json, torch2.13.0+cu130/transformers5.15.0, Qwen/Qwen3-VL-8B-Instruct revision0c351dd01ed87e9c1b53cbc748cba10e6187ff3b; BF16/SDPA/batch1/greedy/160 output tokens. Resources4CPU48GiB, memory-backed model cache32GiB, existing NVIDIA runtime. No packages installed. Code hashes and server-dry-run Job retained. Only image/model-input artifacts are uploaded; reference labels/permit evidence are not uploaded to the inference directory.

User already authorized temporarily pausing qwen3-14b and ollama with automatic restoration. Independent existing restore watcher starts before scaling, snapshotting exact replica counts. Job3900s and watcher4200s bound inference; smoke admission requires projected full job with20% inference margin+90s overhead <3800s. If admission fails, restore before revising execution scheduling; never silently shrink the frozen cohort. Verify both original deployments ready, Qwen health200, Ollama list, endpoints, node memory/disk pressure after completion/failure. Do not finish while services remain paused.

Fresh-agent witness on the actual Job during download, exactly:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-vlm-multiclass-20261026 -- python3 -c "import torch; torch.manual_seed(20261026); x=torch.randn((64,64),device=\"cuda\",dtype=torch.bfloat16); y=x@x; torch.cuda.synchronize(); assert torch.isfinite(y).all(); print(\"WITNESS\", y.shape, torch.cuda.get_device_name(), torch.__version__)"'
```

Independent design reviewer approved the fixed five-arm matrix and scoring masks before inference. Local parser/scoring/image-construction self-checks passed. Production inference, downstream population probabilities, original training labels/weights, existing data and Git history remain untouched.
