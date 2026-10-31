# Target localization and role probe

After the frozen P1 prompt changed all 296 explanations but zero class labels, a separate 12-case probe tests localization, visible description and building role with unknown/other allowed. This is an adaptive diagnostic follow-up, not another accuracy benchmark or evidence of generalization.

Fixed IDs: 4,9,19,35,239,250,277,16,85,83,232,36. Eight hard/ambiguous cases and four previously correct controls; selected before probe outputs. Reuse the same two upsampled512 images and physical footprint area. Do not input labels or expected bounding boxes. The requested output order is normalized target box, roof form, primary/auxiliary/unclear role, visible evidence, then use. These output/prompt changes are joint and cannot isolate ordering alone.

Target-box IoU is measured against the existing mask bbox. It measures whether the model located the outlined target, not whether the source segmentation is correct. Visual role/roof audits are provisional and are not independent human ground truth. Unknown is permitted, and no sample is dropped from reporting.

Reuse the same image, BF16, SDPA, batch1, greedy decoding and160-output-token limit. New Job `instance-impact-stage3/stage2a-vlm-grounding-20261026` has900s deadline, with independent1200s restore watchdog. It starts only after the factorial Job has completed and both original LLMs have been verified restored, to avoid overlapping restoration controllers. The previous512 images mount read-only at `/inputs`; outputs go to `/home/zhjyu11/stage2a-vlm-grounding-20261026`.

Actual-Job CUDA witness, run verbatim after startup:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-vlm-grounding-20261026 -- python3 -c "import torch; torch.manual_seed(20261026); x=torch.randn((64,64),device=\"cuda\",dtype=torch.bfloat16); y=x@x; torch.cuda.synchronize(); assert torch.isfinite(y).all(); print(\"WITNESS\", y.shape, torch.cuda.get_device_name(), torch.__version__)"'
```

Expected: `WITNESS torch.Size([64, 64]) NVIDIA A30 2.13.0+cu130`, exit0. Model output, box/role schema, completion and service restoration are separate checks. Local self-check already passed valid normalized boxes, auxiliary/other schema, malformed-output handling and prompt formatting.

Actual execution: the main ablation completed at20:27:01Z. Both original services
were independently verified healthy at20:31:43Z (one ready/available replica each,
Qwen HTTP200, Ollama model list, ready service endpoints). The new restoration
watcher PID548026 was started before pausing services. GPU memory was0MiB before
the probe was unsuspended. Fresh witness agent Peirce executed the documented
command verbatim and returned the expected CUDA output with exit0. No service
mutation or file edit was performed by the witness.

Completed20:37:19Z,12/12 valid records and no token-limit truncation. Raw records
match the streamed log. Bbox IoU median0.668,11/12>=0.5 against existing mask
bboxes; this is grounding, not segmentation accuracy. Auxiliary roles8/12,
unknown uses0/12, and two of four previously correct controls changed to other.
Results are diagnostic only. Automatic restoration and actual endpoint health
were independently verified20:42:33Z, with both original deployments1/1 ready
and available, Qwen health200, Ollama model list, ready endpoints and no node
memory/disk pressure. See the probe's RESULTS.json, VERIFY.json and
LLM_RESTORATION_VERIFIED.json. No further inference or service pause is pending.
