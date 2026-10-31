# Qwen3-VL pilot compute record

2026-10-26. User identified the personal card as RTX 3080; its SSH address and VRAM remain unverified. Existing private-address candidates timed out. Local Mac is M1 Pro, 16 GB unified memory. TAMU is one physical A30, 24 GiB VRAM (not two GPUs despite two Kubernetes allocation units), with about 179 GiB host memory available. At preflight the Qwen service used 14.6 GiB VRAM; root and /home each had about 14 GiB free.

Choose A30 for the frozen BF16 pilot. Model tensor bytes are 17,534,247,392 (16.33 GiB), before activation/cache/runtime overhead. Reserve 48 GiB host RAM and place the download in a 32 GiB memory-backed emptyDir. Reuse the existing pinned vLLM container image without installing packages. Save predictions to the dedicated user directory and copy them to X9. Restore both original LLM replica counts using the existing independent watcher.

The completed environment specification is `II_package/outputs/stage2a/vlm_pilot_20261026/ENVIRONMENT_v2.json` on X9; `ENVIRONMENT.json` preserves the failed first attempt. The model revision is `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Inputs are 296 fixed buildings, two 256px images per building, batch size 1, BF16, SDPA, greedy decoding and at most 160 output tokens.

Completed on 2026-10-26: all 296 predictions were valid. Framework peak allocated/reserved VRAM was 16.44/16.50 GiB; a whole-device `nvidia-smi` observation was 17,495 MiB (17.09 GiB, a snapshot rather than the device peak). Median latency was 2.35 s/building, mean 2.50 s, p95 3.13 s; inference totaled 12.35 minutes. Download took 172.06 s, loading 8.87 s, and the successful Job took 15m43s including setup. These are measurements for the fixed input/output limits above, not training requirements or guarantees for larger images/batches.

Both original services were verified restored at 2026-10-26T19:07:24Z: desired/ready/available replicas 1/1/1, Qwen health HTTP 200, Ollama model listing successful, and ready service endpoints present. The watcher reported `scaled_back=true, ready=true`; node disk/memory pressure remained false. See [REPORT.md](REPORT.md) for accuracy, interpretation and artifact links. The frozen zero-shot configuration is not suitable as a direct Stage2a replacement on these diagnostic samples.

## Existing-image kernel witness

Run this exact command before pausing the original Qwen deployment. It checks the already-running image; it does not load another language model.

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n llm-inference deployment/qwen3-14b -- python3 -c "import torch; torch.manual_seed(20261026); x=torch.randn((64,64),device=\"cuda\",dtype=torch.bfloat16); y=x@x; torch.cuda.synchronize(); assert torch.isfinite(y).all(); print(\"WITNESS\", y.shape, torch.cuda.get_device_name(), torch.__version__)"'
```

Observed: `WITNESS torch.Size([64, 64]) NVIDIA A30 2.13.0+cu130` and exit 0. This only confirmed kernel compatibility in the existing deployment; the completed pilot and its run record provide the model validation.

Local self-check: `/Users/zhou/miniconda3/envs/py311/bin/python handoff/stage2a_la/vlm_pilot.py self-check` from the repository root. It covers target marking, invalid-output handling and denominator preservation. No existing inference code, checkpoints or labels are changed.

## Corrected Job witness (v2)

Attempt 1 omitted `runtimeClassName: nvidia`; the original deployment witness did not establish GPU visibility in that new Job. It failed before predictions, and both LLMs were restored. Attempt 2 explicitly sets the existing NVIDIA runtime, checks CUDA before downloading, and preserves the same inputs/model/prompt. Its specification is `ENVIRONMENT_v2.json` on X9. Run this exact command against the actual new Job while it downloads:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 k3s-master 'kubectl exec -n instance-impact-stage3 job/stage2a-vlm-pilot-20261026-v2 -- python3 -c "import torch; torch.manual_seed(20261026); x=torch.randn((64,64),device=\"cuda\",dtype=torch.bfloat16); y=x@x; torch.cuda.synchronize(); assert torch.isfinite(y).all(); print(\"WITNESS\", y.shape, torch.cuda.get_device_name(), torch.__version__)"'
```

The actual-Job witness returned the same successful output above and superseded the original deployment-only check. The independent restore watcher for v2 ran as PID532803 on master, with status/log prefix `/tmp/stage2a-vlm-20261026-v2`, and completed restoration. Job `stage2a-vlm-pilot-20261026-v2` ran from 18:48:38Z to 19:04:21Z and succeeded. Failed-attempt artifacts remain preserved; no prompt or sample selection changed between attempts.
