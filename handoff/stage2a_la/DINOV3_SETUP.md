# TAMU 上的 DINOv3

2026-10-27：已配置成功。账号访问、官方权重校验、A30 上的前向计算，以及本轮冻结特征提取均已通过。用户现在无需再安装或登录。

## 登录与下载

主机只安装了独立的 Hugging Face CLI，不改系统 Python。以后检查登录：

```sh
ssh tamu-gpu
~/stage2a-hf-tools/bin/hf auth whoami
```

仅在登录失效时执行 `~/stage2a-hf-tools/bin/hf auth login`，在终端输入 token。账号须已获 [官方模型仓库](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) 的访问权限；token 不放进代码、聊天或训练容器。CLI 用法见 [Hugging Face 官方文档](https://huggingface.co/docs/huggingface_hub/en/guides/cli)。

本轮固定 `facebook/dinov3-vitb16-pretrain-lvd1689m`，revision `5931719e67bbdb9737e363e781fb0c67687896bc`。如主机重启后需要恢复 RAM 中的权重，可在已登录的 TAMU 主机运行：

```sh
~/stage2a-hf-tools/bin/hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  config.json preprocessor_config.json model.safetensors \
  --revision 5931719e67bbdb9737e363e781fb0c67687896bc \
  --local-dir /dev/shm/stage2a-dinov3-model-20261026
```

`model.safetensors` 为 342,662,192 bytes，SHA256 为 `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`，已与官方 LFS 元数据核对。RAM 路径重启后不会保留；本轮已将三个文件持久保存到 X9 的 `II_package/outputs/stage2a/generalizable_pilot_20261026/execution_v2/pretrained/dinov3/`。

## 模型环境与实际用法

Torch/Transformers 使用已有容器环境：torch `2.13.0+cu130`、transformers `5.15.0`。镜像固定为 `vllm/vllm-openai:v0.27.1@sha256:c2f3b1b964e47809b722b5e75b61b1e7b39a50f70388cf2bf2418f16a9f31da2`。这是复用现成 Python/CUDA 环境，DINO 本身没有启动 vLLM 服务。不要把以下代码直接当作 TAMU 主机系统 Python 的命令。

实验容器将上述权重目录只读挂载到 `/dinov3-model`，无需容器内登录。已验证的加载方式：

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "/dinov3-model", local_files_only=True, torch_dtype=torch.bfloat16
).cuda().eval()
with torch.inference_mode():
    output = model(pixel_values=torch.randn(2, 3, 224, 224, device="cuda", dtype=torch.bfloat16))
assert output.last_hidden_state.shape == (2, 201, 768)
assert torch.isfinite(output.last_hidden_state).all()
```

独立复核在实际 A30 Job 中输出 `DINO_WITNESS_PASS NVIDIA A30 [2, 201, 768]`。具体 Job、资源和自动恢复流程见仓库 `.aris/compute/tamu-gpu.md`，原始记录在 X9 `execution_v2/INDEPENDENT_WITNESS_v3.json`。

DINO 提供视觉特征，建筑用途由另外训练的分类头预测。本轮使用 224px RGB、ImageNet 归一化、目标 mask 的 area pooling、宽图平均池化及四个几何量；冻结 DINO，只训练小头。角色没有可靠监督，因此保持 `unknown`。实际提取 1,790 栋双视野特征，batch=32 栋，PyTorch peak reserved 为 0.586 GiB；这不是整卡占用、端到端耗时或 DINO 微调所需显存。

原始特征、模型和数据版本均已冻结。后续运行应复用这些版本；本轮分类结果与论文泛化边界见 [实验报告](analysis/20261027_generalizable_pilot/RESULTS.md)。
