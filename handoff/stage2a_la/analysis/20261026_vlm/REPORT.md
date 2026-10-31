# Stage2a：Qwen3-VL-8B-Instruct 零样本试验

2026-10-26，296 栋已全部完成。**TAMU A30 足以运行本轮 BF16 推理，但这套固定零样本配置不适合直接替换当前 Stage2a。** VLM 减少了住宅被判为商业的误报，同时大量漏掉商业和多户住宅。结论仅针对本轮输入、提示词及开发样本。

| 固定样本与指标 | 当前五模型 CNN 集成 | N1 seed42 | Qwen3-VL |
| --- | ---: | ---: | ---: |
| Harris 均衡 90 栋：macro-F1 | 0.9547 | 0.9319 | **0.7348** |
| Harris 均衡 90 栋：准确率 | 86/90（95.6%） | 84/90（93.3%） | 67/90（74.4%） |
| LA 42 栋商业：召回率 | 40/42（95.2%） | 37/42（88.1%） | **16/42（38.1%）** |
| LA 14 栋多户住宅：召回率 | 4/14（28.6%） | 3/14（21.4%） | **3/14（21.4%）** |
| 选定的 34 个 CNN 商业误报：分类正确数 | 0/34 | 23/34 | 34/34 |
| LA 随机 120 栋：准确率 | 101/120（84.2%） | 115/120（95.8%） | 117/120（97.5%） |

LA 随机组的 97.5% 来自 **120 栋全部预测为小型住宅**：该组参考标签为 117 个小型住宅、3 个商业，没有多户住宅，商业全部漏检。这与“全部预测多数类”的基线相同，不能据此宣称三类识别优于 CNN。34 个误报样本是在模型运行前按现有 CNN 错误选出的，34/34 的纠正率也不能作为总体准确率。

Harris 的多户住宅召回从 CNN 的 30/30 降至 VLM 的 13/30；VLM 将其中 16 栋判为小型住宅。全部 296 个输出均为有效 JSON，且全部自报 `evidence_sufficient=true`，包括与参考标签不一致的 60 栋；本轮的自报证据标记无法筛出错误。图像是否足以区分真实用途、参考标签与建筑实际用途是否一致，仍需进一步核实。

模型为 [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)，固定 revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。每栋输入原始 256×256 RGB、同一图像的青色目标轮廓版本，以及真实尺度的 footprint 面积；目标可能偏离图像中心。未输入城市名、参考标签或 CNN 预测。三类定义沿用数据集，duplex/triplex 归小型住宅，4 户及以上归多户住宅。BF16、batch=1、SDPA、贪心解码、最多 160 输出 token，无微调，未依据输出调整提示词。

固定 seed=20261026。分组为 LA 随机 120、冻结 LA primary 参考集内全部 42 商业和 14 多户住宅、34 个选定商业误报，以及 Harris validation 每类 30 栋。分组有重叠，去重后为 LA 206 + Harris 90 = 296，不能把富集类别和误报组混合成总体准确率。LA 参考来自 assessor parcel 用途映射，未经逐栋人工确认；Harris validation 曾参与 CNN 模型选择。所有结果是开发诊断，未访问 San Diego 独立测试。

| A30 上的实际测量 | 数值 |
| --- | ---: |
| BF16 模型权重 | 17,534,247,392 bytes（16.33 GiB） |
| PyTorch 显存峰值：allocated / reserved | 16.44 / 16.50 GiB |
| 整卡显存观测值（非整卡峰值） | 17,495 MiB（17.09 GiB） |
| 单栋耗时：中位数 / 平均 / p95 | 2.35 / 2.50 / 3.13 秒 |
| 296 栋推理耗时合计 | 12.35 分钟 |
| 下载 / 模型加载 | 172.06 / 8.87 秒 |
| 成功 Job 全程（含准备） | 15 分 43 秒 |

本轮选 TAMU 的一张 A30 24 GiB。申请了 4 CPU、48 GiB 主机内存，权重下载至 32 GiB 的内存盘；48 GiB 是资源申请量，不是实测最低需求。root 与 /home 各仅约 14 GiB 空闲，不能在那里存放原始 BF16 权重。本地 M1 Pro 的 16 GB 统一内存不足以承载本轮未量化权重及运行开销。个人 RTX 3080 的 SSH 候选地址仍超时，实际显存和量化运行能力未验证。上述实测不适用于训练、更大图片、更大 batch 或更长输出。

首次 Job 因遗漏 `runtimeClassName: nvidia`，虽申请 GPU 却未向容器暴露 CUDA，下载后在推理前失败。第二次在实际 Job 内先做 BF16 CUDA 核验，输入和模型保持不变，18:48:38–19:04:21 UTC 成功完成。两次运行的环境、日志与 Job 配置均保留。**19:07:24 UTC 已确认 qwen3-14b 和 ollama 自动恢复至 desired/ready/available=1/1/1**；Qwen `/health` 返回 200，Ollama 模型列表成功，相关服务有 ready endpoints，节点无磁盘或内存压力。未重测外部路由。

核验复用了冻结的原评估器，四组 × 三模型的准确率、macro-F1、逐类指标和混淆矩阵均一致（容差 1e-12）；296 个唯一 ID、原始响应、日志与本地备份逐条对应。输入标注与 JSON 解析的自检已通过。没有修改现有训练/推理入口、模型权重或标签。

完整记录位于 [X9 实验目录](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/vlm_pilot_20261026>)：

- [RESULTS.json](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/vlm_pilot_20261026/RESULTS.json>)、[逐栋对照](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/vlm_pilot_20261026/paired_predictions.json>)、[固定协议](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/vlm_pilot_20261026/PROTOCOL.json>)。
- [资源与耗时](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/vlm_pilot_20261026/PERFORMANCE.json>)、[核验记录](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/vlm_pilot_20261026/VERIFY.json>)、[服务恢复记录](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/vlm_pilot_20261026/LLM_RESTORATION_VERIFIED.json>)。
- [六个诊断例图](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/vlm_pilot_20261026/DIAGNOSTIC_EXAMPLES.png>)按结果选取，用于展示错误形态，不代表总体分布。

本轮没有测试量化、微调、放大目标裁剪或独立地区，也没有验证 VLM 与 CNN 自动融合规则。继续研究应优先解释商业和多户住宅漏检，并保留独立评估数据；当前结果不足以推广为所有 VLM 路线无效。
