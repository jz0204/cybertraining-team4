# Stage2a 建筑用途泛化：文献与下一步判断

核查日期：2026-10-26。目标为单栋实际用途五类、主/附属角色和拒答；不是城市功能区或整幅场景分类。此轮完成原始来源阅读与本地数据盘点，没有训练新模型。推荐为研究判断，尚未经过新的对照实验验证。

## 当前判断

最值得先投入的是 **补足建筑级、多地区监督，同时比较 ConvNeXt 与冻结视觉基础模型**。VLM 零样本失败不代表 SFT 无效；Qwen QLoRA 值得保留为同数据对照。当前不宜从 GRPO、长推理链或大模型全参微调开始。标签归属、类别覆盖和新城市评估是三条路线共同的先决条件。

模型能力与可观测信息要分开：同类屋顶不一定能告诉我们单栋究竟有 3 户还是 4 户，或者某座普通建筑内部是否用于宗教活动。若核实后的错误仍集中在这种情况，应量化额外楼栋记录、街景或高度的价值，并允许拒答；不能靠解释更长替代证据。

## 原始来源与可借鉴之处

| 工作 | 方法与已报告证据 | 对我们有何帮助；不能外推什么 |
|---|---|---|
| Skuppin, Shi & Zhu, **Geographic transferability of building function models**, IEEE JSTARS, 2026；[论文记录](https://elib.dlr.de/227125/)，DOI 10.1109/JSTARS.2026.3737186 | 六国、18 个测试城市；Planet RGB 与 OSM footprint，U-Net；含独立测试城市和留国家实验。联合多地区训练、输入 footprint 值得重视 | 最贴近泛化目标。但住宅/非住宅二分类、建筑像素评分；不是五类单栋任务。附属车棚/garage 等未作为有标签类别学习 |
| Liang, Yap & Biljecki, **Heterogeneous graph neural networks for building attribute prediction from hierarchical urban features and cross-view imagery**, ISPRS JPRS 234, 2026；[全文](https://ual.sg/publication/2026-ijprs-gnn/2026-ijprs-gnn.pdf)，DOI 10.1016/j.isprsjprs.2026.02.016 | DINOv3 特征、建筑/街道/地块上下文、街景；Amsterdam/DC/Berlin 的完整模型 macro-F1 为 .70/.73/.71 | 支持复用冻结视觉特征和上下文；60/20/20 城内随机划分不是跨城迁移。先借鉴特征与输入设计，尚无理由直接复制整套 GNN |
| Su et al., **BuildingSense**, ESSD 18, 2026；[全文/数据链接](https://essd.copernicus.org/articles/18/2609/2026/)，DOI 10.5194/essd-18-2609-2026 | NYC 34,458 栋、26 个功能类别；屋顶/周边影像、街景、POI、楼栋属性 | 可扩充来源类别及补充模态，但只是一个城市。我们已使用其 1000 候选中的 578 栋，不是尚未尝试的新来源；来源标签也须核实到目标楼栋 |
| Siméoni et al., **DINOv3**, 2025 technical report；[全文 §8](https://arxiv.org/html/2508.10104v1#S8)，arXiv:2508.10104；[官方权重列表](https://github.com/facebookresearch/dinov3) | SAT-493M 使用 0.6m Maxar RGB；有 300M ViT-L 卫星权重。冻结 backbone + 下游小头可做多种任务；Web 版本也有强语义表现 | 与现有 VHR RGB 输入匹配，适合先缓存特征做便宜对照；不能预设 SAT 优于 Web，也不能把其他任务的成绩当用途性能 |
| Reed et al., **Scale-MAE**, ICCV 2023；[论文](https://openaccess.thecvf.com/content/ICCV2023/html/Reed_Scale-MAE_A_Scale-Aware_Masked_Autoencoder_for_Multiscale_Geospatial_Representation_Learning_ICCV_2023_paper.html)，arXiv:2212.14532 | 利用真实地面尺度进行位置编码和多尺度重建；跨尺度分类及 SpaceNet 分割有改善 | 记录原生 GSD/物理窗口；近景与邻域保留不同信息。不必马上重新预训练 Scale-MAE，重采样成更大像素也不会增加真实细节 |
| Bastani et al., **SatlasPretrain**, ICCV 2023；[论文](https://arxiv.org/abs/2211.15660) | Sentinel-2 与 NAIP，多任务监督，含 1m 航片 | 可作为影像域更匹配的编码器候选。多任务遥感预训练不是现成的五类用途真值，先完成少量核心模型再扩候选 |
| Liu et al., **RemoteCLIP**, IEEE TGRS, 2024（2023 预印本）；[论文](https://arxiv.org/abs/2306.11029) | 统一影像/文本表征；支持线性探测、少样本分类等，评估多个遥感数据集 | 可用作非生成式视觉语言表征。其场景分类/检索增益不等于可辨识单栋户数；当前不再扩大零样本 prompt 搜索 |
| Wang, Fang & Luo, **Steering Instruction-Tuned Vision-Language Models for Multi-Label Landcover Classification**, WACV Workshops 2026；[论文](https://openaccess.thecvf.com/content/WACV2026W/CV4EO/papers/Wang_Steering_Instruction-Tuned_Vision-Language_Models_for_Multi-Label_Landcover_Classification_WACVW_2026_paper.pdf) | BigEarthNet RGB，10k 训练/1k 测试；Qwen3-VL-4B 的 QLoRA rank128 将 macro-F1 .253→.612，rank4 已达 .503 | 直接支持“零样本弱仍值得试 SFT”；类别分层抽样而非所需的城市外测试，不能承诺同等提升。实用设置需按 A30 改小并实测 |
| Köksal & Alatan, **Few-Shot Vision-Language Reasoning for Satellite Imagery via Verifiable Rewards**, ICCV Workshops 2025；[全文](https://arxiv.org/html/2507.21745v1)，arXiv:2507.21745 | Qwen2-VL-2B，1–128 个训练例，GRPO，用正确性/IoU 奖励；涵盖场景分类、VQA、grounding | 给 RL 一个可验证奖励的方向，但少样本不等于低算力；论文使用 16–128 H100、长序列，并选择所评 checkpoint 中较好者。不能照搬为单 A30 的严格盲测方案 |
| Huang et al., **Urban Building Classification (UBC)**, CVPR Workshops 2022；[论文](https://elib.dlr.de/191304/1/Huang_Urban_Building_Classification_UBC_-_A_Dataset_for_Individual_Building_CVPRW_2022_paper.pdf)，DOI 10.1109/CVPRW56347.2022.00147 | Beijing/Munich、41,586 个实例，0.5/0.8m；粗/细用途体系 | 是更接近单栋的候选来源，但粗类把 garage 归住宅；公开 fine-label 完整性尚未核实。随机 tile 划分和 mask AP 不直接回答跨城分类 |
| Li et al., **Satellite mapping of every building’s function in urban China reveals deep built environment disparities**, Nature Communications 17, 2026；[全文](https://www.nature.com/articles/s41467-026-69589-5)，DOI 10.1038/s41467-026-69589-5 | SinoBF-1：109 城、约 1.1 亿建筑输出；RGB+高度+夜光、八类、CNN/ViT；用土地用途/AOI 叠置生成弱标签，POI 多数一致性验证 | 支持扩大地理覆盖及独立辅助信息，但输出是单栋不代表监督就是单栋实际用途；不能把这类弱标签用作解决车棚错误的独立真值 |
| Gulrajani & Lopez-Paz, **In Search of Lost Domain Generalization**, ICLR 2021；[论文](https://arxiv.org/abs/2007.01434) | DomainBed 比较强调模型选择协议；良好实现的 ERM 是强基线 | 先把训练数据、预算、选模和留域规则做一致；无需预先增加复杂域对抗方法。属于通用方法学依据，不是建筑识别实验 |

## 重点阅读核查

**跨地域建筑用途论文。** 接受稿 §III–IV、Table II 和 Appendix A 是关键。多地区联合训练的价值比“每个城市都单独微调”更值得先检验；但不能越过二分类、像素评分和 OSM 标签边界。论文将 carport/garage/shed/roof 等保留为未标注，正是我们必须另行处理的目标对象。商用影像不能直接重新分发；作者代码/标签为合理请求可得，本轮未联系作者。

**DINOv3 与建筑属性工作。** DINOv3 原报告 §8.1/8.3 支持 VHR 预训练和冻结特征路线，但指出 Web 权重在若干语义任务也更强。Liang et al. §4.2、Table 5、§6.2/Table 7 的编码器比较更贴近本任务；其 DINOv3-B 为 86M，特征先提取缓存。三个城市的测试来自各城随机划分，不能把 .70–.73 称作新城市泛化。用轻量特征融合先验证，若不足再决定图模型是否必要。

**Qwen SFT。** 已读取官方 PDF pp.1477–1479、1481 的训练设置、结果和限制。10k 与 1k 是类别分层、互不重叠的样本池；训练池另划 validation。论文训练用单张 A100 80GB，rank128、训练至多1000步；因此“单卡”不是“24GB 已验证可跑”。rank4 的显著收益使小适配值得试。输出标签适配并不自动生成可信置信度。

**RLVR。** 读全文 §3–4，尤其 §4.1–4.2：少数例重复组成128的批次、4条采样回答、prompt/completion 上限各8192、1000–2000步，以及跨checkpoint择优，均会影响计算量与结果解释。应借鉴可核验 reward 的原则，不能用“一条样本就够”的摘要叙述设计我们的五类实验。

## 与我们已有证据衔接

本地重新盘点并保存 [数据清单与哈希](../handoff/stage2a_la/analysis/20261026_generalization/TRAINING_DATA_INVENTORY.json)：Harris clean train 13,333 栋，11,363 small / 1,097 multi / 873 commercial；NYC 578 栋，445 / 43 / 90。两个训练来源都没有 institutional/other，也没有可靠角色监督。因此原 N2 结果测试的是少量、偏住宅的第二城市数据，不代表完整多城市、五类训练已经失败。

本轮五类 VLM 在 SD 的 multi 0/14、institutional 3/17、所有角色 primary，以及旧模型的高置信错误，都指向监督与校准缺口。另一方面，双视野确有成对改善，所以值得继续作为输入对照。不能把“多户被错判商业”等问题都归因于小裁剪，不能把地块类别的改善等同于单栋用途改善。详见 [MULTICLASS_REPORT.md](../handoff/stage2a_la/analysis/20261026_vlm/MULTICLASS_REPORT.md)。

## 执行顺序与论文可检验问题

先审核可复用的 NYC institution/industry 等候选、补多户与附属角色；然后用相同监督与空间分组比较五类 ConvNeXt、冻结 DINOv3-B、Qwen8B QLoRA SFT。只有领先方向再升级容量或解冻。实施矩阵、拒答、城市划分、算力和停止条件见 [NEXT_MODEL_PLAN.md](../handoff/stage2a_la/analysis/20261026_generalization/NEXT_MODEL_PLAN.md)。

文章有价值的问题是：纠正地块到楼栋的标签归属、引入可迁移视觉特征及真实上下文，能否在独立城市降低少数类和附属结构错误，并改善类型面积/暴露汇总？需要用同数据消融与未用于调参的城市验证。这里提出的是可检验方向；没有宣称该方向独创、必然达到某精度或足以发表。

## 检索记录与未完成项

- 来源：现有项目报告、原论文/出版方/作者机构全文、作者 GitHub 文档。一个只读检索分支负责直接建筑用途文献；按 DOI/arXiv 去重后保留12项。未配置 Zotero/Obsidian，项目无现成 PDF 库；arXiv helper 不存在，使用网页原始来源回退。
- 主要检索式：`building function classification cross city`；`geographic transferability building function`；`remote sensing foundation model building function DINO`；`remote sensing VLM SFT GRPO`；`Qwen3-VL landcover classification`；以及各论文标题的精确检索。范围截至2026-10-26，本轮不是穷尽性系统综述。
- 阅读层级：直接建筑用途4篇由检索分支读原文并提取协议；主流程复核跨地域论文记录、HeteroGNN方法/结果、SinoBF监督/验证；另读 BuildingSense、DINOv3 §8、Qwen SFT全文相关部分和RLVR方法/实验。Scale-MAE、SatlasPretrain、RemoteCLIP、DomainBed只作原文摘要/公开方法层级的候选依据，不声称复现或全面评审。
- Qwen官方训练文档支持多图和 LoRA；[文档](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md)的存在不是我们环境兼容性或显存通过证据。
- 部分CVF网页抓取失败；Qwen论文改从公开官方PDF下载后用已有`pdftotext`读取，未安装依赖。新模型权重访问授权、训练兼容性、新城市标签和独立审核均未完成。
- 后续代码必须使用X9已修复area pooling的实验副本，不能误用handoff里旧nearest实现。此次只新增文档和数据盘点，不改生产训练/推理代码、旧标签或权重。


## 2026-10-27 实验期间的方法核对补记

针对 Wang et al. 的 QLoRA 工作，本次重新核对 CVF 收录的 §3–4 原文索引。
直接 PDF 获取返回 403；本次读取的是检索工具返回的原论文方法与优化段落，
不将这次访问描述为再次完整下载全文。来源仍为上表的 CVF 论文链接。

作者在视觉与语言两侧的 attention/MLP 中加入适配器，报告学习更新主要集中在
语言模块。其训练池约 10k，主要设置 rank128、最多1000个优化步骤，硬件 A100
80GB。本轮只有720栋 NYC 训练样本、语言层 rank8、两轮共90次优化器更新，
因此是成本受限的任务试验，不是该论文复现。也不能把本轮结果归因于缺少视觉
适配：这需要独立消融，论文中更新主要集中于语言层亦不证明视觉层可以省略。
原文 §3 与 §4 对 dropout 分别写0.0与0.05；若复现其设置，应核对实现。

本轮测试城市不用于调整适配器、提示词或训练预算。后续修改方案需要新的开发
协议及未被这些错误分析使用的最终测试区域。
