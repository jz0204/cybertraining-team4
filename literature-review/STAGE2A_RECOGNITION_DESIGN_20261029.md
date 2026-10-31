# Stage2a：识别瓶颈对应的方法与创新边界

日期：2026-10-29；任务为单栋实际用途及主/附属角色。延续[已有综述](STAGE2A_GENERALIZATION_20261026.md)，根据新LA四户实验的漏判/误报做定向补查。使用research-lit流程；无本地PDF库、Zotero/Obsidian会话或可用arXiv helper，改读出版方、作者及机构原始网页/PDF。以下不是穷尽性新颖性审查，也没有复现这些论文。

## 原始来源与本次阅读深度

|工作与来源|本次读到的内容|对我们的启发及边界|
|---|---|---|
|Liang, Yap & Biljecki. *Heterogeneous graph neural networks for building attribute prediction from hierarchical urban features and cross-view imagery*. ISPRS JPRS 234, 2026. [作者全文](https://ual.sg/publication/2026-ijprs-gnn/2026-ijprs-gnn.pdf), DOI10.1016/j.isprsjprs.2026.02.016|全文相关方法、§4.2、Table5与§6消融；同行评审期刊论文|已有DINOv3、建筑/道路/地块异构关系、航片与街景融合。完整模型三城macro-F1约.70/.73/.71，但使用60/20/20随机划分，不能对照成我们未见城市的收益。增加GNN、街景或基础模型本身不构成新点。|
|Su et al. *BuildingSense: a new multimodal building function classification dataset*. ESSD18, 2026. [出版方全文](https://essd.copernicus.org/articles/18/2609/2026/), DOI10.5194/essd-18-2609-2026|摘要、数据获取/核查、§3.3与§4相关段落；同行评审数据论文|34,458栋NYC建筑、26类，已有屋顶、60.96m周边、街景与POI。近景+环境及VLM错误分析已有直接先例；单城市数据不证明跨城泛化。我们的独立楼栋证据和归属问题仍须另行验证。|
|Skuppin, Hoffmann, Shi & Zhu. *Building Type Classification with Incomplete Labels*. IGARSS2022. [机构接受稿](https://elib.dlr.de/186659/1/BUILDING_TYPE_CLASSIFICATION_WITH_INCOMPLETE_LABELS.pdf), DOI10.1109/IGARSS46834.2022.9884076|方法§2、实验§3及Table1；同行评审会议论文|检测与分类损失分离，只对有功能标签的建筑像素计算分类监督；9城训练、Cologne测试。该研究主要处理缺失标签，并将其OSM样本视为不完整而非大量错标；不是地块户数在多个屋顶间的用途分配。可借鉴“不把未知当负例”。|
|Ilse, Tomczak & Welling. *Attention-based Deep Multiple Instance Learning*. ICML2018. [PMLR原始记录](https://proceedings.mlr.press/v80/ilse18a.html)|出版方摘要与元数据；未完整阅读PDF；同行评审会议论文|包标签与注意力聚合已有成熟方法，不能把普通MIL当创新。我们须先定义地块标签和单栋标签的逻辑关系；地块4户并不保证其中存在单栋≥4户。注意力权重本身不能确证用途归属。|
|Chang, Adam & Goldenberg. *Towards Robust Classification Model by Counterfactual and Invariant Data Generation*. CVPR2021. [作者arXiv v2](https://arxiv.org/abs/2106.01127v2)|摘要与会议接受信息；CVF PDF本次获取失败，未全面读方法/实验|已有前景/背景干预以削弱伪相关的工作。任意背景替换或一致性损失不是新概念；建筑用途的环境可能是真实有用证据，不能一概强制忽略。我们的合成干预最多先作为模型敏感性诊断，不能直接证明现实因果归因。|
|Skuppin, Shi & Zhu. *Geographic transferability of building function models*. IEEE JSTARS2026. [原始机构记录](https://elib.dlr.de/227125/), DOI10.1109/JSTARS.2026.3737186|本次核对机构摘要；早期全文阅读记录见上一综述；同行评审期刊论文|已有多国/多地区训练与地理迁移研究，加入footprint和多地区覆盖值得用作强基线。但住宅/非住宅任务不能回答一户与四户或主楼与车棚的细分类问题。|

另检索到Zhang et al. *Using knowledge graph embeddings and meta-path transformers for geographic context-aware predictions of building functions*，IJGIS40(7),2026；[出版方摘要](https://www.tandfonline.com/doi/abs/10.1080/13658816.2025.2596838)。本次只读摘要，未核实数据划分；仅用于确认地理上下文知识图谱也已有先例，不使用其准确率作我们的性能依据。

## 建议聚焦的研究问题

**地块弱标签怎样正确归属于单栋，并利用邻域帮助识别而不把邻居的用途复制给目标？** 本地新增样本中主楼和附属屋顶共存，小型多户与独栋高度相似；这给研究问题提供实际失败案例，但还不是方法有效性证明。[本轮诊断](../handoff/stage2a_la/analysis/20261029_bottlenecks/REPORT.md)

最小起点是目标尺度稳定的视觉表示，以及使用已核实样本的用途/角色双头。保留独立的目标与邻域表示，角色标签约束目标是主楼还是附属结构。对于来源只有地块用途的样本，记录监督粒度；未知归属先不做硬用途监督。等这个简单对照成立后，再研究满足户数或用途配置的弱监督损失。

“地块4户→至少一栋多户”的MIL假设并不总成立：两栋各2户也可以组成同一地块4户。若要利用总户数，需要允许多种分配配置，并有独立楼栋标签作为锚点；否则潜变量不可辨识，训练拟合地块总数也不代表分配正确。测试时不得把产生参考类别的assessor Units/UseCode或其派生量作为特征，否则不能声称影像识别能力提高。

辅助数据应瞄准形态匹配的难负例和地区多样性：同面积/相似坡顶的1户与4户、住宅式办公室与住宅、主楼与车棚。对比学习只作为有必要时的工具。不能因为在原LA错误上找到直观解释，就声称这些线索跨城市成立。

## 什么属于工程，什么可能成为贡献

|改进|判断|必要的证据|
|---|---|---|
|动态增强、较高分辨率特征、目标近景|优先工程基线|同数据、同预算消融，报告训练成本|
|双视野、GNN、VLM SFT、街景融合|已有广泛先例|与简单CNN融合比较其增量；不能只靠组合名称声称新颖|
|形态匹配负例与跨街区采样|关键数据设计，单独算法新颖性弱|同规模正例补充对照，保持测试隔离|
|用途/角色与监督粒度共同建模，显式约束地块到楼栋的归属|最值得深入的候选方法贡献|对比直接标签复制、未知屏蔽、普通包学习与角色模型；独立建筑级及新城市验证|
|少量可靠证据获取，处理航片不可辨识对象|可能的次要研究方向|同覆盖率/标注预算下比较错误与成本；不能只报告拒答后的accuracy|

在本次检索范围内，没有确认与该完整问题定义完全相同的解决方案；这不等于证明“首次”。需要先用简单实验确定瓶颈和收益，再做围绕监督粒度、实例归属和组合数量标签的更完整近邻工作检索。

## 实验与论文边界

现有3332栋LA保持标签和样本冻结，用于开发回归；加入新视野时保留原输入，并把输入变化与架构变化分开。模型选择只在开发验证区域进行。最终论文还需未用于错误分析的空间保留区和未见城市、更多多户样本、独立单栋用途及角色复核。

五类+角色的长期输出与当前三类参考评估必须区分。先在相同三类监督上做表示对照，再在具备institutional/other/role可靠监督的同一数据上比较模型；不能用类别映射变化制造表面增益。unknown须报告覆盖率，全样本主指标不能只保留回答部分。

VLM生成的解释、未经核实的POI和来自同一assessor参考的户数都不能自动成为独立证据。高度、街景和建筑关系也需要时相、定位与缺失情况核对。当前没有新增训练收益、独立泛化结果或已证实的新颖性结论。

## 检索记录

- 检索日期2026-10-29；关键词包含`building function classification parcel multiple instance learning auxiliary buildings label noise`、`building function classification counterfactual context remote sensing invariant`、`building function classification single family multifamily aerial street view height graph`、`building function classification multi scale contrastive learning 2025 2026`，以及上述论文精确标题。
- 按标题/DOI去重；保留与当前错误模式直接相关的原始来源。搜索结果中的代理镜像未作为来源，BuildingSense改读出版方canonical页面。
- 此轮没有联系作者、下载新训练数据或导入参考管理软件；未读全文的条目已标明，未将摘要报道的准确率与LA直接比较。
