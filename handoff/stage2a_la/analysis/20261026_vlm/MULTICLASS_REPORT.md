# Stage2a 五类、角色与双视野对照：2026-10-26

**近景加真实宽视野值得保留；直接把 Qwen3-VL-8B 零样本分类扩成五类，目前不足以获得更可靠的用途识别。** 新城市 San Diego 中，五类双视野的完整五类参考一致率为63/101（62.4%），高于三类双视野58/101（57.4%）；但同一74栋原三类建筑上，从58/74（78.4%）下降至53/74（71.6%）。新增类别识别出10栋，同时损失原本正确的5栋。多户住宅仍全部漏判，主/附属和unknown字段也未发挥预期作用。

本报告的 accuracy 是**给定footprint后的用途分类与参考标签的一致性**，不是建筑检测AP，也不是已经取得独立人工真值的实际用途准确率。

## 固定实验与参考

共164栋、5组，820次正式推理，加20次Harris预检；全部完成。Harris49栋作为开发诊断，San Diego115栋作为冻结方案后的新地域测试。21:56:13Z固定图像、提示、样本、评分资格和文件哈希，San Diego未用于prompt调整或预检。

五类是small residential、multifamily residential、commercial、institutional、other；A–D允许unknown，并另输出primary/auxiliary/unknown角色。Other定义为可识别的工业、仓储、公用设施、农业或停车/存储支持用途，不能代替“不确定”。三类中的unknown同时承担超出类别范围与视觉不确定两种含义，其使用率不能直接当校准结果。

近景窗口为40m与目标包围框长边1.4倍中的较大者；宽视野为300m与近景窗口3倍中的较大者。两图均来自真实原始影像，重采样为512×512，同一目标以青色轮廓标记；模型另获footprint面积，不提供城市、来源类别或参考记录。

新增institutional/other候选58栋：Harris18、SD40。SD按明确assessor代码、2023/2025一致土地用途、单一映射屋顶和严格地块重叠选择；另查询官方历史许可记录。模型运行前发现商店/仓库混用、多建筑记录、住宅记录冲突、屋顶碎片和原图边界截断，固定降级为诊断项，未按模型结果补换样本。

| 地区 | 全部输入 | 可评分 | small | multi | commercial | institutional | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| Harris开发 | 49 | 35 | 8 | 6 | 8 | 2 | 11 |
| San Diego地域测试 | 115 | 101 | 34 | 14 | 26 | 17 | 10 |

其余28栋仍跑齐五组，计入输出格式、unknown和诊断统计，不把无参考或拒答自动算正确。新增两类共40栋可评分。SD有21栋获得官方记录佐证（7small、6commercial、5institutional、3other），仍不等于影像同一日期、独立核实的单栋真值；没有该等级的多户住宅。所有角色都缺独立人工真值，仅9个Harris视觉假设用于诊断。

## 对照结果

原三类子集是判断扩展标签是否损伤已有识别能力的主要比较。三类方案无法输出institutional/other，所以完整五类分数同时衡量命名能力，不能把全部增益解释为辨识能力提高。

| 方案 | SD原三类正确/74 | 三类macro-F1 | SD五类正确/101 | 五类macro-F1 | Harris原三类正确/22 | Harris五类正确/35 |
|---|---:|---:|---:|---:|---:|---:|
| A 三类近景，角色+unknown | 50（67.6%） | 0.499 | 50（49.5%） | 0.258 | 15 | 15 |
| B 五类近景，角色+unknown | 40（54.1%） | 0.527 | 45（44.6%） | 0.348 | 6 | 15 |
| C 三类双视野，角色+unknown | 58（78.4%） | 0.576 | 58（57.4%） | 0.293 | 15 | 15 |
| D 五类双视野，角色+unknown | 53（71.6%） | 0.550 | 63（62.4%） | 0.468 | 11 | 20 |
| E 强制三类双视野，不输出角色 | 58（78.4%） | 0.576 | 58（57.4%） | 0.295 | 17 | 17 |

- **加真实宽视野：** A→C在SD原三类纠正9栋、引入1个错误，净增8/74，即10.8个百分点。B→D在SD完整五类纠正25栋、引入7个错误，净增18/101，即17.8个百分点；原三类部分为纠正19、退步6，净增13/74。Harris五类B→D从15/35到20/35；三类A→C仍为15/22，说明收益依数据而异。
- **同样双视野，扩展三类到五类：** C→D在原三类没有纠正任何一栋，却新增5个错误，分布在4个SD区域。新增正确来自3栋institutional、7栋other；完整五类净增5/101，即5.0个百分点。两栋住宅被改成commercial，三栋commercial被改成other。
- **角色与unknown联合指令：** C和E在SD同为58/74，均未正确识别多户。E同时移除了角色输出和unknown，因此C−E不能单独归因于拒答或角色指令。

SD官方佐证子集也没有逆转该方向：原三类13栋，C为13/13、D为11/13；加上新增用途后，21栋上C为13/21、D为15/21。这个子集很小、没有多户，而且用途时间仍可能不一致。

保留全部候选来源标签的敏感性分析中，SD原三类C59/75、D54/75，仍净损失5栋；完整五类C59/115、D73/115。未经降级的地块标签会把五类的表面净增从5栋放大到14栋，进一步说明不能将地块一致性等同于实际单栋用途准确性。

## 具体失败机制

五类双视野D在SD101栋的结果：

| 类别 | 正确/参考数量 | Recall | Precision |
|---|---:|---:|---:|
| small residential | 30/34 | 88.2% | 85.7% |
| multifamily residential | 0/14 | 0% | 无该类预测 |
| commercial | 23/26 | 88.5% | 48.9% |
| institutional | 3/17 | 17.6% | 100%（仅3次预测） |
| other | 7/10 | 70.0% | 43.8% |

多户14栋分别被判为small5、commercial6、other3；并非只存在“全部偏向小型住宅”的单一错误。模型经常将平屋顶、停车位和多栋布局直接解释为办公、仓储或支持设施。Institutional17栋中11栋被判commercial、3栋other；普通商用建筑与部分宗教/公共用途的屋顶外观不足以稳定分离。Other扩展后吸收了9个非other样本（3multi、3commercial、3institutional）。

SD218的同一双视野输入，在三类中正确判commercial，在五类中因“平顶、停车场、道路”被判为维护/公用设施other。SD199的公寓参考，在多个方案中因“平顶、停车场”持续被判commercial。官方许可佐证为教堂的SX204817754仍被判commercial。SX205213662则从近景的commercial改成宽视野下的other，符合仓库参考。这里既有上下文带来的改善，也有用途推断超出可见证据的失败。

**角色字段没有形成有效区分。** A–D各自164/164都输出primary，合计656个角色输出中没有一次auxiliary或unknown。9个视觉诊断中6个primary假设匹配，但3个疑似车棚H004/H239/H250也全部标primary；不能把6/9当成有效角色识别性能。D把这3栋都判成small。HX015与SD209的解释明确写supporting/auxiliary或“rather than a primary…”，结构化role却仍为primary；原始JSON核查确认这不是解析器默认值。

**Unknown也未形成可靠拒答机制。** 两个五类方案各164栋均为0次unknown；SD在五个方案中均为0次unknown，包括大量参考错误。三类A/C仅在Harris分别输出5次、2次。SD覆盖率因此全部100%，selective accuracy与correct/all相同，无法得到有效的准确率—覆盖率权衡。

[六个改善与失败案例](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/vlm_multiclass_20261026/MULTICLASS_CASES.jpg>)同时展示近景、宽视野、参考与模型解释；角色假设明确标为未独立核实。

## 能支持什么泛化结论

这支持**双视野在本次冻结的SD地域测试中改善参考一致性**，不支持“恢复五类普遍提高已有类别的泛化”。五类B→D完整评分在8个区域中6个改善、1个持平、1个下降；收益并非只来自一个区域。C→D在有原三类样本的6个区域中4个退步、2个持平，没有改善区域。

| SD区域 | 评分样本 | 五类近景B正确 | 五类双视野D正确 |
|---|---:|---:|---:|
| Chula Vista | 16 | 7 | 8 |
| City Heights | 15 | 6 | 6 |
| Clairemont | 15 | 7 | 13 |
| El Cajon | 13 | 5 | 9 |
| Kearny Mesa | 5 | 1 | 2 |
| La Mesa | 16 | 6 | 11 |
| North Park | 15 | 10 | 9 |
| Otay Mesa | 6 | 3 | 5 |

这不是随机城市总体样本；建筑在地域、地块和影像上存在相关性，未按独立建筑计算显著性。SD institutional中14/17是宗教用途，other仅覆盖两个工业区域；不能外推医院、学校、农业及所有公用设施。SD2022影像对照较晚地块/土地用途与历史许可，Harris2017影像也存在时间差。Harris已参与开发；SD只是本次VLM协议未用于调参的地域，不保证基础模型预训练未见过该地区。尚未测试另一个新城市、跨日期性能或端到端检测。

## 建议的下一步

保留**五类＋角色＋unknown作为最终输出目标**，并保留已获支持的**目标近景＋真实宽视野**输入。当前8B零样本方案适合作为可复核的研究基线，不能直接替换生产分类器或自动生成用途真值。

下一步最有价值的是先补建筑级用途与角色监督，再做有明确对照的适配。优先核实多户、宗教/公共用途、commercial与warehouse边界，以及主楼/车棚；建筑记录中的单位数或用途应确实对应这栋建筑，地块用途只作上下文。1–3户与4户以上这类界限不能总从屋顶恢复，必要时需要楼栋记录等额外证据。用核实数据训练/微调后，分别验证用途、角色与拒答，unknown需要独立校准，不能仅依赖自报字段。最后冻结方案，在另一个完整地域验证；SD已经产生本轮反馈，继续调参后的SD成绩应作为开发成绩。该后续适配尚未执行，也不能保证收益。

## 执行与验证

模型为Qwen/Qwen3-VL-8B-Instruct，revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16/SDPA/batch1/greedy/最多160输出token；TAMU单A30，固定已有容器，未安装新依赖。正式平均2.58秒/次，总35分16秒；D平均2.82秒/栋。下载约4分2秒，Job22:00:26Z至22:41:22Z，共40分56秒。PyTorch最大reserved显存16.69GiB；整卡一次观测17,689MiB约17.27GiB，后者不是峰值保证。

840条输出全部有效，无输出达到160token上限；与流式日志逐条精确一致。20条预检在正式运行中原始回答完全重现。333个冻结文件哈希全部匹配；820个样本—方案组合完整、无重复。独立scikit-learn核对了20个地区/标签范围组合的accuracy、macroF1、balanced accuracy及逐类support/F1，全部一致。原始回答、证据、评分资格和空间分组均保留。

原Qwen与Ollama服务已于22:44:30Z完成独立恢复核验：均为原1/1 ready/available，Qwen health200，Ollama模型列表及两个服务端点正常，节点无内存/磁盘压力；watcher报告ready=true、未触发超时。未改原训练标签、权重、生产流程或下游人口概率接口，未提交或推送Git。

完整输出目录：[vlm_multiclass_20261026](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/vlm_multiclass_20261026>)。主要证据：[RESULTS.json](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/vlm_multiclass_20261026/RESULTS.json>)、[逐栋对照CSV](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/vlm_multiclass_20261026/CASE_RESULTS.csv>)、[成对纠正/退步](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/vlm_multiclass_20261026/CONTRASTS.json>)、[恢复核验](</Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/vlm_multiclass_20261026/LLM_RESTORATION_VERIFIED.json>)。
