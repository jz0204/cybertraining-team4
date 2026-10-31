# 执行记录

| 步骤 | 状态 | 证据 |
|---|---|---|
| LA少数类＋1冲突属性、近/宽图像筛查 | 完成，原标签未改，4条疑点保留 | LABEL_AUDIT.csv、VISUAL_REVIEW.json |
| 新增影像、空间隔离与冻结 | 完成1,186栋，748train/438val，无图像QA失败 | PROTOCOL.json、DATASET.json |
| 原CNN复现与适配42/43 | 完成；完整集成macro-F1 .571612 | PRIMARY_RESULTS.json |
| DINO同数据42/43 | 完成；每轮抽样序列与CNN对应种子一致；macro-F1 .572436 | PRIMARY_RESULTS.json |
| 低层多户抽样后续对照42/43 | 完成；macro-F1 .580393，增益未明显超过普通适配CNN | LOWRISE_PLAN.md、RESULTS.json |
| 统一评估与cell bootstrap | 完成；全3332、共同3157、两种敏感性子集 | METRICS.csv、MINORITY_COMPARISON.csv |
| 原服务恢复、磁盘健康 | 完成，服务1/1，Qwen200，无DiskPressure | VERIFICATION.json、外置盘SERVICE_LIFECYCLE.json |
| 下一轮小多户候选 | 31个双轮廓四户训练区候选已列出，未赋单栋用途/未加入训练 | FOURPLEX_REVIEW_QUEUE.csv |
