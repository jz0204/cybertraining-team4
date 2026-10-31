"""Verify backed-up follow-up outputs and summarize the frozen paired tests."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

STAGE = Path('/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a')
ROOT = STAGE / 'la_data_20261024/followup_20261025'


def main():
    result = json.loads((ROOT / 'RESULTS.json').read_text())
    protocol = json.loads((ROOT / 'PROTOCOL.json').read_text())
    ref = pd.read_csv(STAGE / 'la_explore_20261023/data/la_reference.csv')
    ref = ref[ref.primary_single_outline_iou50]
    checked = []
    for name in protocol['models']:
        for kind in ('best', 'last'):
            path = ROOT / f'runs/{name}/stage2a_{kind}_model.pt'
            tensors = torch.load(path, map_location='cpu', weights_only=False)['model_state']
            assert tensors and all(torch.isfinite(t).all() for t in tensors.values())
            checked.append(dict(model=name, checkpoint=kind, bytes=path.stat().st_size, loaded_finite=True))
        for label, expected in [('la', 21888), ('harris_val', 2026), ('harris_test', 2653)]:
            f = pd.read_csv(ROOT / f'predictions/{name}/{label}.csv')
            assert len(f) == expected and f.building_uid.is_unique
    rows = []
    for name, metric in result['metrics'].items():
        folder = ROOT if name in protocol['models'] else STAGE / (
            'la_explore_20261023' if name.startswith('N1') else 'la_data_20261024')
        f = ref.merge(pd.read_csv(folder / f'predictions/{name}/la.csv'), on='building_uid', validate='one_to_one')
        assert len(f) == 3332
        y = f.assessor_type.eq('commercial')
        p, r, _ = precision_recall_curve(y, f.prob_commercial)
        assert np.isclose(p[r >= .9].max(), metric['precision'])
        assert np.isclose(average_precision_score(y, f.prob_commercial), metric['ap'])
        strict = f.prob_commercial.ge(f.loc[y, 'prob_commercial'].nlargest(38).min())
        rows.append(dict(model=name, precision=metric['precision'], recall=metric['recall'],
                         ap=metric['ap'], harris_val_f1=metric['harris_val_f1'],
                         guardrail_pass=metric['harris_val_f1'] >= protocol['guardrail'],
                         exact38_tp=int((strict & y).sum()), exact38_fp=int((strict & ~y).sum()),
                         exact38_precision=float((strict & y).sum()/strict.sum())))
    table = pd.DataFrame(rows).sort_values('model')
    table.to_csv(ROOT / 'PAIRED_COMPARISON.csv', index=False)
    (ROOT / 'BACKUP_CHECK.json').write_text(json.dumps(checked, indent=2)+'\n')
    lines = ['# 两种子与环境特征实验结果', '',
             '四组新增实验已完成，8个best/last权重已加载并验证有限值，预测行数和ID唯一性通过。原seed42结果保留。', '',
             '| 模型 | P@R≥90% | 实际R | AP | Harris验证F1 | 门槛通过 |',
             '|---|---:|---:|---:|---:|---|']
    for row in table.itertuples():
        lines.append(f'| {row.model} | {row.precision:.2%} | {row.recall:.2%} | {row.ap:.4f} | {row.harris_val_f1:.4f} | {"是" if row.guardrail_pass else "否"} |')
    lines += ['',
              '注意：协议指标是召回率≥90%范围内的最佳开发集precision，不保证每个模型实际召回完全相同。N1 seed43的最优点为40/42=95.24%；其他模型为38/42=90.48%。CSV另列严格38个TP时的结果，避免误称完全同召回比较。', '',
              '1. NYC效应不稳定：seed42的P@R≥90%下降6.11个百分点，seed43上升0.26个百分点，但seed43的源域验证未通过。不能继续声称NYC必然导致相同程度的负迁移。',
              '2. N1自身种子敏感：P@R≥90%从12.26%降到4.61%。单种子最佳结果不足以证明稳定泛化。',
              '3. 环境特征在两个种子上分别提高1.71和3.63个百分点，但seed42的Harris验证F1=0.92128，低于0.92402门槛；未满足预先冻结的两种子均改善且均过门槛条件，不能升级为替代模型。AP变化也不一致。',
              '4. 下一步优先提高目标域困难样本覆盖、核实小型商业与中大型住宅标签，并检验保留建筑内部特征的轻量环境融合；这轮结果仅支持继续研究，不能宣布该结构有效泛化。', '',
              'LA已经用于开发选择，42个商业样本仅分布在7个cell；两个种子只构成诊断。SD仍未运行，未替换现有模型。',
              '服务恢复：两部署各1/1且有ready endpoints；Qwen容器内/health返回200，Ollama API能列出保留模型。master直连Pod IP失败，未验证外部推理入口。']
    (ROOT / 'FOLLOWUP_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(table.to_string(index=False))
    print('PASS:', len(checked), 'checkpoints; all prediction exports; independently recomputed precision/AP')


if __name__ == '__main__':
    main()
