"""Compare the frozen small-multifamily data ablation on unchanged LA labels."""
import json
from pathlib import Path
import numpy as np

from analyze_la_transfer import STAGE,measure,bootstrap
from run_la import sha256
from reliable_transfer_common import save

ROOT=STAGE/'la_hard_ablation_20261030/execution'
BASE=STAGE/'la_augmentation_20261030/execution'
OUT=Path(__file__).parent/'analysis/20261030_hard_ablation'


def main():
    rows=json.loads((STAGE/'la_transfer_20261027/dataset.json').read_text())
    frozen=json.loads((ROOT/'FROZEN.json').read_text())
    assert sha256(STAGE/'la_transfer_20261027/dataset.json')==frozen['original_test_dataset_sha256']
    parent=json.loads((BASE/'DATASET.json').read_text())
    prior=json.loads((OUT.parent/'20261030_augmentation/PREDICTIONS.json').read_text())
    predictions={'A':prior['augmentation'],'A_s42':prior['42'],'A_s43':prior['43']};checks={}
    for arm,directory in [('B','B_positive'),('C','C_positive_negative')]:
        dataset=json.loads((ROOT/directory/'DATASET.json').read_text())
        assert dataset[:len(parent)]==parent
        for split in ['val','test']:assert [r for r in dataset if r['split']==split]==[r for r in parent if r['split']==split]
        checks[arm]={}
        for seed in [42,43]:
            folder=ROOT/f'collected/{directory}/runs/cnn_lowrise_s{seed}'
            result=json.loads((folder/'RESULT.json').read_text());history=json.loads((folder/'history.json').read_text())
            initial=json.loads((folder/'initial_validation.json').read_text())
            old_initial=json.loads((BASE/f'collected/runs/cnn_lowrise_s{seed}/initial_validation.json').read_text())
            assert initial['confusion_matrix']==old_initial['confusion_matrix']
            scores=[initial['sampling_weighted_macro_f1']]+[r['validation']['sampling_weighted_macro_f1'] for r in history]
            assert len(history)==6 and result['best_epoch']==int(np.argmax(scores)) and result['optimizer_updates']==1536
            assert sha256(folder/'best.pt')==result['checkpoint_sha256'] and sha256(ROOT/directory/'DATASET.json')==result['dataset_sha256']
            assert sha256(ROOT/directory/'LOWRISE_SAMPLING.json')==result['sampling_factor_sha256']
            values=json.loads((folder/'test_predictions.json').read_text())
            assert [v['sample_id'] for v in values]==[r['sample_id'] for r in rows]
            probabilities=np.array([v['probabilities'] for v in values]);assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1),1,atol=1e-5)
            predictions[f'{arm}_s{seed}']=values
            checks[arm][str(seed)]=dict(best_epoch=result['best_epoch'],validation=result['validation'],checkpoint_sha256=result['checkpoint_sha256'])
        p=np.mean([np.array([r['probabilities'] for r in predictions[f'{arm}_s{seed}']]) for seed in [42,43]],axis=0)
        predictions[arm]=[dict(sample_id=r['sample_id'],use=['residential_small','residential_multi','commercial'][int(v.argmax())],probabilities=v.tolist()) for r,v in zip(rows,p)]
    assert all([v['sample_id'] for v in values]==[r['sample_id'] for r in rows] for values in predictions.values())
    metrics={name:measure(rows,values) for name,values in predictions.items()}
    intervals={f'{a}-{b}':bootstrap(rows,predictions,a,b) for a,b in [('B','A'),('C','A'),('C','B')]}
    cases={name:{kind:[r['sample_id'] for r,v in zip(rows,predictions[name])
                      if (r['reference_use']=='residential_multi',v['use']=='residential_multi')==pair]
                 for kind,pair in [('tp',(True,True)),('fp',(False,True)),('fn',(True,False))]} for name in ['A','B','C']}
    save(OUT/'RESULTS.json',dict(metrics=metrics,paired_cell_bootstrap=intervals,checks=checks,multifamily_cases=cases,data=frozen))
    save(OUT/'PREDICTIONS.json',predictions)
    lines=['# LA小体量多户补样对照','',
        'A：增强修复后的基线；B：增加45栋经官方资料和AI图审支持的多户主楼；C：B再增加6栋混淆独栋。',
        '新样本位于原测试区2km缓冲外且只用既定训练网格。原438条验证和3332条测试记录、标签及输入未变。',
        '源域/LA、LA类别和多户户数组的抽样概率比例不变，实际随机抽样次数可波动；每组两个种子，选模仅依据原验证集。','',
        '|模型|Accuracy|Macro-F1|多户P|多户R|多户F1|','|---|---:|---:|---:|---:|---:|']
    for n in ['A_s42','B_s42','C_s42','A_s43','B_s43','C_s43','A','B','C']:
        m=metrics[n];u=m['per_class']['residential_multi'];lines.append(f"|{n}|{m['accuracy']:.4f}|{m['macro_f1']:.4f}|{u['precision']:.4f}|{u['recall']:.4f}|{u['f1']:.4f}|")
    lines+=['','|集成|TP|FP|FN|','|---|---:|---:|---:|']
    for n in ['A','B','C']:lines.append(f"|{n}|{len(cases[n]['tp'])}|{len(cases[n]['fp'])}|{len(cases[n]['fn'])}|")
    lines+=['','## 结果解读','',
        f"C相对A消除{len(set(cases['A']['fp'])-set(cases['C']['fp']))}条旧误报、新增{len(set(cases['C']['fp'])-set(cases['A']['fp']))}条误报；漏报集合是否相同：{set(cases['A']['fn'])==set(cases['C']['fn'])}。",
        '本轮集成的收益主要是减少误报并改善商业识别。两个单种子的变化方向不一致；不能声称单模型稳定改善，也没有解决多户召回。',
        '下一项优先验证目标近景与上下文输入，保持原标签及分组；先区分目标缩得太小与可见用途证据不足。补样影像来源变化意味着本轮不能排除分辨率/来源差异影响。']
    lines+=['','## 配对空间bootstrap','', '以下区间以已训练检查点为条件，不包含完整训练随机性。']+[f'- {name}: {v}' for name,v in intervals.items()]
    lines+=['','## 解释边界','',
        '新数据同时增加街区覆盖和NAIP2022影像（原生0.6m），不是纯标签数量的因果对照。NAIP输出采样约0.305m以保持原CNN物理视野；上采样不增加真实细节。',
        '训练参考经过官方单记录/单轮廓关联和AI图审，未产生独立人工金标准。6条负例中1条为集成误报，其余为高混淆/种子分歧候选。',
        '2022影像与2024房产用途记录存在时间差；历史轮廓一致不能证明期间用途或户数完全不变。',
        'LA已反复用于开发，多户参考仅14条。本轮不证明跨城泛化；预留网格未用于采图、打分或选模。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({n:dict(macro_f1=metrics[n]['macro_f1'],multi=metrics[n]['per_class']['residential_multi']) for n in ['A','B','C']},indent=2))


if __name__=='__main__':main()
