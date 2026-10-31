"""Paired frozen-test comparison after changing only per-draw augmentation."""
import json
from pathlib import Path
import numpy as np
from analyze_la_transfer import STAGE,measure,bootstrap
from run_la import sha256
from reliable_transfer_common import save
ROOT=STAGE/'la_augmentation_20261030';OLD=STAGE/'la_adaptation_20261029'
OUT=Path(__file__).parent/'analysis/20261030_augmentation'

def main():
    OUT.mkdir(exist_ok=True)
    for n,previous in [('DATASET.json',OLD/'DATASET.json'),('PROTOCOL.json',OLD/'PROTOCOL.json'),('LOWRISE_SAMPLING.json',OLD/'execution/LOWRISE_SAMPLING.json')]:
        assert sha256(ROOT/'execution'/n)==sha256(previous)
    rows=json.loads((STAGE/'la_transfer_20261027/dataset.json').read_text())
    protocol=json.loads((STAGE/'la_transfer_20261027/PROTOCOL.json').read_text())
    assert sha256(STAGE/'la_transfer_20261027/dataset.json')==protocol['dataset_sha256']
    old=json.loads((OUT.parent/'20261029_la_adaptation/PREDICTIONS.json').read_text())
    predictions={'baseline':old['cnn_lowrise_ensemble']};checks={}
    for seed in [42,43]:
        name=f'cnn_lowrise_s{seed}';folder=ROOT/'execution/collected/runs'/name;previous=OLD/'execution/collected/runs'/name
        result=json.loads((folder/'RESULT.json').read_text());history=json.loads((folder/'history.json').read_text());prev=json.loads((previous/'history.json').read_text())
        assert [r['draw_sha256'] for r in history]==[r['draw_sha256'] for r in prev]
        initial=json.loads((folder/'initial_validation.json').read_text());initial_old=json.loads((previous/'initial_validation.json').read_text())
        assert initial['confusion_matrix']==initial_old['confusion_matrix']
        scores=[initial['sampling_weighted_macro_f1']]+[r['validation']['sampling_weighted_macro_f1'] for r in history]
        assert result['best_epoch']==int(np.argmax(scores))
        assert sha256(folder/'best.pt')==result['checkpoint_sha256']
        predictions[str(seed)]=json.loads((folder/'test_predictions.json').read_text())
        predictions[f'baseline{seed}']=old[name]
        checks[str(seed)]=dict(best_epoch=result['best_epoch'],sampling_draws_identical=True,initial_validation_identical=True,checkpoint_sha256=result['checkpoint_sha256'])
    for values in predictions.values():
        assert [v['sample_id'] for v in values]==[r['sample_id'] for r in rows]
    p=np.mean([np.array([r['probabilities'] for r in predictions[str(seed)]]) for seed in [42,43]],axis=0)
    predictions['augmentation']=[dict(sample_id=r['sample_id'],use=['residential_small','residential_multi','commercial'][int(v.argmax())],probabilities=v.tolist()) for r,v in zip(rows,p)]
    metrics={k:measure(rows,v) for k,v in predictions.items()}
    ci=bootstrap(rows,predictions,'augmentation','baseline')
    errors={}
    for name in ['baseline','augmentation']:
        errors[name]={kind:[r['sample_id'] for r,v in zip(rows,predictions[name])
                            if (r['reference_use']=='residential_multi',v['use']=='residential_multi')==pair]
                      for kind,pair in [('tp',(True,True)),('fp',(False,True)),('fn',(True,False))]}
    save(OUT/'RESULTS.json',dict(metrics=metrics,paired_cell_bootstrap=ci,checks=checks,multifamily_cases=errors))
    save(OUT/'PREDICTIONS.json',predictions)
    lines=['# LA增强单因素双种子对照','', '训练数据、逐epoch采样序列、初始化权重、结构、颜色增强和选模规则保持一致，原LA测试标签不变。','', '|模型|Accuracy|Macro-F1|多户Precision|多户Recall|多户F1|','|---|---:|---:|---:|---:|---:|']
    for k in ['baseline42','42','baseline43','43','baseline','augmentation']:
        m=metrics[k];u=m['per_class']['residential_multi'];lines.append(f"|{k}|{m['accuracy']:.4f}|{m['macro_f1']:.4f}|{u['precision']:.4f}|{u['recall']:.4f}|{u['f1']:.4f}|")
    lines+=['',f'配对cell bootstrap（固定模型）：{ci}', '',
            f"多户误报：{len(errors['baseline']['fp'])} → {len(errors['augmentation']['fp'])}；漏报：{len(errors['baseline']['fn'])} → {len(errors['augmentation']['fn'])}。",
            f"消除旧误报{len(set(errors['baseline']['fp'])-set(errors['augmentation']['fp']))}条，新增误报{len(set(errors['augmentation']['fp'])-set(errors['baseline']['fp']))}条；漏报集合相同：{set(errors['baseline']['fn'])==set(errors['augmentation']['fn'])}。", '',
            '两个种子的macro-F1和多户precision/F1均提高；seed43多户recall从4/14下降至3/14。集成recall未提高，macro-F1差值的95%区间跨零，不能宣称稳定显著提升。', '',
            '下一步保持修复后的增强，单独对照可靠的小体量多户正例与相似独栋难负例补样；不依据本测试集调整阈值。优先核实单栋实际用途、主/附属角色与轮廓，不能将整地块户数直接分配给任一屋顶。', '',
            '候选补样没有进入本次训练。原LA是已用于开发的固定回归集，只有14个多户参考，不能据此证明跨城泛化。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(metrics['augmentation'],indent=2))
if __name__=='__main__':main()
