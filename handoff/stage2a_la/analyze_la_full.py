"""Compare complete LA CNN outputs using unchanged reference definitions."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_la_transfer import LABELS, STAGE, bootstrap, measure
from reliable_transfer_common import load, save
from run_la import read_csv, sha256, write_csv

ROOT=STAGE/'la_full_20261030'
OUT=Path(__file__).parent/'analysis/20261030_la_full'


def main():
    OUT.mkdir(exist_ok=True)
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    assert sha256(ROOT/'DATASET.json')==protocol['dataset_sha256']
    for path,expected in protocol['source_sha256'].items():assert sha256(path)==expected,path
    rows=json.loads((ROOT/'DATASET.json').read_text());assert len(rows)==22024
    ids=[r['sample_id'] for r in rows];by_id={r['sample_id']:i for i,r in enumerate(rows)}
    assert len(by_id)==len(rows)
    cached=json.loads((ROOT/'OLD_PREDICTIONS.json').read_text());models={};run_metadata={};old_arrays=[];new_arrays=[]
    for seed in ['42','43']:
        folder=ROOT/'execution/collected'/('seed'+seed)
        result=json.loads((folder/'RESULT.json').read_text());run_metadata[seed]=result
        assert result['dataset_sha256']==protocol['dataset_sha256'] and result['n']==22024
        assert result['checkpoint_sha256']==protocol['checkpoints'][seed]['adapted_sha256']
        assert sha256(folder/'predictions.json')==result['predictions_sha256']
        assert sha256(folder/'old_recovered.json')==result['old_recovered_sha256']
        new=json.loads((folder/'predictions.json').read_text());assert [r['sample_id'] for r in new]==ids
        repaired={r['sample_id']:r['probabilities'] for r in json.loads((folder/'old_recovered.json').read_text())}
        assert set(repaired)=={r['sample_id'] for r in rows if r['input_status']=='recovered_pixel_geometry'}
        old_arrays.append(np.asarray([repaired[r['sample_id']] if r['sample_id'] in repaired else cached[seed][r['building_uid']] for r in rows]))
        new_arrays.append(np.asarray([r['probabilities'] for r in new]))
    for name,arrays in [('old_n2',old_arrays),('best_cnn',new_arrays)]:
        p=np.mean(arrays,axis=0)
        assert p.shape==(22024,3) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5)
        models[name]=[dict(sample_id=r['sample_id'],use=LABELS[int(v.argmax())],probabilities=v.tolist()) for r,v in zip(rows,p)]
    reference,ref_protocol=load(STAGE/'la_transfer_20261027')
    assert ref_protocol['dataset_sha256']==protocol['original_test_sha256']
    prior=json.loads((OUT.parent/'20261029_la_adaptation/PREDICTIONS.json').read_text());parity={}
    for name,old_name in [('old_n2','CNN_old'),('best_cnn','cnn_lowrise_ensemble')]:
        current=[models[name][by_id[r['sample_id']]] for r in reference]
        earlier=prior[old_name]
        assert [r['sample_id'] for r in current]==[r['sample_id'] for r in earlier]
        expected=np.asarray([r.get('probabilities',r.get('use_probabilities')) for r in earlier])
        actual=np.asarray([r['probabilities'] for r in current]);error=float(np.abs(actual-expected).max())
        assert error<3e-4 and [r['use'] for r in current]==[r['use'] for r in earlier],(name,error)
        parity[name]=dict(n=len(reference),maximum_probability_difference=error,predicted_classes_identical=True)
    refs=read_csv(STAGE/'la_explore_20261023/data/la_reference.csv')
    cohorts={'primary3332':reference}
    for flag,name in [('strict_single_assessor_building','strict3315'),('iou50','exploratory_iou50'),('broad_parcel_labels','exploratory_broad_parcel')]:
        rr=[r for r in refs if r[flag]=='True'];assert len(rr)==len({r['building_uid'] for r in rr})
        cohorts[name]=[dict(sample_id='la_'+'_'.join(json.loads(r['building_uid'])),reference_use=r['assessor_type'],cell_id=r['cell_id']) for r in rr]
    results={};table=[];paired={}
    for cohort,rr in cohorts.items():
        pp={name:[vv[by_id[r['sample_id']]] for r in rr] for name,vv in models.items()}
        results[cohort]={name:measure(rr,p) for name,p in pp.items()}
        paired[cohort]=bootstrap(rr,pp,'best_cnn','old_n2')
        for name,metrics in results[cohort].items():
            for label,v in metrics['per_class'].items():
                table.append(dict(cohort=cohort,model=name,n=metrics['n'],accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'],
                    class_name=label,support=v['n'],precision=v['precision'],recall=v['recall'],f1=v['f1']))
    assert abs(results['primary3332']['best_cnn']['macro_f1']-.5803933392384916)<1e-10
    assert abs(results['primary3332']['old_n2']['macro_f1']-.4832792226)<1e-9
    repairs={r['sample_id']:r for r in json.loads((ROOT/'REPAIRS.json').read_text())}
    products={json.dumps([r['cell_id'],r['bldg_uid']],separators=(',',':')):r for r in read_csv(STAGE/'la_eval_20261023/inputs/building_damage_all_cells.csv')}
    primary_labels={r['sample_id']:r['reference_use'] for r in reference};comparison=[]
    for r,old,new in zip(rows,models['old_n2'],models['best_cnn']):
        raw=products[r['building_uid']]
        comparison.append(dict(sample_id=r['sample_id'],cell_id=r['cell_id'],bldg_uid=r['bldg_uid'],input_status=r['input_status'],
            old_n2_type=old['use'],best_cnn_type=new['use'],changed=old['use']!=new['use'],
            **{'old_prob_'+label:old['probabilities'][i] for i,label in enumerate(LABELS)},
            **{'new_prob_'+label:new['probabilities'][i] for i,label in enumerate(LABELS)},
            original_test_reference=primary_labels.get(r['sample_id'],''),footprint_m2_used=r['footprint_m2'],geometry_crs='EPSG:32611',
            geometry_wkt_used=repairs[r['sample_id']]['recovered_geometry_wkt'] if r['sample_id'] in repairs else raw['polygon_wkt_utm']))
    counts={name:dict(Counter(r['use'] for r in vv)) for name,vv in models.items()}
    transitions={status:[[sum(r['input_status']==status and o['use']==a and n['use']==b for r,o,n in zip(rows,models['old_n2'],models['best_cnn'])) for b in LABELS] for a in LABELS] for status in ['original_valid','recovered_pixel_geometry']}
    changed=[r for r in comparison if r['changed']];primary_changes=[r for r in changed if r['original_test_reference']]
    write_csv(OUT/'PREDICTION_COMPARISON.csv',list(comparison[0]),comparison)
    write_csv(OUT/'CHANGED_PREDICTIONS.csv',list(comparison[0]),changed)
    write_csv(OUT/'METRICS.csv',list(table[0]),table)
    summary=dict(n=22024,coverage=1.,input_status_counts=protocol['input_status_counts'],class_counts=counts,
        changed_n=len(changed),primary_changed_n=len(primary_changes),transitions=transitions,transition_classes=LABELS,
        results=results,paired_cell_bootstrap=paired,old3332_prediction_parity=parity,run_metadata=run_metadata,
        original_test_sha256=protocol['original_test_sha256'],source_sha256=protocol['source_sha256'],limits=protocol['limits'])
    save(OUT/'RESULTS.json',summary)
    save(OUT/'VERIFICATION.json',dict(passed=True,all_product_ids=22024,missing_predictions=0,duplicate_predictions=0,
        all_original_input_files_verified_in_each_cpu_run=True,unchanged_test_labels=True,all_original_source_hashes_unchanged=True,
        full3332_prediction_parity=parity,source_protocol_sha256=sha256(ROOT/'PROTOCOL.json'),
        output_sha256={p.name:sha256(p) for p in [OUT/'PREDICTION_COMPARISON.csv',OUT/'CHANGED_PREDICTIONS.csv',OUT/'METRICS.csv',OUT/'RESULTS.json']}))
    primary=results['primary3332'];old=primary['old_n2'];new=primary['best_cnn']
    lines=['# LA全部22,024条建筑记录：CNN用途识别对照','',
        '2026-10-30。当前最佳低层多户重采样CNN的两个种子分别对全部22,024条记录推理，以概率平均得到最终类别。旧模型为适配前N2的两个种子概率平均。没有重新训练、调整阈值或改动原测试标签。','',
        '21,888条沿用原裁剪和mask；136条原坐标混用像素/UTM、mask为空，使用同cell三个有效锚点恢复坐标，并从已对齐的火前影像重新裁剪。128条复用已有底图，8条需要恢复7个cell的底图。新旧模型使用同一份恢复输入。原始产品和输入文件保留；恢复记录未被赋予新的参考标签。','',
        '旧模型21,888条预测复用冻结文件，并核对实际CPU推理的一致性；恢复的136条使用旧权重重新推理。新模型两次均全量运行。所有输入文件SHA256通过，原3,332栋的新旧类别与此前保存结果完全一致。','',
        '## 固定3,332栋参考基准','',
        '|模型|Accuracy|Macro-F1|','|---|---:|---:|',
        f"|旧N2|{old['accuracy']:.2%}|{old['macro_f1']:.4f}|",
        f"|当前最佳CNN|{new['accuracy']:.2%}|{new['macro_f1']:.4f}|",'',
        '|类别|参考数量|旧precision|新precision|旧recall|新recall|旧F1|新F1|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in LABELS:
        a=old['per_class'][label];b=new['per_class'][label]
        lines.append(f"|{label}|{a['n']}|{a['precision']:.2%}|{b['precision']:.2%}|{a['recall']:.2%}|{b['recall']:.2%}|{a['f1']:.4f}|{b['f1']:.4f}|")
    ci=paired['primary3332']
    lines += ['',f"macro-F1差值{ci['delta_macro_f1']:+.4f}，按{ci['cells']}个cell配对bootstrap的95%区间[{ci['ci95'][0]:.4f}, {ci['ci95'][1]:.4f}]。仅以当前固定模型为条件。",'',
        '## 全量预测分布','',
        '|预测类别|旧模型|当前最佳CNN|','|---|---:|---:|']
    for label in LABELS:lines.append(f"|{label}|{counts['old_n2'].get(label,0)}|{counts['best_cnn'].get(label,0)}|")
    lines += ['',f'全部记录中{len(changed):,}条分类发生变化；原3,332栋中{len(primary_changes):,}条发生变化。预测类别数量不等于真实用途数量。','',
        '## 参考可靠性分层','',
        '|分层|n|旧accuracy|新accuracy|旧macro-F1|新macro-F1|','|---|---:|---:|---:|---:|---:|']
    for cohort,value in results.items():
        a=value['old_n2'];b=value['best_cnn'];lines.append(f"|{cohort}|{a['n']}|{a['accuracy']:.2%}|{b['accuracy']:.2%}|{a['macro_f1']:.4f}|{b['macro_f1']:.4f}|")
    lines += ['',
        '主结果保持原3,332栋不变；3,315栋额外要求单条assessor建筑记录，作为更严格的敏感性分析。7,601及9,036栋使用较宽松的地块用途继承，可能把同地块附属屋顶标成住宅/多户，只能作探索性结果。它们不能替代主结果，也不是新增的建筑级金标准。各层每类precision/recall/F1详见METRICS.csv。','',
        '## 验证与限制','',
        'CPU在TAMU独立任务中执行，未暂停Qwen或Ollama。第一次尝试在数据加载检查时遇到默认共享内存不足，停止并保留日志后，使用已有的2Gi共享内存配置重新运行；失败尝试不计入最终预测。运行时长和校验结果保存在RESULTS.json及远端收回的WITNESS/RESULT记录中。','',
        '本次完成用途分类全量推理，不是重新检测建筑或评估损伤、人口及五类/角色输出。可靠参考仍主要依赖地块用途匹配；只有14栋多户。原LA已用于开发，不能据此证明未见城市泛化。恢复的136条包含坐标解释与影像重采样，且部分目标仍受固定256像素窗口裁切，单列input_status便于追溯。','',
        '[全部逐栋对照](PREDICTION_COMPARISON.csv) · [变化记录](CHANGED_PREDICTIONS.csv) · [完整指标](METRICS.csv) · [机器可读结果](RESULTS.json) · [验证记录](VERIFICATION.json)']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(counts=counts,changed=len(changed),primary=results['primary3332'],paired=paired['primary3332']),indent=2))


if __name__=='__main__':main()
