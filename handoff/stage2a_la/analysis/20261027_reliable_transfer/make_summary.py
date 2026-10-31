"""Summarize the independently scored runs; show every external class recall."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
report=json.loads((root/'RESULTS.json').read_text())
families={'DINO frozen':['dino_frozen_s42','dino_frozen_s43'],
          'DINO fine-tuned':['dino_finetune_s42','dino_finetune_s43'],
          'Qwen zero-shot':['qwen_nf4_zero'],
          'Qwen QLoRA':['qwen_qlora_s42','qwen_qlora_s43']}
classes=['residential_small','residential_multi','commercial','institutional','other']
summary=[];matrix=[];labels=[]
for city in ['cambridge','seattle']:
    for family,names in families.items():
        metrics=[report['results'][name][city] for name in names]
        use={key:float(np.mean([m['use'][key] for m in metrics])) for key in ['accuracy','macro_f1','coverage']}
        role={key:float(np.mean([m['role'][key] for m in metrics])) for key in ['accuracy','macro_f1','coverage']}
        recalls=[float(np.mean([m['use']['per_class'][c]['recall'] for m in metrics])) for c in classes]
        precisions=[float(np.mean([m['use']['per_class'][c]['precision'] for m in metrics])) for c in classes]
        aux=[m['role']['per_class']['auxiliary']['recall'] for m in metrics]
        aux_recall=None if aux[0] is None else float(np.mean(aux))
        aux_precision=None if aux[0] is None else float(np.mean([m['role']['per_class']['auxiliary']['precision'] for m in metrics]))
        selective=None
        if 'selective_use' in metrics[0]:
            selective=dict(coverage=float(np.mean([m['selective_use']['coverage'] for m in metrics])),
                           accuracy_among_answered=float(np.mean([m['selective_use']['accuracy']/m['selective_use']['coverage'] for m in metrics])))
        summary.append(dict(city=city,model=family,runs=names,use=use,role=role,
                            use_recall=dict(zip(classes,recalls)),use_precision=dict(zip(classes,precisions)),
                            auxiliary_recall=aux_recall,auxiliary_precision=aux_precision,selective_use=selective))
        matrix.append(recalls+[np.nan if aux_recall is None else aux_recall]);labels.append(city.title()+' | '+family)
values=np.array(matrix);assert values.shape==(8,6)
assert np.all((values[np.isfinite(values)]>=0)&(values[np.isfinite(values)]<=1))
(root/'MEAN_METRICS.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(10,5.8));colors=plt.get_cmap('Blues').copy();colors.set_bad('#e5e5e5')
im=ax.imshow(values,vmin=0,vmax=1,cmap=colors,aspect='auto')
ax.set_xticks(range(6),['Small residential','Multi residential','Commercial','Institutional','Other','Auxiliary role'],rotation=20,ha='right')
ax.set_yticks(range(8),labels);ax.axhline(3.5,color='white',lw=3);ax.axvline(4.5,color='white',lw=3)
for (i,j),value in np.ndenumerate(values):
    ax.text(j,i,'n/a' if np.isnan(value) else f'{value:.0%}',ha='center',va='center',color='white' if value>.65 else 'black',fontsize=11)
ax.set_title('Recall on held-out official-reference buildings',loc='left',pad=14,fontweight='bold')
fig.colorbar(im,ax=ax,fraction=.025,pad=.025,label='Recall')
fig.text(.02,.02,'NYC training and selection. Means of seeds 42/43; zero-shot is one control. Not an ensemble.\nCambridge: 250 use / 50 auxiliary references. Seattle: 244 use references; auxiliary recall unmeasured.',fontsize=8)
fig.tight_layout(rect=(0,.09,1,1));fig.savefig(root/'PER_CLASS_RECALL.png',dpi=220);fig.savefig(root/'PER_CLASS_RECALL.pdf');plt.close(fig)
print(json.dumps(summary,indent=2))
