"""Optional figure generator; install requirements-plots.txt first."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1]/'reports/reproduction'
names=['retrieval_only','gbdt_full','gbdt_top50','lambdarank_top50']
labels=['Trend','GBDT full','GBDT Top50','LambdaRank Top50']
fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
colors=['#235789','#e07a3f']
for ax,metric in zip(axes,['Recall@10','NDCG@10']):
    positions=np.arange(len(names))
    for offset,protocol,color in zip([-.18,.18],['historical','strict'],colors):
        frame=pd.read_csv(root/protocol/'metrics.csv').set_index('experiment')
        ax.bar(positions+offset,frame.loc[names,metric],width=.34,label=protocol.capitalize(),color=color)
    ax.set_xticks(positions,labels,rotation=15,ha='right')
    ax.set_title(metric);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
axes[0].legend(frameon=False)
fig.suptitle('KuaiRec: repaired historical reproduction and strict chronological evaluation',fontsize=12)
fig.savefig(root/'ranking_comparison.png',dpi=180)
fig.savefig(root/'ranking_comparison.svg')
