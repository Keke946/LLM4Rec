"""Optional static figure; install requirements-plots.txt."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]/'reports/reproduction'
main=pd.read_csv(root/'strict/metrics.csv').set_index('experiment')
new=pd.read_csv(root/'completion/neural/metrics.csv').set_index('experiment')
rows=[main.loc['retrieval_only'],main.loc['gbdt_top50'],new.loc['embedding_mlp_top50'],new.loc['deepfm_top50'],new.loc['din_top50']]
labels=['Trend','GBDT Top50','MLP Top50','DeepFM Top50','DIN Top50']
colors=['#718096','#2a6f97','#3b8f75','#d99b43','#8c5e9f']
fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for ax,metric in zip(axes,['Recall@10','NDCG@10','MRR@10']):
    ax.bar(labels,[r[metric] for r in rows],color=colors)
    ax.tick_params(axis='x',rotation=40,labelsize=8)
    ax.set_title(metric,fontsize=11)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Strict KuaiRec: fixed Top50 head reranking experiments',fontsize=13)
fig.savefig(root/'completion/neural_top50_comparison.png',dpi=180)
fig.savefig(root/'completion/neural_top50_comparison.svg')
