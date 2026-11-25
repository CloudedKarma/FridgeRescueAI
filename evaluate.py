import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from engine import Engine

def main():
    p=argparse.ArgumentParser(); p.add_argument('--trials',type=int,default=300); p.add_argument('--seed',type=int,default=42); p.add_argument('--dataset',default='recipes.csv'); p.add_argument('--output',default='results'); a=p.parse_args()
    e=Engine(a.dataset); rng=np.random.default_rng(a.seed); vocab=sorted(e.vocab); rows=[]
    for trial in range(a.trials):
        # Synthetic inventories: sample a target recipe, retain 60% of its ingredients, add distractors.
        target=int(rng.integers(len(e.sets))); ingredients=sorted(e.sets[target]); n=max(1,round(.6*len(ingredients)))
        inv=set(rng.choice(ingredients,n,replace=False)); distractors=sorted(set(vocab)-e.sets[target]); inv.update(rng.choice(distractors,min(4,len(distractors)),replace=False))
        for method in ['Baseline','TF-IDF','KNN','Hybrid']:
            r=e.rank(inv,method); top=r.head(5); hit=target in top.index; pos=list(top.index).index(target)+1 if hit else 0
            rows.append(dict(trial=trial,method=method,hit_at_5=int(hit),reciprocal_rank_at_5=1/pos if pos else 0,mean_missing=top.missing_count.mean(),mean_utilization=top.utilization.mean(),mean_coverage=top.coverage.mean()))
    out=Path(a.output); out.mkdir(exist_ok=True,parents=True); df=pd.DataFrame(rows); df.to_csv(out/'trials.csv',index=False)
    metrics=list(df.columns[2:]); summary=df.groupby('method')[metrics].mean(); summary.to_csv(out/'summary.csv'); print(summary.round(3).to_string())
    # Paired bootstrap intervals for differences versus baseline.
    intervals=[]
    for m in ['TF-IDF','KNN','Hybrid']:
        for metric in metrics:
            pivot=df.pivot(index='trial',columns='method',values=metric); delta=(pivot[m]-pivot['Baseline']).to_numpy(); boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)])
            intervals.append(dict(method=m,metric=metric,difference=delta.mean(),lower=np.quantile(boot,.025),upper=np.quantile(boot,.975)))
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv',index=False)
if __name__=='__main__': main()
