"""Evaluate frozen models once on held-out synthetic target recipes."""
import argparse,json,itertools,time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from engine import Engine
from learned_ranker import features,fingerprint
from train_models import inventory
from optimized_ranker import NAME,score,metrics
from optimize_models import COLS

def run(a):
    out=Path(a.output); dest=out/'final'; dest.mkdir(exist_ok=False)
    p=json.loads((out/'protocol.json').read_text()); e=Engine(a.dataset); bundle=joblib.load(out/'rankers.joblib')
    if fingerprint(a.dataset)!=p['dataset_sha256'] or bundle['fingerprint']!=p['dataset_sha256']: raise ValueError('Dataset changed')
    parts=np.load(out/'partitions.npz'); test=parts['test']; assert not set(test)&set(parts['previous_seen']); assert not set(test)&set(parts['train']); assert not set(test)&set(parts['validation'])
    rng=np.random.default_rng(p['seed']+3); targets=rng.choice(test,p['test_queries'],replace=False); rows=[]
    for q,t in enumerate(targets):
        f=features(e,inventory(e,int(t),rng),bundle['ratings']); methods={'Baseline':lambda:f[:,0],'TF-IDF':lambda:f[:,2],'KNN':lambda:f[:,3],'Hybrid':lambda:.4*f[:,2]+.25*f[:,1]+.35*f[:,0]}
        methods.update({name:lambda m=m:m.predict_proba(f)[:,1] for name,m in bundle['models'].items()}); methods[NAME]=lambda:score(f,bundle['models'],bundle['optimized_weights'])
        for name,fn in methods.items():
            start=time.perf_counter(); result=metrics(fn(),f,t); ms=(time.perf_counter()-start)*1000; rows.append(dict(query=q,target=int(t),method=name,**dict(zip(COLS,result)),ranking_ms=ms))
        if (q+1)%50==0: print('Final queries',q+1,flush=True)
    df=pd.DataFrame(rows); df.to_csv(dest/'trials.csv',index=False); summary=df.groupby('method',sort=False)[COLS+['ranking_ms']].mean(); summary.to_csv(dest/'summary.csv'); print(summary.to_string())
    boot=np.random.default_rng(p['seed']+4); ci=[]
    for metric in COLS:
        table=df.pivot(index='query',columns='method',values=metric); idx=boot.integers(0,len(table),(2000,len(table)))
        for ref,method in itertools.combinations(summary.index,2):
            d=(table[method]-table[ref]).to_numpy(); b=d[idx].mean(axis=1); ci.append(dict(reference=ref,method=method,metric=metric,difference=d.mean(),low=np.quantile(b,.025),high=np.quantile(b,.975)))
    pd.DataFrame(ci).to_csv(dest/'paired_intervals.csv',index=False)
    imp=bundle['models']['Random forest'].feature_importances_; coef=bundle['models']['Logistic regression'].steps[-1][1].coef_[0]; pd.DataFrame(dict(feature=bundle['features'],forest_importance=imp,standardized_logistic_coefficient=coef)).to_csv(dest/'feature_importance.csv',index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset',default='recipes_foodcom.csv'); p.add_argument('--output',default='optimized_results'); run(p.parse_args())
