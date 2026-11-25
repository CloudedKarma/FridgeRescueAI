"""Synthetic target retrieval experiment, not a real-world preference study."""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from engine import Engine
from learned_ranker import features, fingerprint, FEATURES

METHODS=['Baseline','TF-IDF','KNN','Hybrid','Logistic regression','Random forest']
def ratings_features(e,path):
    if not path.exists() or 'recipe_id' not in e.df: return np.tile([3.,0.],(len(e.df),1))
    r=pd.read_csv(path,usecols=['recipe_id','rating']); r=r[r.rating.between(1,5)]
    g=r.groupby('recipe_id').rating.agg(['sum','count']); g=g.reindex(e.df.recipe_id).fillna(0)
    return np.column_stack([(g['sum']+15)/(g['count']+5),np.log1p(g['count'])])

def inventory(e,target,rng):
    items=sorted(e.sets[target]); keep=max(1,int(np.ceil(len(items)*rng.uniform(.4,.9))))
    inv=set(rng.choice(items,keep,replace=False)); extras=sorted(e.vocab-inv)
    inv.update(rng.choice(extras,min(len(extras),int(rng.integers(2,9))),replace=False)); return inv

def main(a):
    e=Engine(a.dataset); n=len(e.df); rng=np.random.default_rng(a.seed)
    if n<10 or np.any(e.sizes==0): raise ValueError('Need at least 10 nonempty recipes')
    order=rng.permutation(n); split=int(n*.7); train=order[:split]; test=order[split:]
    ratings=ratings_features(e,Path(a.ratings)); X=[]; y=[]
    for t in range(a.train_queries):
        target=int(rng.choice(train)); f=features(e,inventory(e,target,rng),ratings)
        # Hard negatives and random negatives, restricted to training target partition.
        pool=train[train!=target]; hard=pool[np.argsort(-f[pool,3],kind='stable')[:12]]
        neg=np.unique(np.r_[hard,rng.choice(pool,min(24,len(pool)),replace=False)])
        ids=np.r_[target,neg]; X.append(f[ids]); y.extend([1]+[0]*len(neg))
    X=np.vstack(X); y=np.array(y)
    models={'Logistic regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',random_state=a.seed)), 'Random forest':RandomForestClassifier(n_estimators=100,min_samples_leaf=5,class_weight='balanced',random_state=a.seed,n_jobs=-1)}
    for name,m in models.items():
        print('Training',name,flush=True); m.fit(X,y)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    joblib.dump({'models':models,'ratings':ratings,'fingerprint':fingerprint(a.dataset),'features':FEATURES},out/'rankers.joblib')
    rows=[]
    for q in range(a.test_queries):
        target=int(rng.choice(test)); inv=inventory(e,target,rng); f=features(e,inv,ratings)
        scores=[f[:,0],f[:,2],f[:,3],f[:,2]*.4+f[:,1]*.25+f[:,0]*.35]
        scores.extend(m.predict_proba(f)[:,1] for m in models.values())
        for name,s in zip(METHODS,scores):
            top=np.lexsort((np.arange(n),f[:,4],-s))[:5]; hits=np.flatnonzero(top==target)
            rows.append(dict(query=q,target=target,method=name,hit_at_5=int(bool(len(hits))),mrr_at_5=1/(hits[0]+1) if len(hits) else 0,missing=f[top,4].mean(),utilization=f[top,1].mean(),complete_rate=(f[top,4]==0).mean()))
        if (q+1)%50==0: print('Evaluated',q+1,flush=True)
    df=pd.DataFrame(rows); df.to_csv(out/'trials.csv',index=False)
    summary=df.groupby('method',sort=False)[['hit_at_5','mrr_at_5','missing','utilization','complete_rate']].mean(); summary.to_csv(out/'summary.csv'); print(summary.to_string())
    ci=[]; boot=np.random.default_rng(a.seed+1)
    for metric in summary.columns:
        p=df.pivot(index='query',columns='method',values=metric)
        for method in METHODS[1:]:
            d=(p[method]-p['Baseline']).to_numpy(); samples=boot.choice(d,(2000,len(d)),replace=True).mean(axis=1)
            ci.append(dict(method=method,metric=metric,difference=d.mean(),low=np.quantile(samples,.025),high=np.quantile(samples,.975)))
    pd.DataFrame(ci).to_csv(out/'paired_intervals.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(seed=a.seed,train_queries=a.train_queries,test_queries=a.test_queries,train_target_recipes=len(train),test_target_recipes=len(test),catalog=n,ratings_file_used=Path(a.ratings).exists() and 'recipe_id' in e.df,features=FEATURES,interpretation='Synthetic target retrieval; non-target candidates are unlabeled, not confirmed dislikes. Full catalog used at test; no human pantry observations.'),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset',default='recipes_foodcom.csv'); p.add_argument('--ratings',default='data/FoodCom/interactions_train.csv'); p.add_argument('--output',default='ml_results'); p.add_argument('--train-queries',type=int,default=1000); p.add_argument('--test-queries',type=int,default=300); p.add_argument('--seed',type=int,default=42); a=p.parse_args();
    if min(a.train_queries,a.test_queries)<1: p.error('Query counts must be positive')
    main(a)
