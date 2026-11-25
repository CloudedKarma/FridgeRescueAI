"""Validation-only model selection followed by a held-out synthetic test."""
import argparse,json,itertools,time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from engine import Engine
from learned_ranker import features,fingerprint,FEATURES
from train_models import inventory,ratings_features
from optimized_ranker import NAME,components,metrics
COLS=['hit_at_5','mrr_at_5','missing','utilization','complete_rate']

def split(e,a):
    n=len(e.df); rng=np.random.default_rng(a.previous_seed); order=rng.permutation(n); cut=int(n*.7); oldtrain=order[:cut]; oldtest=order[cut:]
    # Replay the original random draws; feature computation itself consumed no RNG.
    for _ in range(a.previous_train_queries):
        target=int(rng.choice(oldtrain)); inventory(e,target,rng); pool=oldtrain[oldtrain!=target]; rng.choice(pool,min(24,len(pool)),replace=False)
    seen=set()
    for _ in range(a.previous_test_queries):
        target=int(rng.choice(oldtest)); seen.add(target); inventory(e,target,rng)
    rng2=np.random.default_rng(a.seed); fresh=rng2.permutation([i for i in oldtest if i not in seen]); nt=int(n*.2)
    if len(fresh)<nt: raise ValueError('Not enough previously unused target recipes for final partition')
    test=fresh[:nt]; tr=rng2.permutation(oldtrain)[:int(n*.6)]; val=np.setdiff1d(np.arange(n),np.r_[tr,test]); return tr,val,test,seen

def run(a):
    out=Path(a.output)
    if out.exists(): raise FileExistsError('Choose a new output folder; existing results are preserved.')
    e=Engine(a.dataset); n=len(e.df)
    if n<20 or np.any(e.sizes==0): raise ValueError('Need at least 20 nonempty recipes')
    if a.previous_bundle:
        # Locally generated trusted model only.
        prior=joblib.load(a.previous_bundle)
        if prior['fingerprint']!=fingerprint(a.dataset): raise ValueError('Previous dataset differs; cannot reconstruct held-out split')
    tr,val,test,seen=split(e,a); rng=np.random.default_rng(a.seed+1); ratings=ratings_features(e,Path(a.ratings)); X=[]; y=[]
    for q in range(a.train_queries):
        target=int(rng.choice(tr)); f=features(e,inventory(e,target,rng),ratings); pool=tr[tr!=target]; hard=pool[np.argsort(-f[pool,3],kind='stable')[:12]]; neg=np.unique(np.r_[hard,rng.choice(pool,min(24,len(pool)),replace=False)]); ids=np.r_[target,neg]; X.append(f[ids]); y.extend([1]+[0]*len(neg))
    X=np.vstack(X); y=np.asarray(y)
    candidates={'Logistic regression':[(str(c),make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000,class_weight='balanced',random_state=a.seed))) for c in [.1,1,10]],'Random forest':[(str(leaf),RandomForestClassifier(n_estimators=100,min_samples_leaf=leaf,class_weight='balanced',random_state=a.seed,n_jobs=-1)) for leaf in [3,10,25]]}
    for name,opts in candidates.items():
        for param,m in opts: print('Training',name,param,flush=True); m.fit(X,y)
    # Validation queries are fixed and reused for all parameter combinations.
    vr=np.random.default_rng(a.seed+2); vtargets=vr.choice(val,min(a.validation_queries,len(val)),replace=False); vinvs=[inventory(e,int(t),vr) for t in vtargets]
    stats={ (name,p):[] for name,opts in candidates.items() for p,m in opts}
    for i,(t,inv) in enumerate(zip(vtargets,vinvs)):
        f=features(e,inv,ratings)
        for name,opts in candidates.items():
            for p,m in opts: stats[name,p].append(metrics(m.predict_proba(f)[:,1],f,t))
        if (i+1)%50==0: print('Model validation',i+1,flush=True)
    rows=[]; models={}; chosen={}
    for name,opts in candidates.items():
        for p,m in opts: rows.append(dict(method=name,parameter=p,**dict(zip(COLS,np.mean(stats[name,p],axis=0)))))
        p,m=max(opts,key=lambda z:np.mean(stats[name,z[0]],axis=0)[1]); models[name]=m; chosen[name]=p
    # 70 nonnegative weight vectors, step .25; includes single-objective extremes.
    weights=np.array([w for w in itertools.product(range(5),repeat=5) if sum(w)==4],dtype=float)/4
    accum=np.zeros((len(weights),5))
    for i,(t,inv) in enumerate(zip(vtargets,vinvs)):
        f=features(e,inv,ratings); c=components(f,models); scores=c@weights.T
        for j in range(len(weights)): accum[j]+=metrics(scores[:,j],f,t)
        if (i+1)%50==0: print('Weight validation',i+1,flush=True)
    accum/=len(vtargets)
    # Prespecified compromise utility. Missing penalty normalized by five.
    utility=accum[:,1]+.5*accum[:,3]-.5*accum[:,2]/5+.25*accum[:,4]
    best=int(np.argmax(utility)); w=weights[best]; print('Selected weights',w,flush=True)
    out.mkdir(parents=True); pd.DataFrame(rows).to_csv(out/'model_validation.csv',index=False)
    pd.DataFrame(np.column_stack([weights,accum,utility]),columns=['relevance','jaccard','coverage','utilization_weight','missing_penalty']+COLS+['utility']).to_csv(out/'weight_validation.csv',index=False)
    joblib.dump(dict(models=models,ratings=ratings,fingerprint=fingerprint(a.dataset),features=FEATURES,optimized_weights=w),out/'rankers.joblib')
    protocol=dict(seed=a.seed,previous_seed=a.previous_seed,previous_train_queries=a.previous_train_queries,previous_test_queries=a.previous_test_queries,previous_seen_targets=len(seen),train_targets=len(tr),validation_targets=len(val),test_targets=len(test),train_queries=a.train_queries,validation_queries=len(vtargets),test_queries=min(a.test_queries,len(test)),weights=w.tolist(),parameters=chosen,utility='MRR + 0.5 utilization - 0.5 missing/5 + 0.25 complete_rate',dataset_sha256=fingerprint(a.dataset),prior_bundle_verified=bool(a.previous_bundle),interpretation='Synthetic target retrieval; not human preference or real food waste. Prior target split replay assumes unchanged catalog and original training code.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)); np.savez(out/'partitions.npz',train=tr,validation=val,test=test,previous_seen=np.array(sorted(seen)))
    print('Optimization finished. Run evaluate_final.py for held-out testing.',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset',default='recipes_foodcom.csv'); p.add_argument('--ratings',default='data/FoodCom/interactions_train.csv'); p.add_argument('--output',default='optimized_results'); p.add_argument('--previous-bundle',default='ml_results/rankers.joblib'); p.add_argument('--previous-seed',type=int,default=42); p.add_argument('--previous-train-queries',type=int,default=1000); p.add_argument('--previous-test-queries',type=int,default=300); p.add_argument('--seed',type=int,default=123); p.add_argument('--train-queries',type=int,default=2000); p.add_argument('--validation-queries',type=int,default=200); p.add_argument('--test-queries',type=int,default=500); a=p.parse_args(); run(a)
