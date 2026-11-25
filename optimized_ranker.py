"""Shared scoring and deterministic ranking for experiments and inference."""
import numpy as np
NAME = 'FROR (optimized)'
def components(f, models):
    return np.column_stack([models['Logistic regression'].predict_proba(f)[:,1], f[:,3], f[:,0], f[:,1], -np.minimum(f[:,4]/5,1)])
def score(f, models, weights):
    return components(f, models) @ np.asarray(weights)
def top_indices(s, f, k=5):
    return np.lexsort((np.arange(len(s)), f[:,4], -s))[:k]
def metrics(s,f,target,k=5):
    top=top_indices(s,f,k); hits=np.flatnonzero(top==target)
    return [int(bool(len(hits))), 1/(hits[0]+1) if len(hits) else 0, f[top,4].mean(),f[top,1].mean(),(f[top,4]==0).mean()]
