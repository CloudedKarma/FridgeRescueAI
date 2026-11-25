"""Supervised ranking features shared by training and inference."""
import hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ingredient_names import normalize_items

FEATURES = ['coverage','utilization','cosine','jaccard','missing','recipe_size','overlap','minutes','train_rating','train_rating_count']
def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def features(engine, inventory, ratings=None):
    inv=normalize_items(inventory)
    q=engine.binary.transform([inv & engine.vocab])
    overlap=(engine.incidence @ q.T).toarray().ravel()
    sizes=engine.sizes
    coverage=overlap/np.maximum(sizes,1)
    utilization=overlap/max(len(inv),1)
    cosine=cosine_similarity(engine.vectorizer.transform(['|'.join(sorted(inv))]),engine.matrix)[0]
    jaccard=overlap/np.maximum(sizes+len(inv)-overlap,1)
    return np.column_stack([coverage,utilization,cosine,jaccard,sizes-overlap,sizes,overlap,engine.minutes,ratings if ratings is not None else np.tile([3.,0.],(len(sizes),1))])
