import re
from ingredient_names import normalize_items
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
def normalize(x): return " ".join(str(x).lower().strip().split())
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


def parse(s):
    return normalize_items(str(s).split('|'))

class Engine:
    def __init__(self, path='recipes.csv'):
        self.df = pd.read_csv(path).dropna(subset=['name','ingredients']).reset_index(drop=True)
        self.sets = [parse(s) for s in self.df.ingredients]
        self.vectorizer = TfidfVectorizer(analyzer=lambda s: sorted(parse(s)))
        self.matrix = self.vectorizer.fit_transform(self.df.ingredients)
        self.binary = MultiLabelBinarizer(sparse_output=True).fit(self.sets)
        self.incidence = self.binary.transform(self.sets).tocsr()
        self.sizes = np.asarray(self.incidence.sum(axis=1)).ravel()
        self.vocab = set(self.binary.classes_)
        self.minutes = pd.to_numeric(self.df.get('minutes',pd.Series(0,index=self.df.index)),errors='coerce').fillna(0).clip(0,1440).to_numpy()
        self.path = path
        self.learned = None
    def load_learned(self, path):
        import joblib
        from learned_ranker import fingerprint, FEATURES
        bundle = joblib.load(path)  # only load locally generated models
        if bundle['fingerprint'] != fingerprint(self.path) or bundle['features'] != FEATURES:
            raise ValueError('Model does not match dataset; retrain it.')
        self.learned = bundle
    def rank(self, inventory, method='Hybrid', weights=(.4,.25,.35)):
        inv = normalize_items(inventory)
        query = self.binary.transform([inv & self.vocab])
        overlap = (self.incidence @ query.T).toarray().ravel()
        coverage = overlap / self.sizes
        utilization = overlap / max(len(inv),1)
        cosine = cosine_similarity(self.vectorizer.transform(['|'.join(inv)]),self.matrix)[0]
        jaccard = overlap / np.maximum(self.sizes + len(inv) - overlap, 1)
        w = np.asarray(weights,dtype=float)
        if w.sum() <= 0: raise ValueError('Weights must sum to a positive value')
        scores = {'Baseline':coverage,'TF-IDF':cosine,'KNN':jaccard,'Hybrid':(w @ np.vstack([cosine,utilization,coverage]))/w.sum()}
        if method in scores: scores = scores[method]
        elif self.learned and method == 'FROR (optimized)' and 'optimized_weights' in self.learned:
            from learned_ranker import features
            from optimized_ranker import score
            scores = score(features(self, inventory, self.learned['ratings']), self.learned['models'], self.learned['optimized_weights'])
        elif self.learned and method in self.learned['models']:
            from learned_ranker import features
            scores = self.learned['models'][method].predict_proba(features(self, inventory, self.learned['ratings']))[:,1]
        else: raise ValueError('Unknown or unavailable model: '+method)
        out = self.df.copy(); out['score']=scores; out['coverage']=coverage; out['utilization']=utilization
        out['missing']=[sorted(s-inv) for s in self.sets]; out['missing_count']=out.missing.map(len)
        return out.sort_values(['score','missing_count'],ascending=[False,True],kind='stable')
    def staples(self, inventory, k=5):
        inv=normalize_items(inventory); chosen=[]
        counts=Counter(x for s in self.sets for x in s)
        for _ in range(k):
            candidates=self.vocab-inv
            if not candidates: break
            gains=Counter(next(iter(missing)) for recipe in self.sets if len(missing := recipe-inv)==1)
            x=max(sorted(candidates),key=lambda x:(gains[x],counts[x]))
            chosen.append((x,gains[x],counts[x])); inv.add(x)
        return chosen
    def associations(self, inventory, min_support=.05):
        inv=normalize_items(inventory); n=len(self.sets); counts=Counter(x for s in self.sets for x in s); rows=[]
        for a in sorted(inv & self.vocab):
            pairs=Counter(b for recipe in self.sets if a in recipe for b in recipe-inv)
            for b,both in pairs.items():
                if both/n>=min_support:
                    rows.append((a,b,both/n,both/counts[a],both*n/(counts[a]*counts[b])))
        return sorted(rows,key=lambda r:(-r[4],-r[3],r[0],r[1]))
