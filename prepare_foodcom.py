import argparse, ast, csv, json, random, re
from pathlib import Path
from collections import Counter
from ingredient_names import normalize_items

def parse_list(s):
    x=ast.literal_eval(s)
    if not isinstance(x,list): raise ValueError('Expected list')
    return x

def convert(folder, output, size, seed):
    # Use the supplied structured names without trying to parse recipe prose.
    rng=random.Random(seed); counts=Counter(); sample=[]; seen=set()
    with (folder/'RAW_recipes.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            counts['scanned']+=1
            try:
                ident=int(r['id']); names=parse_list(r['ingredients']); steps=parse_list(r['steps'])
                if not names or not steps or not r['name'].strip(): raise ValueError()
                if any(not isinstance(x,str) or not x.strip() or '|' in x or not any(c.isalpha() for c in x) for x in names): raise ValueError()
                if any(not isinstance(x,str) or not x.strip() for x in steps): raise ValueError()
                if ident in seen: counts['duplicate_id']+=1; continue
                seen.add(ident)
            except (ValueError,SyntaxError,TypeError): counts['invalid']+=1; continue
            normalized=sorted(normalize_items(names))
            if not normalized: counts['invalid']+=1; continue
            row=dict(name=r['name'],ingredients='|'.join(normalized),original_ingredients=json.dumps(names),directions=json.dumps(steps),source_link=f'https://www.food.com/recipe/{ident}',source='Food.com',recipe_id=ident,minutes=r.get('minutes',''))
            counts['valid']+=1
            if len(sample)<size: sample.append(row)
            else:
                j=rng.randrange(counts['valid'])
                if j<size: sample[j]=row
    if not sample: raise ValueError('No valid recipes')
    vocab=Counter(x for r in sample for x in r['ingredients'].split('|'))
    with output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(sample[0])); w.writeheader(); w.writerows(sample)
    report=dict(counts,sample_size=len(sample),seed=seed,unique_ingredients=len(vocab),singleton_ingredients=sum(n==1 for n in vocab.values()),method='structured raw names; conservative aliases and plural normalization; no ingredient ID mapping applied')
    output.with_suffix('.report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with output.with_suffix('.vocabulary.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['ingredient','recipe_count']); w.writerows(vocab.most_common())
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--folder',type=Path,default=Path('data/FoodCom')); p.add_argument('--output',type=Path,default=Path('recipes_foodcom.csv')); p.add_argument('--size',type=int,default=20000); p.add_argument('--seed',type=int,default=42); a=p.parse_args();
    if a.size<1: p.error('--size must be positive')
    convert(a.folder,a.output,a.size,a.seed)
