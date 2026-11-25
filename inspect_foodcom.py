import csv, json, ast
from pathlib import Path
from collections import Counter
root=Path('data/FoodCom'); report={}
for name in ['RAW_recipes.csv','PP_recipes.csv','RAW_interactions.csv','interactions_train.csv','interactions_validation.csv','interactions_test.csv']:
    with (root/name).open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f); examples=[]
        for _,row in zip(range(3),r): examples.append({k:v for k,v in row.items() if k in ['id','name','ingredients','ingredient_ids','recipe_id','rating','u','i']})
        report[name]={'columns':r.fieldnames,'examples':examples}
# Pickle files can execute code. Do not unpickle an unverified download.
report['ingr_map.pkl']={'exists':(root/'ingr_map.pkl').exists(),'bytes':(root/'ingr_map.pkl').stat().st_size if (root/'ingr_map.pkl').exists() else None,'note':'Not opened: pickle requires trusted provenance.'}
Path('foodcom_schema.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
