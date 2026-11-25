"""Audit processed Fridge Rescue AI data without modifying it."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

# Review candidates, not an automatic deletion list.
FLAGS = set('filling|crust|topping|red|t|o|cubes|regular|marinade|n|crusty|chimney|porch|siding|for sauce'.split('|'))

def audit(path, output):
    counts = Counter()
    sources = Counter()
    flagged = []
    rows = 0
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'name', 'ingredients', 'original_ingredients', 'directions', 'source'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Input must be the processed recipe CSV')
        for row in reader:
            rows += 1
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'Malformed CSV record {rows}')
            for col in ('original_ingredients', 'directions'):
                if not isinstance(json.loads(row[col]), list):
                    raise ValueError(f'Invalid {col} at record {rows}')
            terms = {s.strip() for s in row['ingredients'].split('|') if s.strip()}
            counts.update(terms)
            sources[row['source']] += 1
            matches = sorted(terms & FLAGS)
            if matches:
                flagged.append((rows, row['name'], ', '.join(matches)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(recipes=rows, sources=dict(sources), unique_ingredients=len(counts), single_occurrence_ingredients=sum(v == 1 for v in counts.values()), recipes_with_selected_review_flags=len(flagged), selected_flag_counts={k: counts[k] for k in sorted(FLAGS)}, most_common=counts.most_common(30))
    (output/'audit_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (output/'ingredient_vocabulary.txt').open('w', encoding='utf-8') as f:
        f.write('Recipe count\tIngredient\n')
        for term, count in sorted(counts.items()):
            f.write(f'{count}\t{term}\n')
    with (output/'review_candidates.txt').open('w', encoding='utf-8') as f:
        f.write('Review candidates only; not confirmed errors. Record numbers exclude header.\n')
        for number, name, terms in flagged:
            f.write(f'{number}\t{name}\t{terms}\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input')
    p.add_argument('--output', default='dataset_audit')
    args = p.parse_args()
    audit(args.input, args.output)
