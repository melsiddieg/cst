"""Quick analysis of missed roots — test v2 coverage."""
import json, re, sys
from collections import Counter
sys.path.insert(0, '.')
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from training.arabic_experiment_v2 import _build_wildcard_index, ARABIC_FUNCTION_WORDS, PROCLITICS

root_index = _build_wildcard_index()
db = MorphologyDB.builtin_db()
analyzer = Analyzer(db)

with open('data/arabic/sentences-1000.json') as f:
    sentences = json.load(f)

missed = Counter()
hit = Counter()
func_hit = 0
norr = Counter()

def find_field(roots):
    for r in roots:
        if r in root_index:
            return root_index[r]
    return None

for sent in sentences:
    words = re.findall(r'[\u0600-\u06FF]+', sent)
    for w in words:
        clean = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', w)
        if clean in ARABIC_FUNCTION_WORDS:
            func_hit += 1
            continue
        aa = analyzer.analyze(clean)
        roots = [a.get('root','') for a in aa if a.get('root','') not in ('','NTWS','PUNC','DIGIT','FOREIGN')]
        field = find_field(roots)
        if field:
            hit[field] += 1
            continue
        # Try proclitic stripping
        found = False
        for prefix in PROCLITICS:
            if clean.startswith(prefix) and len(clean) > len(prefix) + 1:
                stem = clean[len(prefix):]
                aa2 = analyzer.analyze(stem)
                roots2 = [a.get('root','') for a in aa2 if a.get('root','') not in ('','NTWS','PUNC','DIGIT','FOREIGN')]
                field2 = find_field(roots2)
                if field2:
                    hit[field2] += 1
                    found = True
                    break
        if found:
            continue
        if roots:
            missed[roots[0]] += 1
        else:
            norr[clean] += 1

total_h = sum(hit.values())
total_m = sum(missed.values())
total_n = sum(norr.values())
total = total_h + total_m + total_n + func_hit
print(f'Function words: {func_hit} ({func_hit*100//total}%)')
print(f'Root+field: {total_h} ({total_h*100//total}%)')
print(f'Root no field: {total_m} ({total_m*100//total}%)')
print(f'No root: {total_n} ({total_n*100//total}%)')
print(f'Unique missed roots: {len(missed)}')
print(f'TOTAL words: {total}')
print(f'COVERAGE: {(func_hit + total_h)*100//total}%')
print()
print('TOP 30 MISSED:')
for r, c in missed.most_common(30):
    print(f'  {r:15s} {c:4d}')
