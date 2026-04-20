"""Build word↔token lookup tables from training data."""
import json
from collections import Counter

word2tok_cnt = Counter()
tok2word_cnt = Counter()

with open("data/tokenized/cst-ar-8k/train-100000.jsonl") as f:
    for line in f:
        d = json.loads(line)
        text = d.get("text", "")
        tokens = d["tokens"]
        words = text.split()
        toks = [t for t in tokens if t not in ("[BOS]", "[EOS]")]

        if len(words) == len(toks):
            for w, t in zip(words, toks):
                clean = w.rstrip("،؛.؟")
                word2tok_cnt[(clean, t)] += 1
                tok2word_cnt[(t, clean)] += 1

# word → most common token
w2t = {}
for (word, tok), _ in word2tok_cnt.most_common():
    if word not in w2t:
        w2t[word] = tok

# token → most common word
t2w = {}
for (tok, word), _ in tok2word_cnt.most_common():
    if tok not in t2w:
        t2w[tok] = word

print(f"word→token: {len(w2t)} entries")
print(f"token→word: {len(t2w)} entries")

tests = ["في", "من", "على", "إلى", "كان", "الماء", "الرجل", "يعمل", "المدرسة"]
for w in tests:
    print(f"  {w} → {w2t.get(w, '?')}")

print()
rev_tests = ["FUNC:PREP", "FUNC:CONJ", "FUNC:AUX", "ROOT:exist", "ROOT:speak", "ROOT:know", "ROOT:work"]
for t in rev_tests:
    print(f"  {t} → {t2w.get(t, '?')}")

with open("edge/demo/public/model/word2tok.json", "w") as f:
    json.dump(w2t, f, ensure_ascii=False)
with open("edge/demo/public/model/tok2word.json", "w") as f:
    json.dump(t2w, f, ensure_ascii=False)
print("\nSaved word2tok.json and tok2word.json")
