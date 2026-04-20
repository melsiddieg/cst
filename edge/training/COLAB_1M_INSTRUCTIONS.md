# 1M Experiment — Colab Instructions

## Overview

Same 3-step flow as 100K, just bigger data:

1. **Tokenize** — Download 1M Arabic Wikipedia sentences + CST tokenize
2. **Train** — GPT-2 6.8M params, 5 epochs
3. **Export** — ONNX + int8 quantization

## Colab Steps

### Cell 1: Install deps

```python
!pip install camel-tools torch transformers onnx onnxruntime -q
!camel_data -i morphology-db-msa-r13
```

### Cell 2: Upload scripts

Upload these 2 files to `/content/`:

- `tokenize_1m.py`
- `colab_edge_1m.py`

Or use:

```python
from google.colab import files
uploaded = files.upload()  # select both .py files
```

### Cell 3: Tokenize (the slow part ~2-4 hours)

```python
!python tokenize_1m.py
```

This downloads 1M Arabic sentences from HuggingFace Wikipedia and CST-tokenizes them.
Output: `/content/cst_1m/train-1000000.jsonl` + `train-1000000-vocab.json`

**Tip:** If Colab disconnects, the sentences download is cached in `/content/sentences-1000000.json`.
Re-run and it will skip download.

### Cell 4: Train + Export (~30-45 min on T4)

```python
!python colab_edge_1m.py
```

Output in `/content/edge_model_1m/`:

- `model.pt` — PyTorch checkpoint
- `model.onnx` — ONNX fp32
- `model_int8.onnx` — ONNX int8 (~7-8 MB)
- `vocab.json` — Browser vocab

### Cell 5: Build word↔token lookups

```python
import json
from collections import Counter

word2tok_cnt = Counter()
tok2word_cnt = Counter()

with open("/content/cst_1m/train-1000000.jsonl") as f:
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

w2t = {}
for (word, tok), _ in word2tok_cnt.most_common():
    if word not in w2t:
        w2t[word] = tok

t2w = {}
for (tok, word), _ in tok2word_cnt.most_common():
    if tok not in t2w:
        t2w[tok] = word

print(f"word→token: {len(w2t)} entries")
print(f"token→word: {len(t2w)} entries")

with open("/content/edge_model_1m/word2tok.json", "w") as f:
    json.dump(w2t, f, ensure_ascii=False)
with open("/content/edge_model_1m/tok2word.json", "w") as f:
    json.dump(t2w, f, ensure_ascii=False)
print("Saved!")
```

### Cell 6: Download artifacts

```python
from google.colab import files
import os
for f in ["model_int8.onnx", "vocab.json", "word2tok.json", "tok2word.json"]:
    path = f"/content/edge_model_1m/{f}"
    if os.path.exists(path):
        files.download(path)
```

## After Download

Replace files in `edge/demo/public/model/` with the 1M versions and run the demo.

## Expected Results

| Metric          | 100K   | 1M (expected) |
| --------------- | ------ | ------------- |
| BPC             | 1.14   | ~0.9-1.0      |
| PPL             | 88     | ~40-60        |
| Vocab           | ~8K    | ~15-30K       |
| model_int8.onnx | 7.5 MB | ~7-8 MB       |
| Training time   | ~5 min | ~30-45 min    |
| Tokenize time   | ~1 hr  | ~2-4 hrs      |

## Estimated Total Time

- Tokenization: 2-4 hours (camel-tools is the bottleneck)
- Training: 30-45 min on T4
- Total: ~3-5 hours
