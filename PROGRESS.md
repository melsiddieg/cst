# CST — Contextual Semantic Tokenizer

## What We Built

A 7-stage tokenizer that converts English text into typed semantic tokens instead of surface subwords.

**Pipeline:** normalize → structure detect → word split → NER → morphology → semantic field → emit

**Token types:**

- `CMP:field:role` — composed (e.g., `CMP:force:past` for "dropped")
- `ROOT:field` — semantic field only (e.g., `ROOT:time` for "soon")
- `REL:relation` — logical relation (e.g., `REL:into`)
- `STR:marker` — sentence structure
- `LIT:surface` — literal fallback

## Codebase

```
cst-poc/
  src/tokenizer/       # 7-stage tokenizer (index, normalizer, structureDetector, ner, morphology, semanticFields, emitter, vocabulary)
  src/pipeline/        # download, process, stream, stats
  src/tests/           # 30 tests (4 files)
  training/            # colab_train.py, train_gpt2.py
  data/tokenized/      # CST and BPE .jsonl outputs
```

**Key numbers:**

- 2,398 semantic field entries
- ~190 relation map entries
- Lemma cache for performance (~78K unique words cached)

## Coverage Results (7K sentences)

| Metric         | Value                       |
| -------------- | --------------------------- |
| Total tokens   | 154,828                     |
| CMP            | 16.9%                       |
| ROOT           | 19.2%                       |
| REL            | 28.2%                       |
| STR            | 1.8%                        |
| LIT            | 33.9%                       |
| UNK            | 0.0%                        |
| **Structured** | **66.1%** (gate >55%: PASS) |

## Training Results — Preliminary (100K sentences, 3 epochs, T4 GPU)

| Metric        | CST    | BPE (whitespace) | Delta      |
| ------------- | ------ | ---------------- | ---------- |
| Vocab size    | 67,996 | 77,868           | -12.7%     |
| Parameters    | 22.2M  | 24.7M            | -10.2%     |
| Best val loss | 5.389  | 5.673            | -5.0%      |
| Best val PPL  | 218.9  | 290.8            | **-24.7%** |

**CST wins: lower perplexity with smaller vocabulary and fewer parameters.**

## Known Weaknesses

1. **BPE baseline is too weak** — whitespace splitting, not real subword BPE (SentencePiece/tiktoken). Any reviewer rejects this.
2. **PPL not directly comparable** — CST predicts semantic tokens, BPE predicts surface words. Different output distributions.
3. **No downstream task evaluation** — need classification, NLI, or similarity task to prove utility.
4. **One model size, one dataset, one seed** — no error bars, no scaling curves.

## What's Next — Real Proof

### Phase 1: Fix Baseline

- [ ] Replace whitespace BPE with SentencePiece (real subword tokenizer)
- [ ] Retrain both on same 100K data

### Phase 2: Downstream Task

- [ ] Add SST-2 sentiment classification (or similar)
- [ ] Fine-tune both pretrained models on downstream task
- [ ] Compare accuracy, not just perplexity

### Phase 3: Scaling & Ablation

- [ ] Train 3 model sizes: 10M, 25M, 50M
- [ ] Run 3-5 seeds per config for error bars
- [ ] Scaling curve: 10K → 100K → 1M data
- [ ] Ablation: which CST components matter (remove morphology, remove fields, etc.)

### Phase 4: Paper

- [ ] Write up with proper experimental methodology
- [ ] Figures: loss curves, scaling plots, ablation table
- [ ] Target: workshop paper (ACL/EMNLP workshop)
