# CST Architecture

A one-page overview of how Contextual Semantic Tokenization is organised in this repository.

## High-level picture

```
                 ┌─────────────── DEFAULT LEVEL (T_D) ───────────────┐
text  ──►  normalize  ──►  segment  ──►  morphology  ──►  CST tokens  ──►  LM
                                                           │
                                                           ▼
                                                 REASONING LEVEL (T_R)
                                                 projection rules → dense
                                                 [BOS] CMP:field:role
                                                       LIT:kind:value …
                                                       [EOS] sequences
```

Two tokenization levels share one pipeline:

| Level     | Purpose                                         | Produces                                          | Home                                                                                                                                                |
| --------- | ----------------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Default   | Language-model training / inference             | `ROOT:field`, `INF:tense`, `REL:…`, `STR:…`, etc. | [`src/tokenizer/`](src/tokenizer), [`edge/arabic_tokenizer.py`](edge/arabic_tokenizer.py), [`edge/english_tokenizer.py`](edge/english_tokenizer.py) |
| Reasoning | Dense structured token stream for reasoning LMs | `[BOS] CMP:field:role LIT:kind:value … [EOS]`     | [`reasoning/tokenizer/`](reasoning/tokenizer) + [`arabic-algebra-engine`](../arabic-algebra/arabic-algebra-engine)                                  |

See [docs/spec/two-level-tokenization.md](docs/spec/two-level-tokenization.md) for the formal specification.

## TypeScript pipeline (`src/`)

```
src/
├── tokenizer/           # 7-stage core pipeline
│   ├── normalizer.ts    # NFKC + contractions + number/punct rules
│   ├── morphology.ts    # root / pattern extraction
│   ├── semanticFields.ts
│   ├── structureDetector.ts
│   ├── ner.ts           # compromise.js wrapper
│   ├── emitter.ts       # produces CST tokens + validates shape
│   ├── vocabulary.ts    # id ↔ string tables
│   ├── data.ts          # field + lemma dictionaries
│   ├── cst-spec.ts      # token type + role inventory
│   ├── types.ts
│   └── index.ts         # public CSTTokenizer class
├── pipeline/            # batch jobs on top of the tokenizer
│   ├── download.ts      # corpus acquisition (HF Datasets)
│   ├── process.ts       # tokenize corpus
│   ├── stats.ts         # coverage / BPC stats
│   └── stream.ts        # streaming utilities
├── tests/               # vitest suite
└── demo.ts              # runnable demo
```

## Edge model (`edge/`)

Python reference implementations + Colab training. Produces the quantized ONNX model that powers the in-browser demo.

- `arabic_tokenizer.py`, `english_tokenizer.py` — Python twins of the TS tokenizer.
- `build_lookups.py` — regenerates the semantic-field lookup tables consumed by both TS and Python.
- `artifacts/` — trained models (`model.onnx`, `model_int8.onnx`, `vocab.json`).
- `training/` — Colab scripts for the 100K and 1M experiments.

Parity between TS and Python is enforced by [`scripts/check_tokenizer_parity.py`](scripts/check_tokenizer_parity.py).

## Reasoning track (`reasoning/`)

Builds the reasoning-level corpus and tokenizer:

- `tokenizer/arabic.py`, `tokenizer/english.py` — wrap the default tokenizer and apply projection rules.
- `data/build.py` + `data/sources/` — curates entailment / CoT / instruction data.
- `tokenize_corpus.py` — produces `[BOS] … [EOS]` JSONL ready for training.

The [Arabic Algebra Engine](../arabic-algebra/arabic-algebra-engine) emits the same token vocabulary (`CMP:field:role`, `LIT:kind:value`), so reasoning-model training can concatenate both sources.

## Training (`training/`)

GPT-2 reference training code for the scaling sweeps, ablations and multi-seed runs described in [docs/plans/TRAINING_PLAN.md](docs/plans/TRAINING_PLAN.md).

## Scripts (`scripts/`)

Utility code used outside the tokenizer hot-path. See [scripts/README.md](scripts/README.md).

## Where to go next

- Concepts and math: [docs/paper/cst-paper.md](docs/paper/cst-paper.md)
- Everyday use: [README.md](README.md) quick-start
- Planned work: [ROADMAP.md](ROADMAP.md), [docs/plans/](docs/plans)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
