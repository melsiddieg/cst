# `data/tokenizer/` — shared tokenizer tables

Single source of truth for CST English tokenization rules. Used by both
runtimes so they can't drift out of sync:

- **TypeScript** (`src/tokenizer/`) — browser demo, CLI (`src/tokenize-cli.ts`),
  pipeline scripts, vitest suite. Currently the **authoritative** source.
- **Python** (`edge/english_tokenizer.py`) — training-data generator, unified
  CLI (`python -m edge.tokenize`). Loads these JSONs at import time.

## Contract

TS source → extraction → JSON → Python. Do not edit these JSONs directly.

```
src/tokenizer/semanticFields.ts   │
src/tokenizer/data.ts             │  npx tsx scripts/extract_tokenizer_data.ts
src/tokenizer/morphology.ts       │  ───────────────────────────────────────▶
src/tokenizer/cst-spec.ts         │       data/tokenizer/*.json
src/tokenizer/structureDetector.ts│
```

## Regenerate after editing TS

```bash
npx tsx scripts/extract_tokenizer_data.ts
```

## Verify sync (CI / pre-commit)

```bash
npx tsx scripts/extract_tokenizer_data.ts --check
```

Exits non-zero if any JSON drifts from TS source. Run this in CI.

## Files

| file                      | TS origin                      | shape                              |
| ------------------------- | ------------------------------ | ---------------------------------- |
| `semantic_fields.json`    | `SEMANTIC_FIELDS`              | `{lemma: field}`                   |
| `relation_map.json`       | `RELATION_MAP`                 | `{word: REL:token}`                |
| `function_words.json`     | `FUNCTION_WORDS`               | sorted string list                 |
| `prefix_roles.json`       | `PREFIX_ROLES`                 | `{prefix: role}`                   |
| `suffix_roles.json`       | `SUFFIX_ROLES`                 | `[[suffix, role]]` (order matters) |
| `structure_patterns.json` | inline `STRUCTURE_PATTERNS`    | `[{pattern, token}]`               |
| `cst_spec.json`           | multiple `cst-spec.ts` exports | spec metadata                      |

## Parity

TS and Python cannot be byte-identical — they use different lemmatizers
(`compromise` vs spaCy). On the 50-sentence sanity set:

- sentence-level exact match: ~54%
- token-type distribution: ≤6% divergence per type (CMP/ROOT/REL/STR/LIT)

The aggregate tokenization _shape_ is preserved; individual sentences may
differ in which semantic field a particular word lands in. For training,
Python is the source of truth — use `python -m edge.tokenize` to generate
corpora. TS is kept for the browser demo and as the authoritative rules
source.

Run `python scripts/check_tokenizer_parity.py` to see the current numbers.
