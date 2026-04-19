# CST-POC — Contextual Semantic Tokenization: Proof of Concept

## The Hypothesis

> If tokens carry pre-encoded semantic meaning (type, field, role) instead of arbitrary subword fragments, a transformer model can learn language with fewer parameters and less training data.

## Token Types

| Type | Format | Example | Meaning |
|------|--------|---------|---------|
| CMP | `CMP:field:role` | `CMP:write:agent` | writer (one who writes) |
| ROOT | `ROOT:field` | `ROOT:send` | send concept |
| REL | `REL:relation` | `REL:to` | logical relation |
| STR | `STR:marker` | `STR:question` | sentence structure |
| LIT | `LIT:surface` | `LIT:the` | literal/function word |

## Quick Start

```bash
npm install
npm test              # run all tests
npx tsx src/demo.ts   # see tokenizer in action
```

## Project Structure

```
src/
  tokenizer/
    index.ts            ← main CSTTokenizer class
    types.ts            ← Token, CoverageStats, etc.
    normalizer.ts       ← Stage 1: text cleaning
    structureDetector.ts ← Stage 2: STR token detection
    ner.ts              ← Stage 4: named entity detection
    morphology.ts       ← Stage 5: prefix/suffix decomposition
    semanticFields.ts   ← Stage 6: lemma → semantic field mapping
    emitter.ts          ← Stage 7: final token emission
    vocabulary.ts       ← token ↔ ID registry
    data.ts             ← relation map + function words
  tests/
    examples.test.ts    ← golden test cases from spec
    normalizer.test.ts
    structure.test.ts
    morphology.test.ts
  demo.ts               ← quick demo
```
