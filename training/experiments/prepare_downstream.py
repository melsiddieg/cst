"""
Downstream-task file prep.

Turns per-example pre-tokenized string arrays into the id-based format that
``downstream_eval.py`` consumes. Tokenization itself is done by the existing
project pipelines (TS for EN, ``edge/arabic_tokenizer.py`` for AR,
sentencepiece for SPM). This script only handles:

    - loading the trained vocab JSON (same one used at training time),
    - mapping token strings -> ids (UNK for misses),
    - writing the canonical jsonl shape expected by the evaluator.

Two input shapes are supported.

classification input (one line per example):
    {"tokens": ["tok1", "tok2", ...], "label": 0}

lm_scoring input (one line per example):
    {"context_tokens": ["...", "..."],
     "candidate_tokens": [["...", "..."], ["...", ...], ...],
     "gold": 0}

Output matches ``downstream_eval.py``:

classification:
    {"ids": [..], "label": 0}

lm_scoring:
    {"context_ids": [..], "candidates": [[..], [..], ...], "gold": 0}

Usage
-----
    # Classification (HARD, RuSentiment):
    python training/experiments/prepare_downstream.py \\
        --vocab data/tokenized/cst-ar-8k/train-100000-vocab.json \\
        --task classification \\
        --in  data/downstream/raw/hard-ar-cst-tokens-train.jsonl \\
        --out data/downstream/hard-ar-cst-8k-train.jsonl

    # LM scoring (LAMBADA):
    python training/experiments/prepare_downstream.py \\
        --vocab data/tokenized/cst-8k/train-99963-vocab.json \\
        --task lm_scoring \\
        --in  data/downstream/raw/lambada-en-cst-tokens.jsonl \\
        --out data/downstream/lambada-en-cst-8k.jsonl

Notes
-----
* Vocab JSON may be either a dict ``{tok: {"id": int, ...}}`` or a list of
  entries each with ``token`` and ``id``. Both shapes are produced by the
  existing ``cap_cst_vocab*.py`` scripts; we accept either.
* ``<UNK>`` is used if present in the vocab; otherwise id 1 (the convention
  used by ``cap_cst_vocab*.py``).
* ``<PAD>`` (id 0) is never emitted by this script; it's only used at
  collate time by the evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def load_vocab(path: str) -> tuple[dict[str, int], int]:
    """Return (token->id, unk_id)."""
    with open(path) as f:
        v = json.load(f)
    tok2id: dict[str, int] = {}
    if isinstance(v, dict):
        for tok, entry in v.items():
            if isinstance(entry, dict) and "id" in entry:
                tok2id[tok] = int(entry["id"])
            else:
                tok2id[tok] = int(entry)
    elif isinstance(v, list):
        for entry in v:
            tok2id[entry["token"]] = int(entry["id"])
    else:
        raise ValueError(f"Unrecognized vocab format in {path}")
    unk_id = tok2id.get("<UNK>", tok2id.get("[UNK]", 1))
    return tok2id, unk_id


def to_ids(tokens: list[str], tok2id: dict[str, int], unk_id: int) -> list[int]:
    return [tok2id.get(t, unk_id) for t in tokens]


def convert_classification(in_path: str, out_path: str,
                           tok2id: dict[str, int], unk_id: int) -> dict[str, Any]:
    n = 0
    unk_total = 0
    tok_total = 0
    labels: dict[int, int] = {}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(in_path) as fi, open(out_path, "w") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids = to_ids(obj["tokens"], tok2id, unk_id)
            if not ids:
                continue
            label = int(obj["label"])
            labels[label] = labels.get(label, 0) + 1
            unk_total += sum(1 for i in ids if i == unk_id)
            tok_total += len(ids)
            fo.write(json.dumps({"ids": ids, "label": label}) + "\n")
            n += 1
    return {
        "task": "classification",
        "n": n,
        "label_counts": labels,
        "unk_rate": unk_total / max(tok_total, 1),
        "out": out_path,
    }


def convert_lm_scoring(in_path: str, out_path: str,
                       tok2id: dict[str, int], unk_id: int) -> dict[str, Any]:
    n = 0
    unk_total = 0
    tok_total = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(in_path) as fi, open(out_path, "w") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ctx = to_ids(obj["context_tokens"], tok2id, unk_id)
            cands = [to_ids(c, tok2id, unk_id) for c in obj["candidate_tokens"]]
            if not ctx or not cands or any(not c for c in cands):
                continue
            gold = int(obj["gold"])
            unk_total += sum(1 for i in ctx if i == unk_id)
            unk_total += sum(sum(1 for i in c if i == unk_id) for c in cands)
            tok_total += len(ctx) + sum(len(c) for c in cands)
            fo.write(json.dumps({
                "context_ids": ctx, "candidates": cands, "gold": gold,
            }) + "\n")
            n += 1
    return {
        "task": "lm_scoring",
        "n": n,
        "unk_rate": unk_total / max(tok_total, 1),
        "out": out_path,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", required=True)
    p.add_argument("--task", choices=["classification", "lm_scoring"], required=True)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    tok2id, unk_id = load_vocab(args.vocab)
    print(f"  vocab: {len(tok2id):,} tokens, UNK id = {unk_id}")

    if args.task == "classification":
        info = convert_classification(args.inp, args.out, tok2id, unk_id)
    else:
        info = convert_lm_scoring(args.inp, args.out, tok2id, unk_id)

    print(f"  wrote {info['n']:,} rows -> {info['out']}")
    print(f"  UNK rate: {info['unk_rate']*100:.2f}%")
    if "label_counts" in info:
        print(f"  labels: {info['label_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
