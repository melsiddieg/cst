"""Cap the 1M CST vocabulary AFTER tokenization — no re-tokenize needed.

Why: the Arabic CST tokenizer mints a unique `LIT:<word>` (and sometimes
`ROOT:<root>`) token for every unseen surface form. On 1M Wikipedia
sentences this produced ~517K unique tokens, which inflated a 6.8M-param
GPT-2 into a 137M-param model (embedding matrix = vocab × n_embd).

This script keeps all *structural* tokens (CMP / REL / STR / FEAT / NER /
special / reserved ROOT:<field>) and the top-N most frequent *literal*
tokens (LIT:*, ROOT:<arbitrary>), mapping the rest to `[UNK]`.

Usage (Colab):
  !python cap_vocab_1m.py --cap 8000
  !python cap_vocab_1m.py --cap 4000 \
      --in-jsonl /content/cst_1m/train-1000000.jsonl \
      --in-vocab /content/cst_1m/train-1000000-vocab.json \
      --out-jsonl /content/cst_1m/train-1000000-cap8k.jsonl \
      --out-vocab /content/cst_1m/train-1000000-cap8k-vocab.json

After capping, point `colab_edge_1m.py` at the capped files:
  DATA_FILE  = "train-1000000-cap8k.jsonl"
  VOCAB_FILE = "train-1000000-cap8k-vocab.json"
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter


# Token prefixes treated as *structural* — always kept regardless of frequency.
# Matches the schema in edge/arabic_tokenizer.py.
STRUCTURAL_PREFIXES = ("CMP:", "REL:", "STR:", "FEAT:", "NER:")
SPECIAL_TOKENS = {"[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"}

# Reserved semantic-field ROOT tokens (pre-registered in ArabicCSTTokenizer.__init__).
# Any other ROOT:<x> is treated as literal (unseen root) and subject to capping.
# We detect "reserved" by: ROOT:<lowercase_ascii_word> (fields are ASCII names
# like ROOT:write, ROOT:health, ROOT:size). Roots from raw Arabic text are
# non-ASCII and thus literal.

def is_structural(tok: str) -> bool:
    if tok in SPECIAL_TOKENS:
        return True
    if tok.startswith(STRUCTURAL_PREFIXES):
        return True
    if tok.startswith("ROOT:"):
        tail = tok[5:]
        # ASCII-only tail → reserved semantic-field ROOT → structural
        return tail.isascii() and tail.replace("_", "").isalpha()
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=8000, help="Target max vocab size")
    ap.add_argument("--in-jsonl", default="/content/cst_1m/train-1000000.jsonl")
    ap.add_argument("--in-vocab", default="/content/cst_1m/train-1000000-vocab.json")
    ap.add_argument("--out-jsonl", default=None)
    ap.add_argument("--out-vocab", default=None)
    args = ap.parse_args()

    out_jsonl = args.out_jsonl or args.in_jsonl.replace(
        ".jsonl", f"-cap{args.cap//1000}k.jsonl")
    out_vocab = args.out_vocab or args.in_vocab.replace(
        "-vocab.json", f"-cap{args.cap//1000}k-vocab.json")

    print("=" * 60)
    print(f"  Cap CST vocab → {args.cap:,}")
    print("=" * 60)
    print(f"  Input jsonl:  {args.in_jsonl}")
    print(f"  Input vocab:  {args.in_vocab}")
    print(f"  Output jsonl: {out_jsonl}")
    print(f"  Output vocab: {out_vocab}")

    # Load original vocab {token: old_id}
    with open(args.in_vocab) as f:
        old_vocab: dict[str, int] = json.load(f)
    old_id_to_tok = {v: k for k, v in old_vocab.items()}
    print(f"\n  Original vocab: {len(old_vocab):,} tokens")

    # Pass 1: count frequencies over the jsonl
    print("  Counting token frequencies ...")
    freq: Counter[int] = Counter()
    n_rows = 0
    total_tokens = 0
    with open(args.in_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids = obj["ids"]
            freq.update(ids)
            n_rows += 1
            total_tokens += len(ids)
    print(f"  Rows: {n_rows:,}  | Tokens: {total_tokens:,}")

    # Partition by structural vs literal
    structural_ids = [tid for tid in old_id_to_tok if is_structural(old_id_to_tok[tid])]
    literal_ids = [tid for tid in old_id_to_tok if not is_structural(old_id_to_tok[tid])]
    print(f"  Structural tokens: {len(structural_ids):,}")
    print(f"  Literal tokens:    {len(literal_ids):,}")

    # Budget for literals
    lit_budget = args.cap - len(structural_ids)
    if lit_budget < 0:
        raise SystemExit(
            f"ERROR: structural tokens alone ({len(structural_ids)}) exceed cap {args.cap}. "
            f"Raise --cap."
        )

    # Keep top-N literals by frequency
    literal_ids_sorted = sorted(literal_ids, key=lambda tid: -freq[tid])
    keep_literal = set(literal_ids_sorted[:lit_budget])
    dropped = set(literal_ids_sorted[lit_budget:])

    kept_lit_uses = sum(freq[t] for t in keep_literal)
    dropped_lit_uses = sum(freq[t] for t in dropped)
    print(f"  Keeping {len(keep_literal):,} literals (budget {lit_budget:,})")
    print(f"  Dropping {len(dropped):,} literals  "
          f"({dropped_lit_uses:,} uses = {dropped_lit_uses/max(total_tokens,1)*100:.2f}% of all tokens)")

    # Build new vocab. Keep canonical layout: PAD=0, UNK=1, BOS=2, EOS=3, SEP=4.
    CANONICAL = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]
    new_vocab: dict[str, int] = {}
    for i, tok in enumerate(CANONICAL):
        if tok in old_vocab:
            new_vocab[tok] = i
        else:
            # Still reserve the slot so PAD/UNK/BOS/EOS/SEP stay at fixed IDs
            new_vocab[tok] = i

    next_id = len(CANONICAL)
    # Other structural tokens (alphabetical for determinism)
    for tid in sorted(structural_ids, key=lambda t: old_id_to_tok[t]):
        tok = old_id_to_tok[tid]
        if tok in new_vocab:
            continue
        new_vocab[tok] = next_id
        next_id += 1

    # Kept literals, by descending frequency (most common get low IDs)
    for tid in literal_ids_sorted[:lit_budget]:
        tok = old_id_to_tok[tid]
        new_vocab[tok] = next_id
        next_id += 1

    print(f"  Final vocab size: {len(new_vocab):,}")

    # Build old→new remap (dropped tokens → UNK_ID)
    UNK_ID = new_vocab["[UNK]"]
    remap: dict[int, int] = {}
    for tok, new_id in new_vocab.items():
        if tok in old_vocab:
            remap[old_vocab[tok]] = new_id
    for tid in dropped:
        remap[tid] = UNK_ID
    # Any ID present in data but missing from old_vocab (shouldn't happen) → UNK
    # (handled defensively in the loop below)

    # Pass 2: rewrite jsonl
    print("  Rewriting jsonl ...")
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    unk_uses = 0
    with open(args.in_jsonl) as fin, open(out_jsonl, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            new_ids = []
            for tid in obj["ids"]:
                mapped = remap.get(tid, UNK_ID)
                if mapped == UNK_ID and tid != old_vocab.get("[UNK]", -1):
                    unk_uses += 1
                new_ids.append(mapped)
            obj["ids"] = new_ids
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(out_vocab, "w") as f:
        json.dump(new_vocab, f, ensure_ascii=False)

    print(f"\n  UNK substitutions: {unk_uses:,} "
          f"({unk_uses/max(total_tokens,1)*100:.2f}% of all tokens)")
    print(f"\n  Wrote {out_jsonl}")
    print(f"  Wrote {out_vocab}")
    print("\n  Next: edit colab_edge_1m.py to point at these files, then rerun.")


if __name__ == "__main__":
    main()
