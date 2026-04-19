"""
Cap Arabic CST vocab to match SPM vocab sizes for fair comparison.

Keeps all semantic tokens (ROOT:*, FUNC:*) + top-N most frequent SURF tokens,
maps the rest to UNK.

Usage:
  python3 training/cap_cst_vocab_ar.py 8000
  python3 training/cap_cst_vocab_ar.py 32000
"""

import json
import sys
from collections import Counter
from pathlib import Path

SRC = Path("data/tokenized/cst-ar/train-100000.jsonl")


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Capping Arabic CST vocab to {cap}")

    # Pass 1: count token frequencies
    freq = Counter()
    rows = []
    with open(SRC) as f:
        for line in f:
            obj = json.loads(line.strip())
            tokens = obj["tokens"]
            freq.update(tokens)
            rows.append(obj)

    print(f"Loaded {len(rows)} examples, {len(freq)} unique tokens")

    # Separate semantic (ROOT:*, FUNC:*, [BOS], [EOS]) vs surface (SURF:*)
    semantic = {tok for tok in freq if not tok.startswith("SURF:")}
    surf_tokens = {tok for tok in freq if tok.startswith("SURF:")}
    print(f"Semantic tokens (ROOT/FUNC/special): {len(semantic)}")
    print(f"SURF tokens: {len(surf_tokens)}")

    # Budget: cap - semantic - 2 (for PAD + UNK)
    surf_budget = cap - len(semantic) - 2
    if surf_budget < 0:
        print(f"ERROR: {len(semantic)} semantic tokens already exceed cap {cap}")
        sys.exit(1)

    # Top SURF tokens by frequency
    surf_ranked = sorted(surf_tokens, key=lambda t: freq[t], reverse=True)
    keep_surf = set(surf_ranked[:surf_budget])
    drop_surf = set(surf_ranked[surf_budget:])

    print(f"Keeping {len(keep_surf)} SURF tokens, dropping {len(drop_surf)}")
    if keep_surf:
        min_freq_kept = freq[surf_ranked[min(surf_budget - 1, len(surf_ranked) - 1)]]
        print(f"Lowest kept SURF frequency: {min_freq_kept}")
    if drop_surf:
        total_drop_occurrences = sum(freq[t] for t in drop_surf)
        total_all_occurrences = sum(freq.values())
        print(f"Dropped tokens total occurrences: {total_drop_occurrences} ({total_drop_occurrences/total_all_occurrences*100:.1f}% of all token uses)")

    # Build new vocab: id 0 = PAD, 1 = UNK, then semantic, then SURF
    UNK = "<UNK>"
    new_vocab = {}
    new_vocab["<PAD>"] = {"token": "<PAD>", "id": 0, "type": "SPECIAL"}
    new_vocab[UNK] = {"token": UNK, "id": 1, "type": "SPECIAL"}

    next_id = 2
    for tok in sorted(semantic, key=lambda t: freq[t], reverse=True):
        new_vocab[tok] = {"token": tok, "id": next_id, "type": tok.split(":")[0] if ":" in tok else "SPECIAL"}
        next_id += 1

    for tok in surf_ranked[:surf_budget]:
        new_vocab[tok] = {"token": tok, "id": next_id, "type": "SURF"}
        next_id += 1

    print(f"Final vocab size: {len(new_vocab)}")
    assert len(new_vocab) <= cap, f"Vocab {len(new_vocab)} > cap {cap}"

    # Token-to-id mapping
    tok2id = {entry["token"]: entry["id"] for entry in new_vocab.values()}
    unk_id = tok2id[UNK]

    # Pass 2: remap IDs
    out_dir = Path(f"data/tokenized/cst-ar-{cap // 1000}k")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"train-{len(rows)}.jsonl"
    out_vocab = out_dir / f"train-{len(rows)}-vocab.json"

    unk_count = 0
    total_count = 0
    with open(out_jsonl, "w") as f:
        for obj in rows:
            new_ids = []
            for tok in obj["tokens"]:
                tid = tok2id.get(tok, unk_id)
                if tid == unk_id and tok != UNK:
                    unk_count += 1
                new_ids.append(tid)
                total_count += 1
            f.write(json.dumps({"ids": new_ids, "tokens": obj["tokens"], "text": obj["text"]}, ensure_ascii=False) + "\n")

    vocab_list = sorted(new_vocab.values(), key=lambda x: x["id"])
    with open(out_vocab, "w") as f:
        json.dump(vocab_list, f, ensure_ascii=False)

    print(f"\nWritten: {out_jsonl}")
    print(f"Written: {out_vocab}")
    print(f"UNK rate: {unk_count}/{total_count} = {unk_count / total_count * 100:.1f}%")

    max_id = 0
    with open(out_jsonl) as f:
        for line in f:
            for i in json.loads(line.strip())["ids"]:
                if i > max_id:
                    max_id = i
    print(f"Max token ID: {max_id} (vocab: {len(new_vocab)})")


if __name__ == "__main__":
    main()
