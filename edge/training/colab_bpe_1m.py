"""Phase 1 BPE baseline — train the SAME 6.8M GPT-2 on BPE-tokenized 1M data.

This is a thin wrapper over the CST trainer (`colab_edge_1m.py`) that swaps the
input file/vocab and writes a `summary_bpe.json` at the end. The model config
(N_EMBD=256, N_LAYER=6, N_HEAD=4, MAX_LEN=128), optimizer, LR, and epoch count
are intentionally identical to the CST run so BPC is directly comparable.

Prerequisites:
  1. Run tokenize_1m.py        → produces CST-tokenized 1M + vocab
  2. Run tokenize_bpe_1m.py    → produces BPE-tokenized 1M + vocab
     (both read the SAME sentences pool)

Run:
  !python colab_bpe_1m.py
"""
from __future__ import annotations

import json
import os
import sys

# Reuse every training function + config constant from the CST trainer so the
# two runs are guaranteed identical except for the input data.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import colab_edge_1m as cst  # noqa: E402


# ── Override input data ──
DATA_DIR = "/content/cst_1m"
DATA_FILE = "train-1000000-bpe.jsonl"
VOCAB_FILE = "train-1000000-bpe-vocab.json"
OUT_DIR = "/content/edge_model_1m_bpe"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data_path = os.path.join(DATA_DIR, DATA_FILE)
    vocab_path = os.path.join(DATA_DIR, VOCAB_FILE)

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run tokenize_bpe_1m.py first.")
        return
    if not os.path.exists(vocab_path):
        print(f"ERROR: {vocab_path} not found. Run tokenize_bpe_1m.py first.")
        return

    print(f"Loading {DATA_FILE} …")
    ids_list, char_counts = cst.load_jsonl(data_path, cst.MAX_LEN)

    with open(vocab_path) as f:
        vocab = json.load(f)
    vocab_size = max(vocab.values()) + 1
    print(f"  Sentences: {len(ids_list):,} | Vocab: {vocab_size:,}")

    split_idx = int(len(ids_list) * (1 - cst.VAL_RATIO))
    model, config, best_state, best_bpc, n_params = cst.train_model(
        train_ids=ids_list[:split_idx],
        train_chars=char_counts[:split_idx],
        val_ids=ids_list[split_idx:],
        val_chars=char_counts[split_idx:],
        vocab_size=vocab_size,
    )

    model.load_state_dict(best_state)
    ckpt_path = os.path.join(OUT_DIR, "model.pt")
    import torch
    torch.save(best_state, ckpt_path)
    print(f"\n  Checkpoint: {ckpt_path}")

    summary = {
        "tokenizer": "bpe",
        "data_file": DATA_FILE,
        "n_sentences": len(ids_list),
        "vocab_size": vocab_size,
        "n_params": n_params,
        "best_val_bpc": best_bpc,
        "config": {
            "n_embd": cst.N_EMBD, "n_layer": cst.N_LAYER, "n_head": cst.N_HEAD,
            "max_len": cst.MAX_LEN, "epochs": cst.EPOCHS, "batch_size": cst.BATCH_SIZE,
            "lr": cst.LR,
        },
    }
    summary_path = os.path.join(OUT_DIR, "summary_bpe.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary:    {summary_path}")

    print(f"\n{'='*60}")
    print(f"  DONE — Arabic BPE baseline 1M")
    print(f"{'='*60}")
    print(f"  Params:     {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Vocab:      {vocab_size:,}")
    print(f"  Best BPC:   {best_bpc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
