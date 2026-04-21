"""Phase 1 BPE baseline — tokenize the SAME 1M Arabic Wikipedia pool with BPE.

Reads the sentence pool cached by :mod:`tokenize_1m` (so both tokenizers see
identical input text), trains a SentencePiece BPE model, and emits a JSONL
file in the same schema as the CST output so that `colab_bpe_1m.py` (or
`colab_edge_1m.py`) can consume it unchanged.

Run (Colab):
  !pip install sentencepiece
  !python tokenize_bpe_1m.py                 # default vocab_size=8000
  !python tokenize_bpe_1m.py --vocab-size 4000

Output:
  /content/cst_1m/train-1000000-bpe.jsonl
  /content/cst_1m/train-1000000-bpe-vocab.json   # {token: id}
  /content/cst_1m/train-1000000-bpe.model        # SentencePiece model
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import sentencepiece as spm


# Special IDs — MUST match colab_edge_1m.py (PAD=0, UNK=1, BOS=2, EOS=3 … or
# whatever the CST vocab uses). The CST tokenizer in this repo uses:
#   PAD=0, UNK=1, BOS=2, EOS=3  (see edge/arabic_tokenizer.py)
# We align SentencePiece to the same layout for a fair training setup.
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def load_sentence_pool(target: int) -> list[str]:
    """Load the exact same 1M sentence pool that tokenize_1m.py used."""
    candidates = [
        f"/content/sentences-{target}.json",
        "/content/drive/MyDrive/cst-data/sentences-1M.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"  Loading sentence pool: {path}")
            with open(path) as f:
                sentences = json.load(f)
            print(f"  {len(sentences):,} sentences")
            return sentences[:target]
    raise FileNotFoundError(
        "No sentence pool found. Run tokenize_1m.py first so both tokenizers "
        f"see the same input. Looked in: {candidates}"
    )


def train_bpe(sentences: list[str], vocab_size: int, model_prefix: str) -> spm.SentencePieceProcessor:
    print(f"\n  Training SentencePiece BPE (vocab_size={vocab_size:,}) …")
    t0 = time.time()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for s in sentences:
            f.write(s + "\n")
        tmp_path = f.name
    try:
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=PAD_ID, pad_piece="<pad>",
            unk_id=UNK_ID, unk_piece="<unk>",
            bos_id=BOS_ID, bos_piece="<s>",
            eos_id=EOS_ID, eos_piece="</s>",
            input_sentence_size=min(len(sentences), 2_000_000),
            shuffle_input_sentence=True,
        )
    finally:
        os.unlink(tmp_path)
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    print(f"  Trained in {time.time()-t0:.0f}s  (real vocab: {sp.get_piece_size()})")
    return sp


def tokenize_all(sp: spm.SentencePieceProcessor, sentences: list[str], out_path: str) -> int:
    n = len(sentences)
    total_tok = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        for i, sent in enumerate(sentences):
            ids = sp.encode(sent, out_type=int)
            # Wrap with BOS/EOS to match CST's framing convention
            ids = [BOS_ID] + ids + [EOS_ID]
            if len(ids) < 4:
                continue
            total_tok += len(ids)
            f.write(json.dumps({"text": sent, "ids": ids}, ensure_ascii=False) + "\n")
            if (i + 1) % 100_000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n - i - 1) / rate / 60
                print(f"    {i+1:,} / {n:,}  ({rate:.0f} sent/s, ETA {eta:.0f}min)")
    print(f"  Wrote {out_path}  ({total_tok:,} tokens, {total_tok/n:.1f} avg tok/sent)")
    return total_tok


def save_vocab_json(sp: spm.SentencePieceProcessor, out_path: str) -> None:
    tok2id = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    with open(out_path, "w") as f:
        json.dump(tok2id, f, ensure_ascii=False)
    print(f"  Wrote {out_path}  ({len(tok2id)} tokens)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1_000_000,
                    help="Number of sentences (must match tokenize_1m.py TARGET)")
    ap.add_argument("--vocab-size", type=int, default=8000,
                    help="BPE vocabulary size. 8000 is a reasonable default; pass the "
                         "CST vocab size for a size-matched baseline.")
    ap.add_argument("--out-dir", default="/content/cst_1m")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 60)
    print(f"  Arabic BPE Tokenization — {args.target:,} sentences "
          f"(vocab_size={args.vocab_size:,})")
    print("=" * 60)

    sentences = load_sentence_pool(args.target)

    prefix = os.path.join(args.out_dir, f"train-{args.target}-bpe")
    sp = train_bpe(sentences, args.vocab_size, prefix)

    jsonl_path = f"{prefix}.jsonl"
    vocab_path = f"{prefix}-vocab.json"
    tokenize_all(sp, sentences, jsonl_path)
    save_vocab_json(sp, vocab_path)

    print("\n  ═══ Done ═══")
    print(f"  Data:   {jsonl_path}")
    print(f"  Vocab:  {vocab_path}")
    print(f"  Model:  {prefix}.model")
    print("\n  Next: run colab_bpe_1m.py (same GPT-2 config as CST run).")


if __name__ == "__main__":
    main()
