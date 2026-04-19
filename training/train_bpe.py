"""
Train SentencePiece BPE model on raw text, then tokenize → .jsonl.
This produces a proper subword baseline comparable to CST.

Usage:
  python training/train_bpe.py [--vocab-size 8000] [--data data/sentences-100k.txt]

Reads the parquet-cached sentences or a sentences.json, trains BPE, outputs .jsonl.
"""

import argparse
import json
import os
import tempfile

import sentencepiece as spm


def load_sentences(path: str, max_count: int = 0) -> list[str]:
    """Load sentences from .json (array of strings) or .txt (one per line)."""
    if path.endswith(".json"):
        with open(path) as f:
            sentences = json.load(f)
    else:
        with open(path) as f:
            sentences = [line.strip() for line in f if line.strip()]
    if max_count > 0:
        sentences = sentences[:max_count]
    return sentences


def extract_sentences_from_jsonl(jsonl_path: str) -> list[str]:
    """Extract raw text from a tokenized .jsonl file."""
    sentences = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sentences.append(obj["text"])
    return sentences


def train_sentencepiece(sentences: list[str], vocab_size: int, model_prefix: str):
    """Train a SentencePiece BPE model from a list of sentences."""
    # Write sentences to temp file (SentencePiece needs a file)
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
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            num_threads=os.cpu_count() or 4,
        )
    finally:
        os.unlink(tmp_path)

    return spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")


def tokenize_to_jsonl(
    sp: spm.SentencePieceProcessor,
    sentences: list[str],
    output_path: str,
):
    """Tokenize sentences with SentencePiece and write .jsonl."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0

    with open(output_path, "w") as f:
        for sentence in sentences:
            ids = sp.encode(sentence, out_type=int)
            tokens = sp.encode(sentence, out_type=str)
            if len(ids) < 3:
                continue
            example = {
                "ids": ids,
                "tokens": tokens,
                "text": sentence,
            }
            f.write(json.dumps(example) + "\n")
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece BPE and tokenize")
    parser.add_argument("--vocab-size", type=int, default=8000,
                        help="BPE vocabulary size (default: 8000)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to sentences .json or .txt. If not provided, extracts from CST jsonl.")
    parser.add_argument("--data-size", type=int, default=100000,
                        help="Dataset size (matches stream.ts output naming)")
    args = parser.parse_args()

    n = args.data_size

    # Load sentences
    if args.data:
        print(f"Loading sentences from {args.data}...")
        sentences = load_sentences(args.data)
    else:
        # Extract from the CST jsonl (has the same raw text)
        cst_path = f"data/tokenized/cst/train-{n}.jsonl"
        if os.path.exists(cst_path):
            print(f"Extracting sentences from {cst_path}...")
            sentences = extract_sentences_from_jsonl(cst_path)
        elif os.path.exists("data/sentences.json"):
            print("Loading from data/sentences.json...")
            sentences = load_sentences("data/sentences.json", max_count=n)
        else:
            print("ERROR: No sentence source found. Run stream.ts first.")
            return

    print(f"  {len(sentences):,} sentences")

    # Train SentencePiece
    model_prefix = f"data/tokenized/spm/bpe-{args.vocab_size}"
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

    print(f"\nTraining SentencePiece BPE (vocab_size={args.vocab_size})...")
    sp = train_sentencepiece(sentences, args.vocab_size, model_prefix)
    print(f"  Model saved: {model_prefix}.model")
    print(f"  Actual vocab size: {sp.get_piece_size()}")

    # Tokenize
    output_path = f"data/tokenized/spm/train-{len(sentences)}.jsonl"
    print(f"\nTokenizing → {output_path}...")
    count = tokenize_to_jsonl(sp, sentences, output_path)
    print(f"  {count:,} examples written")

    # Also save vocab as JSON for the training script
    vocab_path = output_path.replace(".jsonl", "-vocab.json")
    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"  Vocab saved: {vocab_path}")

    # Quick stats
    total_tokens = 0
    total_sentences = 0
    with open(output_path) as f:
        for line in f:
            obj = json.loads(line.strip())
            total_tokens += len(obj["ids"])
            total_sentences += 1

    avg_tokens = total_tokens / total_sentences if total_sentences else 0
    print(f"\n═══ SentencePiece BPE Stats ═══")
    print(f"  Sentences:       {total_sentences:,}")
    print(f"  Total tokens:    {total_tokens:,}")
    print(f"  Avg tokens/sent: {avg_tokens:.1f}")
    print(f"  Vocab size:      {sp.get_piece_size():,}")


if __name__ == "__main__":
    main()
