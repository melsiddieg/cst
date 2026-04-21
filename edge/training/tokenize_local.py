"""Local CST tokenization driver — scalable 10M / 50M / 100M runs.

Designed to run on a workstation (e.g. i7 / 16–32 GB RAM) so Colab time
is spent only on training, not on tokenization. Produces a single
``.jsonl`` + vocab ``.json`` pair that you upload to Colab / Drive.

Differences vs ``tokenize_1m.py``:

* **Streams everything.** Sentences are never fully materialized in
  memory; tokenized lines are written as they are produced. 100M runs
  fit in < 4 GB RSS.
* **Multi-source.** Arabic Wikipedia alone caps ≈ 11M usable sentences.
  Above that the script falls back to CulturaX (Arabic) and then OSCAR
  (Arabic). Both are HuggingFace datasets in streaming mode; no full
  download to disk.
* **Multi-process tokenization.** CAMeL analyzer is the bottleneck;
  we fan out across CPU cores with ``multiprocessing.Pool``.
* **Resumable.** If the output file exists and is partial, the script
  skips already-written sentence indices and continues.

Usage
-----

    # On the local workstation
    pip install -r requirements.txt
    camel_data -i morphology-db-msa-r13

    # Optional: HF token for gated datasets (CulturaX, OSCAR)
    export HF_TOKEN=hf_xxx

    python tokenize_local.py --target 10M  --out ./out
    python tokenize_local.py --target 50M  --out ./out
    python tokenize_local.py --target 100M --out ./out

Output layout (per run)::

    out/
      sentences-10M.txt        # one sentence per line (for reuse)
      cst-train-10M.jsonl      # tokenized, one JSON per line
      cst-train-10M-vocab.json # vocab table

Upload only the ``cst-train-<N>.jsonl`` + ``-vocab.json`` pair to
Colab / Drive; the raw sentences file is local-only.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

# ── Make sibling library importable ─────────────────────────────
_HERE = Path(__file__).resolve().parent
_EDGE = _HERE.parent
if str(_EDGE) not in sys.path:
    sys.path.insert(0, str(_EDGE))


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

SIZE_MAP = {
    "1M":   1_000_000,
    "10M":  10_000_000,
    "50M":  50_000_000,
    "100M": 100_000_000,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--target",
        required=True,
        choices=list(SIZE_MAP.keys()),
        help="How many sentences to extract + tokenize.",
    )
    p.add_argument(
        "--out",
        default="./out",
        help="Output directory (default: ./out).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Tokenizer worker processes (default: cpu_count - 1).",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=1000,
        help="Sentences per worker task (default: 1000).",
    )
    p.add_argument(
        "--min-len",
        type=int,
        default=20,
        help="Min sentence length in chars (default: 20).",
    )
    p.add_argument(
        "--max-len",
        type=int,
        default=300,
        help="Max sentence length in chars (default: 300).",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# Sentence extraction — streaming, multi-source
# ═══════════════════════════════════════════════════════════════

_AR_RANGE = ("\u0600", "\u06FF")
_SENT_SPLIT = re.compile(r"[.؟!\n]\s*")


def _is_good(sent: str, min_len: int, max_len: int) -> bool:
    if len(sent) < min_len or len(sent) > max_len:
        return False
    ar = sum(1 for c in sent if _AR_RANGE[0] <= c <= _AR_RANGE[1])
    return ar >= len(sent) * 0.5


def _iter_wikipedia():
    """Yield raw article text from Arabic Wikipedia (streaming)."""
    from datasets import load_dataset

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.ar",
        split="train",
        streaming=True,
    )
    for row in ds:
        text = row.get("text") or ""
        if text:
            yield text


def _iter_culturax():
    """Yield raw article text from CulturaX Arabic (streaming, gated)."""
    from datasets import load_dataset

    ds = load_dataset(
        "uonlp/CulturaX",
        "ar",
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )
    for row in ds:
        text = row.get("text") or ""
        if text:
            yield text


def _iter_oscar():
    """Yield raw text from OSCAR 2301 Arabic (streaming, gated)."""
    from datasets import load_dataset

    ds = load_dataset(
        "oscar-corpus/OSCAR-2301",
        "ar",
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )
    for row in ds:
        text = row.get("text") or ""
        if text:
            yield text


# Order matters: cleanest source first, fall through on exhaustion.
SOURCES = [
    ("wikipedia", _iter_wikipedia),
    ("culturax", _iter_culturax),
    ("oscar", _iter_oscar),
]


def extract_sentences(target: int, sentences_path: Path,
                      min_len: int, max_len: int) -> int:
    """Write ``target`` filtered sentences to ``sentences_path`` (one per line).

    Returns the number of sentences actually written. Deduplicates on
    exact match via an in-memory set (costs ~50 bytes/sentence).
    """
    if sentences_path.exists():
        existing = sum(1 for _ in sentences_path.open("r", encoding="utf-8"))
        if existing >= target:
            print(f"  Sentences already extracted: {existing:,}")
            return existing
        print(f"  Found {existing:,} existing sentences; extending to {target:,}")

    seen: set[int] = set()
    written = 0
    t0 = time.time()

    # Resume: seed `seen` with hashes of existing lines.
    mode = "a" if sentences_path.exists() else "w"
    if sentences_path.exists():
        with sentences_path.open("r", encoding="utf-8") as f:
            for line in f:
                seen.add(hash(line.rstrip("\n")))
                written += 1

    out = sentences_path.open(mode, encoding="utf-8")
    try:
        for src_name, src_iter in SOURCES:
            if written >= target:
                break
            print(f"\n  ── Source: {src_name} ──")
            try:
                stream = src_iter()
            except Exception as e:
                print(f"    Skipping {src_name}: {e}")
                continue

            src_t0 = time.time()
            src_written = 0
            try:
                for text in stream:
                    for sent in _SENT_SPLIT.split(text):
                        sent = sent.strip()
                        if not _is_good(sent, min_len, max_len):
                            continue
                        h = hash(sent)
                        if h in seen:
                            continue
                        seen.add(h)
                        out.write(sent + "\n")
                        written += 1
                        src_written += 1
                        if written >= target:
                            break
                        if written % 100_000 == 0:
                            elapsed = time.time() - t0
                            rate = written / max(elapsed, 1)
                            print(
                                f"    {written:,} / {target:,} "
                                f"({elapsed/60:.1f} min, {rate:.0f} sent/s)"
                            )
                            out.flush()
                    if written >= target:
                        break
            except Exception as e:
                print(f"    Source {src_name} failed mid-stream: {e}")
                continue

            print(
                f"    Source {src_name}: +{src_written:,} sentences "
                f"in {(time.time()-src_t0)/60:.1f} min"
            )
    finally:
        out.close()

    if written < target:
        print(
            f"\n  WARNING: only extracted {written:,} / {target:,}. "
            f"All sources exhausted."
        )
    return written


# ═══════════════════════════════════════════════════════════════
# Tokenization — multiprocess
# ═══════════════════════════════════════════════════════════════

_WORKER_TOK = None  # per-process tokenizer singleton


def _worker_init():
    global _WORKER_TOK
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    from arabic_tokenizer import ArabicCSTTokenizer

    db = MorphologyDB.builtin_db()
    analyzer = Analyzer(db)
    _WORKER_TOK = ArabicCSTTokenizer(analyzer)


def _worker_tokenize(batch):
    out = []
    for sent in batch:
        try:
            r = _WORKER_TOK.tokenize(sent)
        except Exception:
            continue
        if len(r.get("ids", [])) < 4:
            continue
        out.append(r)
    return out


def _chunked(iterable, size):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def tokenize_stream(sentences_path: Path, out_jsonl: Path, vocab_path: Path,
                    workers: int, chunk: int, total: int) -> None:
    """Tokenize sentences from ``sentences_path`` into ``out_jsonl`` in parallel.

    Merges per-worker vocab updates back into a single tokenizer for the
    final vocab save. Each worker maintains its own vocab id space during
    the run; we remap at the end using a canonical id map.
    """
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    from arabic_tokenizer import ArabicCSTTokenizer

    # Canonical tokenizer (main process) — used to remap ids and save vocab.
    print("  Loading main-process tokenizer...")
    canon = ArabicCSTTokenizer(Analyzer(MorphologyDB.builtin_db()))

    def sentences_iter():
        with sentences_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.rstrip("\n")
                if s:
                    yield s

    print(f"  Spawning {workers} workers (chunk={chunk})...")
    t0 = time.time()
    written = 0
    total_tokens = 0

    pool = mp.Pool(processes=workers, initializer=_worker_init)
    try:
        with out_jsonl.open("w", encoding="utf-8") as out:
            batches = _chunked(sentences_iter(), chunk)
            for results in pool.imap_unordered(_worker_tokenize, batches, chunksize=1):
                for r in results:
                    # Remap token strings through the canonical tokenizer
                    # so we have a single consistent vocab at the end.
                    toks = r.get("tokens") or []
                    ids = [canon._get_id(t) for t in toks]
                    r["ids"] = ids
                    out.write(json.dumps(r, ensure_ascii=False))
                    out.write("\n")
                    written += 1
                    total_tokens += len(ids)
                if written % 50_000 < chunk:
                    elapsed = time.time() - t0
                    rate = written / max(elapsed, 1)
                    eta_min = (total - written) / max(rate, 1) / 60
                    print(
                        f"    {written:,} / {total:,} "
                        f"({elapsed/60:.1f} min, {rate:.0f} sent/s, "
                        f"ETA {eta_min:.0f} min)"
                    )
    finally:
        pool.close()
        pool.join()

    canon.save_vocab(str(vocab_path))
    elapsed = time.time() - t0
    print("\n  ═══ Tokenization done ═══")
    print(f"  Sentences:  {written:,}")
    print(f"  Tokens:     {total_tokens:,}")
    print(f"  Avg tok/s:  {total_tokens/max(written,1):.1f}")
    print(f"  Vocab size: {canon.next_id:,}")
    print(f"  Time:       {elapsed/60:.1f} min")
    print(f"  Output:     {out_jsonl}")
    print(f"  Vocab:      {vocab_path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    target = SIZE_MAP[args.target]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sentences_path = out_dir / f"sentences-{args.target}.txt"
    out_jsonl = out_dir / f"cst-train-{args.target}.jsonl"
    vocab_path = out_dir / f"cst-train-{args.target}-vocab.json"

    print("=" * 64)
    print(f"  Arabic CST tokenization — {args.target} ({target:,} sentences)")
    print(f"  Output dir: {out_dir}")
    print(f"  Workers:    {args.workers}")
    print("=" * 64)

    print("\n── Step 1: Extract sentences (streaming, multi-source) ──")
    got = extract_sentences(
        target=target,
        sentences_path=sentences_path,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    if got == 0:
        print("  No sentences extracted. Aborting.")
        sys.exit(1)

    print("\n── Step 2: CST tokenize (multiprocess) ──")
    if out_jsonl.exists():
        print(f"  WARNING: {out_jsonl} exists; overwriting.")
        out_jsonl.unlink()

    tokenize_stream(
        sentences_path=sentences_path,
        out_jsonl=out_jsonl,
        vocab_path=vocab_path,
        workers=args.workers,
        chunk=args.chunk,
        total=got,
    )

    print("\n── Upload to Colab ──")
    print(f"  Files to upload (Drive → /MyDrive/cst-data/):")
    print(f"    {out_jsonl.name}")
    print(f"    {vocab_path.name}")
    print(
        f"  Then in the training notebook set:\n"
        f"    DATA_FILE  = \"{out_jsonl.name}\"\n"
        f"    VOCAB_FILE = \"{vocab_path.name}\""
    )


if __name__ == "__main__":
    main()
