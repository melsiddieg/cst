"""
Arabic Wikipedia — Download & Extract Sentences (Google Colab)

Downloads Arabic Wikipedia via HuggingFace `datasets` library,
extracts clean sentences, saves to Google Drive for reuse.

Run on Colab (in notebook cell first):
  from google.colab import drive
  drive.mount("/content/drive")

Then:
  !pip install datasets
  !python download_data.py

Output saved to Google Drive:
  /content/drive/MyDrive/cst-data/sentences-1M.json
  /content/drive/MyDrive/cst-data/sentences-10M.json
  /content/drive/MyDrive/cst-data/sentences-50M.json
  /content/drive/MyDrive/cst-data/sentences-100M.json

Each file is independent — stop at any scale you need.
"""

import json
import os
import re
import time

# ── Config ──

# Check if Google Drive is already mounted (must be done in notebook cell first)
GDRIVE_PATH = "/content/drive/MyDrive/cst-data"
if os.path.isdir("/content/drive/MyDrive"):
    OUT_DIR = GDRIVE_PATH
    print(f"  Google Drive detected → saving to {OUT_DIR}")
else:
    OUT_DIR = "/content/cst-data"
    print(f"  No Google Drive → saving locally to {OUT_DIR}")
    print(f"  To use Drive, run this in a notebook cell FIRST:")
    print(f"    from google.colab import drive")
    print(f"    drive.mount('/content/drive')")

os.makedirs(OUT_DIR, exist_ok=True)

# Checkpoints: save at each scale
CHECKPOINTS = [
    (1_000_000,   "sentences-1M.json"),
    (10_000_000,  "sentences-10M.json"),
    (50_000_000,  "sentences-50M.json"),
    (100_000_000, "sentences-100M.json"),
]

MAX_TARGET = CHECKPOINTS[-1][0]


# ── Sentence extraction ──

def is_good_sentence(sent):
    """Filter: 20-300 chars, >50% Arabic."""
    if len(sent) < 20 or len(sent) > 300:
        return False
    arabic_count = sum(1 for c in sent if '\u0600' <= c <= '\u06FF')
    return arabic_count >= len(sent) * 0.5


def extract_sentences():
    from datasets import load_dataset

    # Check which checkpoints are already done
    done = set()
    for target, filename in CHECKPOINTS:
        path = os.path.join(OUT_DIR, filename)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {filename} already exists ({size:.0f} MB)")
            done.add(target)

    if len(done) == len(CHECKPOINTS):
        print("\n  All checkpoints done. Nothing to download.")
        return

    # Find the largest completed checkpoint to resume from
    resume_sentences = []
    resume_from = 0
    for target, filename in CHECKPOINTS:
        if target in done:
            resume_from = target
        else:
            break

    if resume_from > 0:
        # Load the last completed checkpoint to resume
        prev_file = None
        for target, filename in CHECKPOINTS:
            if target == resume_from:
                prev_file = os.path.join(OUT_DIR, filename)
                break
        if prev_file:
            print(f"\n  Resuming from {prev_file} ({resume_from:,} sentences)...")
            with open(prev_file) as f:
                resume_sentences = json.load(f)
            print(f"  Loaded {len(resume_sentences):,} sentences")

    print(f"\n  Downloading Arabic Wikipedia via `datasets` library...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train", streaming=True)

    sentences = list(resume_sentences)
    articles = 0
    skipped = 0
    t0 = time.time()
    next_checkpoint_idx = 0

    # Skip past already-done checkpoints
    for i, (target, _) in enumerate(CHECKPOINTS):
        if target in done:
            next_checkpoint_idx = i + 1
        else:
            break

    if len(sentences) >= MAX_TARGET:
        print("  Already at max target.")
        return

    # If resuming, we need to skip articles we already processed
    # Simple approach: just continue from where sentences left off
    print(f"  Starting from {len(sentences):,} sentences, target {MAX_TARGET:,}")

    for row in ds:
        text = row.get("text", "")
        for sent in re.split(r'[.؟!]\s*', text):
            sent = sent.strip()
            if not is_good_sentence(sent):
                skipped += 1
                continue
            sentences.append(sent)

            # Check if we hit a checkpoint
            if next_checkpoint_idx < len(CHECKPOINTS):
                target, filename = CHECKPOINTS[next_checkpoint_idx]
                if len(sentences) >= target:
                    path = os.path.join(OUT_DIR, filename)
                    elapsed = time.time() - t0
                    print(f"\n  ── Checkpoint: {target:,} sentences ({elapsed:.0f}s) ──")
                    save_checkpoint(sentences[:target], path)
                    next_checkpoint_idx += 1

            if len(sentences) >= MAX_TARGET:
                break

        if len(sentences) >= MAX_TARGET:
            break

        articles += 1
        if articles % 10000 == 0:
            elapsed = time.time() - t0
            rate = len(sentences) / elapsed if elapsed > 0 else 0
            eta_min = (MAX_TARGET - len(sentences)) / rate / 60 if rate > 0 else 0
            print(f"    {len(sentences):>12,} sentences | {articles:,} articles | "
                  f"{elapsed:.0f}s | {rate:.0f}/s | ETA {eta_min:.0f}min")

    elapsed = time.time() - t0
    print(f"\n  ═══ Done ═══")
    print(f"  Total sentences: {len(sentences):,}")
    print(f"  Skipped:         {skipped:,}")
    print(f"  Articles:        {articles:,}")
    print(f"  Time:            {elapsed/60:.1f} min")
    print(f"  Output dir:      {OUT_DIR}")


def save_checkpoint(sentences, path):
    """Save sentences as JSON array."""
    size_est = len(sentences) * 100 / (1024 * 1024)  # rough estimate
    print(f"  Saving {len(sentences):,} sentences to {path} (~{size_est:.0f} MB est.)...")
    with open(path, "w") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=0)
    actual_size = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved: {actual_size:.0f} MB")


# ── Main ──

if __name__ == "__main__":
    print("=" * 60)
    print("  Arabic Wikipedia — Sentence Extraction")
    print("=" * 60)
    print(f"\n  Output: {OUT_DIR}")
    print(f"  Targets: {', '.join(f'{t:,}' for t, _ in CHECKPOINTS)}")
    extract_sentences()
    print(f"\n  Files in {OUT_DIR}:")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f)) / (1024 * 1024)
        print(f"    {f:30s} {size:8.1f} MB")
