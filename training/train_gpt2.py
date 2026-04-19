"""
Train small GPT-2 models on CST vs BPE tokenized data.
Compares identical architectures with different tokenization.

Usage:
  # Train both (default 10M params):
  python training/train_gpt2.py

  # Custom size:
  python training/train_gpt2.py --params 50  # 50M params

  # Train only one:
  python training/train_gpt2.py --tokenizer cst
  python training/train_gpt2.py --tokenizer bpe
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel


# ── Model size presets ──────────────────────────────

PRESETS = {
    10: dict(n_embd=256, n_layer=6, n_head=4),
    25: dict(n_embd=384, n_layer=8, n_head=6),
    50: dict(n_embd=512, n_layer=8, n_head=8),
}


# ── Dataset ─────────────────────────────────────────

class JsonlDataset(Dataset):
    """Reads .jsonl with {ids: [...], tokens: [...], text: "..."} lines."""

    def __init__(self, path: str, max_len: int = 128):
        self.examples = []
        self.max_len = max_len

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ids = obj["ids"]
                if len(ids) < 4:
                    continue
                # Truncate or keep as is
                if len(ids) > max_len:
                    ids = ids[:max_len]
                self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_id=0):
    """Pad sequences to same length within batch."""
    max_len = max(len(seq) for seq in batch)
    padded = []
    attention_masks = []
    for seq in batch:
        pad_len = max_len - len(seq)
        padded.append(seq + [pad_id] * pad_len)
        attention_masks.append([1] * len(seq) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(padded, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
    }


# ── Training ────────────────────────────────────────

def train_model(
    name: str,
    train_path: str,
    val_path: str,
    vocab_size: int,
    preset: dict,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 3e-4,
    max_len: int = 128,
    device: str = "auto",
    output_dir: str = "training/checkpoints",
):
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  Device:   {device}")
    print(f"  Vocab:    {vocab_size:,}")
    print(f"  Config:   {preset}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading training data...")
    train_ds = JsonlDataset(train_path, max_len=max_len)
    val_ds = JsonlDataset(val_path, max_len=max_len)
    print(f"  Train: {len(train_ds):,} examples")
    print(f"  Val:   {len(val_ds):,} examples")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Build model
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_len,
        n_embd=preset["n_embd"],
        n_layer=preset["n_layer"],
        n_head=preset["n_head"],
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(config).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader)
    )

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_ppl": []}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        total_loss = 0
        total_tokens = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Shift: input is [:-1], labels is [1:]
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # ignore padding

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item() * attention_mask.sum().item()
            total_tokens += attention_mask.sum().item()

            if (step + 1) % 100 == 0:
                avg = total_loss / total_tokens
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | "
                    f"Loss: {avg:.4f} | {elapsed:.0f}s"
                )

        train_loss = total_loss / total_tokens
        history["train_loss"].append(train_loss)

        # ── Validate ──
        model.eval()
        val_loss_total = 0
        val_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss_total += outputs.loss.item() * attention_mask.sum().item()
                val_tokens += attention_mask.sum().item()

        val_loss = val_loss_total / val_tokens
        val_ppl = math.exp(min(val_loss, 20))  # cap to avoid overflow
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)

        epoch_time = time.time() - t0
        print(
            f"\n  Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PPL: {val_ppl:.1f} | "
            f"Time: {epoch_time:.0f}s\n"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = os.path.join(output_dir, name)
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            print(f"  Saved best model → {save_dir}")

    return {
        "name": name,
        "params": param_count,
        "vocab_size": vocab_size,
        "best_val_loss": best_val_loss,
        "best_val_ppl": math.exp(min(best_val_loss, 20)),
        "history": history,
    }


# ── Train/Val split ────────────────────────────────

def split_jsonl(input_path: str, train_path: str, val_path: str, val_ratio: float = 0.1):
    """Split a .jsonl file into train/val if val doesn't exist yet."""
    if os.path.exists(train_path) and os.path.exists(val_path):
        return

    with open(input_path) as f:
        lines = f.readlines()

    split_idx = int(len(lines) * (1 - val_ratio))
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_path), exist_ok=True)

    with open(train_path, "w") as f:
        f.writelines(lines[:split_idx])
    with open(val_path, "w") as f:
        f.writelines(lines[split_idx:])

    print(f"  Split: {split_idx:,} train / {len(lines)-split_idx:,} val")


# ── Main ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 on CST vs BPE")
    parser.add_argument("--params", type=int, default=10, choices=[10, 25, 50],
                        help="Target model size in millions of params")
    parser.add_argument("--tokenizer", type=str, default="both", choices=["cst", "bpe", "both"],
                        help="Which tokenizer to train")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--data-size", type=int, default=100000,
                        help="Which dataset size to use (matches stream.ts output)")
    args = parser.parse_args()

    preset = PRESETS[args.params]
    n = args.data_size
    results = []

    configs = {
        "cst": {
            "data": f"data/tokenized/cst/train-{n}.jsonl",
            "vocab_json": f"data/tokenized/cst/train-{n}-vocab.json",
        },
        "bpe": {
            "data": f"data/tokenized/bpe/train-{n}.jsonl",
            "vocab_json": f"data/tokenized/bpe/train-{n}-vocab.json",
        },
    }

    tokenizers_to_train = ["cst", "bpe"] if args.tokenizer == "both" else [args.tokenizer]

    for tok_name in tokenizers_to_train:
        cfg = configs[tok_name]

        if not os.path.exists(cfg["data"]):
            print(f"ERROR: {cfg['data']} not found. Run stream.ts first.")
            continue

        # Get vocab size
        with open(cfg["vocab_json"]) as f:
            vocab = json.load(f)
        vocab_size = len(vocab) if isinstance(vocab, dict) else vocab.get("size", 50000)

        # Split into train/val
        train_path = cfg["data"].replace(".jsonl", "-train.jsonl")
        val_path = cfg["data"].replace(".jsonl", "-val.jsonl")
        print(f"\nPreparing {tok_name.upper()} data...")
        split_jsonl(cfg["data"], train_path, val_path)

        result = train_model(
            name=f"gpt2-{args.params}m-{tok_name}",
            train_path=train_path,
            val_path=val_path,
            vocab_size=vocab_size,
            preset=preset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_len=args.max_len,
        )
        results.append(result)

    # ── Compare ──
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("  COMPARISON: CST vs BPE")
        print("=" * 60)
        for r in results:
            print(
                f"  {r['name']:30s} | "
                f"Vocab: {r['vocab_size']:>8,} | "
                f"Val Loss: {r['best_val_loss']:.4f} | "
                f"Val PPL: {r['best_val_ppl']:.1f}"
            )

        cst_r, bpe_r = results[0], results[1]
        ppl_diff = ((bpe_r["best_val_ppl"] - cst_r["best_val_ppl"]) / bpe_r["best_val_ppl"]) * 100
        loss_diff = ((bpe_r["best_val_loss"] - cst_r["best_val_loss"]) / bpe_r["best_val_loss"]) * 100

        print(f"\n  Vocab compression: {((1 - cst_r['vocab_size']/bpe_r['vocab_size'])*100):.1f}%")
        print(f"  Loss improvement:  {loss_diff:+.1f}%")
        print(f"  PPL improvement:   {ppl_diff:+.1f}%")

        if cst_r["best_val_ppl"] < bpe_r["best_val_ppl"]:
            print("\n  ✓ CST achieves LOWER perplexity with SMALLER vocabulary")
        else:
            print("\n  → BPE has lower perplexity (CST still has smaller vocab)")

    # Save results
    results_path = f"training/results-{args.params}m.json"
    os.makedirs("training", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
