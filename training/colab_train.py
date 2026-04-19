"""
CST vs SentencePiece BPE — GPT-2 Training Comparison (Google Colab)

Upload these 7 files to Colab before running:
  - cst-train-100000.jsonl + cst-train-100000-vocab.json
  - spm-8k-train-99963.jsonl + spm-8k-train-99963-vocab.json
  - spm-32k-train-99963.jsonl + spm-32k-train-99963-vocab.json

Primary metric: bits-per-character (BPC) — normalizes across different
tokenizers / vocab sizes so the comparison is apples-to-apples.

Run cells in order. GPU runtime recommended (Runtime → Change runtime type → T4 GPU).
"""

# ── Cell 1: Install deps ────────────────────────────
# !pip install torch transformers -q

# ── Cell 2: Upload data ─────────────────────────────
"""
Upload all 7 files via Colab UI (folder icon → upload)
"""

# ── Cell 3: Training code ───────────────────────────

import json
import math
import os
import time

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel

# ── Config ──

DATA_DIR = "/content"

TOKENIZERS = {
    "CST": {
        "data": os.path.join(DATA_DIR, "cst-train-100000.jsonl"),
        "vocab": os.path.join(DATA_DIR, "cst-train-100000-vocab.json"),
    },
    "SPM-8K": {
        "data": os.path.join(DATA_DIR, "spm-8k-train-99963.jsonl"),
        "vocab": os.path.join(DATA_DIR, "spm-8k-train-99963-vocab.json"),
    },
    "SPM-32K": {
        "data": os.path.join(DATA_DIR, "spm-32k-train-99963.jsonl"),
        "vocab": os.path.join(DATA_DIR, "spm-32k-train-99963-vocab.json"),
    },
}

# Model size presets
PRESETS = {
    10: dict(n_embd=256, n_layer=6, n_head=4),    # ~10M params
    25: dict(n_embd=384, n_layer=8, n_head=6),    # ~25M params
    50: dict(n_embd=512, n_layer=8, n_head=8),    # ~50M params
}

# ── Choose your settings ──
MODEL_SIZE = 10        # 10, 25, or 50 (millions of params)
EPOCHS = 3
BATCH_SIZE = 32
LR = 3e-4
MAX_LEN = 128
VAL_RATIO = 0.1


# ── Dataset ──

class JsonlDataset(Dataset):
    def __init__(self, ids_list, char_counts):
        self.ids_list = ids_list
        self.char_counts = char_counts

    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, idx):
        return self.ids_list[idx], self.char_counts[idx]


def load_jsonl(path, max_len=128):
    """Load jsonl, return (ids_list, char_counts)."""
    ids_list = []
    char_counts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids = obj["ids"]
            text = obj.get("text", "")
            if len(ids) < 4:
                continue
            if len(ids) > max_len:
                # Proportionally reduce char count when truncating
                ratio = max_len / len(ids)
                ids = ids[:max_len]
                char_counts.append(int(len(text) * ratio))
            else:
                char_counts.append(len(text))
            ids_list.append(ids)
    return ids_list, char_counts


def collate_fn(batch, pad_id=0):
    ids_list = [item[0] for item in batch]
    char_counts = [item[1] for item in batch]
    max_len = max(len(seq) for seq in ids_list)
    padded = []
    masks = []
    for seq in ids_list:
        pad_len = max_len - len(seq)
        padded.append(seq + [pad_id] * pad_len)
        masks.append([1] * len(seq) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(padded, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "char_counts": char_counts,
    }


# ── Training function ──

def train_model(name, train_ids, train_chars, val_ids, val_chars, vocab_size, preset):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parameter breakdown
    n_embd = preset["n_embd"]
    embed_params = vocab_size * n_embd  # token embedding
    pos_params = MAX_LEN * n_embd       # position embedding

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Device: {device} | Vocab: {vocab_size:,}")
    print(f"{'='*60}\n")

    train_ds = JsonlDataset(train_ids, train_chars)
    val_ds = JsonlDataset(val_ids, val_chars)
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # Stats
    train_total_tokens = sum(len(ids) for ids in train_ids)
    train_total_chars = sum(train_chars)
    val_total_tokens = sum(len(ids) for ids in val_ids)
    val_total_chars = sum(val_chars)
    print(f"  Val tokens: {val_total_tokens:,} | Val chars: {val_total_chars:,}")
    print(f"  Avg tokens/sentence: {val_total_tokens/len(val_ids):.1f}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=MAX_LEN,
        n_embd=n_embd,
        n_layer=preset["n_layer"],
        n_head=preset["n_head"],
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    transformer_params = total_params - embed_params - pos_params
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"    Embedding:   {embed_params:,} ({embed_params/1e6:.1f}M)")
    print(f"    Position:    {pos_params:,}")
    print(f"    Transformer: {transformer_params:,} ({transformer_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_loader)
    )

    history = {"train_loss": [], "val_loss": [], "val_ppl": [], "val_bpc": []}
    best_val_loss = float("inf")
    best_val_bpc = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        total_tokens = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item() * attention_mask.sum().item()
            total_tokens += attention_mask.sum().item()

            if (step + 1) % 200 == 0:
                avg = total_loss / total_tokens
                elapsed = time.time() - t0
                print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | Loss: {avg:.4f} | {elapsed:.0f}s")

        train_loss = total_loss / total_tokens
        history["train_loss"].append(train_loss)

        # Validate
        model.eval()
        val_nll_total = 0   # total NLL in nats
        val_tok_total = 0
        val_char_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                batch_tokens = attention_mask.sum().item()
                val_nll_total += outputs.loss.item() * batch_tokens
                val_tok_total += batch_tokens
                val_char_total += sum(batch["char_counts"])

        val_loss = val_nll_total / val_tok_total
        val_ppl = math.exp(min(val_loss, 20))
        # BPC = total NLL (nats) / total chars / ln(2)
        val_bpc = val_nll_total / val_char_total / math.log(2)

        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["val_bpc"].append(val_bpc)

        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch}/{EPOCHS} | Loss: {val_loss:.4f} | PPL: {val_ppl:.1f} | BPC: {val_bpc:.4f} | {elapsed:.0f}s\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_bpc < best_val_bpc:
            best_val_bpc = val_bpc

    return {
        "name": name,
        "params": total_params,
        "embed_params": embed_params,
        "transformer_params": transformer_params,
        "vocab_size": vocab_size,
        "val_tokens": val_total_tokens,
        "val_chars": val_total_chars,
        "best_val_loss": best_val_loss,
        "best_val_ppl": math.exp(min(best_val_loss, 20)),
        "best_val_bpc": best_val_bpc,
        "history": history,
    }


# ── Cell 4: Run training ────────────────────────────

def run():
    preset = PRESETS[MODEL_SIZE]
    results = []

    for tok_name, cfg in TOKENIZERS.items():
        data_path = cfg["data"]
        vocab_path = cfg["vocab"]

        if not os.path.exists(data_path):
            print(f"MISSING: {data_path}")
            continue

        # Load and split
        print(f"\nLoading {tok_name} data from {data_path}...")
        ids_list, char_counts = load_jsonl(data_path, MAX_LEN)
        split_idx = int(len(ids_list) * (1 - VAL_RATIO))
        train_ids = ids_list[:split_idx]
        train_chars = char_counts[:split_idx]
        val_ids = ids_list[split_idx:]
        val_chars = char_counts[split_idx:]

        # Get vocab size
        with open(vocab_path) as f:
            vocab = json.load(f)
        if isinstance(vocab, dict):
            vocab_size = len(vocab)
        elif isinstance(vocab, list):
            # CST vocab is [{token, id, ...}, ...] — need max id + 1
            vocab_size = max(entry["id"] for entry in vocab) + 1
        else:
            vocab_size = 50000

        result = train_model(
            name=f"gpt2-{MODEL_SIZE}m-{tok_name.lower()}",
            train_ids=train_ids,
            train_chars=train_chars,
            val_ids=val_ids,
            val_chars=val_chars,
            vocab_size=vocab_size,
            preset=preset,
        )
        results.append(result)

    # Compare
    if len(results) >= 2:
        print("\n" + "=" * 70)
        print("  RESULTS: CST vs SentencePiece BPE")
        print("=" * 70)

        # Header
        header = f"  {'Metric':<25}"
        for r in results:
            header += f" {r['name']:>18}"
        print(header)
        print(f"  {'-'*80}")

        # Rows
        rows = [
            ("Vocab size",      lambda r: f"{r['vocab_size']:>18,}"),
            ("Total params",    lambda r: f"{r['params']:>14,} ({r['params']/1e6:.1f}M)"),
            ("  Embedding",     lambda r: f"{r['embed_params']:>14,} ({r['embed_params']/1e6:.1f}M)"),
            ("  Transformer",   lambda r: f"{r['transformer_params']:>14,} ({r['transformer_params']/1e6:.1f}M)"),
            ("Val tokens",      lambda r: f"{r['val_tokens']:>18,}"),
            ("Val chars",       lambda r: f"{r['val_chars']:>18,}"),
            ("Best val loss",   lambda r: f"{r['best_val_loss']:>18.4f}"),
            ("Best val PPL",    lambda r: f"{r['best_val_ppl']:>18.1f}"),
            ("*** Best BPC ***", lambda r: f"{r['best_val_bpc']:>18.4f}"),
        ]
        for label, fmt in rows:
            line = f"  {label:<25}"
            for r in results:
                line += fmt(r)
            print(line)
        print(f"  {'-'*80}")

        # BPC comparison (the fair metric)
        cst_r = results[0]
        for r in results[1:]:
            bpc_diff = ((r["best_val_bpc"] - cst_r["best_val_bpc"]) / r["best_val_bpc"]) * 100
            param_diff = ((cst_r["params"] - r["params"]) / r["params"]) * 100
            print(f"\n  CST vs {r['name']}:")
            print(f"    BPC:    CST {cst_r['best_val_bpc']:.4f} vs {r['name']} {r['best_val_bpc']:.4f} ({bpc_diff:+.1f}%)")
            print(f"    Params: CST {cst_r['params']/1e6:.1f}M vs {r['name']} {r['params']/1e6:.1f}M ({param_diff:+.1f}% more)")
            if cst_r["best_val_bpc"] < r["best_val_bpc"]:
                print(f"    → CST wins on BPC (lower is better)")
            else:
                print(f"    → {r['name']} wins on BPC (lower is better)")

        # Find best by BPC
        best = min(results, key=lambda r: r["best_val_bpc"])
        print(f"\n  Best overall (BPC): {best['name']} ({best['best_val_bpc']:.4f})")
        print(f"\n  NOTE: BPC (bits-per-character) is the fair metric here.")
        print(f"  PPL is NOT comparable across tokenizers with different vocab sizes.")

    return results


results = run()


# ── Cell 5: Plot (optional) ─────────────────────────

"""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for r in results:
    axes[0].plot(r['history']['train_loss'], label=f"{r['name']} train")
    axes[0].plot(r['history']['val_loss'], label=f"{r['name']} val", linestyle='--')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()

for r in results:
    axes[1].plot(r['history']['val_ppl'], label=r['name'])
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Perplexity')
axes[1].set_title('Val Perplexity (not comparable across tokenizers)')
axes[1].legend()

for r in results:
    axes[2].plot(r['history']['val_bpc'], label=r['name'])
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('BPC')
axes[2].set_title('Val BPC (fair metric)')
axes[2].legend()

plt.tight_layout()
plt.savefig('cst_vs_bpe.png', dpi=150)
plt.show()
"""
