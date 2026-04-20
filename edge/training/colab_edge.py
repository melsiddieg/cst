"""
Arabic CST Edge Model — Train + Export to ONNX (Google Colab)

Upload to Colab:
  - cst-ar-8k-train-100000.jsonl
  - cst-ar-8k-train-100000-vocab.json

Runtime → Change runtime type → T4 GPU.

What it does:
  1. Trains GPT-2 small (~6M params) on Arabic CST-8K tokens
  2. Exports to ONNX (browser-ready)
  3. Quantizes to int8 (shrinks ~3x)
  4. Saves vocab.json for browser inference

Output files (download these):
  - model.onnx           (~25 MB fp32)
  - model_int8.onnx       (~8 MB — this is what the browser loads)
  - vocab.json            (token↔id for browser)

Time: ~20 min on T4 GPU.
"""

# !pip install torch transformers onnx onnxruntime -q

import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel

# ── Config ──

DATA_DIR = "/content"
DATA_FILE = "train-100000.jsonl"
VOCAB_FILE = "train-100000-vocab.json"
OUT_DIR = "/content/edge_model"

PRESET = dict(n_embd=256, n_layer=6, n_head=4)
EPOCHS = 3
BATCH_SIZE = 32
LR = 3e-4
MAX_LEN = 128
VAL_RATIO = 0.1


# ── Dataset (same as colab_train_ar.py) ──

class JsonlDataset(Dataset):
    def __init__(self, ids_list, char_counts):
        self.ids_list = ids_list
        self.char_counts = char_counts

    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, idx):
        return self.ids_list[idx], self.char_counts[idx]


def load_jsonl(path, max_len=128):
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


# ── Train ──

def train_model(name, train_ids, train_chars, val_ids, val_chars, vocab_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_embd = PRESET["n_embd"]

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Device: {device} | Vocab: {vocab_size:,}")
    print(f"{'='*60}\n")

    train_ds = JsonlDataset(train_ids, train_chars)
    val_ds = JsonlDataset(val_ids, val_chars)

    val_total_chars = sum(val_chars)
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=MAX_LEN,
        n_embd=n_embd,
        n_layer=PRESET["n_layer"],
        n_head=PRESET["n_head"],
        bos_token_id=3,
        eos_token_id=4,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_loader)
    )

    best_val_bpc = float("inf")
    best_model_state = None

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

        # Validation
        model.eval()
        val_nll = 0
        val_toks = 0
        val_ch = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                bt = attention_mask.sum().item()
                val_nll += outputs.loss.item() * bt
                val_toks += bt
                val_ch += sum(batch["char_counts"])

        val_loss = val_nll / val_toks
        val_ppl = math.exp(min(val_loss, 20))
        val_bpc = val_nll / val_ch / math.log(2)

        print(f"\n  Epoch {epoch}/{EPOCHS} | Loss: {val_loss:.4f} | PPL: {val_ppl:.1f} | BPC: {val_bpc:.4f}\n")

        if val_bpc < best_val_bpc:
            best_val_bpc = val_bpc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return model, config, best_model_state, best_val_bpc, total_params


# ── ONNX Export (same approach as arabic-algebra-engine) ──

class LogitsOnlyWrapper(torch.nn.Module):
    """Wraps HF model to return only logits tensor (avoids DynamicCache export issue)."""
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
    def forward(self, input_ids):
        return self.model(input_ids=input_ids, use_cache=False).logits

def export_onnx(model, config, out_dir):
    model.eval()
    model.cpu()

    wrapper = LogitsOnlyWrapper(model)
    dummy = torch.zeros(1, MAX_LEN, dtype=torch.long)
    onnx_path = os.path.join(out_dir, "model.onnx")

    print(f"\nExporting ONNX to {onnx_path} ...")
    torch.onnx.export(
        wrapper,
        (dummy,),
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {1: "seq_len"}, "logits": {1: "seq_len"}},
        opset_version=18,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"  model.onnx: {size_mb:.1f} MB")
    return onnx_path, size_mb


def quantize_int8(onnx_path, out_dir):
    from onnxruntime.quantization import quantize_dynamic, QuantType

    int8_path = os.path.join(out_dir, "model_int8.onnx")
    print(f"\nQuantizing to int8 → {int8_path} ...")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
    )
    size_mb = os.path.getsize(int8_path) / 1e6
    print(f"  model_int8.onnx: {size_mb:.1f} MB")
    return int8_path, size_mb


def build_vocab_json(vocab_path, out_dir):
    with open(vocab_path) as f:
        v = json.load(f)
    entries = v if isinstance(v, list) else list(v.values())

    tok2id = {e["token"]: int(e["id"]) for e in entries}
    id2tok = {str(e["id"]): e["token"] for e in entries}
    size = max(int(e["id"]) for e in entries) + 1

    out = {
        "vocab": tok2id,
        "rev_vocab": id2tok,
        "size": size,
        "special": {"PAD": 0, "UNK": 1, "BOS": 3, "EOS": 4},
    }
    out_path = os.path.join(out_dir, "vocab.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  vocab.json: {os.path.getsize(out_path)/1024:.0f} KB ({size} tokens)")
    return out_path


# ── Main ──

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data_path = os.path.join(DATA_DIR, DATA_FILE)
    vocab_path = os.path.join(DATA_DIR, VOCAB_FILE)

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Upload {DATA_FILE} first.")
        return
    if not os.path.exists(vocab_path):
        print(f"ERROR: {vocab_path} not found. Upload {VOCAB_FILE} first.")
        return

    # Load
    print(f"Loading {DATA_FILE} ...")
    ids_list, char_counts = load_jsonl(data_path, MAX_LEN)

    with open(vocab_path) as f:
        vocab = json.load(f)
    if isinstance(vocab, list):
        vocab_size = max(e["id"] for e in vocab) + 1
    else:
        vocab_size = len(vocab)

    # Split
    split_idx = int(len(ids_list) * (1 - VAL_RATIO))

    # Train
    model, config, best_state, best_bpc, n_params = train_model(
        name="AR-CST-8K-Edge",
        train_ids=ids_list[:split_idx],
        train_chars=char_counts[:split_idx],
        val_ids=ids_list[split_idx:],
        val_chars=char_counts[split_idx:],
        vocab_size=vocab_size,
    )

    # Load best state & save checkpoint
    model.load_state_dict(best_state)
    ckpt_path = os.path.join(OUT_DIR, "model.pt")
    torch.save(best_state, ckpt_path)
    print(f"\n  Checkpoint saved: {ckpt_path}")

    # Export ONNX
    onnx_path, fp32_mb = export_onnx(model, config, OUT_DIR)

    # Quantize
    int8_path, int8_mb = quantize_int8(onnx_path, OUT_DIR)

    # Vocab
    build_vocab_json(vocab_path, OUT_DIR)

    # Summary
    print(f"\n{'='*60}")
    print(f"  DONE — Arabic CST-8K Edge Model")
    print(f"{'='*60}")
    print(f"  Params:     {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Best BPC:   {best_bpc:.4f}")
    print(f"  model.onnx:      {fp32_mb:.1f} MB")
    print(f"  model_int8.onnx: {int8_mb:.1f} MB")
    print(f"  Output: {OUT_DIR}/")
    print(f"{'='*60}")
    print(f"\nDownload these 3 files from {OUT_DIR}/:")
    print(f"  1. model_int8.onnx  (browser model)")
    print(f"  2. vocab.json       (token mappings)")
    print(f"  3. model.onnx       (optional, fp32 backup)")


if __name__ == "__main__":
    main()
