"""Sanity trainer: causal LM over reasoning tokens.

One-epoch run on stage-2b syllogisms as a first end-to-end check that
the two-level pipeline can actually learn something.

Usage::

    python -m reasoning.train.train \
        --data  ./reasoning/tokenized/stage-2b-syllogisms.tokenized.jsonl \
        --vocab ./reasoning/tokenized/vocab-reasoning.json \
        --out   ./reasoning/train/runs/syllog-v0

Writes ``ckpt.pt`` (model state + config) and ``train_log.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .dataset import (
    ReasoningJsonlDataset,
    load_vocab,
    make_batch,
    split_indices,
)
from .model import GPTConfig, TinyGPT


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model: TinyGPT, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            n_valid = (targets != -100).sum().item()
            total_loss += loss.item()
            total_tokens += n_valid
    mean = total_loss / max(1, total_tokens)
    return {"val_loss": mean, "val_ppl": math.exp(min(mean, 20)), "val_tokens": total_tokens}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = _device()
    args.out.mkdir(parents=True, exist_ok=True)

    # ── Data ───────────────────────────────────────────────
    vocab = load_vocab(args.vocab)
    dataset = ReasoningJsonlDataset(args.data, vocab, max_len=args.max_len)
    train_idx, val_idx = split_indices(len(dataset), val_frac=args.val_frac, seed=args.seed)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=make_batch, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=make_batch,
    )

    # ── Model ──────────────────────────────────────────────
    cfg = GPTConfig(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
    )
    model = TinyGPT(cfg).to(device)
    n_params = model.num_parameters()
    print(f"device={device}  vocab={cfg.vocab_size}  params={n_params/1e6:.2f}M")
    print(f"train={len(train_ds)}  val={len(val_ds)}  batch={args.batch_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── Loop ───────────────────────────────────────────────
    log_path = args.out / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_fh:
        step = 0
        t0 = time.time()
        for epoch in range(args.epochs):
            model.train()
            for batch in train_loader:
                inputs = batch["inputs"].to(device)
                targets = batch["targets"].to(device)
                logits = model(inputs)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
                if step % args.log_every == 0:
                    ppl = math.exp(min(loss.item(), 20))
                    line = {"epoch": epoch, "step": step,
                            "train_loss": loss.item(), "train_ppl": ppl,
                            "elapsed_s": round(time.time() - t0, 1)}
                    log_fh.write(json.dumps(line) + "\n")
                    log_fh.flush()
                    print(f"  ep{epoch} step{step:>5}  loss={loss.item():.4f}  ppl={ppl:.1f}")
            ev = evaluate(model, val_loader, device)
            line = {"epoch": epoch, "step": step, **ev,
                    "elapsed_s": round(time.time() - t0, 1)}
            log_fh.write(json.dumps(line) + "\n")
            log_fh.flush()
            print(f"── epoch {epoch}  val_loss={ev['val_loss']:.4f}  val_ppl={ev['val_ppl']:.2f}")

    # ── Save ───────────────────────────────────────────────
    ckpt = {
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "vocab_path": str(args.vocab),
        "args": vars(args) | {"data": str(args.data), "out": str(args.out), "vocab": str(args.vocab)},
    }
    torch.save(ckpt, args.out / "ckpt.pt")
    print(f"Saved {args.out / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
