"""
Downstream task evaluation for Phase 0.

Reads a trained checkpoint saved by ``train_and_eval(..., ckpt_path=...)`` and
evaluates it on one of two pre-tokenized task formats. The tokenizer used for
the downstream data MUST match the one used to train the model (same CST or
SPM pipeline, same vocab). That tokenization happens in ``prepare_downstream.py``
(per-language) \u2014 this script only consumes the prepared jsonl.

Task formats
------------

1. LM scoring (LAMBADA-style, zero-shot contrastive sentiment, etc.)
   One JSON object per line:
       {"context_ids": [int, ...],
        "candidates":  [[int, ...], [int, ...], ...],
        "gold":        <index into candidates>}
   Metric: accuracy = fraction of rows where the gold candidate has the
   lowest sum-NLL (equivalently, highest joint log-prob) among candidates
   when appended to the context.

2. Classification (HARD sentiment, RuSentiment, ...)
   One JSON object per line:
       {"ids": [int, ...], "label": int}
   Method: freeze the LM, mean-pool the last hidden state over non-pad
   positions, train a linear classifier on the [CLS] representation for
   ``--ft-epochs`` epochs. Metric: accuracy on the held-out split.

The input file must be split-ready: if it ends in ``-train.jsonl`` we also
expect a sibling ``-test.jsonl``. For classification, both are required;
for lm_scoring, only the test file is used.

Usage
-----
    python training/experiments/downstream_eval.py \\
        --ckpt /content/checkpoints/CST-8K-seed0 \\
        --task lm_scoring \\
        --test  /content/downstream/lambada-en-cst-8k.jsonl \\
        --out   /content/results_lambada_CST-8K-seed0.json

    python training/experiments/downstream_eval.py \\
        --ckpt /content/checkpoints/AR-CST-8K-seed0 \\
        --task classification --num-labels 2 \\
        --train /content/downstream/hard-ar-cst-8k-train.jsonl \\
        --test  /content/downstream/hard-ar-cst-8k-test.jsonl \\
        --out   /content/results_hard_AR-CST-8K-seed0.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from experiments._core import set_seed  # noqa: E402


# \u2500\u2500 LM scoring \u2500\u2500

def _seq_nll(model, device, context_ids: list[int], completion_ids: list[int]) -> float:
    """Sum NLL (in nats) of completion_ids given context_ids, under the LM."""
    full = context_ids + completion_ids
    if len(full) < 2:
        return float("inf")
    x = torch.tensor([full], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=x).logits[0]  # (T, V)
    # next-token prediction: logits[t] predicts x[t+1]
    log_probs = torch.log_softmax(logits[:-1], dim=-1)
    targets = x[0, 1:]
    per_tok_nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (T-1,)
    # Only count loss over the completion tokens.
    cstart = max(len(context_ids) - 1, 0)
    return per_tok_nll[cstart:].sum().item()


def eval_lm_scoring(ckpt: str, test_path: str, max_len: int = 128) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(ckpt).to(device)
    model.eval()

    correct = 0
    total = 0
    per_row: list[dict[str, Any]] = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ctx = obj["context_ids"][-(max_len - 2):]  # truncate left to fit
            cands = obj["candidates"]
            gold = int(obj["gold"])
            nlls = []
            for c in cands:
                room = max_len - len(ctx)
                cc = c[:max(room, 1)]
                nlls.append(_seq_nll(model, device, ctx, cc))
            pred = int(min(range(len(nlls)), key=lambda i: nlls[i]))
            correct += int(pred == gold)
            total += 1
            per_row.append({"gold": gold, "pred": pred, "nlls": nlls})

    acc = correct / max(total, 1)
    return {"task": "lm_scoring", "n": total, "accuracy": acc,
            "ckpt": ckpt, "test": test_path}


# \u2500\u2500 Classification \u2500\u2500

class ClsDataset(Dataset):
    def __init__(self, path: str, max_len: int):
        self.rows: list[tuple[list[int], int]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ids = obj["ids"][:max_len]
                if len(ids) < 2:
                    continue
                self.rows.append((ids, int(obj["label"])))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def _cls_collate(batch, pad_id: int = 0):
    ids_list = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    max_len = max(len(s) for s in ids_list)
    padded = [s + [pad_id] * (max_len - len(s)) for s in ids_list]
    masks = [[1] * len(s) + [0] * (max_len - len(s)) for s in ids_list]
    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(masks, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def eval_classification(
    ckpt: str,
    train_path: str,
    test_path: str,
    num_labels: int,
    max_len: int = 128,
    ft_epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict[str, Any]:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(ckpt, output_hidden_states=True).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_dim = model.config.n_embd
    head = torch.nn.Linear(hidden_dim, num_labels).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_ds = ClsDataset(train_path, max_len)
    test_ds = ClsDataset(test_path, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_cls_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=_cls_collate)

    history = []
    for epoch in range(1, ft_epochs + 1):
        head.train()
        t0 = time.time()
        total_loss = 0.0
        n = 0
        for input_ids, attn, labels in train_loader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                out = model.transformer(input_ids=input_ids, attention_mask=attn)
                pooled = _mean_pool(out.last_hidden_state, attn)
            logits = head(pooled)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)
        train_loss = total_loss / max(n, 1)

        head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_ids, attn, labels in test_loader:
                input_ids = input_ids.to(device)
                attn = attn.to(device)
                labels = labels.to(device)
                out = model.transformer(input_ids=input_ids, attention_mask=attn)
                pooled = _mean_pool(out.last_hidden_state, attn)
                pred = head(pooled).argmax(dim=-1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        acc = correct / max(total, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "test_acc": acc,
                        "time_s": round(time.time() - t0, 1)})
        print(f"  epoch {epoch}/{ft_epochs}  train_loss={train_loss:.4f}  test_acc={acc:.4f}")

    best = max(history, key=lambda h: h["test_acc"])
    return {
        "task": "classification",
        "num_labels": num_labels,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "best_test_acc": best["test_acc"],
        "best_epoch": best["epoch"],
        "history": history,
        "ckpt": ckpt,
        "train": train_path,
        "test": test_path,
    }


# \u2500\u2500 CLI \u2500\u2500

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Checkpoint dir saved by train_and_eval().")
    p.add_argument("--task", choices=["lm_scoring", "classification"], required=True)
    p.add_argument("--train", default=None, help="Pre-tokenized train jsonl (classification only).")
    p.add_argument("--test", required=True, help="Pre-tokenized test jsonl.")
    p.add_argument("--num-labels", type=int, default=2)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--ft-epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.task == "lm_scoring":
        result = eval_lm_scoring(args.ckpt, args.test, max_len=args.max_len)
    else:
        if not args.train:
            p.error("--train is required for classification")
        result = eval_classification(
            ckpt=args.ckpt,
            train_path=args.train,
            test_path=args.test,
            num_labels=args.num_labels,
            max_len=args.max_len,
            ft_epochs=args.ft_epochs,
            seed=args.seed,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\n  \u2192 {args.out}")
    if args.task == "lm_scoring":
        print(f"  accuracy: {result['accuracy']:.4f}  (n={result['n']})")
    else:
        print(f"  best test acc: {result['best_test_acc']:.4f}  (epoch {result['best_epoch']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
