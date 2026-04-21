"""Tiny causal-LM dataset over tokenized reasoning JSONL.

Each record from ``tokenize_corpus.py`` has three token fields
(``question_tokens``, ``cot_tokens``, ``answer_tokens``), each of which
is a ``{"default": [...], "reasoning": [...]}`` dict. For the sanity
trainer we concatenate the **reasoning** streams only::

    seq = question_tokens.reasoning
        + flatten(cot_tokens[i].reasoning for i in …)
        + answer_tokens.reasoning

Each sub-sequence already starts with ``[BOS]`` and ends with ``[EOS]``
(inserted by the tokenizers), so concatenating gives a natural
segment-boundary signal the LM can learn from.

Sequences longer than ``max_len`` are truncated from the *left* so the
answer tokens (the supervision signal) are always present at the tail.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset


def load_vocab(vocab_path: Path) -> dict[str, int]:
    return json.loads(vocab_path.read_text(encoding="utf-8"))


def ids_from_tokens(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    unk = vocab["[UNK]"]
    return [vocab.get(t, unk) for t in tokens]


def _flatten_record(rec: dict, vocab: dict[str, int]) -> list[int]:
    seq: list[str] = []
    seq.extend(rec["question_tokens"]["reasoning"])
    for step in rec["cot_tokens"]:
        seq.extend(step["reasoning"])
    seq.extend(rec["answer_tokens"]["reasoning"])
    return ids_from_tokens(seq, vocab)


class ReasoningJsonlDataset(Dataset):
    """Streaming-friendly dataset that materializes id sequences in memory.

    For the sanity run (syllogisms = 20k records ≈ 1M tokens) this fits
    comfortably; swap to a streaming impl for larger stages.
    """

    def __init__(
        self,
        jsonl_path: Path,
        vocab: dict[str, int],
        *,
        max_len: int = 256,
    ) -> None:
        self.vocab = vocab
        self.max_len = max_len
        self.pad_id = vocab["[PAD]"]
        self._seqs: list[list[int]] = []
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                ids = _flatten_record(rec, vocab)
                if len(ids) > max_len:
                    ids = ids[-max_len:]           # keep answer at tail
                self._seqs.append(ids)

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self._seqs[idx]
        x = torch.full((self.max_len,), self.pad_id, dtype=torch.long)
        x[: len(seq)] = torch.tensor(seq, dtype=torch.long)
        return {"ids": x, "length": torch.tensor(len(seq), dtype=torch.long)}


def make_batch(batch: list[dict]) -> dict[str, torch.Tensor]:
    ids = torch.stack([b["ids"] for b in batch], dim=0)
    lengths = torch.stack([b["length"] for b in batch], dim=0)
    # Inputs are seq[:-1], targets seq[1:]
    inputs = ids[:, :-1]
    targets = ids[:, 1:].clone()
    # Mask padding in targets with -100 so CE ignores them.
    for i, L in enumerate(lengths.tolist()):
        if L - 1 < targets.size(1):
            targets[i, L - 1:] = -100
    return {"inputs": inputs, "targets": targets, "lengths": lengths}


def split_indices(n: int, *, val_frac: float = 0.05, seed: int = 42) -> tuple[list[int], list[int]]:
    import random
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    n_val = max(1, int(n * val_frac))
    return idxs[n_val:], idxs[:n_val]
