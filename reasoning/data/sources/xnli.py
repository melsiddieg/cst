"""XNLI (Arabic + English) entailment source — Category 1.

Streams XNLI from HuggingFace and emits reasoning-data JSONL records.
Category 1 items have empty ``cot`` and an ``answer`` drawn from
``{entails, contradicts, neutral}``.

Usage::

    python -m reasoning.data.sources.xnli --n 10000 --out out/xnli.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from ..schema import Meta, Record, write_jsonl


_LABELS_EN = {0: "entails", 1: "neutral", 2: "contradicts"}
_LABELS_AR = {0: "يستلزم", 1: "محايد", 2: "يناقض"}


def _iter(lang: str, n: int) -> Iterable[Record]:
    from datasets import load_dataset

    ds = load_dataset("xnli", lang, split="train", streaming=True)
    emitted = 0
    labels = _LABELS_EN if lang == "en" else _LABELS_AR
    for i, row in enumerate(ds):
        if emitted >= n:
            break
        prem = (row.get("premise") or "").strip()
        hyp = (row.get("hypothesis") or "").strip()
        if not prem or not hyp:
            continue
        label = labels[row["label"]]
        question = (
            f"Premise: {prem}\nHypothesis: {hyp}\n"
            f"Does the premise entail, contradict, or stay neutral to the hypothesis?"
            if lang == "en"
            else
            f"المقدمة: {prem}\nالفرضية: {hyp}\n"
            f"هل تستلزم المقدمة الفرضية أم تناقضها أم محايدة؟"
        )
        yield Record(
            id=f"xnli-{lang}-{i:07d}",
            lang=lang,  # type: ignore[arg-type]
            category=1,
            question=question,
            answer=label,
            meta=Meta(source="xnli", license="cc-by-nc-4.0"),
        )
        emitted += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000,
                    help="Items per language (total = 2n).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    def both() -> Iterable[Record]:
        yield from _iter("en", args.n)
        yield from _iter("ar", args.n)

    n = write_jsonl(args.out, both())
    print(f"Wrote {n:,} XNLI records to {args.out}")


if __name__ == "__main__":
    main()
