"""GSM8K — English CoT, optionally translated to Arabic.

English records have ``lang="en"`` and the original CoT. Arabic records
are produced by running the translator over the English records.

Usage::

    # English only
    python -m reasoning.data.sources.gsm8k --out out/gsm8k.jsonl --langs en

    # Both languages (requires NLLB-200 on first run)
    python -m reasoning.data.sources.gsm8k --out out/gsm8k.jsonl --langs en,ar
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

from ..schema import Meta, Record, write_jsonl


_STEP_SPLIT = re.compile(r"(?<=[.?!])\s+|\n+")


def _english_records(n: int | None) -> Iterable[Record]:
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="train")
    count = 0
    for i, row in enumerate(ds):
        if n is not None and count >= n:
            break
        question = row["question"].strip()
        full_answer = row["answer"].strip()
        # GSM8K answers look like: "... #### 42" — split into trace + final
        if "####" in full_answer:
            trace, final = full_answer.rsplit("####", 1)
            final = final.strip()
        else:
            trace, final = full_answer, ""
        cot = [s.strip() for s in _STEP_SPLIT.split(trace) if s.strip()]
        yield Record(
            id=f"gsm8k-en-{i:06d}",
            lang="en",
            category=3,
            question=question,
            answer=final,
            cot=cot,
            meta=Meta(
                source="gsm8k",
                license="mit",
                difficulty="medium",
            ),
        )
        count += 1


def _translate_records(records: Iterable[Record]) -> Iterable[Record]:
    from ..translate import translate_en_to_ar

    for r in records:
        ar_question = translate_en_to_ar(r.question)
        ar_cot = [translate_en_to_ar(s) for s in r.cot]
        ar_answer = r.answer  # numeric — keep as is
        yield Record(
            id=r.id.replace("-en-", "-ar-"),
            lang="ar",
            category=3,
            question=ar_question,
            answer=ar_answer,
            cot=ar_cot,
            meta=Meta(
                source="gsm8k",
                license="mit",
                translated_from="en",
                translation_quality=None,  # filled by QA pass
                difficulty=r.meta.difficulty,
            ),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="Limit per language (default: all).")
    ap.add_argument("--langs", default="en",
                    help="Comma-separated subset of {en,ar}.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    langs = {x.strip() for x in args.langs.split(",") if x.strip()}

    def stream() -> Iterable[Record]:
        en = list(_english_records(args.n))
        if "en" in langs:
            yield from en
        if "ar" in langs:
            yield from _translate_records(en)

    n = write_jsonl(args.out, stream())
    print(f"Wrote {n:,} GSM8K records to {args.out}")


if __name__ == "__main__":
    main()
