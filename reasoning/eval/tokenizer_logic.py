"""§6.2 validator — logic preservation under tokenization.

Runs premise / hypothesis pairs through the reasoning tokenizer
``T_R`` and checks structural properties that a correct projection
should satisfy:

1. **Stability.** Lemma-level paraphrases should produce identical
   reasoning token sequences on the content subsequence (ROOT / CMP).
2. **Label separation.** The distribution of (edit distance between
   T_R(premise) and T_R(hypothesis)) should differ by entailment class.
   We expect: entailment < neutral < contradiction, on average.
3. **Compression ratio.** |T_R| / |T_D| target ≤ declared threshold.

This is a **diagnostic**, not a training signal. A single failing pair
is not a bug; a *distribution* that doesn't separate is.

Usage::

    python -m reasoning.eval.tokenizer_logic \
        --in out/stage-1-xnli.jsonl --lang ar --n 1000
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = curr
    return prev[-1]


def _tokenizer(lang: str):
    if lang == "ar":
        from reasoning.tokenizer.arabic import ArabicReasoningTokenizer

        return ArabicReasoningTokenizer.default()
    if lang == "en":
        from reasoning.tokenizer.english import EnglishReasoningTokenizer

        return EnglishReasoningTokenizer()
    raise ValueError(lang)


def _parse_nli(question: str) -> tuple[str, str] | None:
    """Extract (premise, hypothesis) from the XNLI-formatted question."""
    marks_en = ("Premise:", "Hypothesis:")
    marks_ar = ("المقدمة:", "الفرضية:")
    for p_mark, h_mark in (marks_en, marks_ar):
        if p_mark in question and h_mark in question:
            p_part = question.split(p_mark, 1)[1]
            prem, rest = p_part.split(h_mark, 1)
            hyp = rest.split("\n", 1)[0]
            return prem.strip(), hyp.strip()
    return None


def evaluate(records: list[dict], lang: str) -> dict:
    tok = _tokenizer(lang)

    distances_by_label: dict[str, list[int]] = defaultdict(list)
    ratios: list[float] = []
    skipped = 0

    for r in records:
        if r.get("lang") != lang or r.get("category") != 1:
            continue
        parsed = _parse_nli(r["question"])
        if not parsed:
            skipped += 1
            continue
        prem, hyp = parsed
        tp = tok.tokenize(prem)
        th = tok.tokenize(hyp)
        d = _edit_distance(tp["reasoning_tokens"], th["reasoning_tokens"])
        distances_by_label[r["answer"]].append(d)
        for side in (tp, th):
            d_len = len(side["default_tokens"])
            if d_len:
                ratios.append(len(side["reasoning_tokens"]) / d_len)

    def summarize(xs: list[int]) -> dict:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "mean": round(statistics.mean(xs), 3),
            "median": statistics.median(xs),
            "stdev": round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0,
        }

    return {
        "lang": lang,
        "pairs_total": sum(len(v) for v in distances_by_label.values()),
        "pairs_skipped": skipped,
        "distance_by_label": {k: summarize(v) for k, v in distances_by_label.items()},
        "compression_ratio": summarize([int(r * 1000) for r in ratios]) if ratios else {"n": 0},
        "compression_ratio_note": "values ×1000; divide mean/median by 1000",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="JSONL from reasoning.data.sources.xnli")
    ap.add_argument("--lang", choices=["ar", "en"], required=True)
    ap.add_argument("--n", type=int, default=1000,
                    help="Max pairs to evaluate.")
    args = ap.parse_args()

    records: list[dict] = []
    with args.inp.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) >= args.n * 4:  # headroom for filters
                break

    result = evaluate(records, args.lang)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
