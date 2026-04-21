"""Master reasoning-data build pipeline.

Invokes per-stage builders and writes one JSONL per stage under
``--out``::

    out/
      stage-2a-prop_logic.jsonl
      stage-2b-syllogisms.jsonl
      stage-2c-algebra_engine.jsonl    (if --with-algebra)
      stage-3-gsm8k.jsonl              (if --with-gsm8k)
      stage-1-xnli.jsonl               (category 1; eval set)
      manifest.json                    summary of counts + licenses

Usage::

    python -m reasoning.data.build --stage all --out ./out
    python -m reasoning.data.build --stage 2  --out ./out   # synthetic only
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _run_prop_logic(out: Path, count: int) -> Path:
    from .generators import prop_logic

    path = out / "stage-2a-prop_logic.jsonl"
    from .schema import write_jsonl
    n = write_jsonl(path, prop_logic.generate(count))
    print(f"  prop_logic:  {n:,} → {path.name}")
    return path


def _run_syllogisms(out: Path, count: int) -> Path:
    from .generators import syllogisms
    from .schema import write_jsonl

    path = out / "stage-2b-syllogisms.jsonl"
    n = write_jsonl(path, syllogisms.generate(count))
    print(f"  syllogisms:  {n:,} → {path.name}")
    return path


def _run_xnli(out: Path, n_per_lang: int) -> Path:
    from .sources.xnli import _iter
    from .schema import write_jsonl

    def stream():
        yield from _iter("en", n_per_lang)
        yield from _iter("ar", n_per_lang)

    path = out / "stage-1-xnli.jsonl"
    n = write_jsonl(path, stream())
    print(f"  xnli:        {n:,} → {path.name}")
    return path


def _run_gsm8k(out: Path, langs: set[str]) -> Path:
    from .sources.gsm8k import _english_records, _translate_records
    from .schema import write_jsonl

    def stream():
        en = list(_english_records(None))
        if "en" in langs:
            yield from en
        if "ar" in langs:
            yield from _translate_records(en)

    path = out / "stage-3-gsm8k.jsonl"
    n = write_jsonl(path, stream())
    print(f"  gsm8k:       {n:,} → {path.name}")
    return path


def _manifest(out: Path, files: list[Path]) -> None:
    summary: dict[str, dict] = {}
    for f in files:
        counts: Counter = Counter()
        langs: Counter = Counter()
        licenses: Counter = Counter()
        categories: Counter = Counter()
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                counts["total"] += 1
                langs[r["lang"]] += 1
                licenses[r["meta"]["license"]] += 1
                categories[str(r["category"])] += 1
        summary[f.name] = {
            "total": counts["total"],
            "by_lang": dict(langs),
            "by_license": dict(licenses),
            "by_category": dict(categories),
        }
    with (out / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"  manifest  → {out / 'manifest.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="2",
                    choices=["1", "2", "3", "all"],
                    help="Which stage(s) to build.")
    ap.add_argument("--out", type=Path, default=Path("./out"))
    ap.add_argument("--prop-count", type=int, default=25_000,
                    help="Prop-logic count per language pair (yields 2×).")
    ap.add_argument("--syllog-count", type=int, default=10_000)
    ap.add_argument("--xnli-count", type=int, default=10_000)
    ap.add_argument("--gsm8k-langs", default="en",
                    help="Comma-separated subset of {en,ar}.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    if args.stage in ("2", "all"):
        print("── Stage 2 — synthetic formal logic ──")
        files.append(_run_prop_logic(args.out, args.prop_count))
        files.append(_run_syllogisms(args.out, args.syllog_count))

    if args.stage in ("1", "all"):
        print("── Stage 1 — entailment (XNLI) ──")
        try:
            files.append(_run_xnli(args.out, args.xnli_count))
        except Exception as e:
            print(f"  skipped: {e}")

    if args.stage in ("3", "all"):
        print("── Stage 3 — CoT (GSM8K) ──")
        try:
            langs = {x.strip() for x in args.gsm8k_langs.split(",")}
            files.append(_run_gsm8k(args.out, langs))
        except Exception as e:
            print(f"  skipped: {e}")

    if files:
        _manifest(args.out, files)


if __name__ == "__main__":
    main()
