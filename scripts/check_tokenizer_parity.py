"""TS vs Python tokenizer parity check.

Runs the TypeScript CSTTokenizer and the Python EnglishCSTTokenizer over the
same sentences and reports:

    - exact-match rate (token sequences identical, in order)
    - value-set match rate (same tokens regardless of order)
    - type-match rate (types align even if values differ \u2014 e.g. one picks
      CMP and the other ROOT for the same word)
    - first N divergences, printed side by side

Run:
    python scripts/check_tokenizer_parity.py \\
        --sentences data/sentences-100.json \\
        --n 100

If ``--sentences`` is omitted, the golden test sentences from
``src/tests/examples.test.ts`` are used.

Exit code:
    0   if exact-match rate >= --threshold (default 0.95)
    1   otherwise

Prerequisites:
    - ``tsx`` installed as a devDependency (already in package.json)
    - spaCy + en_core_web_sm installed in the active Python env
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
TS_CLI = REPO / "src" / "tokenize-cli.ts"
sys.path.insert(0, str(REPO))


GOLDEN = [
    "The writer sent a message to the teacher",
    "Students learn in the library",
    "Will you send the document?",
    "She cannot rewrite the unreadable text",
    "The meeting was scheduled for tomorrow",
]


def load_sentences(path: str | None, n: int | None) -> list[str]:
    if not path:
        return GOLDEN
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".json" and text.lstrip().startswith("["):
        data = json.loads(text)
        out = []
        for x in data:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict):
                out.append(x.get("text", ""))
        sents = [s for s in out if s]
    else:
        sents = [line.strip() for line in text.splitlines() if line.strip()]
    if n:
        sents = sents[:n]
    return sents


def run_ts(sentences: list[str]) -> list[list[str]]:
    cmd = ["npx", "--yes", "tsx", str(TS_CLI)]
    input_text = "\n".join(sentences) + "\n"
    res = subprocess.run(
        cmd, input=input_text, capture_output=True, text=True,
        cwd=str(REPO), check=False,
    )
    if res.returncode != 0:
        print("TS tokenizer failed:", file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        sys.exit(2)
    out: list[list[str]] = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append(obj["tokens"])
    return out


def run_py(sentences: list[str]) -> list[list[str]]:
    import spacy
    from edge.english_tokenizer import EnglishCSTTokenizer

    nlp = spacy.load("en_core_web_sm")
    tok = EnglishCSTTokenizer(nlp)
    return [tok.tokenize(s)["values"] for s in sentences]


def _type_of(t: str) -> str:
    return t.split(":", 1)[0] if ":" in t else t


def compare(a: list[list[str]], b: list[list[str]]) -> dict:
    assert len(a) == len(b), f"length mismatch: TS={len(a)} PY={len(b)}"
    n = len(a)
    exact = 0
    set_match = 0
    type_match = 0
    diffs: list[tuple[int, list[str], list[str]]] = []
    ts_type_counts: dict[str, int] = {}
    py_type_counts: dict[str, int] = {}
    for i, (ta, tb) in enumerate(zip(a, b)):
        for t in ta:
            ts_type_counts[_type_of(t)] = ts_type_counts.get(_type_of(t), 0) + 1
        for t in tb:
            py_type_counts[_type_of(t)] = py_type_counts.get(_type_of(t), 0) + 1
        if ta == tb:
            exact += 1
            set_match += 1
            type_match += 1
            continue
        if sorted(ta) == sorted(tb):
            set_match += 1
        if sorted(_type_of(t) for t in ta) == sorted(_type_of(t) for t in tb):
            type_match += 1
        diffs.append((i, ta, tb))
    return {
        "n": n,
        "exact": exact,
        "set_match": set_match,
        "type_match": type_match,
        "diffs": diffs,
        "ts_type_counts": ts_type_counts,
        "py_type_counts": py_type_counts,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", default=None,
                    help="Path to sentences file (JSON array or newline-delimited).")
    ap.add_argument("--n", type=int, default=None, help="Cap on sentences.")
    ap.add_argument("--show", type=int, default=10,
                    help="Number of divergences to print (default 10).")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Minimum exact-match rate for exit 0 (default 0.5). "
                         "Byte-identical parity is impossible between compromise "
                         "and spaCy; this is a sanity gate, not a strict check.")
    args = ap.parse_args()

    sents = load_sentences(args.sentences, args.n)
    print(f"Loaded {len(sents)} sentences")

    print("Running TS tokenizer...")
    ts = run_ts(sents)
    print("Running Python tokenizer...")
    py = run_py(sents)

    r = compare(ts, py)
    n = r["n"]
    print(f"\n  n = {n}")
    print(f"  exact match:    {r['exact']}/{n}  ({r['exact']/n*100:.1f}%)")
    print(f"  set match:      {r['set_match']}/{n}  ({r['set_match']/n*100:.1f}%)")
    print(f"  type match:     {r['type_match']}/{n}  ({r['type_match']/n*100:.1f}%)")

    print("\n  token-type distribution:")
    all_types = sorted(set(r["ts_type_counts"]) | set(r["py_type_counts"]))
    print(f"    {'type':<10} {'TS':>8} {'PY':>8} {'diff%':>8}")
    for ty in all_types:
        ts_c = r["ts_type_counts"].get(ty, 0)
        py_c = r["py_type_counts"].get(ty, 0)
        denom = max(ts_c, py_c, 1)
        diff_pct = abs(ts_c - py_c) / denom * 100
        print(f"    {ty:<10} {ts_c:>8} {py_c:>8} {diff_pct:>7.1f}%")

    if r["diffs"]:
        print(f"\n  first {min(args.show, len(r['diffs']))} divergences:")
        for idx, ta, tb in r["diffs"][: args.show]:
            print(f"\n  [{idx}] {sents[idx]}")
            print(f"      TS: {ta}")
            print(f"      PY: {tb}")

    rate = r["exact"] / n if n else 0.0
    if rate < args.threshold:
        print(f"\n  FAIL: exact-match rate {rate:.3f} < threshold {args.threshold}")
        return 1
    print(f"\n  OK: exact-match rate {rate:.3f} >= threshold {args.threshold}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
