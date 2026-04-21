"""Phase 1 result comparator — CST vs BPE BPC on the same 1M pool.

Reads the summary files written by `colab_edge_1m.py` (summary_cst.json) and
`colab_bpe_1m.py` (summary_bpe.json) and prints a side-by-side table.

Usage:
  python compare_bpc.py
  python compare_bpc.py --cst /content/edge_model_1m/summary_cst.json \
                       --bpe /content/edge_model_1m_bpe/summary_bpe.json
"""
from __future__ import annotations

import argparse
import json
import os


def load(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"Missing summary: {path}")
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cst", default="/content/edge_model_1m/summary_cst.json")
    ap.add_argument("--bpe", default="/content/edge_model_1m_bpe/summary_bpe.json")
    args = ap.parse_args()

    cst = load(args.cst)
    bpe = load(args.bpe)

    rows = [
        ("sentences",   cst.get("n_sentences"),  bpe.get("n_sentences")),
        ("vocab_size",  cst.get("vocab_size"),   bpe.get("vocab_size")),
        ("n_params",    cst.get("n_params"),     bpe.get("n_params")),
        ("best_val_bpc", cst.get("best_val_bpc"), bpe.get("best_val_bpc")),
    ]

    print("=" * 60)
    print("  Phase 1 — CST vs BPE (same 1M sentences, same 6.8M GPT-2)")
    print("=" * 60)
    print(f"  {'metric':<16} {'CST':>18} {'BPE':>18}")
    print(f"  {'-'*16} {'-'*18:>18} {'-'*18:>18}")
    for name, c, b in rows:
        if isinstance(c, float) or isinstance(b, float):
            cs = f"{c:.4f}" if c is not None else "—"
            bs = f"{b:.4f}" if b is not None else "—"
        else:
            cs = f"{c:,}" if c is not None else "—"
            bs = f"{b:,}" if b is not None else "—"
        print(f"  {name:<16} {cs:>18} {bs:>18}")

    c_bpc = cst.get("best_val_bpc")
    b_bpc = bpe.get("best_val_bpc")
    if c_bpc is not None and b_bpc is not None:
        delta = b_bpc - c_bpc
        pct = (delta / b_bpc) * 100
        winner = "CST" if c_bpc < b_bpc else ("BPE" if b_bpc < c_bpc else "tie")
        print()
        print(f"  Δ BPC (BPE - CST): {delta:+.4f}  ({pct:+.1f}% of BPE)")
        print(f"  Winner: {winner}")
    print("=" * 60)


if __name__ == "__main__":
    main()
