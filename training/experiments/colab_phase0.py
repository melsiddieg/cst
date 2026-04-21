"""
Phase 0 Colab driver.

One-shot runner for the full Phase 0 experiment grid:
    - languages:   EN, AR, RU (RU runs only if tokenized pairs are present)
    - vocab sizes: 8K, 32K
    - tokenizers:  CST, SPM
    - seeds:       5 by default (0..4)
    - downstream:  LAMBADA (EN), HARD (AR), RuSentiment (RU) \u2014
                   only the ones whose prepared files are present in
                   ``$DATA_DIR/downstream/``.

Assumed Colab layout (upload or mount before running):

    /content/
        cst-8k/train-99963.jsonl            (+ -vocab.json)
        spm/train-99963-8k.jsonl            (+ -vocab.json)
        cst-32k/...
        spm/...
        cst-ar-8k/train-100000.jsonl        (+ -vocab.json)
        spm-ar/ar-bpe-8000.jsonl            (+ -vocab.json)
        cst-ar-32k/...  spm-ar/ar-bpe-32000...
        cst-ru-8k/...   spm-ru/ru-bpe-8000...      (optional \u2014 see plan/PHASE0_RUSSIAN.md)
        cst-ru-32k/...  spm-ru/ru-bpe-32000...     (optional)
        downstream/
            lambada-en-cst-8k.jsonl         lambada-en-spm-8k.jsonl
            lambada-en-cst-32k.jsonl        lambada-en-spm-32k.jsonl
            hard-ar-cst-8k-train.jsonl      hard-ar-cst-8k-test.jsonl
            ... (one pair per (task, tokenizer, vocab))

Outputs (under ``$OUT_DIR``, default /content/phase0_out):
    results_<lang>.json               \u2014 multi-seed BPC per tokenizer
    checkpoints/<NAME>-seed<S>/       \u2014 best-BPC checkpoint per run
    downstream/<task>_<NAME>-seed<S>.json
    summary.md                        \u2014 aggregated tables

Usage in a Colab cell:

    !pip install -q torch transformers
    !git clone https://github.com/<you>/cst-poc /content/cst-poc
    %cd /content/cst-poc
    !python training/experiments/colab_phase0.py \\
        --data-dir /content \\
        --out-dir  /content/phase0_out \\
        --langs en ar \\
        --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

from experiments.run_multiseed import PAIRS  # noqa: E402


# Downstream task wiring. Each entry maps (lang, name) \u2192 prepared files.
# ``name`` matches the first element of the PAIRS tuples (e.g., "CST-8K").
# The prepared files are expected at ``$DATA_DIR/downstream/<pattern>``.
DOWNSTREAM = {
    "en": {
        "task": "lm_scoring",
        "num_labels": None,
        # {name} is the tokenizer name, lowercased.
        "test_pattern":  "downstream/lambada-en-{name}.jsonl",
        "train_pattern": None,
    },
    "ar": {
        "task": "classification",
        "num_labels": 2,
        "train_pattern": "downstream/hard-ar-{name}-train.jsonl",
        "test_pattern":  "downstream/hard-ar-{name}-test.jsonl",
    },
    "ru": {
        "task": "classification",
        "num_labels": 3,  # RuSentiment: positive / neutral / negative
        "train_pattern": "downstream/rusentiment-{name}-train.jsonl",
        "test_pattern":  "downstream/rusentiment-{name}-test.jsonl",
    },
}


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def _run_multiseed(lang: str, runs: list[str], seeds: list[int],
                   data_dir: str, out_dir: str, epochs: int) -> str:
    out = os.path.join(out_dir, f"results_{lang}.json")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    cmd = [
        sys.executable,
        os.path.join(HERE, "run_multiseed.py"),
        "--lang", lang,
        "--runs", *runs,
        "--seeds", *[str(s) for s in seeds],
        "--data-dir", data_dir,
        "--out", out,
        "--epochs", str(epochs),
        "--ckpt-dir", ckpt_dir,
    ]
    _run(cmd)
    return out


def _run_downstream(lang: str, runs: list[str], seeds: list[int],
                    data_dir: str, out_dir: str) -> list[str]:
    cfg = DOWNSTREAM[lang]
    produced: list[str] = []
    for run in runs:
        for (name, _, _) in PAIRS[lang][run]:
            name_lc = name.lower()
            test_path = os.path.join(data_dir, cfg["test_pattern"].format(name=name_lc))
            if not os.path.exists(test_path):
                print(f"  downstream skip: missing {test_path}")
                continue
            train_path = None
            if cfg["train_pattern"]:
                train_path = os.path.join(data_dir, cfg["train_pattern"].format(name=name_lc))
                if not os.path.exists(train_path):
                    print(f"  downstream skip: missing {train_path}")
                    continue

            for seed in seeds:
                ckpt = os.path.join(out_dir, "checkpoints", f"{name}-seed{seed}")
                if not os.path.isdir(ckpt):
                    print(f"  downstream skip: missing ckpt {ckpt}")
                    continue
                out_file = os.path.join(out_dir, "downstream",
                                        f"{cfg['task']}_{name}-seed{seed}.json")
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                cmd = [
                    sys.executable,
                    os.path.join(HERE, "downstream_eval.py"),
                    "--ckpt", ckpt,
                    "--task", cfg["task"],
                    "--test", test_path,
                    "--out", out_file,
                    "--seed", str(seed),
                ]
                if train_path:
                    cmd += ["--train", train_path,
                            "--num-labels", str(cfg["num_labels"])]
                _run(cmd)
                produced.append(out_file)
    return produced


def _summarize(out_dir: str, bpc_files: list[str], downstream_files: list[str]) -> str:
    """Emit a small Markdown summary. BPC table + downstream table."""
    from collections import defaultdict
    import math

    def _stats(xs: list[float]) -> tuple[float, float]:
        if not xs:
            return float("nan"), float("nan")
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
        return m, math.sqrt(v)

    lines = ["# Phase 0 \u2014 Results Summary\n"]

    # BPC
    lines.append("## Language modeling (mean \u00b1 std, n=seeds)\n")
    lines.append("| Language | Run | Tokenizer | BPC (mean \u00b1 std) | n |")
    lines.append("|---|---|---|---|---|")
    for f in bpc_files:
        try:
            data = json.load(open(f))
        except Exception:
            continue
        by_key: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for r in data:
            key = (r.get("lang", "?"), r.get("run", "?"), r["name"])
            by_key[key].append(r["best_val_bpc"])
        for (lang, run, name), bpcs in sorted(by_key.items()):
            m, s = _stats(bpcs)
            lines.append(f"| {lang} | {run} | {name} | {m:.4f} \u00b1 {s:.4f} | {len(bpcs)} |")

    # Downstream
    lines.append("\n## Downstream tasks (mean \u00b1 std, n=seeds)\n")
    lines.append("| Tokenizer | Task | Metric | Score (mean \u00b1 std) | n |")
    lines.append("|---|---|---|---|---|")
    by_name: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for f in downstream_files:
        try:
            r = json.load(open(f))
        except Exception:
            continue
        # Name derived from filename: <task>_<NAME>-seed<S>.json
        base = os.path.basename(f).rsplit(".", 1)[0]
        parts = base.split("_", 1)
        task_tag = parts[0] if parts else "?"
        name = parts[1].rsplit("-seed", 1)[0] if len(parts) > 1 else "?"
        if r.get("task") == "lm_scoring":
            metric, score = "accuracy", r["accuracy"]
        else:
            metric, score = "test_acc", r["best_test_acc"]
        by_name[(name, task_tag, metric)].append(score)
    for (name, task, metric), xs in sorted(by_name.items()):
        m, s = _stats(xs)
        lines.append(f"| {name} | {task} | {metric} | {m:.4f} \u00b1 {s:.4f} | {len(xs)} |")

    path = os.path.join(out_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/content")
    p.add_argument("--out-dir", default="/content/phase0_out")
    p.add_argument("--langs", nargs="+", default=["en", "ar"],
                   choices=["en", "ar", "ru"])
    p.add_argument("--runs", nargs="+", default=["8k", "32k"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--skip-downstream", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bpc_files: list[str] = []
    for lang in args.langs:
        print(f"\n=== Multi-seed \u2014 {lang.upper()} ===")
        out = _run_multiseed(lang, args.runs, args.seeds,
                             args.data_dir, args.out_dir, args.epochs)
        bpc_files.append(out)

    downstream_files: list[str] = []
    if not args.skip_downstream:
        for lang in args.langs:
            print(f"\n=== Downstream \u2014 {lang.upper()} ===")
            downstream_files += _run_downstream(
                lang, args.runs, args.seeds, args.data_dir, args.out_dir,
            )

    summary = _summarize(args.out_dir, bpc_files, downstream_files)
    print(f"\n\u2192 Summary written to {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
