#!/usr/bin/env tsx
/**
 * TS tokenizer CLI — stdin/stdout, matches `python -m edge.tokenize` output.
 *
 * Each input line is tokenized and emitted as a single jsonl row:
 *     {"text": "<input>", "tokens": ["tok1", "tok2", ...]}
 *
 * Used by scripts/check_tokenizer_parity.py to verify the TS and Python
 * implementations produce the same token sequences on the same input.
 *
 * Usage:
 *   cat sentences.txt | tsx src/tokenize-cli.ts > tokens.jsonl
 */

import { CSTTokenizer } from "./tokenizer/index.ts";
import { createInterface } from "node:readline";

const tokenizer = new CSTTokenizer();
const rl = createInterface({ input: process.stdin, terminal: false });

rl.on("line", (line) => {
  const text = line;
  if (!text) return;
  const out = tokenizer.tokenize(text);
  const values = out.tokens.map((t) => t.value);
  process.stdout.write(JSON.stringify({ text, tokens: values }) + "\n");
});
