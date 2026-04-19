/**
 * Process sentences with CST tokenizer → .jsonl files for training.
 * Also produces BPE baseline .jsonl (simple whitespace token IDs).
 */

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { CSTTokenizer } from "../tokenizer/index.ts";

interface TrainingExample {
  ids: number[];
  tokens: string[];
  text: string;
}

/**
 * Tokenize all sentences with CST and write .jsonl
 */
export function processCST(
  sentences: string[],
  outputPath: string,
): { examples: number; vocabSize: number } {
  const tokenizer = new CSTTokenizer();

  mkdirSync(dirname(outputPath), { recursive: true });

  let lines: string[] = [];
  for (const sentence of sentences) {
    const output = tokenizer.tokenize(sentence);
    if (output.ids.length < 3) continue; // skip tiny

    const example: TrainingExample = {
      ids: output.ids,
      tokens: output.tokens.map((t) => t.value),
      text: sentence,
    };
    lines.push(JSON.stringify(example));
  }

  writeFileSync(outputPath, lines.join("\n"));

  // Save vocabulary alongside
  const vocabPath = outputPath.replace(/\.jsonl$/, "-vocab.json");
  tokenizer.saveVocab(vocabPath);

  return { examples: lines.length, vocabSize: tokenizer.getVocabSize() };
}

/**
 * Simple whitespace BPE-like baseline: each unique lowercased word gets an ID.
 */
export function processBPEBaseline(
  sentences: string[],
  outputPath: string,
): { examples: number; vocabSize: number } {
  const vocab = new Map<string, number>();
  let nextId = 0;

  function getId(word: string): number {
    const existing = vocab.get(word);
    if (existing !== undefined) return existing;
    const id = nextId++;
    vocab.set(word, id);
    return id;
  }

  // Reserve special tokens
  getId("[PAD]");
  getId("[UNK]");
  getId("[BOS]");
  getId("[EOS]");

  mkdirSync(dirname(outputPath), { recursive: true });

  const lines: string[] = [];
  for (const sentence of sentences) {
    const words = sentence
      .toLowerCase()
      .replace(/[.,!?;:'"()\-]/g, " $& ")
      .split(/\s+/)
      .filter((w) => w.length > 0);
    if (words.length < 3) continue;

    const ids = words.map((w) => getId(w));
    const example: TrainingExample = {
      ids,
      tokens: words,
      text: sentence,
    };
    lines.push(JSON.stringify(example));
  }

  writeFileSync(outputPath, lines.join("\n"));

  // Save vocab
  const vocabPath = outputPath.replace(/\.jsonl$/, "-vocab.json");
  const vocabObj = Object.fromEntries(vocab);
  writeFileSync(vocabPath, JSON.stringify(vocabObj, null, 2));

  return { examples: lines.length, vocabSize: vocab.size };
}

// ── CLI entry point ──

if (process.argv[1]?.endsWith("process.ts")) {
  const inputPath = process.argv[2] ?? "data/sentences.json";
  const cstOutput = process.argv[3] ?? "data/tokenized/cst/train.jsonl";
  const bpeOutput = process.argv[4] ?? "data/tokenized/bpe/train.jsonl";

  console.log(`Loading sentences from ${inputPath}...`);
  const sentences = JSON.parse(readFileSync(inputPath, "utf-8")) as string[];
  console.log(`Loaded ${sentences.length} sentences`);

  console.log("\nProcessing with CST tokenizer...");
  const cst = processCST(sentences, cstOutput);
  console.log(`  CST: ${cst.examples} examples, vocab size ${cst.vocabSize}`);

  console.log("\nProcessing with BPE baseline...");
  const bpe = processBPEBaseline(sentences, bpeOutput);
  console.log(`  BPE: ${bpe.examples} examples, vocab size ${bpe.vocabSize}`);

  console.log("\nDone!");
}
