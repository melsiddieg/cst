/**
 * Coverage analysis — run CST tokenizer on sentences and report
 * detailed statistics about token type distribution.
 */

import { readFileSync } from "node:fs";
import { CSTTokenizer } from "../tokenizer/index.ts";

export interface DetailedCoverage {
  total: number;
  cmp: number;
  root: number;
  str: number;
  rel: number;
  lit: number;
  unk: number;
  structuredRatio: number;
  // Top semantic fields by frequency
  topFields: [string, number][];
  // Top roles by frequency
  topRoles: [string, number][];
  // Top LIT tokens (coverage gap candidates)
  topLiterals: [string, number][];
  // Missing: words that became LIT but could potentially be mapped
  missedWords: [string, number][];
}

export function analyzeCorpusCoverage(sentences: string[]): DetailedCoverage {
  const tokenizer = new CSTTokenizer();

  const fieldCounts = new Map<string, number>();
  const roleCounts = new Map<string, number>();
  const litCounts = new Map<string, number>();
  const missedCounts = new Map<string, number>();

  let total = 0,
    cmp = 0,
    root = 0,
    str = 0,
    rel = 0,
    lit = 0,
    unk = 0;

  for (const sentence of sentences) {
    const output = tokenizer.tokenize(sentence);
    for (const tok of output.tokens) {
      total++;
      switch (tok.type) {
        case "CMP":
          cmp++;
          if (tok.field)
            fieldCounts.set(tok.field, (fieldCounts.get(tok.field) ?? 0) + 1);
          if (tok.role)
            roleCounts.set(tok.role, (roleCounts.get(tok.role) ?? 0) + 1);
          break;
        case "ROOT":
          root++;
          if (tok.field)
            fieldCounts.set(tok.field, (fieldCounts.get(tok.field) ?? 0) + 1);
          break;
        case "STR":
          str++;
          break;
        case "REL":
          rel++;
          break;
        case "LIT":
          lit++;
          litCounts.set(tok.value, (litCounts.get(tok.value) ?? 0) + 1);
          // Track non-function-word LITs as missed opportunities
          if (
            !tok.value.match(
              /^LIT:(the|a|an|i|me|my|you|he|she|it|we|they|is|am|are|was|were|be|been|do|does|did|has|have|had|will|would|shall|should|can|could|may|might|must|not|no|very|too|also|just|only)$/,
            )
          ) {
            missedCounts.set(
              tok.surface,
              (missedCounts.get(tok.surface) ?? 0) + 1,
            );
          }
          break;
        default:
          unk++;
      }
    }
  }

  const structured = cmp + root + str + rel;
  const structuredRatio = total > 0 ? structured / total : 0;

  const sort = (m: Map<string, number>) =>
    Array.from(m.entries()).sort((a, b) => b[1] - a[1]);

  return {
    total,
    cmp,
    root,
    str,
    rel,
    lit,
    unk,
    structuredRatio,
    topFields: sort(fieldCounts).slice(0, 20),
    topRoles: sort(roleCounts).slice(0, 15),
    topLiterals: sort(litCounts).slice(0, 30),
    missedWords: sort(missedCounts).slice(0, 50),
  };
}

function printCoverage(cov: DetailedCoverage) {
  console.log("\n═══ COVERAGE ANALYSIS ═══");
  console.log(`  Total tokens: ${cov.total}`);
  console.log(`  CMP:          ${pct(cov.cmp, cov.total)}  (${cov.cmp})`);
  console.log(`  ROOT:         ${pct(cov.root, cov.total)}  (${cov.root})`);
  console.log(`  STR:          ${pct(cov.str, cov.total)}  (${cov.str})`);
  console.log(`  REL:          ${pct(cov.rel, cov.total)}  (${cov.rel})`);
  console.log(`  LIT:          ${pct(cov.lit, cov.total)}  (${cov.lit})`);
  console.log(`  UNK:          ${pct(cov.unk, cov.total)}  (${cov.unk})`);
  console.log(
    `  STRUCTURED:   ${pct(cov.cmp + cov.root + cov.str + cov.rel, cov.total)}`,
  );

  const gate = cov.structuredRatio >= 0.55 ? "✓ PASS" : "✗ FAIL";
  console.log(`\n  Coverage gate (>55%): ${gate}`);

  console.log("\n── Top Semantic Fields ──");
  for (const [field, count] of cov.topFields.slice(0, 15)) {
    console.log(`  ${field.padEnd(15)} ${count}`);
  }

  console.log("\n── Top Roles ──");
  for (const [role, count] of cov.topRoles.slice(0, 10)) {
    console.log(`  ${role.padEnd(15)} ${count}`);
  }

  console.log("\n── Top Missed Words (coverage gap candidates) ──");
  for (const [word, count] of cov.missedWords.slice(0, 30)) {
    console.log(`  ${word.padEnd(20)} ${count}`);
  }
}

function pct(n: number, total: number): string {
  return `${((n / total) * 100).toFixed(1)}%`.padStart(6);
}

// ── CLI entry point ──

if (process.argv[1]?.endsWith("stats.ts")) {
  const inputPath = process.argv[2] ?? "data/sentences.json";
  console.log(`Loading sentences from ${inputPath}...`);
  const sentences = JSON.parse(readFileSync(inputPath, "utf-8")) as string[];
  console.log(`Analyzing ${sentences.length} sentences...`);

  const coverage = analyzeCorpusCoverage(sentences);
  printCoverage(coverage);
}
