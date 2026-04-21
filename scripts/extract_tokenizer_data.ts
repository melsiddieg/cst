/**
 * Dumps TS tokenizer constants → `data/tokenizer/*.json`.
 *
 * TS is the authoritative source. The JSONs are derived artefacts consumed
 * by the Python tokenizer (`edge/english_tokenizer.py`) so both runtimes
 * share a single semantic-field / role / function-word table.
 *
 * Usage:
 *   npx tsx scripts/extract_tokenizer_data.ts
 *   npx tsx scripts/extract_tokenizer_data.ts --check   # drift detection
 *
 * Exit codes:
 *   0 — files up to date (or written successfully)
 *   1 — --check mode found drift
 */
import { writeFileSync, readFileSync, existsSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { SEMANTIC_FIELDS } from "../src/tokenizer/semanticFields.ts";
import { RELATION_MAP, FUNCTION_WORDS } from "../src/tokenizer/data.ts";
import { PREFIX_ROLES, SUFFIX_ROLES } from "../src/tokenizer/morphology.ts";
import {
  SPECIAL_TOKENS,
  SEMANTIC_FIELDS as CST_SPEC_SEMANTIC_FIELDS,
  MORPHOLOGICAL_ROLES,
  RELATION_CATEGORIES,
  STRUCTURE_MARKERS,
} from "../src/tokenizer/cst-spec.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(__dirname, "..", "data", "tokenizer");

// Manually authored: structure patterns are regex, kept in a small table so
// both runtimes use identical patterns. Update here if the TS source changes.
const STRUCTURE_PATTERNS = [
  { pattern: "\\?$", token: "STR:question" },
  {
    pattern: "\\b(not|never|no|cannot|can't|won't|don't|doesn't|didn't)\\b",
    token: "STR:negation",
  },
  {
    pattern: "\\b(if|unless|when|whenever|provided that)\\b",
    token: "STR:condition",
  },
  { pattern: "\\b(will|shall|going to|gonna)\\b", token: "STR:future" },
  { pattern: "\\b(was|were|had|did)\\b", token: "STR:past" },
  { pattern: "!$", token: "STR:emphasis" },
];

const files: Record<string, unknown> = {
  "semantic_fields.json": SEMANTIC_FIELDS,
  "relation_map.json": RELATION_MAP,
  "function_words.json": [...FUNCTION_WORDS].sort(),
  "prefix_roles.json": PREFIX_ROLES,
  "suffix_roles.json": SUFFIX_ROLES,
  "structure_patterns.json": STRUCTURE_PATTERNS,
  "cst_spec.json": {
    special_tokens: SPECIAL_TOKENS,
    semantic_fields_list: CST_SPEC_SEMANTIC_FIELDS,
    morphological_roles: MORPHOLOGICAL_ROLES,
    relation_categories: RELATION_CATEGORIES,
    structure_markers: STRUCTURE_MARKERS,
  },
};

const checkMode = process.argv.includes("--check");

if (!existsSync(OUT_DIR)) mkdirSync(OUT_DIR, { recursive: true });

let drift = false;
for (const [name, data] of Object.entries(files)) {
  const path = join(OUT_DIR, name);
  const next = JSON.stringify(data, null, 2) + "\n";
  if (checkMode) {
    if (!existsSync(path)) {
      console.error(`[drift] missing: ${name}`);
      drift = true;
      continue;
    }
    const current = readFileSync(path, "utf8");
    if (current !== next) {
      console.error(`[drift] ${name} differs from TS source`);
      drift = true;
    }
  } else {
    writeFileSync(path, next);
    console.log(`wrote ${name}`);
  }
}

if (checkMode) {
  if (drift) {
    console.error(
      "\nDrift detected. Run: npx tsx scripts/extract_tokenizer_data.ts",
    );
    process.exit(1);
  }
  console.log("OK: data/tokenizer/*.json in sync with TS source");
}
