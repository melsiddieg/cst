import type { Decomposition } from "./types.ts";

/**
 * Stage 5 — Morphological decomposition
 * Extracts root (lemma) + role (affix-based) from a word.
 * Uses compromise.js for lemmatization, own tables for affix→role mapping.
 */

// ── PREFIX → ROLE mapping ──

export const PREFIX_ROLES: Record<string, string> = {
  un: "negate",
  non: "negate",
  dis: "negate",
  re: "repeat",
  pre: "before",
  mis: "wrong",
  over: "excess",
  co: "mutual",
  out: "exceed",
};

// ── SUFFIX → ROLE mapping (longest suffixes first for greedy matching) ──

export const SUFFIX_ROLES: [string, string][] = [
  ["tion", "instance"],
  ["sion", "instance"],
  ["ment", "instance"],
  ["ance", "instance"],
  ["ence", "instance"],
  ["ness", "state"],
  ["able", "possible"],
  ["ible", "possible"],
  ["less", "negate"],
  ["ing", "instance"],
  ["ful", "has"],
  ["ery", "place"],
  ["ory", "place"],
  ["ary", "place"],
  ["age", "instance"],
  ["ist", "agent"],
  ["ian", "agent"],
  ["ity", "state"],
  ["er", "agent"],
  ["or", "agent"],
  ["ee", "patient"],
  ["ly", "manner"],
  ["al", "quality"],
  ["ed", "past"],
  ["s", "plural"],
];

/**
 * Detect prefix role from the original word.
 * Only fires if the word is notably longer than the prefix (≥ prefix+3 chars).
 */
export function detectPrefix(
  word: string,
): { prefix: string; role: string; stem: string } | null {
  for (const [prefix, role] of Object.entries(PREFIX_ROLES)) {
    if (word.startsWith(prefix) && word.length > prefix.length + 2) {
      const stem = word.slice(prefix.length);
      return { prefix, role, stem };
    }
  }
  return null;
}

/**
 * Detect suffix role by comparing the surface word to the lemma.
 * If the word differs from the lemma at the end, extract the suffix.
 */
export function detectSuffix(
  word: string,
  lemma: string,
): { suffix: string; role: string } | null {
  // If word equals lemma, check the word form itself for known noun/adj suffixes
  const target = word !== lemma ? word : word;

  for (const [suffix, role] of SUFFIX_ROLES) {
    if (target.endsWith(suffix) && target.length > suffix.length + 2) {
      // Avoid false positives: the remaining stem must be ≥ 3 chars
      // Also, if word == lemma and suffix is 's' (plural), only match if lemma differs
      if (suffix === "s" && word === lemma) continue;
      if (suffix === "ed" && word === lemma) continue;
      if (suffix === "ly" && word === lemma) continue;
      if (suffix === "ing" && word === lemma) continue;
      return { suffix, role };
    }
  }

  // Plural: word ends in 's' but lemma doesn't
  if (word !== lemma && word === lemma + "s") {
    return { suffix: "s", role: "plural" };
  }
  if (word !== lemma && word === lemma + "es") {
    return { suffix: "es", role: "plural" };
  }

  return null;
}

/**
 * Full decomposition: given a word and its lemma, extract root + role.
 */
export function decompose(word: string, lemma: string): Decomposition {
  // Check prefix first (un-, re-, dis-, etc.)
  const prefix = detectPrefix(word);
  if (prefix) {
    // Use the stem (word minus prefix) as root — e.g. "rewrite" → "write"
    return { root: prefix.stem, role: prefix.role };
  }

  // Check suffix (comparing word to lemma)
  const suffix = detectSuffix(word, lemma);
  if (suffix) {
    // Use the stem (word minus suffix) as root — e.g. "writer" → "writ" → prefer lemma if available
    const stem = word.slice(0, word.length - suffix.suffix.length);
    // Prefer lemma if it looks like a real root, otherwise use stem
    const root = lemma !== word ? lemma : stem;
    return { root, role: suffix.role };
  }

  // No affix detected — root only
  return { root: lemma, role: null };
}
