import type { Token, TokenType, Decomposition } from "./types.ts";
import { RELATION_MAP, FUNCTION_WORDS } from "./data.ts";
import { getSemanticField } from "./semanticFields.ts";
import type { EntityMap } from "./ner.ts";

/**
 * Stage 7 — Token emission
 * Combines all analysis stages into final token sequence.
 * Each word becomes exactly one token.
 */

/**
 * Try to resolve a semantic field from a root/lemma.
 * Handles English silent-e dropping: "writ" → try "write", etc.
 * Also handles nested affixes: "readable" → strip "-able" → "read"
 */
function resolveField(root: string | null, lemma: string): string | null {
  if (!root && !lemma) return null;

  const candidates: string[] = [];
  if (root) candidates.push(root, root + "e");
  candidates.push(lemma, lemma + "e");

  // Try each candidate directly
  for (const c of candidates) {
    const f = getSemanticField(c);
    if (f) return f;
  }

  // Try stripping known suffixes from root (nested decomposition)
  // e.g. "readable" → strip "able" → "read" → field "know"
  if (root) {
    const SUFFIXES = [
      "able",
      "ible",
      "tion",
      "sion",
      "ment",
      "ance",
      "ence",
      "ness",
      "ful",
      "less",
      "ery",
      "ory",
      "ary",
      "age",
      "ing",
      "ist",
      "ian",
      "ity",
      "er",
      "or",
      "ee",
      "ly",
      "al",
      "ed",
    ];
    for (const sfx of SUFFIXES) {
      if (root.endsWith(sfx) && root.length > sfx.length + 2) {
        const stem = root.slice(0, root.length - sfx.length);
        const f = getSemanticField(stem) ?? getSemanticField(stem + "e");
        if (f) return f;
      }
    }
  }

  return null;
}

export function emitToken(
  word: string,
  lemma: string,
  isEntityWord: boolean,
  decomposition: Decomposition,
): Token {
  // Named entity → LIT
  if (isEntityWord) {
    return makeToken("LIT", `LIT:${word}`, word, 1.0);
  }

  // Numeric literal → ROOT:size
  if (/^\d+$/.test(word)) {
    return makeToken("ROOT", "ROOT:size", word, 0.9, "size");
  }

  // Relation word → REL
  const rel = RELATION_MAP[word];
  if (rel) {
    return makeToken("REL", rel, word, 1.0);
  }

  // Function word → LIT
  if (FUNCTION_WORDS.has(word)) {
    return makeToken("LIT", `LIT:${word}`, word, 0.9);
  }

  const { root, role } = decomposition;
  const field = resolveField(root, lemma);

  // Best case: composed token (field + role)
  if (field && role) {
    return makeToken("CMP", `CMP:${field}:${role}`, word, 0.9, field, role);
  }

  // Good case: root only
  if (field) {
    return makeToken("ROOT", `ROOT:${field}`, word, 0.8, field);
  }

  // Fallback: literal
  return makeToken("LIT", `LIT:${word}`, word, 0.5);
}

function makeToken(
  type: TokenType,
  value: string,
  surface: string,
  confidence: number,
  field?: string,
  role?: string,
): Token {
  return {
    type,
    value,
    field,
    role,
    surface,
    id: 0, // assigned later by vocabulary
    confidence,
  };
}
