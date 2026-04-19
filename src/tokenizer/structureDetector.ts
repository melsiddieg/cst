import type { StructureResult } from './types.ts'

/**
 * Stage 2 — Sentence structure detection
 * Scans full sentence BEFORE word-by-word processing.
 * Emits STR tokens that apply to the whole sentence.
 */

interface StructurePattern {
  pattern: RegExp
  token: string
}

const STRUCTURE_PATTERNS: StructurePattern[] = [
  { pattern: /\?$/, token: 'STR:question' },
  { pattern: /\b(not|never|no|cannot|can't|won't|don't|doesn't|didn't)\b/, token: 'STR:negation' },
  { pattern: /\b(if|unless|when|whenever|provided that)\b/, token: 'STR:condition' },
  { pattern: /\b(will|shall|going to|gonna)\b/, token: 'STR:future' },
  { pattern: /\b(was|were|had|did)\b/, token: 'STR:past' },
  { pattern: /!$/, token: 'STR:emphasis' },
]

export function detectStructure(normalizedText: string): StructureResult {
  const tokens: string[] = []
  for (const { pattern, token } of STRUCTURE_PATTERNS) {
    if (pattern.test(normalizedText)) {
      if (!tokens.includes(token)) {
        tokens.push(token)
      }
    }
  }
  return { tokens }
}
