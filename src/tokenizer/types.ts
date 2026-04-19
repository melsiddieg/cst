// ── Token Types ──────────────────────────────────────────

export type TokenType =
  | "ROOT"
  | "ROLE"
  | "CMP"
  | "REL"
  | "STR"
  | "LIT"
  | "SPECIAL";

export interface Token {
  type: TokenType;
  /** Full token string, e.g. "CMP:write:agent" */
  value: string;
  /** Semantic field, e.g. "write" */
  field?: string;
  /** Role, e.g. "agent" */
  role?: string;
  /** Original surface word this came from */
  surface: string;
  /** Integer ID for model consumption */
  id: number;
  /** Confidence 0.0–1.0 */
  confidence: number;
}

export interface TokenizerOutput {
  tokens: Token[];
  ids: number[];
  coverage: CoverageStats;
}

export interface CoverageStats {
  total: number;
  cmp: number;
  root: number;
  str: number;
  rel: number;
  lit: number;
  unk: number;
  structured: number; // cmp + root + str + rel
  ratio: number; // structured / total
}

export interface VocabEntry {
  token: string;
  id: number;
  type: TokenType;
  frequency: number;
  gloss?: string;
}

// ── Decomposition result from morphology stage ──

export interface Decomposition {
  root: string | null;
  role: string | null;
}

// ── Structure detection result ──

export interface StructureResult {
  tokens: string[]; // e.g. ["STR:past", "STR:negation"]
}
