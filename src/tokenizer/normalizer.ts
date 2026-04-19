/**
 * Stage 1 — Text normalization
 * Lowercase, fix encoding, normalize punctuation, trim whitespace.
 */
export function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[\u2018\u2019\u201A\u2039\u203A]/g, "'")
    .replace(/[\u201C\u201D\u201E\u00AB\u00BB]/g, '"')
    .replace(/\s+/g, " ")
    .trim();
}
