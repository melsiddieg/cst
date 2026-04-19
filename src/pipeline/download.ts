/**
 * Download English Wikipedia sentences for coverage testing and training.
 * Uses HuggingFace datasets API to stream plain text.
 *
 * For POC: we fetch a small subset (~10K sentences) for coverage testing,
 * and optionally a larger set for training.
 */

const WIKI_DATASET_URL =
  "https://datasets-server.huggingface.co/rows?dataset=agentlans%2Fhigh-quality-english-sentences&config=default&split=train";

/**
 * Fetch N rows of Wikipedia text from HuggingFace datasets API.
 * Each row contains a `text` field with full article text.
 */
async function fetchWikiRows(
  offset: number,
  length: number,
): Promise<string[]> {
  const url = `${WIKI_DATASET_URL}&offset=${offset}&length=${length}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  const data = (await res.json()) as {
    rows: { row: { text: string } }[];
  };
  return data.rows.map((r) => r.row.text);
}

/**
 * Download target number of clean sentences.
 * This dataset has one sentence per row — no splitting needed.
 */
export async function downloadSentences(
  targetCount: number,
  onProgress?: (count: number) => void,
): Promise<string[]> {
  const sentences: string[] = [];
  const batchSize = 100; // HuggingFace API limit per request
  let offset = 0;

  while (sentences.length < targetCount) {
    try {
      const rows = await fetchWikiRows(offset, batchSize);
      if (rows.length === 0) break;

      for (const text of rows) {
        const trimmed = text.trim();
        if (trimmed.length > 10) {
          sentences.push(trimmed);
        }
        if (sentences.length >= targetCount) break;
      }

      offset += batchSize;
      onProgress?.(sentences.length);
    } catch (err) {
      console.error(`Error at offset ${offset}:`, err);
      offset += batchSize;
      if (offset > targetCount + 5000) break;
    }
  }

  return sentences.slice(0, targetCount);
}

// ── CLI entry point ──

if (process.argv[1]?.endsWith("download.ts")) {
  const count = parseInt(process.argv[2] ?? "10000", 10);
  const outPath = process.argv[3] ?? "data/sentences.json";

  console.log(`Downloading ${count} Wikipedia sentences...`);

  const { mkdirSync, writeFileSync } = await import("node:fs");
  const { dirname } = await import("node:path");

  mkdirSync(dirname(outPath), { recursive: true });

  const sentences = await downloadSentences(count, (n) => {
    if (n % 1000 === 0) console.log(`  ${n} sentences collected...`);
  });

  writeFileSync(outPath, JSON.stringify(sentences, null, 2));
  console.log(`Done. Saved ${sentences.length} sentences to ${outPath}`);
}
