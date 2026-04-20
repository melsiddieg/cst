/**
 * CST Edge Demo — Arabic LM in the browser
 *
 * Loads model_int8.onnx + vocab.json + word2tok.json + tok2word.json
 * Input: normal Arabic text → word2tok lookup → CST token IDs
 * Output: CST tokens → tok2word reverse lookup → clean Arabic text
 */

import * as ort from "onnxruntime-web";

ort.env.wasm.numThreads = 1;

// ─── Vocab & Lookups ────────────────────────────────────────────────────────

interface Vocab {
  idToToken: Map<number, string>;
  tokenToId: Map<string, number>;
  PAD: number;
  UNK: number;
  BOS: number;
  EOS: number;
}

let vocab: Vocab;
let word2tok: Record<string, string> = {};
let tok2word: Record<string, string> = {};

function buildVocab(raw: Record<string, string>): Vocab {
  const idToToken = new Map<number, string>();
  const tokenToId = new Map<string, number>();
  let PAD = 0,
    UNK = 1,
    BOS = 3,
    EOS = 4;

  for (const [idStr, token] of Object.entries(raw)) {
    const id = Number(idStr);
    idToToken.set(id, token);
    tokenToId.set(token, id);
    if (token === "<PAD>") PAD = id;
    else if (token === "<UNK>") UNK = id;
    else if (token === "[BOS]") BOS = id;
    else if (token === "[EOS]") EOS = id;
  }

  return { idToToken, tokenToId, PAD, UNK, BOS, EOS };
}

// ─── Tokenizer ──────────────────────────────────────────────────────────────

function encodeText(text: string): number[] {
  const ids: number[] = [vocab.BOS];
  const words = text.trim().split(/\s+/).filter(Boolean);
  for (const word of words) {
    const clean = word.replace(/[،؛.؟!:]+$/, "");
    const cstToken = word2tok[clean] ?? word2tok[word];
    if (cstToken) {
      const id = vocab.tokenToId.get(cstToken);
      ids.push(id ?? vocab.UNK);
    } else {
      const surfId =
        vocab.tokenToId.get(`SURF:${clean}`) ??
        vocab.tokenToId.get(`SURF:${word}`);
      ids.push(surfId ?? vocab.UNK);
    }
  }
  return ids;
}

function tokenToArabic(token: string): string {
  const word = tok2word[token];
  if (word) return word;
  if (token.startsWith("SURF:")) return token.slice(5);
  return "";
}

function tokenPrefix(token: string): string {
  const colonIdx = token.indexOf(":");
  return colonIdx > 0 ? token.slice(0, colonIdx) : token;
}

// ─── Sampling ───────────────────────────────────────────────────────────────

function sampleTopK(
  logits: Float32Array,
  temperature: number,
  topK = 40,
): number {
  if (temperature <= 0) {
    let maxIdx = 0,
      maxVal = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxVal) {
        maxVal = logits[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  const scaled = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / temperature;

  const indices = Array.from({ length: scaled.length }, (_, i) => i);
  indices.sort((a, b) => scaled[b] - scaled[a]);
  const topIndices = indices.slice(0, topK);

  let maxLogit = -Infinity;
  for (const i of topIndices) if (scaled[i] > maxLogit) maxLogit = scaled[i];
  let sum = 0;
  const probs: number[] = [];
  for (const i of topIndices) {
    const p = Math.exp(scaled[i] - maxLogit);
    probs.push(p);
    sum += p;
  }

  let r = Math.random() * sum;
  for (let j = 0; j < topIndices.length; j++) {
    r -= probs[j];
    if (r <= 0) return topIndices[j];
  }
  return topIndices[0];
}

// ─── ONNX Inference ─────────────────────────────────────────────────────────

let session: ort.InferenceSession | null = null;

async function loadModel(): Promise<void> {
  const statusEl = document.getElementById("model-status")!;
  statusEl.className = "loading";
  statusEl.textContent = "Loading model...";

  try {
    const [vocabResp, w2tResp, t2wResp] = await Promise.all([
      fetch("./model/vocab.json"),
      fetch("./model/word2tok.json"),
      fetch("./model/tok2word.json"),
    ]);

    vocab = buildVocab(await vocabResp.json());
    word2tok = await w2tResp.json();
    tok2word = await t2wResp.json();

    session = await ort.InferenceSession.create("./model/model_int8.onnx", {
      executionProviders: ["wasm"],
    });

    const wordCount = Object.keys(word2tok).length;
    statusEl.className = "loaded";
    statusEl.textContent = `Model loaded · ${vocab.idToToken.size.toLocaleString()} tokens · ${wordCount.toLocaleString()} words`;
    document.getElementById("send-btn")!.removeAttribute("disabled");
  } catch (e) {
    console.error("Load failed:", e);
    statusEl.className = "error";
    statusEl.textContent = "Failed to load model";
  }
}

async function generate(
  seedIds: number[],
  maxNewTokens: number,
  temperature: number,
): Promise<{ ids: number[]; timeMs: number }> {
  if (!session) throw new Error("Model not loaded");

  const t0 = performance.now();
  const ids = [...seedIds];

  for (let i = 0; i < maxNewTokens; i++) {
    const input = ids.length > 128 ? ids.slice(ids.length - 128) : ids;
    const tensor = new ort.Tensor(
      "int64",
      BigInt64Array.from(input.map(BigInt)),
      [1, input.length],
    );

    const result = await session.run({ input_ids: tensor });
    const logits = result.logits.data as Float32Array;
    const vocabSize = result.logits.dims[2];
    const offset = (input.length - 1) * vocabSize;
    const lastLogits = logits.slice(offset, offset + vocabSize);

    lastLogits[vocab.PAD] = -Infinity;
    lastLogits[vocab.UNK] = -Infinity;

    const nextId = sampleTopK(lastLogits, temperature);
    ids.push(nextId);
    if (nextId === vocab.EOS) break;
  }

  return { ids, timeMs: performance.now() - t0 };
}

// ─── UI ─────────────────────────────────────────────────────────────────────

const SUGGESTIONS = [
  "كان الرجل يعمل في",
  "ذهبت إلى المدرسة",
  "في عام ألف وتسعمائة",
  "المدينة الكبيرة التي",
  "الماء مادة شفافة",
  "قال الرئيس إن",
];

function initUI() {
  const input = document.getElementById("input") as HTMLInputElement;
  const sendBtn = document.getElementById("send-btn") as HTMLButtonElement;
  const sugContainer = document.getElementById("suggestions")!;
  const maxTokensSlider = document.getElementById(
    "max-tokens",
  ) as HTMLInputElement;
  const maxTokensVal = document.getElementById("max-tokens-val")!;
  const anatomyToggle = document.getElementById(
    "anatomy-toggle",
  ) as HTMLButtonElement;
  const anatomyDiv = document.getElementById("anatomy")!;

  for (const text of SUGGESTIONS) {
    const btn = document.createElement("button");
    btn.className = "sug-btn";
    btn.textContent = text;
    btn.onclick = () => {
      input.value = text;
      handleGenerate();
    };
    sugContainer.appendChild(btn);
  }

  maxTokensSlider.oninput = () => {
    maxTokensVal.textContent = maxTokensSlider.value;
  };

  anatomyToggle.onclick = () => {
    const open = anatomyDiv.classList.toggle("visible");
    anatomyToggle.textContent = open ? "Token Anatomy ▾" : "Token Anatomy ▸";
  };

  sendBtn.onclick = handleGenerate;
  input.onkeydown = (e) => {
    if (e.key === "Enter") handleGenerate();
  };
}

async function handleGenerate() {
  const input = document.getElementById("input") as HTMLInputElement;
  const sendBtn = document.getElementById("send-btn") as HTMLButtonElement;
  const text = input.value.trim();
  if (!text || !session) return;

  sendBtn.disabled = true;
  sendBtn.textContent = "...";

  const maxTokens = Number(
    (document.getElementById("max-tokens") as HTMLInputElement).value,
  );
  const temperature = Number(
    (document.getElementById("temperature") as HTMLSelectElement).value,
  );

  try {
    const seedIds = encodeText(text);
    const seedLen = seedIds.length;

    const encodingEl = document.getElementById("encoding-info")!;
    const unkCount = seedIds.filter((id) => id === vocab.UNK).length;
    const wordCount = seedLen - 1;
    encodingEl.textContent =
      unkCount > 0
        ? `Encoded ${wordCount} words (${unkCount} unknown)`
        : `Encoded ${wordCount} words`;
    encodingEl.className =
      unkCount > 0 ? "encoding-info warn" : "encoding-info ok";

    const { ids, timeMs } = await generate(seedIds, maxTokens, temperature);
    renderOutput(ids, seedLen, timeMs);
  } catch (e) {
    console.error("Generation failed:", e);
  }

  sendBtn.disabled = false;
  sendBtn.textContent = "أكمل";
}

function renderOutput(allIds: number[], seedLen: number, timeMs: number) {
  const outputSection = document.getElementById("output-section")!;
  const arabicOutput = document.getElementById("arabic-output")!;
  const tokenList = document.getElementById("token-list")!;
  const statsEl = document.getElementById("stats")!;

  outputSection.classList.add("visible");

  const seedWords: string[] = [];
  const genWords: string[] = [];

  for (let i = 0; i < allIds.length; i++) {
    const token = vocab.idToToken.get(allIds[i]) ?? `<${allIds[i]}>`;
    if (token === "[BOS]" || token === "[EOS]" || token === "<PAD>") continue;
    const arabic = tokenToArabic(token);
    if (!arabic) continue;
    if (i < seedLen) seedWords.push(arabic);
    else genWords.push(arabic);
  }

  arabicOutput.innerHTML =
    seedWords.join(" ") +
    (genWords.length
      ? ' <span class="generated">' + genWords.join(" ") + "</span>"
      : "");

  // Token anatomy
  tokenList.innerHTML = "";
  for (let i = 0; i < allIds.length; i++) {
    const token = vocab.idToToken.get(allIds[i]) ?? `<${allIds[i]}>`;
    if (token === "[BOS]" || token === "[EOS]" || token === "<PAD>") continue;
    const arabic = tokenToArabic(token);
    const prefix = tokenPrefix(token);

    const chip = document.createElement("div");
    chip.className = `token-chip ${i < seedLen ? "seed" : "gen"}`;
    chip.innerHTML =
      `<span class="arabic">${arabic || "—"}</span>` +
      `<span class="tag ${prefix.toLowerCase()}">${token}</span>`;
    tokenList.appendChild(chip);
  }

  const genCount = allIds.length - seedLen;
  const tokPerSec = genCount > 0 ? genCount / (timeMs / 1000) : 0;
  statsEl.innerHTML =
    `Seed: <span>${seedLen - 1}</span> tokens · ` +
    `Generated: <span>${genCount}</span> tokens · ` +
    `Time: <span>${timeMs.toFixed(0)}ms</span> · ` +
    `Speed: <span>${tokPerSec.toFixed(1)}</span> tok/s`;
}

// ─── Init ───────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  initUI();
  loadModel();
});
