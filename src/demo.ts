import { CSTTokenizer } from './tokenizer/index.ts'

const tokenizer = new CSTTokenizer()

const examples = [
  'The writer sent a message to the teacher',
  'Students learn in the library',
  'Will you send the document?',
  'She cannot rewrite the unreadable text',
  'The meeting was scheduled for tomorrow',
]

for (const sentence of examples) {
  const output = tokenizer.tokenize(sentence)
  console.log(`\n─── "${sentence}"`)
  console.log(`  Tokens: ${output.tokens.map(t => `[${t.value}]`).join(' ')}`)
  console.log(`  Structured: ${(output.coverage.ratio * 100).toFixed(1)}%`)
}

// Overall coverage
const coverage = tokenizer.getCoverage(examples)
console.log('\n═══ OVERALL COVERAGE ═══')
console.log(`  CMP:        ${(coverage.cmp / coverage.total * 100).toFixed(1)}%`)
console.log(`  ROOT:       ${(coverage.root / coverage.total * 100).toFixed(1)}%`)
console.log(`  STR:        ${(coverage.str / coverage.total * 100).toFixed(1)}%`)
console.log(`  REL:        ${(coverage.rel / coverage.total * 100).toFixed(1)}%`)
console.log(`  LIT:        ${(coverage.lit / coverage.total * 100).toFixed(1)}%`)
console.log(`  STRUCTURED: ${(coverage.ratio * 100).toFixed(1)}%`)
console.log(`  Vocab size: ${tokenizer.getVocabSize()}`)
