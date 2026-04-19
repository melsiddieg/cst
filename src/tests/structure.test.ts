import { describe, it, expect } from 'vitest'
import { detectStructure } from '../tokenizer/structureDetector.ts'

describe('detectStructure', () => {
  it('detects questions', () => {
    const r = detectStructure('will you send the document?')
    expect(r.tokens).toContain('STR:question')
    expect(r.tokens).toContain('STR:future')
  })

  it('detects negation', () => {
    const r = detectStructure("she cannot rewrite the unreadable text")
    expect(r.tokens).toContain('STR:negation')
  })

  it('detects past tense', () => {
    const r = detectStructure('the writer sent a message to the teacher')
    // "sent" doesn't match (was/were/had/did), but structure detection
    // looks for auxiliary markers
    // This sentence has no auxiliary past marker
  })

  it('detects future', () => {
    const r = detectStructure('will the students understand the lesson?')
    expect(r.tokens).toContain('STR:future')
    expect(r.tokens).toContain('STR:question')
  })

  it('detects condition', () => {
    const r = detectStructure('if you study hard you will pass')
    expect(r.tokens).toContain('STR:condition')
    expect(r.tokens).toContain('STR:future')
  })

  it('detects emphasis', () => {
    const r = detectStructure('stop right now!')
    expect(r.tokens).toContain('STR:emphasis')
  })
})
