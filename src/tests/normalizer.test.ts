import { describe, it, expect } from 'vitest'
import { normalize } from '../tokenizer/normalizer.ts'

describe('normalize', () => {
  it('lowercases text', () => {
    expect(normalize('Hello World')).toBe('hello world')
  })

  it('normalizes smart quotes', () => {
    expect(normalize('\u2018hello\u2019')).toBe("'hello'")
    expect(normalize('\u201CHello\u201D')).toBe('"hello"')
  })

  it('collapses whitespace', () => {
    expect(normalize('  hello   world  ')).toBe('hello world')
  })

  it('handles combined normalization', () => {
    expect(normalize('  The \u201CWriter\u201D  Sent  a  MESSAGE  '))
      .toBe('the "writer" sent a message')
  })
})
