import type { VocabEntry, TokenType, Token } from './types.ts'
import { readFileSync, writeFileSync } from 'node:fs'

/**
 * Token registry — maps token strings ↔ integer IDs.
 * Builds vocabulary from corpus, saves/loads to JSON.
 */

const SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', '[SEP]']

export class Vocabulary {
  private tokenToId = new Map<string, number>()
  private idToEntry = new Map<number, VocabEntry>()
  private nextId = 0

  constructor() {
    // Reserve special tokens
    for (const tok of SPECIAL_TOKENS) {
      this.add(tok, 'SPECIAL')
    }
  }

  add(token: string, type: TokenType, gloss?: string): number {
    const existing = this.tokenToId.get(token)
    if (existing !== undefined) {
      // Increment frequency
      const entry = this.idToEntry.get(existing)!
      entry.frequency++
      return existing
    }

    const id = this.nextId++
    this.tokenToId.set(token, id)
    this.idToEntry.set(id, { token, id, type, frequency: 1, gloss })
    return id
  }

  getId(token: string): number {
    return this.tokenToId.get(token) ?? this.tokenToId.get('[UNK]')!
  }

  getToken(id: number): VocabEntry | undefined {
    return this.idToEntry.get(id)
  }

  has(token: string): boolean {
    return this.tokenToId.has(token)
  }

  get size(): number {
    return this.nextId
  }

  /** Assign IDs to all tokens in the array (mutates .id field) */
  assignIds(tokens: Token[]): number[] {
    return tokens.map(t => {
      if (!this.has(t.value)) {
        this.add(t.value, t.type, t.surface)
      }
      const id = this.getId(t.value)
      t.id = id
      return id
    })
  }

  save(path: string): void {
    const entries = Array.from(this.idToEntry.values())
    writeFileSync(path, JSON.stringify(entries, null, 2))
  }

  load(path: string): void {
    const data = JSON.parse(readFileSync(path, 'utf-8')) as VocabEntry[]
    this.tokenToId.clear()
    this.idToEntry.clear()
    this.nextId = 0
    for (const entry of data) {
      this.tokenToId.set(entry.token, entry.id)
      this.idToEntry.set(entry.id, entry)
      if (entry.id >= this.nextId) {
        this.nextId = entry.id + 1
      }
    }
  }
}
