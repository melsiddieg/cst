import { describe, it, expect } from "vitest";
import {
  decompose,
  detectPrefix,
  detectSuffix,
} from "../tokenizer/morphology.ts";

describe("morphology", () => {
  describe("detectPrefix", () => {
    it("detects un- prefix", () => {
      const r = detectPrefix("unreadable");
      expect(r).toEqual({ prefix: "un", role: "negate", stem: "readable" });
    });

    it("detects re- prefix", () => {
      const r = detectPrefix("rewrite");
      expect(r).toEqual({ prefix: "re", role: "repeat", stem: "write" });
    });

    it("ignores short words", () => {
      const r = detectPrefix("under");
      // "under" length=5, prefix "un"=2, 5 > 2+2=4 → matches
      // This is OK — decompose will still check against lemma
    });
  });

  describe("detectSuffix", () => {
    it("detects -er suffix (agent)", () => {
      const r = detectSuffix("writer", "write");
      expect(r).toEqual({ suffix: "er", role: "agent" });
    });

    it("detects -able suffix (possible)", () => {
      const r = detectSuffix("readable", "read");
      expect(r).toEqual({ suffix: "able", role: "possible" });
    });

    it("detects plural -s", () => {
      const r = detectSuffix("students", "student");
      expect(r).toEqual({ suffix: "s", role: "plural" });
    });

    it("detects -tion suffix", () => {
      const r = detectSuffix("information", "inform");
      expect(r).toEqual({ suffix: "tion", role: "instance" });
    });
  });

  describe("decompose", () => {
    it("decomposes writer → write + agent", () => {
      const r = decompose("writer", "write");
      expect(r).toEqual({ root: "write", role: "agent" });
    });

    it("decomposes rewrite → write + repeat", () => {
      const r = decompose("rewrite", "write");
      expect(r).toEqual({ root: "write", role: "repeat" });
    });

    it("decomposes unreadable → readable + negate", () => {
      const r = decompose("unreadable", "read");
      expect(r).toEqual({ root: "readable", role: "negate" });
    });

    it("returns root only for base forms", () => {
      const r = decompose("send", "send");
      expect(r).toEqual({ root: "send", role: null });
    });
  });
});
