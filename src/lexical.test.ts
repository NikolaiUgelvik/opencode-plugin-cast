import { describe, expect, test } from "bun:test"
import { bm25Search, buildLexicalIndex, reciprocalRankFusion, tokenizeCodeText } from "./lexical.js"
import type { ChunkRecord, SymbolRecord } from "./types.js"

const baseChunk: Omit<ChunkRecord, "id" | "filePath" | "text" | "range"> = {
  language: "typescript",
  kind: "function",
  nonWhitespaceChars: 1,
  nodeTypes: ["function_declaration"],
  symbolIds: [],
  childChunkIds: [],
}

describe("lexical retrieval", () => {
  test("tokenizes code identifiers, env vars, and paths", () => {
    expect(
      tokenizeCodeText("src/retriever.test.ts semantic_get_chunk semanticGetChunk OPENCODE_CAST_CACHE_DIR"),
    ).toEqual(
      expect.arrayContaining([
        "src",
        "retriever",
        "test",
        "ts",
        "semantic_get_chunk",
        "semantic",
        "get",
        "chunk",
        "semanticgetchunk",
        "opencode_cast_cache_dir",
        "opencode",
        "cast",
        "cache",
        "dir",
      ]),
    )
  })

  test("preserves raw kebab-case and path-like tokens", () => {
    expect(tokenizeCodeText("foo-bar src/retriever.test.ts")).toEqual(
      expect.arrayContaining([
        "foo-bar",
        "foo",
        "bar",
        "src/retriever.test.ts",
        "retriever.test.ts",
        "retriever",
        "test",
        "ts",
      ]),
    )
  })

  test("builds lexical stats from chunks, paths, symbols, kind, and node types", () => {
    const chunks = {
      a: chunk("a", "src/retriever.ts", "export function retrieveSemanticChunk() {}", ["s1"]),
      b: chunk("b", "src/options.ts", "const cacheDir = process.env.OPENCODE_CAST_CACHE_DIR", []),
    }
    const symbols: Record<string, SymbolRecord> = {
      s1: symbol("s1", "retrieveSemanticChunk", "function", "src/retriever.ts"),
    }

    const indexed = buildLexicalIndex(chunks, symbols)

    expect(indexed.chunks).not.toBe(chunks)
    expect(indexed.chunks.a).not.toBe(chunks.a)
    expect(indexed.lexical.documentCount).toBe(2)
    expect(indexed.lexical.averageDocumentLength).toBeGreaterThan(0)
    expect(indexed.lexical.documentFrequencies.retrieve).toBe(1)
    expect(indexed.lexical.documentFrequencies.retrievesemanticchunk).toBe(1)
    expect(indexed.lexical.documentFrequencies.src).toBe(2)
    expect(indexed.lexical.documentFrequencies.function).toBe(2)
    expect(indexed.lexical.documentFrequencies.function_declaration).toBe(2)
    expect(indexed.chunks.a.lexical?.termFrequencies.retrieve).toBeGreaterThan(0)
    expect(indexed.chunks.a.lexical?.termFrequencies.retriever).toBeGreaterThan(0)
    expect(indexed.chunks.a.lexical?.termFrequencies.function_declaration).toBe(1)
    expect(indexed.chunks.b.lexical?.termFrequencies.opencode_cast_cache_dir).toBe(1)
  })

  test("counts prototype property names as numeric lexical terms", () => {
    const { lexical, chunks } = buildLexicalIndex(
      {
        a: chunk("a", "src/plugin.ts", "class IndexUnavailableError { constructor(message: string) {} }", []),
      },
      {},
    )

    expect(Object.getOwnPropertyDescriptor(chunks.a.lexical?.termFrequencies, "constructor")?.value).toBe(2)
    expect(Object.getOwnPropertyDescriptor(lexical.documentFrequencies, "constructor")?.value).toBe(1)
  })

  test("bm25 search prefers exact lexical matches", () => {
    const { lexical, chunks } = buildLexicalIndex(
      {
        exact: chunk("exact", "src/cache.ts", "const OPENCODE_CAST_CACHE_DIR = readCacheDir()", []),
        partial: chunk("partial", "src/cache.ts", "function readCacheDir() { return cacheDir }", []),
        other: chunk("other", "src/retriever.ts", "export function retrieve() {}", []),
      },
      {},
    )

    const results = bm25Search("OPENCODE_CAST_CACHE_DIR", Object.values(chunks), lexical, 2)

    expect(results.map((result) => result.id)).toEqual(["exact", "partial"])
    expect(results[0].score).toBeGreaterThan(results[1].score)
    expect(results.every((result) => result.score > 0)).toBe(true)
  })

  test("reciprocal rank fusion merges rankings without score normalization", () => {
    const results = reciprocalRankFusion({
      lists: [
        { weight: 2, results: [{ id: "a", score: 100 }] },
        {
          weight: 1,
          results: [
            { id: "b", score: 10_000 },
            { id: "a", score: 1 },
          ],
        },
      ],
      rrfK: 60,
      topK: 2,
    })

    expect(results).toEqual([
      { id: "a", score: 2 / 61 + 1 / 62 },
      { id: "b", score: 1 / 61 },
    ])
  })
})

function chunk(id: string, filePath: string, text: string, symbolIds: string[]): ChunkRecord {
  return {
    ...baseChunk,
    id,
    filePath,
    text,
    range: { byteStart: 0, byteEnd: text.length, lineStart: 1, lineEnd: 1 },
    symbolIds,
  }
}

function symbol(id: string, name: string, kind: SymbolRecord["kind"], filePath: string): SymbolRecord {
  return {
    id,
    name,
    kind,
    filePath,
    range: { byteStart: 0, byteEnd: name.length, lineStart: 1, lineEnd: 1 },
    childSymbolIds: [],
  }
}
