import { describe, expect, test } from "bun:test"
import { buildLexicalIndex } from "./lexical.js"
import { retrieve } from "./retriever.js"
import { createEmptyIndex } from "./store.js"
import type { CastIndex, HybridRetrievalOptions, RerankOptions } from "./types.js"

const hybridOptions = (overrides: Partial<HybridRetrievalOptions> = {}): HybridRetrievalOptions => ({
  enabled: true,
  mode: "parallel",
  rrfK: 60,
  vectorCandidateMultiplier: 2,
  bm25CandidateMultiplier: 2,
  vectorWeight: 1,
  bm25Weight: 4,
  ...overrides,
})

const rerankOptions = (overrides: Partial<RerankOptions> = {}): RerankOptions => ({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: "key",
  model: "cohere/rerank-4-fast",
  candidateMultiplier: 4,
  ...overrides,
})

function addLexicalStats(index: CastIndex) {
  const lexical = buildLexicalIndex(index.chunks, index.symbols)
  index.lexical = lexical.lexical
  index.chunks = lexical.chunks
}

describe("retrieve", () => {
  test("uses store vector candidates without hydrated embeddings", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.c1 = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
      text: "function a() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
    }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 1, includeParents: true, maxContextChars: 100 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => "function a() {}",
      indexStore: {
        searchVectorCandidates: async () => [{ id: "c1", score: 0.75 }],
      },
    } as Parameters<typeof retrieve>[0] & {
      indexStore: {
        searchVectorCandidates(
          queryEmbedding: number[],
          topK: number,
          paths?: string[],
        ): Promise<Array<{ id: string; score: number }>>
      }
    })

    expect(output.results[0].filePath).toBe("a.ts")
    expect(output.results[0].score).toBe(0.75)
  })

  test("returns normal embedding results without HyDE above threshold", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.c1 = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
      text: "function a() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [1, 0],
    }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 1, includeParents: true, maxContextChars: 100 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: true, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => {
        throw new Error("not called")
      },
      readSource: async () => "function a() {}",
    })

    expect(output.status.hydeUsed).toBe(false)
    expect(output.results[0].filePath).toBe("a.ts")
  })

  test("returns labeled topology in search results", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    const functionText = "function parseOptions() {\n  return {}\n}"
    const source = `${functionText}\n\nconst value = 1\n`
    index.metadata.status = "ready"
    index.symbols.s1 = {
      id: "s1",
      name: "parseOptions",
      kind: "function",
      filePath: "src/options.ts",
      range: { byteStart: 0, byteEnd: functionText.length, lineStart: 1, lineEnd: 3 },
      childSymbolIds: [],
    }
    index.symbols.sChild = {
      id: "sChild",
      name: "readOptions",
      kind: "function",
      filePath: "src/options.ts",
      range: { byteStart: 26, byteEnd: 35, lineStart: 2, lineEnd: 2 },
      childSymbolIds: [],
    }
    index.symbols.sPrevious = {
      id: "sPrevious",
      name: "loadDefaults",
      kind: "function",
      filePath: "src/options.ts",
      range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
      childSymbolIds: [],
    }
    index.symbols.sNext = {
      id: "sNext",
      name: "formatOptions",
      kind: "function",
      filePath: "src/options.ts",
      range: { byteStart: 41, byteEnd: 55, lineStart: 5, lineEnd: 5 },
      childSymbolIds: [],
    }
    index.chunks.parent = {
      id: "parent",
      filePath: "src/options.ts",
      language: "typescript",
      kind: "file",
      range: { byteStart: 0, byteEnd: source.trimEnd().length, lineStart: 1, lineEnd: 6 },
      text: source.trimEnd(),
      nonWhitespaceChars: 44,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: ["c1"],
    }
    index.chunks.c1 = {
      id: "c1",
      filePath: "src/options.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: functionText.length, lineStart: 1, lineEnd: 3 },
      text: functionText,
      nonWhitespaceChars: 33,
      nodeTypes: [],
      symbolIds: ["s1"],
      parentChunkId: "parent",
      childChunkIds: ["child"],
      previousSiblingChunkId: "previous",
      nextSiblingChunkId: "next",
      embedding: [1, 0],
    }
    index.chunks.child = {
      id: "child",
      filePath: "src/options.ts",
      language: "typescript",
      kind: "block",
      range: { byteStart: 26, byteEnd: 35, lineStart: 2, lineEnd: 2 },
      text: "return {}",
      nonWhitespaceChars: 8,
      nodeTypes: [],
      symbolIds: ["sChild"],
      parentChunkId: "c1",
      childChunkIds: [],
    }
    index.chunks.previous = {
      id: "previous",
      filePath: "src/options.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
      text: "defaults()",
      nonWhitespaceChars: 10,
      nodeTypes: [],
      symbolIds: ["sPrevious"],
      childChunkIds: [],
    }
    index.chunks.next = {
      id: "next",
      filePath: "src/options.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 41, byteEnd: 55, lineStart: 5, lineEnd: 5 },
      text: "formatOptions()",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: ["sNext"],
      childChunkIds: [],
    }

    const output = await retrieve({
      index,
      input: { query: "parseOptions", topK: 1, includeParents: true, maxContextChars: 100 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => source,
    })

    expect(output.results[0].topology).toEqual({
      chunk: { id: "c1", label: "function parseOptions", range: "src/options.ts:1-3" },
      parent: { id: "parent", label: "file src/options.ts", range: "src/options.ts:1-6" },
      children: [{ id: "child", label: "function readOptions", range: "src/options.ts:2" }],
      previousSibling: { id: "previous", label: "function loadDefaults", range: "src/options.ts:1" },
      nextSibling: { id: "next", label: "function formatOptions", range: "src/options.ts:5" },
      symbols: ["function parseOptions"],
    })
  })

  test("filters retrieval candidates by exact paths and directory prefixes", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks["src/a.ts"] = {
      id: "src/a.ts",
      filePath: "src/a.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
      text: "function a() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [1, 0],
    }
    index.chunks["src/nested/b.ts"] = {
      id: "src/nested/b.ts",
      filePath: "src/nested/b.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
      text: "function b() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0.9, 0],
    }
    index.chunks["src/c.ts"] = {
      id: "src/c.ts",
      filePath: "src/c.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
      text: "function c() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0.7, 0.7],
    }
    index.chunks["test/c.ts"] = {
      id: "test/c.ts",
      filePath: "test/c.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
      text: "function c() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0.8, 0],
    }

    const exact = await retrieve({
      index,
      input: { query: "a", topK: 3, includeParents: true, maxContextChars: 100, paths: ["test/c.ts"] },
      options: { topK: 3, maxContextChars: 100, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async (filePath) => index.chunks[filePath].text,
    })
    const directory = await retrieve({
      index,
      input: { query: "a", topK: 3, includeParents: true, maxContextChars: 100, paths: ["src/"] },
      options: { topK: 3, maxContextChars: 100, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async (filePath) => index.chunks[filePath].text,
    })
    const glob = await retrieve({
      index,
      input: { query: "a", topK: 3, includeParents: true, maxContextChars: 100, paths: ["src/**/*.ts"] },
      options: { topK: 3, maxContextChars: 100, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async (filePath) => index.chunks[filePath].text,
    })
    const bracketGlob = await retrieve({
      index,
      input: { query: "a", topK: 3, includeParents: true, maxContextChars: 100, paths: ["src/[ab].ts"] },
      options: { topK: 3, maxContextChars: 100, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async (filePath) => index.chunks[filePath].text,
    })

    expect(exact.results.map((result) => result.filePath)).toEqual(["test/c.ts"])
    expect(directory.results.map((result) => result.filePath)).toEqual(["src/a.ts", "src/nested/b.ts", "src/c.ts"])
    expect(glob.results.map((result) => result.filePath)).toEqual(["src/a.ts", "src/nested/b.ts", "src/c.ts"])
    expect(bracketGlob.results.map((result) => result.filePath)).toEqual(["src/a.ts"])
  })

  test("uses HyDE when best score is below threshold", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.c1 = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
      text: "function a() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0, 1],
    }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 1, includeParents: true, maxContextChars: 100 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: true, threshold: 0.5 } },
      embed: async (text) => (text === "hyde text" ? [0, 1] : [1, 0]),
      generateHyde: async () => "hyde text",
      readSource: async () => "function a() {}",
    })

    expect(output.status.hydeUsed).toBe(true)
    expect(output.status.bestScore).toBe(0)
    expect(output.results[0].score).toBe(0)
    expect(output.results[0].finalScore).toBe(1)
  })

  test("falls back to initial results with diagnostics when HyDE fails", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
      diagnostics: ["existing"],
    })
    index.metadata.status = "ready"
    index.chunks.c1 = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
      text: "function a() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0, 1],
    }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 1, includeParents: true, maxContextChars: 100, minFinalScore: 0 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: true, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => {
        throw new Error("hyde exploded")
      },
      readSource: async () => "function a() {}",
    })

    expect(output.status.hydeUsed).toBe(false)
    expect(output.results[0].topology.chunk.id).toBe("c1")
    expect(output.results[0].score).toBe(0)
    expect(output.results[0].finalScore).toBe(0)
    expect(output.diagnostics).toContain("existing")
    expect(output.diagnostics.at(-1)).toContain("HyDE failed: hyde exploded")
  })

  test("uses HyDE when initial search has no embedded chunks", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"

    const output = await retrieve({
      index,
      input: { query: "a", topK: 1, includeParents: true, maxContextChars: 100 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: true, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => "",
    })

    expect(output.status.hydeUsed).toBe(true)
    expect(output.status.bestScore).toBeUndefined()
    expect(output.results).toEqual([])
  })

  test("filters zero-score results with the default minFinalScore", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.c1 = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 16, lineStart: 1, lineEnd: 1 },
      text: "function a() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0, 1],
    }

    const output = await retrieve({
      index,
      input: { query: "nonsense", topK: 1, includeParents: true, maxContextChars: 100 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => "function a() {}",
    })

    expect(output.results).toEqual([])
    expect(output.status.minFinalScore).toBe(0.01)
    expect(output.status.filteredCount).toBe(1)
    expect(output.status.candidateCount).toBe(1)
  })

  test("allows callers to lower minFinalScore", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.c1 = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 16, lineStart: 1, lineEnd: 1 },
      text: "function a() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0, 1],
    }

    const output = await retrieve({
      index,
      input: { query: "nonsense", topK: 1, includeParents: true, maxContextChars: 100, minFinalScore: 0 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => "function a() {}",
    })

    expect(output.results).toHaveLength(1)
    expect(output.results[0].finalScore).toBe(0)
    expect(output.status.minFinalScore).toBe(0)
    expect(output.status.filteredCount).toBe(0)
    expect(output.status.candidateCount).toBe(1)
  })

  test("allows callers to raise minFinalScore", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.c1 = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 16, lineStart: 1, lineEnd: 1 },
      text: "function a() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [1, 0],
    }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 1, includeParents: true, maxContextChars: 100, minFinalScore: 1.1 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => "function a() {}",
    })

    expect(output.results).toEqual([])
    expect(output.status.minFinalScore).toBe(1.1)
    expect(output.status.filteredCount).toBe(1)
    expect(output.status.candidateCount).toBe(1)
  })

  test("clamps negative minFinalScore to zero", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.c1 = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 16, lineStart: 1, lineEnd: 1 },
      text: "function a() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0, 1],
    }

    const output = await retrieve({
      index,
      input: { query: "nonsense", topK: 1, includeParents: true, maxContextChars: 100, minFinalScore: -1 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => "function a() {}",
    })

    expect(output.results).toHaveLength(1)
    expect(output.results[0].finalScore).toBe(0)
    expect(output.status.minFinalScore).toBe(0)
    expect(output.status.filteredCount).toBe(0)
    expect(output.status.candidateCount).toBe(1)
  })

  test("reranks vector candidates after initial ranking", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.c1 = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 19, lineStart: 1, lineEnd: 1 },
      text: "function first() {}",
      nonWhitespaceChars: 18,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [1, 0],
    }
    index.chunks.c2 = {
      id: "c2",
      filePath: "b.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 20, lineStart: 1, lineEnd: 1 },
      text: "function second() {}",
      nonWhitespaceChars: 19,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0.9, Math.sqrt(0.19)],
    }

    const output = await retrieve({
      index,
      input: { query: "best match", topK: 1, includeParents: true, maxContextChars: 100 },
      options: {
        topK: 1,
        maxContextChars: 100,
        hyde: { enabled: false, threshold: 0.5 },
        rerank: rerankOptions({ candidateMultiplier: 2 }),
      },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      rerank: async (_query, documents) => {
        expect(documents).toEqual([
          "a.ts:1\nkind: function\nfunction first() {}",
          "b.ts:1\nkind: function\nfunction second() {}",
        ])
        return [
          { index: 1, score: 0.99 },
          { index: 0, score: 0.1 },
        ]
      },
      readSource: async (filePath) => index.chunks[filePath === "a.ts" ? "c1" : "c2"].text,
    })

    expect(output.status.rerankUsed).toBe(true)
    expect(output.results.map((result) => result.topology.chunk.id)).toEqual(["c2"])
    expect(output.results[0].score).toBe(0.9)
    expect(output.results[0].finalScore).toBe(0.99)
    expect(output.results[0].retrieval).toMatchObject({
      mode: "vector",
      vectorRank: 2,
      rerankRank: 1,
      rerankScore: 0.99,
    })
  })

  test("falls back to pre-rerank results with diagnostics when rerank fails", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.c1 = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 19, lineStart: 1, lineEnd: 1 },
      text: "function first() {}",
      nonWhitespaceChars: 18,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [1, 0],
    }

    const output = await retrieve({
      index,
      input: { query: "best match", topK: 1, includeParents: true, maxContextChars: 100 },
      options: {
        topK: 1,
        maxContextChars: 100,
        hyde: { enabled: false, threshold: 0.5 },
        rerank: rerankOptions(),
      },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      rerank: async () => {
        throw new Error("provider unavailable")
      },
      readSource: async () => "function first() {}",
    })

    expect(output.status.rerankUsed).toBe(false)
    expect(output.results[0].topology.chunk.id).toBe("c1")
    expect(output.results[0].finalScore).toBe(1)
    expect(output.results[0].retrieval).toMatchObject({ mode: "vector", vectorRank: 1 })
    expect(output.results[0].retrieval?.rerankRank).toBeUndefined()
    expect(output.diagnostics.at(-1)).toBe("Rerank failed: provider unavailable")
  })

  test("uses active score when HyDE promotes a result outside initial candidates", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.c1 = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
      text: "function a() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0.9, Math.sqrt(0.19)],
    }
    index.chunks.c2 = {
      id: "c2",
      filePath: "b.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
      text: "function b() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0.8, 0.6],
    }
    index.chunks.c3 = {
      id: "c3",
      filePath: "c.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
      text: "function c() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0.7, Math.sqrt(0.51)],
    }
    index.chunks.c4 = {
      id: "c4",
      filePath: "d.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
      text: "function d() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0.1, Math.sqrt(0.99)],
    }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 1, includeParents: true, maxContextChars: 100 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: true, threshold: 0.95 } },
      embed: async (text) => (text === "hyde text" ? [0, 1] : [1, 0]),
      generateHyde: async () => "hyde text",
      readSource: async () => "function d() {}",
    })

    expect(output.results[0].topology.chunk.id).toBe("c4")
    expect(output.results[0].finalScore).toBeGreaterThan(0.99)
    expect(output.results[0].score).toBe(output.results[0].finalScore)
  })

  test("returns file-level diagnostics when no chunks have embeddings", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.files["a.ts"] = {
      path: "a.ts",
      language: "typescript",
      fingerprint: "abc",
      chunkIds: ["c1"],
      diagnostics: ["embedding failed: boom"],
    }
    index.chunks.c1 = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
      text: "function a() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embeddingError: "boom",
    }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 1, includeParents: true, maxContextChars: 100 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => "function a() {}",
    })

    expect(output.results).toEqual([])
    expect(output.diagnostics).toContain("a.ts: embedding failed: boom")
  })

  test("omits parent text and range when source read fails", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.symbols.s1 = {
      id: "s1",
      name: "A",
      kind: "class",
      filePath: "a.ts",
      range: { byteStart: 20, byteEnd: 40, lineStart: 2, lineEnd: 4 },
      childSymbolIds: [],
    }
    index.chunks.c1 = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "method",
      range: { byteStart: 30, byteEnd: 38, lineStart: 3, lineEnd: 3 },
      text: "a() {}",
      nonWhitespaceChars: 5,
      nodeTypes: [],
      symbolIds: ["s1"],
      childChunkIds: [],
      embedding: [1, 0],
    }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 1, includeParents: true, maxContextChars: 100 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => {
        throw new Error("read failed")
      },
    })

    expect(output.results[0].breadcrumbs).toEqual(["class A"])
    expect(output.results[0].parentText).toBeUndefined()
    expect(output.results[0].parentRange).toBeUndefined()
    expect(output.diagnostics).toContain("source read failed for a.ts; parent context omitted")
  })

  test("omits parent text and range when source no longer matches indexed chunk", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.symbols.s1 = {
      id: "s1",
      name: "A",
      kind: "class",
      filePath: "a.ts",
      range: { byteStart: 0, byteEnd: 20, lineStart: 1, lineEnd: 3 },
      childSymbolIds: [],
    }
    index.chunks.c1 = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "method",
      range: { byteStart: 12, byteEnd: 18, lineStart: 2, lineEnd: 2 },
      text: "a() {}",
      nonWhitespaceChars: 5,
      nodeTypes: [],
      symbolIds: ["s1"],
      childChunkIds: [],
      embedding: [1, 0],
    }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 1, includeParents: true, maxContextChars: 100 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => "class A {\n  z() {}\n}\n",
    })

    expect(output.results[0].breadcrumbs).toEqual(["class A"])
    expect(output.results[0].parentText).toBeUndefined()
    expect(output.results[0].parentRange).toBeUndefined()
    expect(output.diagnostics).toContain("source mismatch for a.ts:c1; parent context omitted")
  })

  test("skips malformed chunk map keys with diagnostics", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.wrong = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
      text: "function a() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [1, 0],
    }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 1, includeParents: true, maxContextChars: 100 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => "function a() {}",
    })

    expect(output.results).toEqual([])
    expect(output.diagnostics).toContain("chunk key mismatch: wrong contains c1; chunk skipped")
  })

  test("suppresses duplicate parent context for repeated parent ranges", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.symbols.s1 = {
      id: "s1",
      name: "A",
      kind: "class",
      filePath: "a.ts",
      range: { byteStart: 0, byteEnd: 29, lineStart: 1, lineEnd: 4 },
      childSymbolIds: [],
    }
    index.chunks.c1 = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "method",
      range: { byteStart: 12, byteEnd: 18, lineStart: 2, lineEnd: 2 },
      text: "a() {}",
      nonWhitespaceChars: 5,
      nodeTypes: [],
      symbolIds: ["s1"],
      childChunkIds: [],
      embedding: [1, 0],
    }
    index.chunks.c2 = {
      id: "c2",
      filePath: "a.ts",
      language: "typescript",
      kind: "method",
      range: { byteStart: 21, byteEnd: 27, lineStart: 3, lineEnd: 3 },
      text: "b() {}",
      nonWhitespaceChars: 5,
      nodeTypes: [],
      symbolIds: ["s1"],
      childChunkIds: [],
      embedding: [0.9, 0],
    }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 2, includeParents: true, maxContextChars: 100 },
      options: { topK: 2, maxContextChars: 100, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => "class A {\n  a() {}\n  b() {}\n}\n",
    })

    expect(output.results[0].parentText).toBe("class A {\n  a() {}\n  b() {}\n}")
    expect(output.results[0].parentRange).toEqual(index.symbols.s1.range)
    expect(output.results[1].breadcrumbs).toEqual(["class A"])
    expect(output.results[1].parentText).toBeUndefined()
    expect(output.results[1].parentRange).toBeUndefined()
  })

  test("keeps distinct compact parent excerpts for oversized repeated parent ranges", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    const source = "class A {\n  aLongName() {}\n  bLongName() {}\n}\n"
    index.metadata.status = "ready"
    index.symbols.s1 = {
      id: "s1",
      name: "A",
      kind: "class",
      filePath: "a.ts",
      range: { byteStart: 0, byteEnd: source.trimEnd().length, lineStart: 1, lineEnd: 4 },
      childSymbolIds: [],
    }
    index.chunks.c1 = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "method",
      range: {
        byteStart: source.indexOf("aLongName"),
        byteEnd: source.indexOf("aLongName") + "aLongName() {}".length,
        lineStart: 2,
        lineEnd: 2,
      },
      text: "aLongName() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: ["s1"],
      childChunkIds: [],
      embedding: [1, 0],
    }
    index.chunks.c2 = {
      id: "c2",
      filePath: "a.ts",
      language: "typescript",
      kind: "method",
      range: {
        byteStart: source.indexOf("bLongName"),
        byteEnd: source.indexOf("bLongName") + "bLongName() {}".length,
        lineStart: 3,
        lineEnd: 3,
      },
      text: "bLongName() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: ["s1"],
      childChunkIds: [],
      embedding: [0.9, 0],
    }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 2, includeParents: true, maxContextChars: 24 },
      options: { topK: 2, maxContextChars: 24, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => source,
    })

    expect(output.results[0].parentText).toContain("aLongName")
    expect(output.results[1].parentText).toContain("bLongName")
    expect(output.results[0].parentText).not.toBe(output.results[1].parentText)
    expect(output.results[1].parentRange).toEqual(index.symbols.s1.range)
  })

  test("parallel hybrid returns a BM25-only exact identifier match when vector score is weak", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.semantic = {
      id: "semantic",
      filePath: "src/semantic.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 25, lineStart: 1, lineEnd: 1 },
      text: "function nearbyConcept() {}",
      nonWhitespaceChars: 24,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [1, 0],
    }
    index.chunks.exact = {
      id: "exact",
      filePath: "src/exact.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 36, lineStart: 1, lineEnd: 1 },
      text: "function throwCriticalParserError() {}",
      nonWhitespaceChars: 35,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0, 1],
    }
    addLexicalStats(index)

    const output = await retrieve({
      index,
      input: { query: "throwCriticalParserError", topK: 1, includeParents: true, maxContextChars: 100 },
      options: {
        topK: 1,
        maxContextChars: 100,
        hyde: { enabled: false, threshold: 0.5 },
        hybrid: hybridOptions({ vectorCandidateMultiplier: 1, bm25CandidateMultiplier: 1 }),
      },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async (filePath) => index.chunks[filePath === "src/semantic.ts" ? "semantic" : "exact"].text,
    })

    expect(output.results.map((result) => result.topology.chunk.id)).toContain("exact")
    expect(output.results.find((result) => result.topology.chunk.id === "exact")?.retrieval).toMatchObject({
      mode: "hybrid",
      hybridMode: "parallel",
      bm25Rank: 1,
    })
    expect(output.results[0].retrieval?.vectorRank).toBeUndefined()
  })

  test("hybrid respects path filters for BM25 candidates", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.src = {
      id: "src",
      filePath: "src/allowed.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 28, lineStart: 1, lineEnd: 1 },
      text: "function allowedNeedle() {}",
      nonWhitespaceChars: 27,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0, 1],
    }
    index.chunks.test = {
      id: "test",
      filePath: "test/blocked.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 35, lineStart: 1, lineEnd: 1 },
      text: "function allowedNeedleBlocked() {}",
      nonWhitespaceChars: 34,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [1, 0],
    }
    addLexicalStats(index)

    const output = await retrieve({
      index,
      input: { query: "allowedNeedleBlocked", topK: 3, includeParents: true, maxContextChars: 100, paths: ["src/"] },
      options: {
        topK: 3,
        maxContextChars: 100,
        hyde: { enabled: false, threshold: 0.5 },
        hybrid: hybridOptions(),
      },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => "function allowedNeedle() {}",
    })

    expect(output.results.map((result) => result.filePath)).toEqual(["src/allowed.ts"])
  })

  test("hybrid with missing lexical data degrades to vector-only with diagnostic", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.c1 = {
      id: "c1",
      filePath: "a.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
      text: "function a() {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [1, 0],
    }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 1, includeParents: true, maxContextChars: 100 },
      options: {
        topK: 1,
        maxContextChars: 100,
        hyde: { enabled: false, threshold: 0.5 },
        hybrid: hybridOptions(),
      },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => "function a() {}",
    })

    expect(output.results[0].retrieval).toEqual({ mode: "vector", vectorRank: 1 })
    expect(output.diagnostics).toContain(
      "hybrid retrieval requested but lexical data is unavailable; using vector-only retrieval",
    )
  })

  test("hybrid disabled preserves vector-only behavior", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.vector = {
      id: "vector",
      filePath: "vector.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 24, lineStart: 1, lineEnd: 1 },
      text: "function vectorOnly() {}",
      nonWhitespaceChars: 23,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [1, 0],
    }
    index.chunks.lexical = {
      id: "lexical",
      filePath: "lexical.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 32, lineStart: 1, lineEnd: 1 },
      text: "function exactDisabledHybrid() {}",
      nonWhitespaceChars: 31,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0, 1],
    }
    addLexicalStats(index)

    const output = await retrieve({
      index,
      input: { query: "exactDisabledHybrid", topK: 1, includeParents: true, maxContextChars: 100 },
      options: {
        topK: 1,
        maxContextChars: 100,
        hyde: { enabled: false, threshold: 0.5 },
        hybrid: hybridOptions({ enabled: false }),
      },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => "function vectorOnly() {}",
    })

    expect(output.results.map((result) => result.topology.chunk.id)).toEqual(["vector"])
    expect(output.results[0].retrieval).toEqual({ mode: "vector", vectorRank: 1 })
  })

  test("bm25-prefilter hybrid ranks deterministically within the lexical pool", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.alpha = {
      id: "alpha",
      filePath: "alpha.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 28, lineStart: 1, lineEnd: 1 },
      text: "function prefilterNeedle() {}",
      nonWhitespaceChars: 27,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0, 1],
    }
    index.chunks.beta = {
      id: "beta",
      filePath: "beta.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 36, lineStart: 1, lineEnd: 1 },
      text: "function prefilterNeedleBetter() {}",
      nonWhitespaceChars: 35,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [1, 0],
    }
    index.chunks.gamma = {
      id: "gamma",
      filePath: "gamma.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 24, lineStart: 1, lineEnd: 1 },
      text: "function unrelated() {}",
      nonWhitespaceChars: 23,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [1, 0],
    }
    addLexicalStats(index)

    const output = await retrieve({
      index,
      input: { query: "prefilterNeedle", topK: 2, includeParents: true, maxContextChars: 100 },
      options: {
        topK: 2,
        maxContextChars: 100,
        hyde: { enabled: false, threshold: 0.5 },
        hybrid: hybridOptions({ mode: "bm25-prefilter", vectorWeight: 10, bm25Weight: 1 }),
      },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async (filePath) => index.chunks[filePath.replace(".ts", "")].text,
    })

    expect(output.results.map((result) => result.topology.chunk.id)).toEqual(["beta", "alpha"])
    expect(output.results.map((result) => result.retrieval?.hybridMode)).toEqual(["bm25-prefilter", "bm25-prefilter"])
  })

  test("vector-prefilter uses deterministic vector tie ordering before restricting BM25", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks["a-unrelated"] = {
      id: "a-unrelated",
      filePath: "a-unrelated.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 24, lineStart: 1, lineEnd: 1 },
      text: "function unrelated() {}",
      nonWhitespaceChars: 23,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [1, 0],
    }
    index.chunks["z-exact"] = {
      id: "z-exact",
      filePath: "z-exact.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 34, lineStart: 1, lineEnd: 1 },
      text: "function exactVectorTieNeedle() {}",
      nonWhitespaceChars: 33,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [1, 0],
    }
    addLexicalStats(index)

    const output = await retrieve({
      index,
      input: { query: "exactVectorTieNeedle", topK: 1, includeParents: true, maxContextChars: 100 },
      options: {
        topK: 1,
        maxContextChars: 100,
        hyde: { enabled: false, threshold: 0.5 },
        hybrid: hybridOptions({ mode: "vector-prefilter", vectorCandidateMultiplier: 1, bm25CandidateMultiplier: 1 }),
      },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async (filePath) => index.chunks[filePath.replace(".ts", "")].text,
    })

    expect(output.results[0].topology.chunk.id).toBe("z-exact")
    expect(output.results[0].retrieval).toMatchObject({
      mode: "hybrid",
      hybridMode: "vector-prefilter",
      vectorRank: 2,
      bm25Rank: 1,
    })
  })

  test("HyDE-triggered hybrid uses HyDE vector candidates while preserving BM25 fusion", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.initial = {
      id: "initial",
      filePath: "initial.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 24, lineStart: 1, lineEnd: 1 },
      text: "function initial() {}",
      nonWhitespaceChars: 23,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [1, 0],
    }
    index.chunks.hyde = {
      id: "hyde",
      filePath: "hyde.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 20, lineStart: 1, lineEnd: 1 },
      text: "function hyde() {}",
      nonWhitespaceChars: 19,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0, 1],
    }
    index.chunks.exact = {
      id: "exact",
      filePath: "exact.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 34, lineStart: 1, lineEnd: 1 },
      text: "function rareHybridNeedle() {}",
      nonWhitespaceChars: 33,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0.1, 0],
    }
    addLexicalStats(index)

    const output = await retrieve({
      index,
      input: { query: "rareHybridNeedle", topK: 2, includeParents: true, maxContextChars: 100 },
      options: {
        topK: 2,
        maxContextChars: 100,
        hyde: { enabled: true, threshold: 0.5 },
        hybrid: hybridOptions({ vectorWeight: 1, bm25Weight: 1 }),
      },
      embed: async (text) => (text === "hyde text" ? [0, 1] : [0, 0]),
      generateHyde: async () => "hyde text",
      readSource: async (filePath) => index.chunks[filePath.replace(".ts", "")].text,
    })

    expect(output.status.hydeUsed).toBe(true)
    expect(output.results.map((result) => result.topology.chunk.id)).toContain("hyde")
    expect(output.results.map((result) => result.topology.chunk.id)).toContain("exact")
    expect(output.results.find((result) => result.topology.chunk.id === "exact")?.retrieval?.bm25Rank).toBe(1)
  })

  test("HyDE-triggered bm25-prefilter uses HyDE vector ranks within the BM25 pool", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.metadata.status = "ready"
    index.chunks.exact = {
      id: "exact",
      filePath: "exact.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 36, lineStart: 1, lineEnd: 1 },
      text: "function sharedHydePrefilterNeedle() {}",
      nonWhitespaceChars: 35,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0.8, 0.6],
    }
    index.chunks.hyde = {
      id: "hyde",
      filePath: "hyde.ts",
      language: "typescript",
      kind: "function",
      range: { byteStart: 0, byteEnd: 41, lineStart: 1, lineEnd: 1 },
      text: "function sharedHydePrefilterNeedleHyde() {}",
      nonWhitespaceChars: 40,
      nodeTypes: [],
      symbolIds: [],
      childChunkIds: [],
      embedding: [0.1, Math.sqrt(0.99)],
    }
    addLexicalStats(index)

    const output = await retrieve({
      index,
      input: { query: "sharedHydePrefilterNeedle", topK: 1, includeParents: true, maxContextChars: 100 },
      options: {
        topK: 1,
        maxContextChars: 100,
        hyde: { enabled: true, threshold: 0.95 },
        hybrid: hybridOptions({ mode: "bm25-prefilter", vectorWeight: 10, bm25Weight: 1 }),
      },
      embed: async (text) => (text === "hyde text" ? [0, 1] : [1, 0]),
      generateHyde: async () => "hyde text",
      readSource: async (filePath) => index.chunks[filePath.replace(".ts", "")].text,
    })

    expect(output.status.hydeUsed).toBe(true)
    expect(output.results[0].topology.chunk.id).toBe("hyde")
    expect(output.results[0].retrieval).toMatchObject({
      mode: "hybrid",
      hybridMode: "bm25-prefilter",
      vectorRank: 1,
    })
  })
})
