import { describe, expect, test } from "bun:test"
import { retrieve } from "./retriever.js"
import { createEmptyIndex } from "./store.js"

describe("retrieve", () => {
  test("returns normal embedding results without HyDE above threshold", async () => {
    const index = createEmptyIndex({ projectId: "p", worktree: "/repo", cacheKey: "key", maxChunkNonWhitespaceChars: 2000 })
    index.metadata.status = "ready"
    index.chunks["c1"] = { id: "c1", filePath: "a.ts", language: "typescript", kind: "function", range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 }, text: "function a() {}", nonWhitespaceChars: 13, nodeTypes: [], symbolIds: [], childChunkIds: [], embedding: [1, 0] }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 1, includeParents: true, maxContextChars: 100 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: true, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => { throw new Error("not called") },
      readSource: async () => "function a() {}",
    })

    expect(output.status.hydeUsed).toBe(false)
    expect(output.results[0].filePath).toBe("a.ts")
  })

  test("filters retrieval candidates by exact paths and directory prefixes", async () => {
    const index = createEmptyIndex({ projectId: "p", worktree: "/repo", cacheKey: "key", maxChunkNonWhitespaceChars: 2000 })
    index.metadata.status = "ready"
    index.chunks["src/a.ts"] = { id: "src/a.ts", filePath: "src/a.ts", language: "typescript", kind: "function", range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 }, text: "function a() {}", nonWhitespaceChars: 13, nodeTypes: [], symbolIds: [], childChunkIds: [], embedding: [1, 0] }
    index.chunks["src/nested/b.ts"] = { id: "src/nested/b.ts", filePath: "src/nested/b.ts", language: "typescript", kind: "function", range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 }, text: "function b() {}", nonWhitespaceChars: 13, nodeTypes: [], symbolIds: [], childChunkIds: [], embedding: [0.9, 0] }
    index.chunks["test/c.ts"] = { id: "test/c.ts", filePath: "test/c.ts", language: "typescript", kind: "function", range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 }, text: "function c() {}", nonWhitespaceChars: 13, nodeTypes: [], symbolIds: [], childChunkIds: [], embedding: [0.8, 0] }

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

    expect(exact.results.map((result) => result.filePath)).toEqual(["test/c.ts"])
    expect(directory.results.map((result) => result.filePath)).toEqual(["src/a.ts", "src/nested/b.ts"])
  })

  test("uses HyDE when best score is below threshold", async () => {
    const index = createEmptyIndex({ projectId: "p", worktree: "/repo", cacheKey: "key", maxChunkNonWhitespaceChars: 2000 })
    index.metadata.status = "ready"
    index.chunks["c1"] = { id: "c1", filePath: "a.ts", language: "typescript", kind: "function", range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 }, text: "function a() {}", nonWhitespaceChars: 13, nodeTypes: [], symbolIds: [], childChunkIds: [], embedding: [0, 1] }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 1, includeParents: true, maxContextChars: 100 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: true, threshold: 0.5 } },
      embed: async (text) => text === "hyde text" ? [0, 1] : [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => "function a() {}",
    })

    expect(output.status.hydeUsed).toBe(true)
    expect(output.status.bestScore).toBe(0)
    expect(output.results[0].score).toBe(0)
    expect(output.results[0].finalScore).toBe(1)
  })

  test("falls back to initial results with diagnostics when HyDE fails", async () => {
    const index = createEmptyIndex({ projectId: "p", worktree: "/repo", cacheKey: "key", maxChunkNonWhitespaceChars: 2000, diagnostics: ["existing"] })
    index.metadata.status = "ready"
    index.chunks["c1"] = { id: "c1", filePath: "a.ts", language: "typescript", kind: "function", range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 }, text: "function a() {}", nonWhitespaceChars: 13, nodeTypes: [], symbolIds: [], childChunkIds: [], embedding: [0, 1] }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 1, includeParents: true, maxContextChars: 100 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: true, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => { throw new Error("hyde exploded") },
      readSource: async () => "function a() {}",
    })

    expect(output.status.hydeUsed).toBe(false)
    expect(output.results[0].topology.chunkId).toBe("c1")
    expect(output.results[0].score).toBe(0)
    expect(output.results[0].finalScore).toBe(0)
    expect(output.diagnostics).toContain("existing")
    expect(output.diagnostics.at(-1)).toContain("HyDE failed: hyde exploded")
  })

  test("uses HyDE when initial search has no embedded chunks", async () => {
    const index = createEmptyIndex({ projectId: "p", worktree: "/repo", cacheKey: "key", maxChunkNonWhitespaceChars: 2000 })
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

  test("uses active score when HyDE promotes a result outside initial candidates", async () => {
    const index = createEmptyIndex({ projectId: "p", worktree: "/repo", cacheKey: "key", maxChunkNonWhitespaceChars: 2000 })
    index.metadata.status = "ready"
    index.chunks["c1"] = { id: "c1", filePath: "a.ts", language: "typescript", kind: "function", range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 }, text: "function a() {}", nonWhitespaceChars: 13, nodeTypes: [], symbolIds: [], childChunkIds: [], embedding: [0.9, Math.sqrt(0.19)] }
    index.chunks["c2"] = { id: "c2", filePath: "b.ts", language: "typescript", kind: "function", range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 }, text: "function b() {}", nonWhitespaceChars: 13, nodeTypes: [], symbolIds: [], childChunkIds: [], embedding: [0.8, 0.6] }
    index.chunks["c3"] = { id: "c3", filePath: "c.ts", language: "typescript", kind: "function", range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 }, text: "function c() {}", nonWhitespaceChars: 13, nodeTypes: [], symbolIds: [], childChunkIds: [], embedding: [0.7, Math.sqrt(0.51)] }
    index.chunks["c4"] = { id: "c4", filePath: "d.ts", language: "typescript", kind: "function", range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 }, text: "function d() {}", nonWhitespaceChars: 13, nodeTypes: [], symbolIds: [], childChunkIds: [], embedding: [0.1, Math.sqrt(0.99)] }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 1, includeParents: true, maxContextChars: 100 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: true, threshold: 0.95 } },
      embed: async (text) => text === "hyde text" ? [0, 1] : [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => "function d() {}",
    })

    expect(output.results[0].topology.chunkId).toBe("c4")
    expect(output.results[0].finalScore).toBeGreaterThan(0.99)
    expect(output.results[0].score).toBe(output.results[0].finalScore)
  })

  test("returns file-level diagnostics when no chunks have embeddings", async () => {
    const index = createEmptyIndex({ projectId: "p", worktree: "/repo", cacheKey: "key", maxChunkNonWhitespaceChars: 2000 })
    index.metadata.status = "ready"
    index.files["a.ts"] = { path: "a.ts", language: "typescript", fingerprint: "abc", chunkIds: ["c1"], diagnostics: ["embedding failed: boom"] }
    index.chunks["c1"] = { id: "c1", filePath: "a.ts", language: "typescript", kind: "function", range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 }, text: "function a() {}", nonWhitespaceChars: 13, nodeTypes: [], symbolIds: [], childChunkIds: [], embeddingError: "boom" }

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
    const index = createEmptyIndex({ projectId: "p", worktree: "/repo", cacheKey: "key", maxChunkNonWhitespaceChars: 2000 })
    index.metadata.status = "ready"
    index.symbols["s1"] = { id: "s1", name: "A", kind: "class", filePath: "a.ts", range: { byteStart: 20, byteEnd: 40, lineStart: 2, lineEnd: 4 }, childSymbolIds: [] }
    index.chunks["c1"] = { id: "c1", filePath: "a.ts", language: "typescript", kind: "method", range: { byteStart: 30, byteEnd: 38, lineStart: 3, lineEnd: 3 }, text: "a() {}", nonWhitespaceChars: 5, nodeTypes: [], symbolIds: ["s1"], childChunkIds: [], embedding: [1, 0] }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 1, includeParents: true, maxContextChars: 100 },
      options: { topK: 1, maxContextChars: 100, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => { throw new Error("read failed") },
    })

    expect(output.results[0].breadcrumbs).toEqual(["class A"])
    expect(output.results[0].parentText).toBeUndefined()
    expect(output.results[0].parentRange).toBeUndefined()
    expect(output.diagnostics).toContain("source read failed for a.ts; parent context omitted")
  })

  test("omits parent text and range when source no longer matches indexed chunk", async () => {
    const index = createEmptyIndex({ projectId: "p", worktree: "/repo", cacheKey: "key", maxChunkNonWhitespaceChars: 2000 })
    index.metadata.status = "ready"
    index.symbols["s1"] = { id: "s1", name: "A", kind: "class", filePath: "a.ts", range: { byteStart: 0, byteEnd: 20, lineStart: 1, lineEnd: 3 }, childSymbolIds: [] }
    index.chunks["c1"] = { id: "c1", filePath: "a.ts", language: "typescript", kind: "method", range: { byteStart: 12, byteEnd: 18, lineStart: 2, lineEnd: 2 }, text: "a() {}", nonWhitespaceChars: 5, nodeTypes: [], symbolIds: ["s1"], childChunkIds: [], embedding: [1, 0] }

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
    const index = createEmptyIndex({ projectId: "p", worktree: "/repo", cacheKey: "key", maxChunkNonWhitespaceChars: 2000 })
    index.metadata.status = "ready"
    index.chunks["wrong"] = { id: "c1", filePath: "a.ts", language: "typescript", kind: "function", range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 }, text: "function a() {}", nonWhitespaceChars: 13, nodeTypes: [], symbolIds: [], childChunkIds: [], embedding: [1, 0] }

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
    const index = createEmptyIndex({ projectId: "p", worktree: "/repo", cacheKey: "key", maxChunkNonWhitespaceChars: 2000 })
    index.metadata.status = "ready"
    index.symbols["s1"] = { id: "s1", name: "A", kind: "class", filePath: "a.ts", range: { byteStart: 0, byteEnd: 29, lineStart: 1, lineEnd: 4 }, childSymbolIds: [] }
    index.chunks["c1"] = { id: "c1", filePath: "a.ts", language: "typescript", kind: "method", range: { byteStart: 12, byteEnd: 18, lineStart: 2, lineEnd: 2 }, text: "a() {}", nonWhitespaceChars: 5, nodeTypes: [], symbolIds: ["s1"], childChunkIds: [], embedding: [1, 0] }
    index.chunks["c2"] = { id: "c2", filePath: "a.ts", language: "typescript", kind: "method", range: { byteStart: 21, byteEnd: 27, lineStart: 3, lineEnd: 3 }, text: "b() {}", nonWhitespaceChars: 5, nodeTypes: [], symbolIds: ["s1"], childChunkIds: [], embedding: [0.9, 0] }

    const output = await retrieve({
      index,
      input: { query: "a", topK: 2, includeParents: true, maxContextChars: 100 },
      options: { topK: 2, maxContextChars: 100, hyde: { enabled: false, threshold: 0.5 } },
      embed: async () => [1, 0],
      generateHyde: async () => "hyde text",
      readSource: async () => "class A {\n  a() {}\n  b() {}\n}\n",
    })

    expect(output.results[0].parentText).toBe("class A {\n  a() {}\n  b() {}\n}")
    expect(output.results[0].parentRange).toEqual(index.symbols["s1"].range)
    expect(output.results[1].breadcrumbs).toEqual(["class A"])
    expect(output.results[1].parentText).toBeUndefined()
    expect(output.results[1].parentRange).toBeUndefined()
  })

  test("keeps distinct compact parent excerpts for oversized repeated parent ranges", async () => {
    const index = createEmptyIndex({ projectId: "p", worktree: "/repo", cacheKey: "key", maxChunkNonWhitespaceChars: 2000 })
    const source = "class A {\n  aLongName() {}\n  bLongName() {}\n}\n"
    index.metadata.status = "ready"
    index.symbols["s1"] = { id: "s1", name: "A", kind: "class", filePath: "a.ts", range: { byteStart: 0, byteEnd: source.trimEnd().length, lineStart: 1, lineEnd: 4 }, childSymbolIds: [] }
    index.chunks["c1"] = { id: "c1", filePath: "a.ts", language: "typescript", kind: "method", range: { byteStart: source.indexOf("aLongName"), byteEnd: source.indexOf("aLongName") + "aLongName() {}".length, lineStart: 2, lineEnd: 2 }, text: "aLongName() {}", nonWhitespaceChars: 13, nodeTypes: [], symbolIds: ["s1"], childChunkIds: [], embedding: [1, 0] }
    index.chunks["c2"] = { id: "c2", filePath: "a.ts", language: "typescript", kind: "method", range: { byteStart: source.indexOf("bLongName"), byteEnd: source.indexOf("bLongName") + "bLongName() {}".length, lineStart: 3, lineEnd: 3 }, text: "bLongName() {}", nonWhitespaceChars: 13, nodeTypes: [], symbolIds: ["s1"], childChunkIds: [], embedding: [0.9, 0] }

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
    expect(output.results[1].parentRange).toEqual(index.symbols["s1"].range)
  })
})
