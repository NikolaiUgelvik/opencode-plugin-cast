import { describe, expect, test } from "bun:test"
import { assignSymbolsToChunks, attachTopology, expandWithParentContext, extractSymbols } from "./topology.js"
import type { ChunkRecord, SymbolRecord } from "./types.js"

const base = {
  filePath: "src/a.ts",
  language: "typescript",
  nodeTypes: [],
  symbolIds: [],
  childChunkIds: [],
  nonWhitespaceChars: 1,
} satisfies Partial<ChunkRecord>

describe("topology", () => {
  test("links children to parent symbols and siblings", () => {
    const parent: SymbolRecord = { id: "sym:class:A", name: "A", kind: "class", filePath: "src/a.ts", range: { byteStart: 0, byteEnd: 40, lineStart: 1, lineEnd: 4 }, childSymbolIds: [] }
    const chunks = attachTopology(
      [
        { ...base, id: "chunk:1", kind: "method", range: { byteStart: 10, byteEnd: 20, lineStart: 2, lineEnd: 2 }, text: "a() {}", symbolIds: [parent.id] },
        { ...base, id: "chunk:2", kind: "method", range: { byteStart: 21, byteEnd: 31, lineStart: 3, lineEnd: 3 }, text: "b() {}", symbolIds: [parent.id] },
      ] as ChunkRecord[],
      { [parent.id]: parent },
    )

    expect(chunks[0].previousSiblingChunkId).toBeUndefined()
    expect(chunks[0].nextSiblingChunkId).toBe("chunk:2")
    expect(chunks[1].previousSiblingChunkId).toBe("chunk:1")
  })

  test("does not link unrelated chunks from different nearest symbols as siblings", () => {
    const parentA: SymbolRecord = { id: "sym:class:A", name: "A", kind: "class", filePath: "src/a.ts", range: { byteStart: 0, byteEnd: 20, lineStart: 1, lineEnd: 3 }, childSymbolIds: [] }
    const parentB: SymbolRecord = { id: "sym:class:B", name: "B", kind: "class", filePath: "src/a.ts", range: { byteStart: 21, byteEnd: 42, lineStart: 4, lineEnd: 6 }, childSymbolIds: [] }
    const chunks = attachTopology(
      [
        { ...base, id: "chunk:a", kind: "method", range: { byteStart: 10, byteEnd: 18, lineStart: 2, lineEnd: 2 }, text: "a() {}", symbolIds: [parentA.id] },
        { ...base, id: "chunk:b", kind: "method", range: { byteStart: 31, byteEnd: 39, lineStart: 5, lineEnd: 5 }, text: "b() {}", symbolIds: [parentB.id] },
      ] as ChunkRecord[],
      { [parentA.id]: parentA, [parentB.id]: parentB },
    )

    expect(chunks[0].nextSiblingChunkId).toBeUndefined()
    expect(chunks[1].previousSiblingChunkId).toBeUndefined()
  })

  test("does not link chunks from different methods in the same class as siblings", () => {
    const parent: SymbolRecord = { id: "sym:class:A", name: "A", kind: "class", filePath: "src/a.ts", range: { byteStart: 0, byteEnd: 60, lineStart: 1, lineEnd: 5 }, childSymbolIds: ["sym:method:a", "sym:method:b"] }
    const methodA: SymbolRecord = { id: "sym:method:a", name: "a", kind: "method", filePath: "src/a.ts", range: { byteStart: 10, byteEnd: 25, lineStart: 2, lineEnd: 2 }, parentSymbolId: parent.id, childSymbolIds: [] }
    const methodB: SymbolRecord = { id: "sym:method:b", name: "b", kind: "method", filePath: "src/a.ts", range: { byteStart: 30, byteEnd: 45, lineStart: 3, lineEnd: 3 }, parentSymbolId: parent.id, childSymbolIds: [] }
    const chunks = attachTopology(
      [
        { ...base, id: "chunk:a", kind: "method", range: { byteStart: 12, byteEnd: 20, lineStart: 2, lineEnd: 2 }, text: "a() {}", symbolIds: [parent.id, methodA.id] },
        { ...base, id: "chunk:b", kind: "method", range: { byteStart: 32, byteEnd: 40, lineStart: 3, lineEnd: 3 }, text: "b() {}", symbolIds: [parent.id, methodB.id] },
      ] as ChunkRecord[],
      { [parent.id]: parent, [methodA.id]: methodA, [methodB.id]: methodB },
    )

    expect(chunks[0].nextSiblingChunkId).toBeUndefined()
    expect(chunks[1].previousSiblingChunkId).toBeUndefined()
  })

  test("links symbol-less siblings only within the same direct parent", () => {
    const chunks = attachTopology(
      [
        { ...base, id: "parent:a", kind: "block", range: { byteStart: 0, byteEnd: 40, lineStart: 1, lineEnd: 4 }, text: "parent a", symbolIds: [] },
        { ...base, id: "child:a1", kind: "block", range: { byteStart: 5, byteEnd: 10, lineStart: 2, lineEnd: 2 }, text: "child a1", symbolIds: [] },
        { ...base, id: "child:a2", kind: "block", range: { byteStart: 12, byteEnd: 18, lineStart: 3, lineEnd: 3 }, text: "child a2", symbolIds: [] },
        { ...base, id: "parent:b", kind: "block", range: { byteStart: 41, byteEnd: 60, lineStart: 5, lineEnd: 6 }, text: "parent b", symbolIds: [] },
      ] as ChunkRecord[],
      {},
    )
    const byId = Object.fromEntries(chunks.map((chunk) => [chunk.id, chunk]))

    expect(byId["parent:a"].previousSiblingChunkId).toBeUndefined()
    expect(byId["parent:a"].nextSiblingChunkId).toBe("parent:b")
    expect(byId["child:a1"].previousSiblingChunkId).toBeUndefined()
    expect(byId["child:a1"].nextSiblingChunkId).toBe("child:a2")
    expect(byId["child:a2"].previousSiblingChunkId).toBe("child:a1")
    expect(byId["child:a2"].nextSiblingChunkId).toBeUndefined()
    expect(byId["parent:b"].previousSiblingChunkId).toBe("parent:a")
  })

  test("does not link parent or child chunks across files", () => {
    const chunks = attachTopology(
      [
        { ...base, id: "chunk:a", kind: "block", range: { byteStart: 0, byteEnd: 40, lineStart: 1, lineEnd: 4 }, text: "a", symbolIds: [] },
        { ...base, id: "chunk:b", filePath: "src/b.ts", kind: "block", range: { byteStart: 10, byteEnd: 20, lineStart: 2, lineEnd: 2 }, text: "b", symbolIds: [] },
      ] as ChunkRecord[],
      {},
    )

    expect(chunks[0].parentChunkId).toBeUndefined()
    expect(chunks[0].childChunkIds).toEqual([])
    expect(chunks[1].parentChunkId).toBeUndefined()
    expect(chunks[1].childChunkIds).toEqual([])
  })

  test("extracts enclosing class and assigns it to contained chunks", () => {
    const source = "class A {\n  a() {}\n}\n"
    const symbols = extractSymbols({
      filePath: "src/a.ts",
      source,
      nodes: [{ type: "class_declaration", startIndex: 0, endIndex: 20, children: [{ type: "method_definition", startIndex: 12, endIndex: 18, children: [] }] }],
    })
    const [chunk] = assignSymbolsToChunks(
      [{ ...base, id: "chunk:method", kind: "method", range: { byteStart: 12, byteEnd: 18, lineStart: 2, lineEnd: 2 }, text: "a() {}", symbolIds: [] } as ChunkRecord],
      Object.fromEntries(symbols.map((symbol) => [symbol.id, symbol])),
    )

    expect(symbols.map((symbol) => `${symbol.kind}:${symbol.name}`)).toEqual(["class:A", "method:a"])
    expect(symbols[0].childSymbolIds).toEqual(["src/a.ts:method:a:12:18"])
    expect(symbols[1].parentSymbolId).toBe("src/a.ts:class:A:0:20")
    expect(chunk.symbolIds).toEqual(["src/a.ts:class:A:0:20", "src/a.ts:method:a:12:18"])
  })

  test("does not assign symbols from another file with overlapping byte ranges", () => {
    const symbol: SymbolRecord = { id: "src/a.ts:class:A:0:40", name: "A", kind: "class", filePath: "src/a.ts", range: { byteStart: 0, byteEnd: 40, lineStart: 1, lineEnd: 4 }, childSymbolIds: [] }
    const [chunk] = assignSymbolsToChunks(
      [{ ...base, id: "chunk:b", filePath: "src/b.ts", kind: "method", range: { byteStart: 10, byteEnd: 20, lineStart: 2, lineEnd: 2 }, text: "b() {}", symbolIds: [] } as ChunkRecord],
      { [symbol.id]: symbol },
    )

    expect(chunk.symbolIds).toEqual([])
  })

  test("uses UTF-8 byte ranges when extracting names and parent text", () => {
    const source = "const café = true\nclass Éclair {\n  goût() {}\n}\n"
    const classStart = new TextEncoder().encode("const café = true\n").length
    const methodStart = new TextEncoder().encode("const café = true\nclass Éclair {\n  ").length
    const methodEnd = new TextEncoder().encode("const café = true\nclass Éclair {\n  goût() {}").length
    const classEnd = new TextEncoder().encode(source.trimEnd()).length
    const symbols = extractSymbols({
      filePath: "src/a.ts",
      source,
      nodes: [{ type: "class_declaration", startIndex: classStart, endIndex: classEnd, children: [{ type: "method_definition", startIndex: methodStart, endIndex: methodEnd, children: [] }] }],
    })
    const result = expandWithParentContext({
      chunk: { ...base, id: "chunk:method", kind: "method", range: { byteStart: methodStart, byteEnd: methodEnd, lineStart: 3, lineEnd: 3 }, text: "goût() {}", symbolIds: symbols.map((symbol) => symbol.id) } as ChunkRecord,
      symbols: Object.fromEntries(symbols.map((symbol) => [symbol.id, symbol])),
      source,
      maxContextChars: 100,
    })

    expect(symbols.map((symbol) => `${symbol.kind}:${symbol.name}`)).toEqual(["class:Éclair", "method:goût"])
    expect(result.parentText).toBe("class Éclair {\n  goût() {}\n}")
    expect(result.breadcrumbs).toEqual(["class Éclair", "method goût"])
  })

  test("adds parent context when it fits budget", () => {
    const chunk = { ...base, id: "chunk:method", kind: "method", range: { byteStart: 12, byteEnd: 20, lineStart: 2, lineEnd: 2 }, text: "a() {}", symbolIds: ["sym:class:A"] } as ChunkRecord
    const parent: SymbolRecord = { id: "sym:class:A", name: "A", kind: "class", filePath: "src/a.ts", range: { byteStart: 0, byteEnd: 28, lineStart: 1, lineEnd: 3 }, childSymbolIds: [] }
    const result = expandWithParentContext({ chunk, symbols: { [parent.id]: parent }, source: "class A {\n  a() {}\n}\n", maxContextChars: 100 })

    expect(result.parentText).toBe("class A {\n  a() {}\n}")
    expect(result.parentRange).toEqual(parent.range)
    expect(result.breadcrumbs).toEqual(["class A"])
  })

  test("returns breadcrumb and excerpt when parent exceeds budget", () => {
    const chunk = { ...base, id: "chunk:method", kind: "method", range: { byteStart: 30, byteEnd: 38, lineStart: 3, lineEnd: 3 }, text: "a() {}", symbolIds: ["sym:class:A"] } as ChunkRecord
    const parent: SymbolRecord = { id: "sym:class:A", name: "A", kind: "class", filePath: "src/a.ts", range: { byteStart: 0, byteEnd: 46, lineStart: 1, lineEnd: 4 }, childSymbolIds: [] }
    const result = expandWithParentContext({ chunk, symbols: { [parent.id]: parent }, source: "class A {\n  longField = true\n  a() {}\n}\n", maxContextChars: 20 })

    expect(result.parentText).toContain("class A")
    expect(result.parentText!.length).toBeLessThanOrEqual(20)
    expect(result.parentRange).toEqual(parent.range)
    expect(result.breadcrumbs).toEqual(["class A"])
  })

  test("returns excerpt when whitespace-heavy parent text exceeds budget", () => {
    const source = "class A {\n\n\n\n\n  a() {}\n}\n"
    const chunk = { ...base, id: "chunk:method", kind: "method", range: { byteStart: 17, byteEnd: 25, lineStart: 6, lineEnd: 6 }, text: "a() {}", symbolIds: ["sym:class:A"] } as ChunkRecord
    const parent: SymbolRecord = { id: "sym:class:A", name: "A", kind: "class", filePath: "src/a.ts", range: { byteStart: 0, byteEnd: 28, lineStart: 1, lineEnd: 7 }, childSymbolIds: [] }
    const result = expandWithParentContext({ chunk, symbols: { [parent.id]: parent }, source, maxContextChars: 20 })

    expect(result.parentText).not.toBe(source.trimEnd())
    expect(result.parentText).toContain("class A")
    expect(result.parentText!.length).toBeLessThanOrEqual(20)
  })
})
