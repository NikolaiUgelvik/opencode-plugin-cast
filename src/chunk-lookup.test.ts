import { describe, expect, test } from "bun:test"
import { getChunkById } from "./chunk-lookup.js"
import { createEmptyIndex } from "./store.js"

describe("chunk lookup", () => {
  test("returns chunk with labeled topology and selected related context", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
      diagnostics: ["index diagnostic"],
    })
    index.metadata.status = "ready"
    const source = "class Parser {\n  previous() {}\n  parse() {\n    return value\n  }\n}\n"
    const parentText = source.trimEnd()
    const previousText = "previous() {}"
    const targetText = "parse() {\n    return value\n  }"
    const childText = "return value"
    const previousStart = source.indexOf(previousText)
    const targetStart = source.indexOf(targetText)
    const childStart = source.indexOf(childText)

    index.symbols.classParser = {
      id: "classParser",
      name: "Parser",
      kind: "class",
      filePath: "src/parser.ts",
      range: { byteStart: 0, byteEnd: parentText.length, lineStart: 1, lineEnd: 6 },
      childSymbolIds: ["methodParse"],
    }
    index.symbols.methodPrevious = {
      id: "methodPrevious",
      name: "previous",
      kind: "method",
      filePath: "src/parser.ts",
      range: { byteStart: previousStart, byteEnd: previousStart + previousText.length, lineStart: 2, lineEnd: 2 },
      parentSymbolId: "classParser",
      childSymbolIds: [],
    }
    index.symbols.methodParse = {
      id: "methodParse",
      name: "parse",
      kind: "method",
      filePath: "src/parser.ts",
      range: { byteStart: targetStart, byteEnd: targetStart + targetText.length, lineStart: 3, lineEnd: 5 },
      parentSymbolId: "classParser",
      childSymbolIds: [],
    }
    index.chunks.parent = {
      id: "parent",
      filePath: "src/parser.ts",
      language: "typescript",
      kind: "class",
      range: { byteStart: 0, byteEnd: parentText.length, lineStart: 1, lineEnd: 6 },
      text: parentText,
      nonWhitespaceChars: 45,
      nodeTypes: [],
      symbolIds: ["classParser"],
      childChunkIds: ["previous", "target"],
    }
    index.chunks.previous = {
      id: "previous",
      filePath: "src/parser.ts",
      language: "typescript",
      kind: "method",
      range: { byteStart: previousStart, byteEnd: previousStart + previousText.length, lineStart: 2, lineEnd: 2 },
      text: previousText,
      nonWhitespaceChars: 12,
      nodeTypes: [],
      symbolIds: ["classParser", "methodPrevious"],
      parentChunkId: "parent",
      childChunkIds: [],
      nextSiblingChunkId: "target",
    }
    index.chunks.target = {
      id: "target",
      filePath: "src/parser.ts",
      language: "typescript",
      kind: "method",
      range: { byteStart: targetStart, byteEnd: targetStart + targetText.length, lineStart: 3, lineEnd: 5 },
      text: targetText,
      nonWhitespaceChars: 25,
      nodeTypes: [],
      symbolIds: ["classParser", "methodParse"],
      parentChunkId: "parent",
      childChunkIds: ["child"],
      previousSiblingChunkId: "previous",
    }
    index.chunks.child = {
      id: "child",
      filePath: "src/parser.ts",
      language: "typescript",
      kind: "block",
      range: { byteStart: childStart, byteEnd: childStart + childText.length, lineStart: 4, lineEnd: 4 },
      text: childText,
      nonWhitespaceChars: 11,
      nodeTypes: [],
      symbolIds: ["classParser", "methodParse"],
      parentChunkId: "target",
      childChunkIds: [],
    }

    const output = await getChunkById({
      index,
      input: { id: "target", maxContextChars: 12 },
      readSource: async () => source,
    })

    expect(output.diagnostics).toEqual(["index diagnostic"])
    expect(output.chunk).toMatchObject({
      filePath: "src/parser.ts",
      language: "typescript",
      range: { byteStart: targetStart, byteEnd: targetStart + targetText.length, lineStart: 3, lineEnd: 5 },
      kind: "method",
      breadcrumbs: ["class Parser", "method parse"],
      text: targetText,
      parentRange: { byteStart: 0, byteEnd: parentText.length, lineStart: 1, lineEnd: 6 },
      topology: {
        chunk: { id: "target", label: "method parse", range: "src/parser.ts:3-5" },
        parent: { id: "parent", label: "class Parser", range: "src/parser.ts:1-6" },
        children: [{ id: "child", label: "block return value", range: "src/parser.ts:4" }],
        previousSibling: { id: "previous", label: "method previous", range: "src/parser.ts:2" },
        symbols: ["class Parser", "method parse"],
      },
      related: {
        parent: { id: "parent", label: "class Parser", range: "src/parser.ts:1-6", text: "class Parser" },
        previousSibling: { id: "previous", label: "method previous", range: "src/parser.ts:2", text: "previous() {" },
        children: [{ id: "child", label: "block return value", range: "src/parser.ts:4", text: "return value" }],
      },
    })
    expect(output.chunk?.parentText).toStartWith("class Parser")
  })

  test("returns a diagnostic when the chunk id is missing", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
      diagnostics: ["index diagnostic"],
    })
    index.metadata.status = "ready"

    const output = await getChunkById({
      index,
      input: { id: "missing" },
      readSource: async () => "",
    })

    expect(output.chunk).toBeUndefined()
    expect(output.diagnostics).toEqual(["index diagnostic", "chunk not found: missing"])
  })

  test("reports an unavailable index before treating the chunk id as missing", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })

    const output = await getChunkById({
      index,
      input: { id: "missing" },
      readSource: async () => "",
    })

    expect(output.chunk).toBeUndefined()
    expect(output.diagnostics).toEqual(["index unavailable: empty"])
  })

  test("keeps symbol breadcrumbs when parent context is disabled", async () => {
    const index = createEmptyIndex({
      projectId: "p",
      worktree: "/repo",
      cacheKey: "key",
      maxChunkNonWhitespaceChars: 2000,
    })
    index.symbols.classParser = {
      id: "classParser",
      name: "Parser",
      kind: "class",
      filePath: "src/parser.ts",
      range: { byteStart: 0, byteEnd: 30, lineStart: 1, lineEnd: 3 },
      childSymbolIds: ["methodParse"],
    }
    index.symbols.methodParse = {
      id: "methodParse",
      name: "parse",
      kind: "method",
      filePath: "src/parser.ts",
      range: { byteStart: 15, byteEnd: 25, lineStart: 2, lineEnd: 2 },
      parentSymbolId: "classParser",
      childSymbolIds: [],
    }
    index.chunks.parent = {
      id: "parent",
      filePath: "src/parser.ts",
      language: "typescript",
      kind: "class",
      range: { byteStart: 0, byteEnd: 30, lineStart: 1, lineEnd: 3 },
      text: "class Parser {}",
      nonWhitespaceChars: 13,
      nodeTypes: [],
      symbolIds: ["classParser"],
      childChunkIds: ["target"],
    }
    index.chunks.target = {
      id: "target",
      filePath: "src/parser.ts",
      language: "typescript",
      kind: "method",
      range: { byteStart: 15, byteEnd: 25, lineStart: 2, lineEnd: 2 },
      text: "parse() {}",
      nonWhitespaceChars: 9,
      nodeTypes: [],
      symbolIds: ["classParser", "methodParse"],
      parentChunkId: "parent",
      childChunkIds: [],
    }

    const output = await getChunkById({
      index,
      input: { id: "target", includeParents: false },
      readSource: async () => "class Parser {\n  parse() {}\n}\n",
    })

    expect(output.chunk?.breadcrumbs).toEqual(["class Parser", "method parse"])
    expect(output.chunk?.parentText).toBeUndefined()
    expect(output.chunk?.related.parent).toBeUndefined()
  })
})
