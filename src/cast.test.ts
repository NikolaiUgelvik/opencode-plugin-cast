import { describe, expect, test } from "bun:test"
import { castChunks, type SyntaxNode } from "./cast.js"

function node(type: string, startIndex: number, endIndex: number, children: SyntaxNode[] = []): SyntaxNode {
  return { type, startIndex, endIndex, children }
}

describe("castChunks", () => {
  test("keeps small files as one file chunk", () => {
    const chunks = castChunks({
      filePath: "src/a.ts",
      language: "typescript",
      source: "class A {}\n",
      root: node("program", 0, 11, [node("class_declaration", 0, 10)]),
      maxNonWhitespaceChars: 20,
    })

    expect(chunks).toHaveLength(1)
    expect(chunks[0].kind).toBe("file")
    expect(chunks[0].text).toBe("class A {}\n")
  })

  test("greedily merges adjacent siblings up to budget", () => {
    const source = "function a() {}\nfunction b() {}\nfunction c() {}\n"
    const chunks = castChunks({
      filePath: "src/a.ts",
      language: "typescript",
      source,
      root: node("program", 0, source.length, [
        node("function_declaration", 0, 16),
        node("function_declaration", 16, 32),
        node("function_declaration", 32, 48),
      ]),
      maxNonWhitespaceChars: 26,
    })

    expect(chunks.map((chunk) => chunk.text)).toEqual(["function a() {}\nfunction b() {}\n", "function c() {}\n"])
    expect(chunks[0].nodeTypes).toEqual(["function_declaration", "function_declaration"])
    expect(chunks[0].nextSiblingChunkId).toBe(chunks[1].id)
  })

  test("recurses into oversized nodes", () => {
    const source = "class A {\n  a() {}\n  b() {}\n}\n"
    const chunks = castChunks({
      filePath: "src/a.ts",
      language: "typescript",
      source,
      root: node("program", 0, source.length, [
        node("class_declaration", 0, source.length, [node("method_definition", 10, 19), node("method_definition", 19, 27)]),
      ]),
      maxNonWhitespaceChars: 8,
    })

    expect(chunks.map((chunk) => chunk.text)).toEqual(["  a() {}\n", "  b() {}"])
    expect(chunks.every((chunk) => chunk.parentChunkId === "src/a.ts:0:30")).toBe(true)
  })

  test("falls back for oversized childless roots", () => {
    const source = "abcdef\nghijkl\n"
    const chunks = castChunks({
      filePath: "src/a.txt",
      language: "text",
      source,
      root: node("document", 0, source.length),
      maxNonWhitespaceChars: 4,
    })

    expect(chunks.length).toBeGreaterThan(0)
    expect(chunks.every((chunk) => chunk.kind === "fallback")).toBe(true)
    expect(chunks.map((chunk) => chunk.text).join("")).toBe(source)
    expect(chunks.every((chunk) => chunk.nonWhitespaceChars <= 4)).toBe(true)
  })

  test("links siblings only within recursive chunk groups", () => {
    const source = "a()\nclass A {\n  m() {}\n  n() {}\n}\nz()\n"
    const chunks = castChunks({
      filePath: "src/a.ts",
      language: "typescript",
      source,
      root: node("program", 0, source.length, [
        node("call_expression", 0, 4),
        node("class_declaration", 4, 34, [node("method_definition", 14, 23), node("method_definition", 23, 32)]),
        node("call_expression", 34, 38),
      ]),
      maxNonWhitespaceChars: 8,
    })

    const byText = Object.fromEntries(chunks.map((chunk) => [chunk.text, chunk]))

    expect(byText["  m() {}\n"].previousSiblingChunkId).toBeUndefined()
    expect(byText["  m() {}\n"].nextSiblingChunkId).toBe(byText["  n() {}\n"].id)
    expect(byText["  n() {}\n"].previousSiblingChunkId).toBe(byText["  m() {}\n"].id)
    expect(byText["  n() {}\n"].nextSiblingChunkId).toBeUndefined()
    expect(byText["a()\n"].nextSiblingChunkId).not.toBe(byText["  m() {}\n"].id)
    expect(byText["z()\n"].previousSiblingChunkId).not.toBe(byText["  n() {}\n"].id)
  })

  test("remaps fallback sibling links for oversized childless nodes", () => {
    const source = "wrap {\n  abcdef\n  ghijkl\n}\n"
    const chunks = castChunks({
      filePath: "src/a.txt",
      language: "text",
      source,
      root: node("document", 0, source.length, [node("block", 7, 25)]),
      maxNonWhitespaceChars: 4,
    })
    const chunkIds = new Set(chunks.map((chunk) => chunk.id))

    expect(chunks).toHaveLength(4)
    expect(chunks.map((chunk) => chunk.range.byteStart)).toEqual([7, 13, 16, 22])
    expect(chunks.map((chunk) => chunk.range.byteEnd)).toEqual([13, 16, 22, 25])
    expect(chunks.every((chunk) => !chunk.previousSiblingChunkId || chunkIds.has(chunk.previousSiblingChunkId))).toBe(true)
    expect(chunks.every((chunk) => !chunk.nextSiblingChunkId || chunkIds.has(chunk.nextSiblingChunkId))).toBe(true)
  })
})
