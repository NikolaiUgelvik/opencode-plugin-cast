import { describe, expect, test } from "bun:test"
import { castChunks, type SyntaxNode } from "./cast.js"

const IDENTIFIER_PATTERN = /[A-Za-z_$][\w$]*/

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
      chunking: { overlap: 0, expansion: false, minSemanticNonWhitespaceChars: 8 },
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
      chunking: { overlap: 0, expansion: false, minSemanticNonWhitespaceChars: 8 },
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
        node("class_declaration", 0, source.length, [
          node("method_definition", 10, 19),
          node("method_definition", 19, 27),
        ]),
      ]),
      maxNonWhitespaceChars: 8,
      chunking: { overlap: 0, expansion: false, minSemanticNonWhitespaceChars: 8 },
    })

    expect(chunks.map((chunk) => chunk.text)).toEqual(["  a() {}\n", "  b() {}"])
    expect(chunks.every((chunk) => chunk.parentChunkId === "src/a.ts:0:30")).toBe(true)
  })

  test("falls back for oversized childless roots", () => {
    // biome-ignore lint/security/noSecrets: fixture text is not a credential.
    const source = "abcdef\nghijkl\n"
    const chunks = castChunks({
      filePath: "src/a.txt",
      language: "text",
      source,
      root: node("document", 0, source.length),
      maxNonWhitespaceChars: 4,
      chunking: { overlap: 0, expansion: false, minSemanticNonWhitespaceChars: 8 },
    })

    expect(chunks.length).toBeGreaterThan(0)
    expect(chunks.every((chunk) => chunk.kind === "fallback")).toBe(true)
    expect(chunks.map((chunk) => chunk.text).join("")).toBe(source)
    expect(chunks.every((chunk) => chunk.nonWhitespaceChars <= 4)).toBe(true)
  })

  test("links siblings across emitted recursive chunk groups", () => {
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
      chunking: { overlap: 0, expansion: false, minSemanticNonWhitespaceChars: 8 },
    })

    const byText = Object.fromEntries(chunks.map((chunk) => [chunk.text, chunk]))

    expect(byText["  m() {}\n"].previousSiblingChunkId).toBe(byText["a()\n"].id)
    expect(byText["  m() {}\n"].nextSiblingChunkId).toBe(byText["  n() {}\n"].id)
    expect(byText["  n() {}\n"].previousSiblingChunkId).toBe(byText["  m() {}\n"].id)
    expect(byText["  n() {}\n"].nextSiblingChunkId).toBe(byText["z()\n"].id)
    expect(byText["a()\n"].nextSiblingChunkId).toBe(byText["  m() {}\n"].id)
    expect(byText["z()\n"].previousSiblingChunkId).toBe(byText["  n() {}\n"].id)
  })

  test("remaps fallback sibling links for oversized childless nodes", () => {
    const source = "wrap {\n  abcdef\n  ghijkl\n}\n"
    const chunks = castChunks({
      filePath: "src/a.txt",
      language: "text",
      source,
      root: node("document", 0, source.length, [node("block", 7, 25)]),
      maxNonWhitespaceChars: 4,
      chunking: { overlap: 0, expansion: false, minSemanticNonWhitespaceChars: 8 },
    })
    const chunkIds = new Set(chunks.map((chunk) => chunk.id))

    expect(chunks).toHaveLength(4)
    expect(chunks.every((chunk) => chunk.kind === "fallback")).toBe(true)
    expect(chunks.map((chunk) => chunk.range.byteStart)).toEqual([7, 13, 16, 22])
    expect(chunks.map((chunk) => chunk.range.byteEnd)).toEqual([13, 16, 22, 25])
    expect(chunks.every((chunk) => !chunk.previousSiblingChunkId || chunkIds.has(chunk.previousSiblingChunkId))).toBe(
      true,
    )
    expect(chunks.every((chunk) => !chunk.nextSiblingChunkId || chunkIds.has(chunk.nextSiblingChunkId))).toBe(true)
  })

  test("merges adjacent child windows after oversized recursion", () => {
    const source = "class A {\n  a() {}\n  b() {}\n  c() {}\n}\n"
    const chunks = castChunks({
      filePath: "src/a.ts",
      language: "typescript",
      source,
      root: node("program", 0, source.length, [
        node("class_declaration", 0, source.length, [
          node("method_definition", 10, 19),
          node("method_definition", 19, 28),
          node("method_definition", 28, 36),
        ]),
      ]),
      maxNonWhitespaceChars: 14,
      chunking: { overlap: 0, expansion: false, minSemanticNonWhitespaceChars: 8 },
    })

    expect(chunks.map((chunk) => chunk.text)).toEqual(["  a() {}\n  b() {}\n", "  c() {}"])
    expect(chunks[0].nextSiblingChunkId).toBe(chunks[1].id)
    expect(chunks[1].previousSiblingChunkId).toBe(chunks[0].id)
  })

  test("merges trivial syntax fragments with adjacent semantic windows", () => {
    const source = "export function example() {\n  return value\n}\n"
    const chunks = castChunks({
      filePath: "src/a.ts",
      language: "typescript",
      source,
      root: node("program", 0, source.length, [
        node("export", 0, 6),
        node("function_declaration", 7, 42, [
          node("identifier", 16, 23),
          node("statement_block", 26, 42, [node("return_statement", 30, 42), node("}", 41, 42)]),
        ]),
      ]),
      maxNonWhitespaceChars: 35,
      chunking: { overlap: 0, expansion: false, minSemanticNonWhitespaceChars: 8 },
    })

    expect(chunks.map((chunk) => chunk.text)).not.toContain("export")
    expect(chunks.map((chunk) => chunk.text)).not.toContain("}")
    expect(chunks.every((chunk) => IDENTIFIER_PATTERN.test(chunk.text))).toBe(true)
  })

  test("keeps trivial chunks only when the budget prevents merging", () => {
    const source = "a)"
    const chunks = castChunks({
      filePath: "src/a.ts",
      language: "typescript",
      source,
      root: node("program", 0, source.length, [node("identifier", 0, 1), node(")", 1, 2)]),
      maxNonWhitespaceChars: 1,
      chunking: { overlap: 0, expansion: false, minSemanticNonWhitespaceChars: 8 },
    })

    expect(chunks.map((chunk) => chunk.text).join("")).toBe(source)
  })

  test("does not merge trivial punctuation when semantic window is already at budget", () => {
    const source = "abcdefgh)"
    const chunks = castChunks({
      filePath: "src/a.ts",
      language: "typescript",
      source,
      root: node("program", 0, source.length, [node("identifier", 0, 8), node(")", 8, 9)]),
      maxNonWhitespaceChars: 8,
      chunking: { overlap: 0, expansion: false, minSemanticNonWhitespaceChars: 8 },
    })

    expect(chunks.map((chunk) => chunk.text)).toEqual(["abcdefgh", ")"])
    expect(chunks.every((chunk) => chunk.nonWhitespaceChars <= 8)).toBe(true)
  })

  test("keeps call callee with the following argument window", () => {
    const source = 'describe("language registry", () => {\n  test("loads", () => {})\n})\n'
    const argumentsStart = source.indexOf("(")
    const callbackStart = source.indexOf("() =>")
    const testStart = source.indexOf("  test")
    const chunks = castChunks({
      filePath: "src/a.ts",
      language: "typescript",
      source,
      root: node("program", 0, source.length, [
        node("call_expression", 0, source.length, [
          node("identifier", 0, argumentsStart),
          node("arguments", argumentsStart, source.length, [
            node("string", argumentsStart + 1, callbackStart - 2),
            node("arrow_function", callbackStart, source.length - 2, [
              node("call_expression", testStart, source.length - 3),
            ]),
          ]),
        ]),
      ]),
      maxNonWhitespaceChars: 32,
      chunking: { overlap: 0, expansion: false, minSemanticNonWhitespaceChars: 8 },
    })

    expect(chunks.map((chunk) => chunk.text)).not.toContain("describe")
    expect(chunks[0].text).toStartWith("describe(")
  })

  test("defaults to no overlap", () => {
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
      maxNonWhitespaceChars: 13,
      chunking: { overlap: 0, expansion: false, minSemanticNonWhitespaceChars: 8 },
    })

    expect(chunks.map((chunk) => chunk.text)).toEqual(["function a() {}\n", "function b() {}\n", "function c() {}\n"])
  })

  test("adds adjacent window overlap when configured", () => {
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
      maxNonWhitespaceChars: 13,
      chunking: { overlap: 1, expansion: false, minSemanticNonWhitespaceChars: 8 },
    })

    expect(chunks.map((chunk) => chunk.text)).toEqual([
      "function a() {}\nfunction b() {}\n",
      "function a() {}\nfunction b() {}\nfunction c() {}\n",
      "function b() {}\nfunction c() {}\n",
    ])
  })

  test("keeps overlapped chunk ids unique when two windows expand to the same range", () => {
    const source = "function a() {}\nfunction b() {}\n"
    const chunks = castChunks({
      filePath: "src/a.ts",
      language: "typescript",
      source,
      root: node("program", 0, source.length, [
        node("function_declaration", 0, 16),
        node("function_declaration", 16, 32),
      ]),
      maxNonWhitespaceChars: 13,
      chunking: { overlap: 1, expansion: false, minSemanticNonWhitespaceChars: 8 },
    })

    expect(chunks.map((chunk) => chunk.text)).toEqual([
      "function a() {}\nfunction b() {}\n",
      "function a() {}\nfunction b() {}\n",
    ])
    expect(new Set(chunks.map((chunk) => chunk.id)).size).toBe(chunks.length)
    expect(chunks.every((chunk) => chunk.previousSiblingChunkId !== chunk.id)).toBe(true)
    expect(chunks.every((chunk) => chunk.nextSiblingChunkId !== chunk.id)).toBe(true)
  })

  test("clears parent metadata when overlap crosses recursive boundaries", () => {
    const source = "a()\nclass A {\n  m() {}\n}\nz()\n"
    const chunks = castChunks({
      filePath: "src/a.ts",
      language: "typescript",
      source,
      root: node("program", 0, source.length, [
        node("call_expression", 0, 4),
        node("class_declaration", 4, 25, [node("method_definition", 14, 23)]),
        node("call_expression", 25, 29),
      ]),
      maxNonWhitespaceChars: 8,
      chunking: { overlap: 1, expansion: false, minSemanticNonWhitespaceChars: 1 },
    })

    expect(chunks.map((chunk) => chunk.parentChunkId)).toEqual([undefined, undefined, undefined])
  })

  test("preserves nested parent boundaries when checking overlapped parent metadata", () => {
    const source = "class A {\n  p()\n  method {\n    m1\n    m2\n  }\n  q()\n}\n"
    const methodStart = source.indexOf("  method")
    const methodEnd = source.indexOf("  q()")
    const pStart = source.indexOf("  p()")
    const m1Start = source.indexOf("    m1")
    const m2Start = source.indexOf("    m2")
    const qStart = source.indexOf("  q()")
    const classEnd = source.length
    const chunks = castChunks({
      filePath: "src/a.ts",
      language: "typescript",
      source,
      root: node("program", 0, source.length, [
        node("class_declaration", 0, classEnd, [
          node("method_definition", pStart, methodStart),
          node("method_definition", methodStart, methodEnd, [
            node("expression_statement", m1Start, m2Start),
            node("expression_statement", m2Start, methodEnd),
          ]),
          node("method_definition", qStart, classEnd - 2),
        ]),
      ]),
      maxNonWhitespaceChars: 4,
      chunking: { overlap: 1, expansion: false, minSemanticNonWhitespaceChars: 1 },
    })

    const innerParentChunkId = `src/a.ts:${methodStart}:${methodEnd}`
    const expandedInnerChunk = chunks.find(
      (chunk) => chunk.text.includes("p()") && chunk.text.includes("m1") && chunk.text.includes("m2"),
    )

    expect(expandedInnerChunk?.parentChunkId).not.toBe(innerParentChunkId)
  })

  test("clears parent metadata when adjacent non-overlap merge escapes nested parent", () => {
    const source = "class A {\n  method {\n    m1\n  }\n  q\n}\n"
    const methodStart = source.indexOf("  method")
    const methodEnd = source.indexOf("  q")
    const m1Start = source.indexOf("    m1")
    const methodCloseStart = source.indexOf("  }\n")
    const qStart = source.indexOf("  q")
    const classEnd = source.length
    const chunks = castChunks({
      filePath: "src/a.ts",
      language: "typescript",
      source,
      root: node("program", 0, source.length, [
        node("class_declaration", 0, classEnd, [
          node("method_definition", methodStart, methodEnd, [
            node("expression_statement", m1Start, methodCloseStart),
            node("}", methodCloseStart, methodEnd),
          ]),
          node("property_identifier", qStart, classEnd - 2),
        ]),
      ]),
      maxNonWhitespaceChars: 4,
      chunking: { overlap: 0, expansion: false, minSemanticNonWhitespaceChars: 1 },
    })

    const innerParentChunkId = `src/a.ts:${methodStart}:${methodEnd}`
    const escapedChunk = chunks.find((chunk) => chunk.text.includes("m1") && chunk.text.includes("q"))

    expect(escapedChunk).toBeDefined()
    expect(escapedChunk?.parentChunkId).not.toBe(innerParentChunkId)
  })
})
