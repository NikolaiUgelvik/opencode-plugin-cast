import { describe, expect, test } from "bun:test"
import { fallbackChunks } from "./fallback.js"
import { nonWhitespaceLength, rangeForSlice } from "./range.js"

describe("fallbackChunks", () => {
  test("keeps small files as one chunk", () => {
    const [chunk] = fallbackChunks({
      filePath: "src/a.ts",
      language: "typescript",
      text: "const a = 1\n",
      maxNonWhitespaceChars: 20,
    })

    expect(chunk.text).toBe("const a = 1\n")
    expect(chunk.kind).toBe("fallback")
    expect(chunk.range).toEqual({ byteStart: 0, byteEnd: 12, lineStart: 1, lineEnd: 2 })
    expect(chunk.nonWhitespaceChars).toBe(8)
  })

  test("splits large files on line boundaries", () => {
    const chunks = fallbackChunks({
      filePath: "src/a.ts",
      language: "typescript",
      text: "aaaa\nbbbb\ncccc\n",
      maxNonWhitespaceChars: 8,
    })

    expect(chunks.map((chunk) => chunk.text)).toEqual(["aaaa\nbbbb\n", "cccc\n"])
    expect(chunks.map((chunk) => chunk.range.lineStart)).toEqual([1, 3])
    expect(chunks.map((chunk) => chunk.range.lineEnd)).toEqual([3, 4])
    expect(chunks[0].nextSiblingChunkId).toBe(chunks[1].id)
    expect(chunks[1].previousSiblingChunkId).toBe(chunks[0].id)
  })

  test("uses UTF-8 byte ranges for multibyte text", () => {
    const text = "é\n"
    const [chunk] = fallbackChunks({ filePath: "src/a.ts", language: "typescript", text, maxNonWhitespaceChars: 20 })

    expect(chunk.range).toEqual({
      byteStart: 0,
      byteEnd: new TextEncoder().encode(text).length,
      lineStart: 1,
      lineEnd: 2,
    })
    expect(chunk.id).toBe(`src/a.ts:0:${new TextEncoder().encode(text).length}`)
  })

  test("splits long single lines within the non-whitespace budget", () => {
    const text = "abc defghi\n"
    const chunks = fallbackChunks({ filePath: "src/a.ts", language: "typescript", text, maxNonWhitespaceChars: 3 })

    expect(chunks.every((chunk) => chunk.nonWhitespaceChars <= 3)).toBe(true)
    expect(chunks.map((chunk) => chunk.text).join("")).toBe(text)
    expect(chunks[0].range.byteStart).toBe(0)
    expect(chunks.at(-1)?.range.byteEnd).toBe(new TextEncoder().encode(text).length)
    expect(chunks.slice(1).every((chunk, index) => chunk.range.byteStart === chunks[index].range.byteEnd)).toBe(true)
    expect(chunks.map((chunk) => chunk.id)).toEqual(
      chunks.map((chunk) => `src/a.ts:${chunk.range.byteStart}:${chunk.range.byteEnd}`),
    )
    expect(chunks.map((chunk) => chunk.previousSiblingChunkId)).toEqual(chunks.map((_, index) => chunks[index - 1]?.id))
    expect(chunks.map((chunk) => chunk.nextSiblingChunkId)).toEqual(chunks.map((_, index) => chunks[index + 1]?.id))
  })

  test("counts astral unicode characters consistently with chunk budget", () => {
    const text = "😀😀\n"
    const chunks = fallbackChunks({ filePath: "src/a.ts", language: "typescript", text, maxNonWhitespaceChars: 1 })

    expect(chunks.every((chunk) => chunk.nonWhitespaceChars <= 1)).toBe(true)
    expect(chunks.map((chunk) => chunk.text).join("")).toBe(text)
    expect(chunks[0].range.byteStart).toBe(0)
    expect(chunks.at(-1)?.range.byteEnd).toBe(new TextEncoder().encode(text).length)
    expect(chunks.slice(1).every((chunk, index) => chunk.range.byteStart === chunks[index].range.byteEnd)).toBe(true)
    expect(chunks.map((chunk) => chunk.id)).toEqual(
      chunks.map((chunk) => `src/a.ts:${chunk.range.byteStart}:${chunk.range.byteEnd}`),
    )
  })

  test("calculates ranges and non-whitespace sizes", () => {
    expect(nonWhitespaceLength("a b\n\tc")).toBe(3)
    expect(rangeForSlice("one\ntwo\nthree\n", 4, 8)).toEqual({ byteStart: 4, byteEnd: 8, lineStart: 2, lineEnd: 3 })
  })
})
