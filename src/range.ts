import type { SourceRange } from "./types.js"

const encoder = new TextEncoder()
const WHITESPACE_PATTERN = /\s/

export function nonWhitespaceLength(text: string) {
  return [...text].filter((character) => !WHITESPACE_PATTERN.test(character)).length
}

export function rangeForSlice(source: string, byteStart: number, byteEnd: number): SourceRange {
  const before = source.slice(0, stringOffsetForByteOffset(source, byteStart))
  const slice = source.slice(stringOffsetForByteOffset(source, byteStart), stringOffsetForByteOffset(source, byteEnd))
  return {
    byteStart,
    byteEnd,
    lineStart: before.split("\n").length,
    lineEnd: before.split("\n").length + slice.split("\n").length - 1,
  }
}

export function stableChunkId(filePath: string, byteStart: number, byteEnd: number) {
  return `${filePath}:${byteStart}:${byteEnd}`
}

function stringOffsetForByteOffset(source: string, byteOffset: number) {
  let bytes = 0
  let offset = 0

  for (const character of source) {
    if (bytes >= byteOffset) {
      return offset
    }
    bytes += encoder.encode(character).length
    offset += character.length
  }

  return offset
}
