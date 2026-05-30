import { fallbackChunks } from "./fallback.js"
import { nonWhitespaceLength, rangeForSlice, stableChunkId } from "./range.js"
import type { ChunkingOptions, ChunkKind, ChunkRecord } from "./types.js"

const encoder = new TextEncoder()
const decoder = new TextDecoder()
const IDENTIFIER_PATTERN = /[A-Za-z_$][\w$]*/
const PUNCTUATION_OR_SYMBOL_PATTERN = /^[\p{P}\p{S}\s]+$/u

export type SyntaxNode = {
  type: string
  startIndex: number
  endIndex: number
  children: SyntaxNode[]
}

type ChunkWindow = {
  nodes: SyntaxNode[]
  byteStart: number
  byteEnd: number
  idByteStart?: number
  idByteEnd?: number
  nodeTypes: string[]
  kind?: ChunkKind
  parentChunkId?: string
  parentByteStart?: number
  parentByteEnd?: number
}

export function castChunks(input: {
  filePath: string
  language: string
  source: string
  root: SyntaxNode
  maxNonWhitespaceChars: number
  chunking: ChunkingOptions
}) {
  if (nonWhitespaceLength(input.source) <= input.maxNonWhitespaceChars) {
    return [makeChunk(input, "file", input.root.startIndex, input.root.endIndex, [input.root.type])]
  }

  if (input.root.children.length === 0) {
    return fallbackChunks({
      filePath: input.filePath,
      language: input.language,
      text: input.source,
      maxNonWhitespaceChars: input.maxNonWhitespaceChars,
    })
  }

  const windows = buildWindows(input, input.root.children, undefined)
  const normalized = normalizeTrivialWindows(input, windows)
  const overlapped = applyOverlap(normalized, input.chunking.overlap)
  return linkSiblings(overlapped.map((window) => makeChunkFromWindow(input, window)))
}

function buildWindows(
  input: {
    filePath: string
    language: string
    source: string
    maxNonWhitespaceChars: number
    chunking: ChunkingOptions
  },
  nodes: SyntaxNode[],
  parentChunkId: string | undefined,
) {
  const windows: ChunkWindow[] = []
  let pending: SyntaxNode[] = []

  for (const child of nodes) {
    if (nodeNonWhitespace(input, child) > input.maxNonWhitespaceChars) {
      const prefix = flushWindow(pending, parentChunkId)
      pending = []
      windows.push(...mergePrefixWithFirstSplitWindow(input, prefix, splitOversizedNode(input, child)))
      continue
    }

    if (
      pending.length > 0 &&
      rangeNonWhitespace(input, pending[0].startIndex, child.endIndex) > input.maxNonWhitespaceChars
    ) {
      windows.push(...flushWindow(pending, parentChunkId))
      pending = []
    }

    pending.push(child)
  }

  windows.push(...flushWindow(pending, parentChunkId))
  return windows
}

function mergePrefixWithFirstSplitWindow(
  input: { source: string; maxNonWhitespaceChars: number },
  prefix: ChunkWindow[],
  split: ChunkWindow[],
) {
  if (prefix.length !== 1 || !split[0]) {
    return [...prefix, ...split]
  }

  if (rangeNonWhitespace(input, prefix[0].byteStart, split[0].byteEnd) > input.maxNonWhitespaceChars) {
    return [...prefix, ...split]
  }

  return [mergeWindows(prefix[0], split[0]), ...split.slice(1)]
}

function splitOversizedNode(
  input: {
    filePath: string
    language: string
    source: string
    maxNonWhitespaceChars: number
    chunking: ChunkingOptions
  },
  node: SyntaxNode,
) {
  const parentChunkId = stableChunkId(input.filePath, node.startIndex, node.endIndex)
  if (node.children.length === 0) {
    return fallbackChunks({
      filePath: input.filePath,
      language: input.language,
      text: textForByteSlice(input.source, node.startIndex, node.endIndex),
      maxNonWhitespaceChars: input.maxNonWhitespaceChars,
    }).map((chunk) => ({
      nodes: [],
      byteStart: node.startIndex + chunk.range.byteStart,
      byteEnd: node.startIndex + chunk.range.byteEnd,
      nodeTypes: [node.type],
      kind: chunk.kind,
      parentChunkId,
      parentByteStart: node.startIndex,
      parentByteEnd: node.endIndex,
    }))
  }
  return mergeAdjacentWindows(input, buildWindows(input, node.children, parentChunkId)).map((window) => ({
    ...window,
    parentByteStart: window.parentByteStart ?? node.startIndex,
    parentByteEnd: window.parentByteEnd ?? node.endIndex,
  }))
}

function flushWindow(pending: SyntaxNode[], parentChunkId: string | undefined): ChunkWindow[] {
  const first = pending[0]
  const last = pending.at(-1)
  if (!(first && last)) {
    return []
  }
  return [windowForNodes(first.startIndex, last.endIndex, pending, parentChunkId)]
}

function windowForNodes(
  byteStart: number,
  byteEnd: number,
  nodes: SyntaxNode[],
  parentChunkId: string | undefined,
): ChunkWindow {
  return {
    nodes,
    byteStart,
    byteEnd,
    nodeTypes: nodes.map((node) => node.type),
    parentChunkId,
  }
}

function mergeAdjacentWindows(input: { source: string; maxNonWhitespaceChars: number }, windows: ChunkWindow[]) {
  const merged: ChunkWindow[] = []
  for (const window of windows) {
    const previous = merged.at(-1)
    if (previous && rangeNonWhitespace(input, previous.byteStart, window.byteEnd) <= input.maxNonWhitespaceChars) {
      merged[merged.length - 1] = mergeWindows(previous, window)
      continue
    }
    merged.push(window)
  }
  return merged
}

function mergeWindows(left: ChunkWindow, right: ChunkWindow): ChunkWindow {
  const byteStart = left.byteStart
  const byteEnd = right.byteEnd
  return {
    nodes: [...left.nodes, ...right.nodes],
    byteStart,
    byteEnd,
    nodeTypes: [...left.nodeTypes, ...right.nodeTypes],
    kind: left.kind ?? right.kind,
    ...parentMetadataForRange(byteStart, byteEnd, left, right),
  }
}

function parentMetadataForRange(byteStart: number, byteEnd: number, left: ChunkWindow, right: ChunkWindow) {
  const parentChunkId = left.parentChunkId ?? right.parentChunkId
  if (!parentChunkId) {
    return {}
  }

  const parentByteStart = left.parentByteStart ?? right.parentByteStart
  const parentByteEnd = left.parentByteEnd ?? right.parentByteEnd
  if (parentByteStart === undefined || parentByteEnd === undefined) {
    return { parentChunkId }
  }
  if (byteStart < parentByteStart || byteEnd > parentByteEnd) {
    return {}
  }
  return { parentChunkId, parentByteStart, parentByteEnd }
}

function normalizeTrivialWindows(
  input: { source: string; maxNonWhitespaceChars: number; chunking: ChunkingOptions },
  windows: ChunkWindow[],
) {
  const normalized: ChunkWindow[] = []
  for (let index = 0; index < windows.length; index++) {
    const window = windows[index]
    if (!window) {
      continue
    }
    if (!isTrivialWindow(input, window)) {
      normalized.push(window)
      continue
    }
    const result = mergeTrivialWindow(input, normalized, windows, index, window)
    if (result.window) {
      normalized.push(result.window)
    }
    if (result.skipNext) {
      index++
    }
  }
  return normalized
}

function mergeTrivialWindow(
  input: { source: string; maxNonWhitespaceChars: number; chunking: ChunkingOptions },
  normalized: ChunkWindow[],
  windows: ChunkWindow[],
  index: number,
  window: ChunkWindow,
) {
  const previous = normalized.at(-1)
  if (previous && rangeNonWhitespace(input, previous.byteStart, window.byteEnd) <= input.maxNonWhitespaceChars) {
    normalized[normalized.length - 1] = mergeWindows(previous, window)
    return { skipNext: false }
  }

  const next = windows[index + 1]
  if (next && rangeNonWhitespace(input, window.byteStart, next.byteEnd) <= input.maxNonWhitespaceChars) {
    return { window: mergeWindows(window, next), skipNext: true }
  }
  return { window, skipNext: false }
}

function isTrivialWindow(input: { source: string; chunking: ChunkingOptions }, window: ChunkWindow) {
  const text = textForByteSlice(input.source, window.byteStart, window.byteEnd).trim()
  return (
    nonWhitespaceLength(text) < input.chunking.minSemanticNonWhitespaceChars ||
    !IDENTIFIER_PATTERN.test(text) ||
    PUNCTUATION_OR_SYMBOL_PATTERN.test(text)
  )
}

function applyOverlap(windows: ChunkWindow[], overlap: number) {
  if (overlap === 0) {
    return windows
  }
  return windows.map((window, index) => {
    const previous = windows.slice(Math.max(0, index - overlap), index)
    const next = windows.slice(index + 1, index + 1 + overlap)
    const expanded = [...previous, window, ...next]
    const merged = expanded.reduce((merged, current) => mergeWindows(merged, current))
    return {
      ...merged,
      idByteStart: window.byteStart,
      idByteEnd: window.byteEnd,
      parentChunkId: parentChunkIdForOverlap(window, merged),
      parentByteStart: window.parentByteStart,
      parentByteEnd: window.parentByteEnd,
    }
  })
}

function parentChunkIdForOverlap(origin: ChunkWindow, expanded: ChunkWindow) {
  if (!(origin.parentChunkId && origin.parentByteStart !== undefined && origin.parentByteEnd !== undefined)) {
    return
  }
  if (expanded.byteStart < origin.parentByteStart || expanded.byteEnd > origin.parentByteEnd) {
    return
  }
  return origin.parentChunkId
}

function makeChunkFromWindow(
  input: { filePath: string; language: string; source: string },
  window: ChunkWindow,
): ChunkRecord {
  return makeChunk(
    input,
    window.kind ?? kindFor(window.nodeTypes),
    window.byteStart,
    window.byteEnd,
    window.nodeTypes,
    window.parentChunkId,
    window.idByteStart,
    window.idByteEnd,
  )
}

function nodeNonWhitespace(input: { source: string }, node: SyntaxNode) {
  return rangeNonWhitespace(input, node.startIndex, node.endIndex)
}

function rangeNonWhitespace(input: { source: string }, byteStart: number, byteEnd: number) {
  return nonWhitespaceLength(textForByteSlice(input.source, byteStart, byteEnd))
}

function makeChunk(
  input: { filePath: string; language: string; source: string },
  kind: ChunkKind,
  byteStart: number,
  byteEnd: number,
  nodeTypes: string[],
  parentChunkId?: string,
  idByteStart?: number,
  idByteEnd?: number,
): ChunkRecord {
  return {
    id: stableChunkId(input.filePath, idByteStart ?? byteStart, idByteEnd ?? byteEnd),
    filePath: input.filePath,
    language: input.language,
    kind,
    range: rangeForSlice(input.source, byteStart, byteEnd),
    text: textForByteSlice(input.source, byteStart, byteEnd),
    nonWhitespaceChars: nonWhitespaceLength(textForByteSlice(input.source, byteStart, byteEnd)),
    nodeTypes,
    symbolIds: [],
    parentChunkId,
    childChunkIds: [],
  }
}

function linkSiblings(chunks: ChunkRecord[]) {
  return chunks.map((chunk, index) => ({
    ...chunk,
    previousSiblingChunkId: chunks[index - 1]?.id,
    nextSiblingChunkId: chunks[index + 1]?.id,
  }))
}

function kindFor(nodeTypes: string[]) {
  if (nodeTypes.some((type) => type.includes("class"))) {
    return "class"
  }
  if (nodeTypes.some((type) => type.includes("method"))) {
    return "method"
  }
  if (nodeTypes.some((type) => type.includes("function"))) {
    return "function"
  }
  return "block"
}

function textForByteSlice(source: string, byteStart: number, byteEnd: number) {
  return decoder.decode(encoder.encode(source).slice(byteStart, byteEnd))
}
