import { fallbackChunks } from "./fallback.js"
import { nonWhitespaceLength, rangeForSlice, stableChunkId } from "./range.js"
import type { ChunkKind, ChunkRecord } from "./types.js"

const encoder = new TextEncoder()
const decoder = new TextDecoder()

export type SyntaxNode = {
  type: string
  startIndex: number
  endIndex: number
  children: SyntaxNode[]
}

export function castChunks(input: {
  filePath: string
  language: string
  source: string
  root: SyntaxNode
  maxNonWhitespaceChars: number
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

  return chunkChildren(input, input.root.children, undefined)
}

function chunkChildren(
  input: { filePath: string; language: string; source: string; maxNonWhitespaceChars: number },
  nodes: SyntaxNode[],
  parentChunkId: string | undefined,
) {
  const chunks: ChunkRecord[] = []
  const siblingChunks: ChunkRecord[] = []
  let pending: SyntaxNode[] = []

  for (const child of nodes) {
    if (
      nonWhitespaceLength(textForByteSlice(input.source, child.startIndex, child.endIndex)) >
      input.maxNonWhitespaceChars
    ) {
      const pendingChunks = flushPending(input, pending, parentChunkId)
      chunks.push(...pendingChunks)
      siblingChunks.push(...pendingChunks)
      pending = []
      chunks.push(...chunkOversizedNode(input, child))
      continue
    }

    if (
      pending.length > 0 &&
      nonWhitespaceLength(textForByteSlice(input.source, pending[0].startIndex, child.endIndex)) >
        input.maxNonWhitespaceChars
    ) {
      const pendingChunks = flushPending(input, pending, parentChunkId)
      chunks.push(...pendingChunks)
      siblingChunks.push(...pendingChunks)
      pending = []
    }

    pending.push(child)
  }

  const pendingChunks = flushPending(input, pending, parentChunkId)
  chunks.push(...pendingChunks)
  siblingChunks.push(...pendingChunks)
  const linkedSiblingChunks = linkSiblings(siblingChunks)
  return chunks.map((chunk) => linkedSiblingChunks.find((siblingChunk) => siblingChunk.id === chunk.id) ?? chunk)
}

function chunkOversizedNode(
  input: { filePath: string; language: string; source: string; maxNonWhitespaceChars: number },
  node: SyntaxNode,
) {
  if (node.children.length > 0) {
    return chunkChildren(input, node.children, stableChunkId(input.filePath, node.startIndex, node.endIndex))
  }
  return linkSiblings(
    fallbackChunks({
      filePath: input.filePath,
      language: input.language,
      text: textForByteSlice(input.source, node.startIndex, node.endIndex),
      maxNonWhitespaceChars: input.maxNonWhitespaceChars,
    }).map((chunk) => ({
      ...chunk,
      id: stableChunkId(input.filePath, node.startIndex + chunk.range.byteStart, node.startIndex + chunk.range.byteEnd),
      range: rangeForSlice(
        input.source,
        node.startIndex + chunk.range.byteStart,
        node.startIndex + chunk.range.byteEnd,
      ),
      parentChunkId: stableChunkId(input.filePath, node.startIndex, node.endIndex),
    })),
  )
}

function flushPending(
  input: { filePath: string; language: string; source: string; maxNonWhitespaceChars: number },
  pending: SyntaxNode[],
  parentChunkId: string | undefined,
) {
  if (pending.length === 0) {
    return []
  }
  const first = pending[0]
  const last = pending.at(-1)
  if (!(first && last)) {
    return []
  }
  return [
    makeChunk(
      input,
      kindFor(pending),
      first.startIndex,
      last.endIndex,
      pending.map((node) => node.type),
      parentChunkId,
    ),
  ]
}

function makeChunk(
  input: { filePath: string; language: string; source: string },
  kind: ChunkKind,
  byteStart: number,
  byteEnd: number,
  nodeTypes: string[],
  parentChunkId?: string,
): ChunkRecord {
  return {
    id: stableChunkId(input.filePath, byteStart, byteEnd),
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

function kindFor(nodes: SyntaxNode[]) {
  if (nodes.some((node) => node.type.includes("class"))) {
    return "class"
  }
  if (nodes.some((node) => node.type.includes("method"))) {
    return "method"
  }
  if (nodes.some((node) => node.type.includes("function"))) {
    return "function"
  }
  return "block"
}

function textForByteSlice(source: string, byteStart: number, byteEnd: number) {
  return decoder.decode(encoder.encode(source).slice(byteStart, byteEnd))
}
