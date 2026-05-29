import type { SyntaxNode } from "./cast.js"
import { rangeForSlice } from "./range.js"
import type { ChunkRecord, SymbolRecord } from "./types.js"

const encoder = new TextEncoder()
const decoder = new TextDecoder()
const CLASS_NAME_PATTERN = /class\s+([\p{L}_$][\p{L}\p{N}_$]*)/u
const INTERFACE_NAME_PATTERN = /interface\s+([\p{L}_$][\p{L}\p{N}_$]*)/u
const FUNCTION_DECLARATION_NAME_PATTERN = /function\s+([\p{L}_$][\p{L}\p{N}_$]*)/u
const FUNCTION_ASSIGNMENT_NAME_PATTERN = /([\p{L}_$][\p{L}\p{N}_$]*)\s*=\s*(?:async\s+)?(?:function|\()/u
const METHOD_NAME_PATTERN = /(?:async\s+)?([\p{L}_$][\p{L}\p{N}_$]*)\s*\(/u

export function extractSymbols(input: { filePath: string; source: string; nodes: SyntaxNode[] }) {
  const symbols = input.nodes.flatMap((node) => extractNodeSymbols(input, node, undefined))
  const symbolsById = Object.fromEntries(symbols.map((symbol) => [symbol.id, symbol]))

  return symbols.map((symbol) => ({
    ...symbol,
    childSymbolIds: symbols
      .filter((child) => child.parentSymbolId === symbol.id && symbolsById[child.id])
      .map((child) => child.id),
  }))
}

export function assignSymbolsToChunks(chunks: ChunkRecord[], symbols: Record<string, SymbolRecord>) {
  return chunks.map((chunk) => ({
    ...chunk,
    symbolIds: Object.values(symbols)
      .filter(
        (symbol) =>
          symbol.filePath === chunk.filePath &&
          containsRange(symbol.range.byteStart, symbol.range.byteEnd, chunk.range.byteStart, chunk.range.byteEnd),
      )
      .sort((left, right) => left.range.byteStart - right.range.byteStart || right.range.byteEnd - left.range.byteEnd)
      .map((symbol) => symbol.id),
  }))
}

export function attachTopology(chunks: ChunkRecord[], symbols: Record<string, SymbolRecord>) {
  const chunksWithParents = chunks.map((chunk) => ({
    ...chunk,
    parentChunkId: chunks
      .filter(
        (candidate) =>
          candidate.filePath === chunk.filePath &&
          candidate.id !== chunk.id &&
          containsRange(candidate.range.byteStart, candidate.range.byteEnd, chunk.range.byteStart, chunk.range.byteEnd),
      )
      .sort((left, right) => rangeSize(left) - rangeSize(right))[0]?.id,
    childChunkIds: chunks
      .filter(
        (candidate) =>
          candidate.filePath === chunk.filePath &&
          candidate.id !== chunk.id &&
          containsRange(chunk.range.byteStart, chunk.range.byteEnd, candidate.range.byteStart, candidate.range.byteEnd),
      )
      .filter(
        (candidate) =>
          !chunks.some(
            (other) =>
              other.filePath === chunk.filePath &&
              other.id !== chunk.id &&
              other.id !== candidate.id &&
              containsRange(chunk.range.byteStart, chunk.range.byteEnd, other.range.byteStart, other.range.byteEnd) &&
              containsRange(
                other.range.byteStart,
                other.range.byteEnd,
                candidate.range.byteStart,
                candidate.range.byteEnd,
              ),
          ),
      )
      .sort((left, right) => left.range.byteStart - right.range.byteStart)
      .map((candidate) => candidate.id),
  }))
  const siblingGroups: Record<string, ChunkRecord[]> = {}
  for (const chunk of chunksWithParents) {
    const contextId = siblingContextId(chunk, symbols)
    siblingGroups[contextId] = [...(siblingGroups[contextId] ?? []), chunk]
  }
  const siblingIdsByChunkId = Object.fromEntries(
    Object.values(siblingGroups).flatMap((group) =>
      [...group]
        .sort((left, right) => left.range.byteStart - right.range.byteStart)
        .map((chunk, index, groupChunks) => [
          chunk.id,
          { previousSiblingChunkId: groupChunks[index - 1]?.id, nextSiblingChunkId: groupChunks[index + 1]?.id },
        ]),
    ),
  )

  return chunksWithParents.map((chunk) => ({
    ...chunk,
    previousSiblingChunkId: siblingIdsByChunkId[chunk.id]?.previousSiblingChunkId,
    nextSiblingChunkId: siblingIdsByChunkId[chunk.id]?.nextSiblingChunkId,
  }))
}

export function expandWithParentContext(input: {
  chunk: ChunkRecord
  symbols: Record<string, SymbolRecord>
  source: string
  maxContextChars: number
}) {
  const chunkSymbols = input.chunk.symbolIds
    .map((id) => input.symbols[id])
    .filter((symbol): symbol is SymbolRecord => Boolean(symbol))
  const parent = chunkSymbols[0]
  const breadcrumbs = chunkSymbols.map((symbol) => `${symbol.kind} ${symbol.name}`)

  if (!parent) {
    return { breadcrumbs }
  }

  const parentText = textForByteSlice(input.source, parent.range.byteStart, parent.range.byteEnd).trimEnd()
  if (parentText.length <= input.maxContextChars) {
    return { breadcrumbs, parentText, parentRange: parent.range }
  }

  return {
    breadcrumbs,
    parentText:
      `${breadcrumbs[0]}\n${textForByteSlice(input.source, input.chunk.range.byteStart, input.chunk.range.byteEnd).trimEnd()}`.slice(
        0,
        input.maxContextChars,
      ),
    parentRange: parent.range,
  }
}

function extractNodeSymbols(
  input: { filePath: string; source: string },
  node: SyntaxNode,
  parentSymbolId: string | undefined,
): SymbolRecord[] {
  const kind = symbolKindFor(node.type)
  const symbol = kind
    ? ({
        id: `${input.filePath}:${kind}:${nameFor(input.source, node, kind)}:${node.startIndex}:${node.endIndex}`,
        name: nameFor(input.source, node, kind),
        kind,
        filePath: input.filePath,
        range: rangeForSlice(input.source, node.startIndex, node.endIndex),
        parentSymbolId,
        childSymbolIds: [],
      } satisfies SymbolRecord)
    : undefined

  return [
    ...(symbol ? [symbol] : []),
    ...node.children.flatMap((child) => extractNodeSymbols(input, child, symbol?.id ?? parentSymbolId)),
  ]
}

function symbolKindFor(type: string): SymbolRecord["kind"] | undefined {
  if (type.includes("interface")) {
    return "interface"
  }
  if (type.includes("class")) {
    return "class"
  }
  if (type.includes("method")) {
    return "method"
  }
  if (type.includes("function")) {
    return "function"
  }
  return
}

function nameFor(source: string, node: SyntaxNode, kind: SymbolRecord["kind"]) {
  const text = textForByteSlice(source, node.startIndex, node.endIndex)
  if (kind === "class") {
    return text.match(CLASS_NAME_PATTERN)?.[1] ?? "anonymous"
  }
  if (kind === "interface") {
    return text.match(INTERFACE_NAME_PATTERN)?.[1] ?? "anonymous"
  }
  if (kind === "function") {
    return (
      text.match(FUNCTION_DECLARATION_NAME_PATTERN)?.[1] ??
      text.match(FUNCTION_ASSIGNMENT_NAME_PATTERN)?.[1] ??
      "anonymous"
    )
  }
  return text.match(METHOD_NAME_PATTERN)?.[1] ?? "anonymous"
}

function siblingContextId(chunk: ChunkRecord, symbols: Record<string, SymbolRecord>) {
  const symbol = symbols[chunk.symbolIds.at(-1) ?? ""]
  return `${chunk.filePath}:${chunk.parentChunkId ?? "root"}:${symbol?.id ?? "no-symbol"}`
}

function containsRange(parentStart: number, parentEnd: number, childStart: number, childEnd: number) {
  return parentStart <= childStart && childEnd <= parentEnd
}

function rangeSize(chunk: ChunkRecord) {
  return chunk.range.byteEnd - chunk.range.byteStart
}

function textForByteSlice(source: string, byteStart: number, byteEnd: number) {
  return decoder.decode(encoder.encode(source).slice(byteStart, byteEnd))
}
