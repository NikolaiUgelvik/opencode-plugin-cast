import type { SyntaxNode } from "./cast.js"
import { rangeForSlice } from "./range.js"
import type { ChunkRecord, SearchResultTopology, SymbolRecord, TopologyNode } from "./types.js"

const encoder = new TextEncoder()
const decoder = new TextDecoder()
const CLASS_NAME_PATTERN = /class\s+([\p{L}_$][\p{L}\p{N}_$]*)/u
const INTERFACE_NAME_PATTERN = /interface\s+([\p{L}_$][\p{L}\p{N}_$]*)/u
const FUNCTION_DECLARATION_NAME_PATTERN = /function\s+([\p{L}_$][\p{L}\p{N}_$]*)/u
const FUNCTION_ASSIGNMENT_NAME_PATTERN = /([\p{L}_$][\p{L}\p{N}_$]*)\s*=\s*(?:async\s+)?(?:function|\()/u
const METHOD_NAME_PATTERN = /(?:async\s+)?([\p{L}_$][\p{L}\p{N}_$]*)\s*\(/u
const PROPERTY_NAME_PATTERN =
  /^\s*([\p{L}_$][\p{L}\p{N}_$]*)\s*:\s*(?:[\p{L}_$][\p{L}\p{N}_$]*\s*\(|async\s+|function\b|\()/u
const TEST_CALL_NAME_PATTERN = /^\s*(?:test|it)\s*\(\s*(["'`])((?:\\.|(?!\1)[^\\\n])*)\1/u
const SNIPPET_LABEL_MAX_LENGTH = 80
const SNIPPET_LABEL_PREFIX_LENGTH = 77

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

export function summarizeTopology(
  chunk: ChunkRecord,
  chunks: Record<string, ChunkRecord>,
  symbols: Record<string, SymbolRecord>,
): SearchResultTopology {
  return {
    chunk: summarizeChunk(chunk, symbols),
    ...optionalChunk(chunks, chunk.parentChunkId, symbols, "parent"),
    children: chunk.childChunkIds.flatMap((id) => {
      const child = chunks[id]
      return child ? [summarizeChunk(child, symbols)] : []
    }),
    ...optionalChunk(chunks, chunk.previousSiblingChunkId, symbols, "previousSibling"),
    ...optionalChunk(chunks, chunk.nextSiblingChunkId, symbols, "nextSibling"),
    symbols: symbolBreadcrumbs(chunk, symbols),
  }
}

export function summarizeChunk(chunk: ChunkRecord, symbols: Record<string, SymbolRecord>): TopologyNode {
  return {
    id: chunk.id,
    label: labelForChunk(chunk, symbols),
    range: rangeLabel(chunk),
  }
}

function optionalChunk<Key extends "parent" | "previousSibling" | "nextSibling">(
  chunks: Record<string, ChunkRecord>,
  id: string | undefined,
  symbols: Record<string, SymbolRecord>,
  key: Key,
): Partial<Record<Key, TopologyNode>> {
  const chunk = id ? chunks[id] : undefined
  return chunk ? ({ [key]: summarizeChunk(chunk, symbols) } as Partial<Record<Key, TopologyNode>>) : {}
}

function labelForChunk(chunk: ChunkRecord, symbols: Record<string, SymbolRecord>) {
  if (chunk.kind === "file") {
    return `file ${chunk.filePath}`
  }
  const localSymbol = nearestLocalSymbol(chunk, symbols)
  if (localSymbol) {
    return `${localSymbol.kind} ${localSymbol.name}`
  }
  const localLabel = localSnippetLabel(chunk)
  if (localLabel) {
    return `${chunk.kind} ${localLabel}`
  }
  const enclosingSymbol = nearestEnclosingSymbol(chunk, symbols)
  return enclosingSymbol ? `${enclosingSymbol.kind} ${enclosingSymbol.name}` : `${chunk.kind} chunk`
}

function nearestLocalSymbol(chunk: ChunkRecord, symbols: Record<string, SymbolRecord>) {
  return [...chunk.symbolIds]
    .reverse()
    .flatMap((id) => (symbols[id] ? [symbols[id]] : []))
    .find((symbol) => rangesTightlyOverlap(symbol, chunk))
}

function nearestEnclosingSymbol(chunk: ChunkRecord, symbols: Record<string, SymbolRecord>) {
  return [...chunk.symbolIds]
    .reverse()
    .flatMap((id) => (symbols[id] ? [symbols[id]] : []))
    .find((symbol) => symbol.range.byteStart <= chunk.range.byteStart && chunk.range.byteEnd <= symbol.range.byteEnd)
}

function rangesTightlyOverlap(symbol: SymbolRecord, chunk: ChunkRecord) {
  return symbol.range.byteStart >= chunk.range.byteStart && symbol.range.byteStart <= chunk.range.byteEnd
}

function localSnippetLabel(chunk: ChunkRecord) {
  const firstLine = chunk.text
    .split("\n")
    .map((line) => line.trim())
    .find((line) => line.length > 0)
  if (!firstLine) {
    return
  }
  return firstLine.length > SNIPPET_LABEL_MAX_LENGTH
    ? `${firstLine.slice(0, SNIPPET_LABEL_PREFIX_LENGTH)}...`
    : firstLine
}

function rangeLabel(chunk: ChunkRecord) {
  if (chunk.range.lineStart === chunk.range.lineEnd) {
    return `${chunk.filePath}:${chunk.range.lineStart}`
  }
  return `${chunk.filePath}:${chunk.range.lineStart}-${chunk.range.lineEnd}`
}

function symbolBreadcrumbs(chunk: ChunkRecord, symbols: Record<string, SymbolRecord>) {
  return chunk.symbolIds.flatMap((id) => {
    const symbol = symbols[id]
    return symbol ? [`${symbol.kind} ${symbol.name}`] : []
  })
}

function extractNodeSymbols(
  input: { filePath: string; source: string },
  node: SyntaxNode,
  parentSymbolId: string | undefined,
): SymbolRecord[] {
  const kind = symbolKindFor(input.source, node)
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

function symbolKindFor(source: string, node: SyntaxNode): SymbolRecord["kind"] | undefined {
  if (typeLooksLikeObjectProperty(node.type)) {
    const text = textForByteSlice(source, node.startIndex, node.endIndex)
    if (PROPERTY_NAME_PATTERN.test(text)) {
      return "function"
    }
  }
  if (typeLooksLikeCall(node.type)) {
    const text = textForByteSlice(source, node.startIndex, node.endIndex)
    if (TEST_CALL_NAME_PATTERN.test(text)) {
      return "function"
    }
  }
  if (node.type.includes("interface")) {
    return "interface"
  }
  if (node.type.includes("class")) {
    return "class"
  }
  if (node.type.includes("method")) {
    return "method"
  }
  if (node.type.includes("function")) {
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
    const testName = text.match(TEST_CALL_NAME_PATTERN)?.[2]
    return (
      (testName ? `test ${unescapeTestName(testName)}` : undefined) ??
      text.match(PROPERTY_NAME_PATTERN)?.[1] ??
      text.match(FUNCTION_DECLARATION_NAME_PATTERN)?.[1] ??
      text.match(FUNCTION_ASSIGNMENT_NAME_PATTERN)?.[1] ??
      "anonymous"
    )
  }
  return text.match(METHOD_NAME_PATTERN)?.[1] ?? "anonymous"
}

function unescapeTestName(name: string) {
  return name.replace(/\\(.)/g, "$1")
}

function typeLooksLikeObjectProperty(type: string) {
  return type === "pair" || (type.includes("property") && !type.includes("signature"))
}

function typeLooksLikeCall(type: string) {
  return type.includes("call")
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
