import { expandWithParentContext, summarizeChunk, summarizeTopology } from "./topology.js"
import type { CastIndex, ChunkLookupInput, ChunkLookupOutput, ChunkLookupRelatedChunk, ChunkRecord } from "./types.js"

async function getChunkById(input: {
  index: CastIndex
  input: ChunkLookupInput
  readSource(filePath: string): Promise<string>
}): Promise<ChunkLookupOutput> {
  const diagnostics = [...input.index.metadata.diagnostics]
  const sourceCache = new Map<string, Promise<SourceReadResult>>()
  const getSource = (filePath: string) => readSourceCached(filePath, input.readSource, sourceCache)
  const chunk = input.index.chunks[input.input.id]
  if (!chunk) {
    if (input.index.metadata.status !== "ready") {
      return {
        status: input.index.metadata,
        diagnostics: [...diagnostics, `index unavailable: ${input.index.metadata.status}`],
      }
    }
    return {
      status: input.index.metadata,
      diagnostics: [...diagnostics, `chunk not found: ${input.input.id}`],
    }
  }

  const source = await getSource(chunk.filePath)
  const chunkText = validatedChunkText(chunk, source, diagnostics, "chunk text omitted")

  const context = parentContext({
    chunk,
    diagnostics,
    includeParents: input.input.includeParents,
    index: input.index,
    maxContextChars: input.input.maxContextChars,
    source,
  })
  const related = await relatedChunks({
    chunk,
    chunks: input.index.chunks,
    diagnostics,
    getSource,
    includeChildren: input.input.includeChildren,
    includeParents: input.input.includeParents,
    includeSiblings: input.input.includeSiblings,
    maxContextChars: input.input.maxContextChars,
    symbols: input.index.symbols,
  })

  return {
    status: input.index.metadata,
    chunk: {
      filePath: chunk.filePath,
      language: chunk.language,
      range: chunk.range,
      kind: chunk.kind,
      breadcrumbs: context.breadcrumbs,
      text: chunkText,
      parentText: context.parentText,
      parentRange: context.parentRange,
      topology: summarizeTopology(chunk, input.index.chunks, input.index.symbols),
      related,
    },
    diagnostics,
  }
}

const encoder = new TextEncoder()
const decoder = new TextDecoder()

type SourceReadResult = { text: string; ok: true } | { text: ""; ok: false }

function parentContext(input: {
  chunk: ChunkRecord
  diagnostics: string[]
  includeParents: boolean | undefined
  index: CastIndex
  maxContextChars: number | undefined
  source: SourceReadResult
}): { breadcrumbs: string[]; parentText?: string; parentRange?: ChunkRecord["range"] } {
  if (input.includeParents === false) {
    return { breadcrumbs: breadcrumbsFor(input.chunk, input.index.symbols) }
  }

  if (input.source.ok && indexedChunkMatchesSource(input.source.text, input.chunk)) {
    return expandWithParentContext({
      chunk: input.chunk,
      symbols: input.index.symbols,
      source: input.source.text,
      maxContextChars: input.maxContextChars ?? Number.MAX_SAFE_INTEGER,
    })
  }
  if (input.source.ok) {
    input.diagnostics.push(`source mismatch for ${input.chunk.filePath}:${input.chunk.id}; parent context omitted`)
  } else {
    input.diagnostics.push(`source read failed for ${input.chunk.filePath}; parent context omitted`)
  }
  return { breadcrumbs: breadcrumbsFor(input.chunk, input.index.symbols) }
}

async function relatedChunks(input: {
  chunk: ChunkRecord
  chunks: CastIndex["chunks"]
  diagnostics: string[]
  getSource(filePath: string): Promise<SourceReadResult>
  includeChildren: boolean | undefined
  includeParents: boolean | undefined
  includeSiblings: boolean | undefined
  maxContextChars: number | undefined
  symbols: CastIndex["symbols"]
}): Promise<NonNullable<ChunkLookupOutput["chunk"]>["related"]> {
  const [parent, previousSibling, nextSibling, children] = await Promise.all([
    input.includeParents === false
      ? undefined
      : relatedChunk({
          chunk: input.chunks[input.chunk.parentChunkId ?? ""],
          symbols: input.symbols,
          maxContextChars: input.maxContextChars,
          diagnostics: input.diagnostics,
          getSource: input.getSource,
        }),
    input.includeSiblings === false
      ? undefined
      : relatedChunk({
          chunk: input.chunks[input.chunk.previousSiblingChunkId ?? ""],
          symbols: input.symbols,
          maxContextChars: input.maxContextChars,
          diagnostics: input.diagnostics,
          getSource: input.getSource,
        }),
    input.includeSiblings === false
      ? undefined
      : relatedChunk({
          chunk: input.chunks[input.chunk.nextSiblingChunkId ?? ""],
          symbols: input.symbols,
          maxContextChars: input.maxContextChars,
          diagnostics: input.diagnostics,
          getSource: input.getSource,
        }),
    input.includeChildren === false
      ? []
      : Promise.all(
          input.chunk.childChunkIds.map((id) =>
            relatedChunk({
              chunk: input.chunks[id],
              symbols: input.symbols,
              maxContextChars: input.maxContextChars,
              diagnostics: input.diagnostics,
              getSource: input.getSource,
            }),
          ),
        ),
  ])
  return {
    parent,
    previousSibling,
    nextSibling,
    children: children.flatMap((child) => (child ? [child] : [])),
  }
}

async function relatedChunk(input: {
  chunk: ChunkRecord | undefined
  symbols: CastIndex["symbols"]
  maxContextChars: number | undefined
  diagnostics: string[]
  getSource(filePath: string): Promise<SourceReadResult>
}): Promise<ChunkLookupRelatedChunk | undefined> {
  if (!input.chunk) {
    return
  }
  const source = await input.getSource(input.chunk.filePath)
  const text = validatedChunkText(input.chunk, source, input.diagnostics, "related chunk text omitted")
  return {
    ...summarizeChunk(input.chunk, input.symbols),
    text: input.maxContextChars === undefined ? text : text.slice(0, input.maxContextChars),
  }
}

function validatedChunkText(
  chunk: ChunkRecord,
  source: SourceReadResult,
  diagnostics: string[],
  omittedReason: string,
) {
  if (!source.ok) {
    diagnostics.push(`source read failed for ${chunk.filePath}:${chunk.id}; ${omittedReason}`)
    return ""
  }
  if (!indexedChunkMatchesSource(source.text, chunk)) {
    diagnostics.push(`source mismatch for ${chunk.filePath}:${chunk.id}; ${omittedReason}`)
    return ""
  }
  return chunk.text
}

function readSourceCached(
  filePath: string,
  readSource: (filePath: string) => Promise<string>,
  sourceCache: Map<string, Promise<SourceReadResult>>,
) {
  const cached = sourceCache.get(filePath)
  if (cached) {
    return cached
  }
  const source = readSource(filePath)
    .then((text): SourceReadResult => ({ text, ok: true }))
    .catch((): SourceReadResult => ({ text: "", ok: false }))
  sourceCache.set(filePath, source)
  return source
}

function indexedChunkMatchesSource(source: string, chunk: ChunkRecord) {
  return textForByteSlice(source, chunk.range.byteStart, chunk.range.byteEnd) === chunk.text
}

function breadcrumbsFor(chunk: ChunkRecord, symbols: CastIndex["symbols"]) {
  return chunk.symbolIds.flatMap((id) => (symbols[id] ? [`${symbols[id].kind} ${symbols[id].name}`] : []))
}

function textForByteSlice(source: string, byteStart: number, byteEnd: number) {
  return decoder.decode(encoder.encode(source).slice(byteStart, byteEnd))
}

export { getChunkById }
