import { expandWithParentContext, summarizeChunk, summarizeTopology } from "./topology.js"
import type { CastIndex, ChunkLookupInput, ChunkLookupOutput, ChunkLookupRelatedChunk, ChunkRecord } from "./types.js"

async function getChunkById(input: {
  index: CastIndex
  input: ChunkLookupInput
  readSource(filePath: string): Promise<string>
}): Promise<ChunkLookupOutput> {
  const diagnostics = [...input.index.metadata.diagnostics]
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

  const context = await parentContext({
    chunk,
    diagnostics,
    includeParents: input.input.includeParents,
    index: input.index,
    maxContextChars: input.input.maxContextChars,
    readSource: input.readSource,
  })

  return {
    status: input.index.metadata,
    chunk: {
      filePath: chunk.filePath,
      language: chunk.language,
      range: chunk.range,
      kind: chunk.kind,
      breadcrumbs: context.breadcrumbs,
      text: chunk.text,
      parentText: context.parentText,
      parentRange: context.parentRange,
      topology: summarizeTopology(chunk, input.index.chunks, input.index.symbols),
      related: relatedChunks({
        chunk,
        chunks: input.index.chunks,
        includeChildren: input.input.includeChildren,
        includeParents: input.input.includeParents,
        includeSiblings: input.input.includeSiblings,
        maxContextChars: input.input.maxContextChars,
        symbols: input.index.symbols,
      }),
    },
    diagnostics,
  }
}

const encoder = new TextEncoder()
const decoder = new TextDecoder()

async function parentContext(input: {
  chunk: ChunkRecord
  diagnostics: string[]
  includeParents: boolean | undefined
  index: CastIndex
  maxContextChars: number | undefined
  readSource(filePath: string): Promise<string>
}): Promise<{ breadcrumbs: string[]; parentText?: string; parentRange?: ChunkRecord["range"] }> {
  if (input.includeParents === false) {
    return { breadcrumbs: breadcrumbsFor(input.chunk, input.index.symbols) }
  }

  const source = await input
    .readSource(input.chunk.filePath)
    .then((text) => ({ text, ok: true }))
    .catch(() => {
      input.diagnostics.push(`source read failed for ${input.chunk.filePath}; parent context omitted`)
      return { text: "", ok: false }
    })
  if (source.ok && indexedChunkMatchesSource(source.text, input.chunk)) {
    return expandWithParentContext({
      chunk: input.chunk,
      symbols: input.index.symbols,
      source: source.text,
      maxContextChars: input.maxContextChars ?? Number.MAX_SAFE_INTEGER,
    })
  }
  if (source.ok) {
    input.diagnostics.push(`source mismatch for ${input.chunk.filePath}:${input.chunk.id}; parent context omitted`)
  }
  return { breadcrumbs: breadcrumbsFor(input.chunk, input.index.symbols) }
}

function relatedChunks(input: {
  chunk: ChunkRecord
  chunks: CastIndex["chunks"]
  includeChildren: boolean | undefined
  includeParents: boolean | undefined
  includeSiblings: boolean | undefined
  maxContextChars: number | undefined
  symbols: CastIndex["symbols"]
}): NonNullable<ChunkLookupOutput["chunk"]>["related"] {
  return {
    parent:
      input.includeParents === false
        ? undefined
        : relatedChunk(input.chunks[input.chunk.parentChunkId ?? ""], input.symbols, input.maxContextChars),
    previousSibling:
      input.includeSiblings === false
        ? undefined
        : relatedChunk(input.chunks[input.chunk.previousSiblingChunkId ?? ""], input.symbols, input.maxContextChars),
    nextSibling:
      input.includeSiblings === false
        ? undefined
        : relatedChunk(input.chunks[input.chunk.nextSiblingChunkId ?? ""], input.symbols, input.maxContextChars),
    children:
      input.includeChildren === false
        ? []
        : input.chunk.childChunkIds.flatMap((id) => {
            const child = relatedChunk(input.chunks[id], input.symbols, input.maxContextChars)
            return child ? [child] : []
          }),
  }
}

function relatedChunk(
  chunk: ChunkRecord | undefined,
  symbols: CastIndex["symbols"],
  maxContextChars: number | undefined,
): ChunkLookupRelatedChunk | undefined {
  if (!chunk) {
    return
  }
  return {
    ...summarizeChunk(chunk, symbols),
    text: maxContextChars === undefined ? chunk.text : chunk.text.slice(0, maxContextChars),
  }
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
