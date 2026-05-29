import { searchVectors } from "./store.js"
import { expandWithParentContext } from "./topology.js"
import type { CastIndex, ChunkRecord, SearchInput, SearchOutput } from "./types.js"

const encoder = new TextEncoder()
const decoder = new TextDecoder()
const CANDIDATE_MULTIPLIER = 3

export async function retrieve(input: {
  index: CastIndex
  input: SearchInput
  options: { topK: number; maxContextChars: number; hyde: { enabled: boolean; threshold: number } }
  embed(text: string): Promise<number[]>
  generateHyde(query: string): Promise<string>
  readSource(filePath: string): Promise<string>
}): Promise<SearchOutput> {
  const topK = input.input.topK ?? input.options.topK
  const maxContextChars = input.input.maxContextChars ?? input.options.maxContextChars
  const diagnostics = [
    ...input.index.metadata.diagnostics,
    ...Object.values(input.index.files)
      .filter((file) => file.diagnostics.length > 0 && matchesPaths(file.path, input.input.paths))
      .flatMap((file) => file.diagnostics.map((diagnostic) => `${file.path}: ${diagnostic}`)),
  ]
  const chunks = Object.entries(input.index.chunks)
    .flatMap(([key, chunk]) => {
      if (key === chunk.id) {
        return [chunk]
      }
      diagnostics.push(`chunk key mismatch: ${key} contains ${chunk.id}; chunk skipped`)
      return []
    })
    .filter((chunk) => matchesPaths(chunk.filePath, input.input.paths))
  const chunksById = Object.fromEntries(chunks.map((chunk) => [chunk.id, chunk]))
  const vectors = chunks
    .filter((chunk): chunk is ChunkRecord & { embedding: number[] } => Boolean(chunk.embedding))
    .map((chunk) => ({ id: chunk.id, vector: chunk.embedding }))
  const queryVector = await input.embed(input.input.query)
  const candidateCount = Math.max(topK * CANDIDATE_MULTIPLIER, topK)
  const initial = searchVectors(queryVector, vectors, candidateCount)
  const bestScore = initial[0]?.score
  const initialScores = Object.fromEntries(initial.map((result) => [result.id, result.score]))
  const hyde =
    input.options.hyde.enabled && (bestScore ?? -1) < input.options.hyde.threshold
      ? await input
          .generateHyde(input.input.query)
          .then((text) => input.embed(text))
          .then((vector) => ({
            scored: searchVectors(vector, vectors, candidateCount),
            hydeUsed: true,
            diagnostics: [] as string[],
          }))
          .catch((error) => ({
            scored: initial,
            hydeUsed: false,
            diagnostics: [`HyDE failed: ${error instanceof Error ? error.message : String(error)}`],
          }))
      : { scored: initial, hydeUsed: false, diagnostics: [] }
  const seenParentRanges = new Set<string>()
  const results = (
    await Promise.all(
      hyde.scored.slice(0, topK).flatMap(async (result) => {
        const chunk = chunksById[result.id]
        if (!chunk) {
          return []
        }

        const source = await input
          .readSource(chunk.filePath)
          .then((text) => ({ text, ok: true }))
          .catch(() => {
            diagnostics.push(`source read failed for ${chunk.filePath}; parent context omitted`)
            return { text: chunk.text, ok: false }
          })
        if (source.ok && !indexedChunkMatchesSource(source.text, chunk)) {
          diagnostics.push(`source mismatch for ${chunk.filePath}:${chunk.id}; parent context omitted`)
        }
        const context = parentContext({
          chunk,
          includeParents: input.input.includeParents,
          maxContextChars,
          source,
          symbols: input.index.symbols,
        })

        return [
          {
            filePath: chunk.filePath,
            language: chunk.language,
            range: chunk.range,
            score: initialScores[result.id] ?? result.score,
            finalScore: result.score,
            kind: chunk.kind,
            breadcrumbs: context.breadcrumbs,
            text: chunk.text,
            parentText: context.parentText,
            parentRange: context.parentRange,
            topology: {
              chunkId: chunk.id,
              parentChunkId: chunk.parentChunkId,
              childChunkIds: chunk.childChunkIds,
              previousSiblingChunkId: chunk.previousSiblingChunkId,
              nextSiblingChunkId: chunk.nextSiblingChunkId,
              symbolIds: chunk.symbolIds,
            },
          },
        ]
      }),
    )
  )
    .flat()
    .map((result) => {
      if (!result.parentRange) {
        return result
      }
      const parentRangeKey = `${result.filePath}:${result.parentRange.byteStart}:${result.parentRange.byteEnd}:${result.parentText}`
      if (seenParentRanges.has(parentRangeKey)) {
        return { ...result, parentText: undefined, parentRange: undefined }
      }
      seenParentRanges.add(parentRangeKey)
      return result
    })

  return {
    status: { ...input.index.metadata, hydeUsed: hyde.hydeUsed, bestScore },
    results,
    diagnostics: [...diagnostics, ...hyde.diagnostics],
  }
}

function parentContext(input: {
  chunk: ChunkRecord
  includeParents: boolean | undefined
  maxContextChars: number
  source: { text: string; ok: boolean }
  symbols: CastIndex["symbols"]
}) {
  if (input.includeParents === false) {
    return { breadcrumbs: [] }
  }
  if (input.source.ok && indexedChunkMatchesSource(input.source.text, input.chunk)) {
    return expandWithParentContext({
      chunk: input.chunk,
      symbols: input.symbols,
      source: input.source.text,
      maxContextChars: input.maxContextChars,
    })
  }
  return { breadcrumbs: breadcrumbsFor(input.chunk, input.symbols) }
}

function matchesPaths(filePath: string, paths: string[] | undefined) {
  if (!paths || paths.length === 0) {
    return true
  }
  return paths.some(
    (filter) => filePath === filter || filePath.startsWith(filter.endsWith("/") ? filter : `${filter}/`),
  )
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
