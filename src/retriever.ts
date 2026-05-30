import { bm25Search, type RankedResult, reciprocalRankFusion } from "./lexical.js"
import { searchVectors } from "./store.js"
import { expandWithParentContext, summarizeTopology } from "./topology.js"
import type {
  CastIndex,
  ChunkRecord,
  HybridRetrievalMode,
  HybridRetrievalOptions,
  RerankOptions,
  SearchInput,
  SearchOutput,
  SearchResultRetrievalDetails,
} from "./types.js"

const encoder = new TextEncoder()
const decoder = new TextDecoder()
const CANDIDATE_MULTIPLIER = 3
const DEFAULT_MIN_FINAL_SCORE = 0.01
const GLOB_SYNTAX_PATTERN = /[*?[]/
const REGEXP_SPECIAL_CHAR_PATTERN = /[\\^$.*+?()[\]{}|]/
const CHARACTER_CLASS_SPECIAL_PATTERN = /[\\\]^]/
const globRegExpCache = new Map<string, RegExp>()

interface VectorCandidateSource {
  searchVectorCandidates(queryEmbedding: number[], topK: number, paths?: string[]): Promise<RankedResult[]>
}

export async function retrieve(input: {
  index: CastIndex
  input: SearchInput
  options: {
    topK: number
    maxContextChars: number
    hyde: { enabled: boolean; threshold: number }
    hybrid?: HybridRetrievalOptions
    rerank?: RerankOptions
  }
  embed(text: string): Promise<number[]>
  generateHyde(query: string): Promise<string>
  rerank?(query: string, documents: string[]): Promise<Array<{ index: number; score: number }>>
  readSource(filePath: string): Promise<string>
  indexStore?: VectorCandidateSource
}): Promise<SearchOutput> {
  const topK = input.input.topK ?? input.options.topK
  const maxContextChars = input.input.maxContextChars ?? input.options.maxContextChars
  const minFinalScore = Math.max(0, input.input.minFinalScore ?? DEFAULT_MIN_FINAL_SCORE)
  const rerank = input.options.rerank
  const rankingTopK = rerank ? Math.max(topK * rerank.candidateMultiplier, topK) : topK
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
    .sort((left, right) => left.id.localeCompare(right.id))
  const queryVector = await input.embed(input.input.query)
  const hybrid = input.options.hybrid
  const canUseHybrid = Boolean(hybrid?.enabled && input.index.lexical && chunks.some((chunk) => chunk.lexical))
  if (hybrid?.enabled && !canUseHybrid) {
    diagnostics.push("hybrid retrieval requested but lexical data is unavailable; using vector-only retrieval")
  }
  const searchCandidateCount = Math.max(
    canUseHybrid && hybrid?.mode === "vector-prefilter"
      ? Math.max(vectors.length, chunks.length)
      : rankingTopK * (canUseHybrid ? (hybrid?.vectorCandidateMultiplier ?? 1) : CANDIDATE_MULTIPLIER),
    rankingTopK,
  )
  const searchVectorCandidates = async (vector: number[]) => {
    if (input.indexStore?.searchVectorCandidates) {
      const candidates = await input.indexStore.searchVectorCandidates(vector, searchCandidateCount, input.input.paths)
      return candidates.filter((candidate) => chunksById[candidate.id])
    }
    return searchVectors(vector, vectors, searchCandidateCount)
  }
  const initial = await searchVectorCandidates(queryVector)
  const bestScore = initial[0]?.score
  const initialScores = Object.fromEntries(initial.map((result) => [result.id, result.score]))
  const hyde =
    input.options.hyde.enabled && (bestScore ?? -1) < input.options.hyde.threshold
      ? await input
          .generateHyde(input.input.query)
          .then((text) => input.embed(text))
          .then(async (vector) => ({
            scored: await searchVectorCandidates(vector),
            hydeUsed: true,
            diagnostics: [] as string[],
          }))
          .catch((error) => {
            if (isIndexUnavailableError(error)) {
              throw error
            }
            return {
              scored: initial,
              hydeUsed: false,
              diagnostics: [`HyDE failed: ${error instanceof Error ? error.message : String(error)}`],
            }
          })
      : { scored: initial, hydeUsed: false, diagnostics: [] }
  let ranked = canUseHybrid
    ? hybridResults({
        query: input.input.query,
        chunks,
        lexical: input.index.lexical,
        topK: rankingTopK,
        vectorCandidates: hyde.scored,
        hybrid: hybrid as HybridRetrievalOptions,
      })
    : {
        results: hyde.scored.slice(0, rankingTopK),
        retrieval: new Map(
          hyde.scored.map((result, index) => [result.id, { mode: "vector" as const, vectorRank: index + 1 }]),
        ),
      }
  let rerankUsed = false
  if (rerank && ranked.results.length > 0) {
    try {
      if (!input.rerank) {
        throw new Error("rerank dependency unavailable")
      }
      ranked = {
        ...ranked,
        results: await rerankResults({
          query: input.input.query,
          results: ranked.results,
          chunksById,
          retrieval: ranked.retrieval,
          rerank: input.rerank,
        }),
      }
      rerankUsed = true
    } catch (error) {
      diagnostics.push(`Rerank failed: ${error instanceof Error ? error.message : String(error)}`)
    }
  }
  const candidateResults = ranked.results.slice(0, topK)
  const filteredRankedResults = candidateResults.filter((result) => result.score >= minFinalScore)
  const filteredCount = candidateResults.length - filteredRankedResults.length
  const seenParentRanges = new Set<string>()
  const results = (
    await Promise.all(
      filteredRankedResults.flatMap(async (result) => {
        const chunk = chunksById[result.id]
        if (!chunk) {
          return []
        }

        const source = await input
          .readSource(chunk.filePath)
          .then((text) => ({ text, ok: true }))
          .catch(() => {
            diagnostics.push(`source read failed for ${chunk.filePath}; parent context omitted`)
            return { text: "", ok: false }
          })
        if (source.ok && !indexedChunkMatchesSource(source.text, chunk)) {
          diagnostics.push(`source mismatch for ${chunk.filePath}:${chunk.id}; parent context omitted`)
        }
        const outputText = source.ok && indexedChunkMatchesSource(source.text, chunk) ? chunk.text : ""
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
            text: outputText,
            parentText: context.parentText,
            parentRange: context.parentRange,
            topology: summarizeTopology(chunk, chunksById, input.index.symbols),
            retrieval: ranked.retrieval.get(result.id),
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
    status: {
      ...input.index.metadata,
      hydeUsed: hyde.hydeUsed,
      bestScore,
      rerankUsed,
      minFinalScore,
      filteredCount,
      candidateCount: candidateResults.length,
    },
    results,
    diagnostics: [...diagnostics, ...hyde.diagnostics],
  }
}

function isIndexUnavailableError(error: unknown) {
  return error instanceof Error && error.name === "IndexUnavailableError"
}

function hybridResults(input: {
  query: string
  chunks: ChunkRecord[]
  lexical: CastIndex["lexical"]
  topK: number
  vectorCandidates: RankedResult[]
  hybrid: HybridRetrievalOptions
}) {
  const bm25CandidateCount = Math.max(input.topK * input.hybrid.bm25CandidateMultiplier, input.topK)
  const vectorCandidateCount = Math.max(input.topK * input.hybrid.vectorCandidateMultiplier, input.topK)
  const allBm25 = bm25Search(input.query, input.chunks, input.lexical, bm25CandidateCount)
  const vectorCandidates =
    input.hybrid.mode === "vector-prefilter"
      ? includeScoreTies(input.vectorCandidates, vectorCandidateCount)
      : input.vectorCandidates.slice(0, vectorCandidateCount)
  const bm25Candidates = candidatesForMode(input.hybrid.mode, {
    query: input.query,
    chunks: input.chunks,
    lexical: input.lexical,
    allBm25,
    vectorCandidates,
  })
  const filteredVectorCandidates = vectorCandidatesForMode(input.hybrid.mode, {
    vectorCandidates,
    bm25Candidates,
  })
  const results = reciprocalRankFusion({
    lists: [
      { weight: input.hybrid.vectorWeight, results: filteredVectorCandidates },
      { weight: input.hybrid.bm25Weight, results: bm25Candidates },
    ],
    rrfK: input.hybrid.rrfK,
    topK: input.topK,
  })
  const vectorRanks = rankMap(filteredVectorCandidates)
  const bm25Ranks = rankMap(bm25Candidates)
  const bm25Scores = new Map(bm25Candidates.map((result) => [result.id, result.score]))
  const retrieval = new Map(
    results.map((result) => [
      result.id,
      {
        mode: "hybrid" as const,
        hybridMode: input.hybrid.mode,
        vectorRank: vectorRanks.get(result.id),
        bm25Rank: bm25Ranks.get(result.id),
        bm25Score: bm25Scores.get(result.id),
      },
    ]),
  )

  return { results, retrieval }
}

function candidatesForMode(
  mode: HybridRetrievalMode,
  input: {
    query: string
    chunks: ChunkRecord[]
    lexical: CastIndex["lexical"]
    allBm25: RankedResult[]
    vectorCandidates: RankedResult[]
  },
) {
  if (mode !== "vector-prefilter") {
    return input.allBm25
  }
  const vectorIds = new Set(input.vectorCandidates.map((result) => result.id))
  return bm25Search(
    input.query,
    input.chunks.filter((chunk) => vectorIds.has(chunk.id)),
    input.lexical,
    input.allBm25.length,
  )
}

function vectorCandidatesForMode(
  mode: HybridRetrievalMode,
  input: {
    vectorCandidates: RankedResult[]
    bm25Candidates: RankedResult[]
  },
) {
  if (mode !== "bm25-prefilter") {
    return input.vectorCandidates
  }
  const bm25Ids = new Set(input.bm25Candidates.map((result) => result.id))
  return input.vectorCandidates.filter((result) => bm25Ids.has(result.id))
}

function rankMap(results: RankedResult[]) {
  return new Map(results.map((result, index) => [result.id, index + 1]))
}

function includeScoreTies(results: RankedResult[], limit: number) {
  const cutoffScore = results[limit - 1]?.score
  if (cutoffScore === undefined) {
    return results.slice()
  }
  return results.filter((result) => result.score >= cutoffScore)
}

async function rerankResults(input: {
  query: string
  results: RankedResult[]
  chunksById: Record<string, ChunkRecord>
  retrieval: Map<string, SearchResultRetrievalDetails>
  rerank(query: string, documents: string[]): Promise<Array<{ index: number; score: number }>>
}) {
  const candidates = input.results.flatMap((result) => {
    const chunk = input.chunksById[result.id]
    return chunk ? [{ result, chunk }] : []
  })
  const reranked = await input.rerank(
    input.query,
    candidates.map(({ chunk }) => rerankDocument(chunk)),
  )

  return reranked.flatMap((rerankedResult, index) => {
    const candidate = candidates[rerankedResult.index]
    if (!candidate) {
      return []
    }
    const existing = input.retrieval.get(candidate.result.id)
    input.retrieval.set(candidate.result.id, {
      ...(existing ?? { mode: "vector" }),
      rerankRank: index + 1,
      rerankScore: rerankedResult.score,
    })
    return [{ id: candidate.result.id, score: rerankedResult.score }]
  })
}

function rerankDocument(chunk: ChunkRecord) {
  return `${chunk.filePath}:${formatLineRange(chunk.range.lineStart, chunk.range.lineEnd)}\nkind: ${chunk.kind}\n${chunk.text}`
}

function formatLineRange(lineStart: number, lineEnd: number) {
  return lineStart === lineEnd ? String(lineStart) : `${lineStart}-${lineEnd}`
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
  return paths.some((filter) => {
    if (hasGlobSyntax(filter)) {
      return globToRegExp(filter).test(filePath)
    }
    return filePath === filter || filePath.startsWith(filter.endsWith("/") ? filter : `${filter}/`)
  })
}

function hasGlobSyntax(filter: string) {
  return GLOB_SYNTAX_PATTERN.test(filter)
}

function globToRegExp(glob: string) {
  const cached = globRegExpCache.get(glob)
  if (cached) {
    return cached
  }
  let pattern = "^"
  for (let index = 0; index < glob.length; index++) {
    const part = globPatternPart(glob, index)
    pattern += part.pattern
    index = part.endIndex
  }
  const expression = new RegExp(`${pattern}$`)
  globRegExpCache.set(glob, expression)
  return expression
}

function globPatternPart(glob: string, index: number) {
  const char = glob[index]
  const next = glob[index + 1]
  if (char === "*" && next === "*" && glob[index + 2] === "/") {
    return { pattern: "(?:.*/)?", endIndex: index + 2 }
  }
  if (char === "*" && next === "*") {
    return { pattern: ".*", endIndex: index + 1 }
  }
  if (char === "*") {
    return { pattern: "[^/]*", endIndex: index }
  }
  if (char === "?") {
    return { pattern: "[^/]", endIndex: index }
  }
  if (char === "[") {
    return globCharacterClass(glob, index) ?? { pattern: escapeRegExp(char), endIndex: index }
  }
  if (char === "\\" && next) {
    return { pattern: escapeRegExp(next), endIndex: index + 1 }
  }
  return { pattern: escapeRegExp(char), endIndex: index }
}

function globCharacterClass(glob: string, startIndex: number) {
  let endIndex = -1
  for (let index = startIndex + 1; index < glob.length; index++) {
    if (glob[index] === "]" && glob[index - 1] !== "\\") {
      endIndex = index
      break
    }
  }
  if (endIndex <= startIndex + 1) {
    return
  }

  const content = glob.slice(startIndex + 1, endIndex)
  if (content.includes("/")) {
    return
  }

  return { pattern: `[${escapeCharacterClassContent(content)}]`, endIndex }
}

function escapeCharacterClassContent(content: string) {
  let escaped = ""
  for (let index = 0; index < content.length; index++) {
    const char = content[index]
    if (char === "-" && index > 0 && index < content.length - 1) {
      escaped += char
      continue
    }
    escaped += CHARACTER_CLASS_SPECIAL_PATTERN.test(char) ? `\\${char}` : char
  }
  return escaped
}

function escapeRegExp(char: string) {
  return REGEXP_SPECIAL_CHAR_PATTERN.test(char) ? `\\${char}` : char
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
