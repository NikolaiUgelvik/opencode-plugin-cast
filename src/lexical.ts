import type { ChunkRecord, LexicalIndex, SymbolRecord } from "./types.js"

const BM25_K1 = 1.2
const BM25_B = 0.75
const BM25_IDF_SMOOTHING = 0.5
const TOKEN_PATTERN = /[A-Za-z0-9_]+(?:[./-][A-Za-z0-9_]+)*/g
const CAMEL_CASE_BOUNDARY_PATTERN = /([a-z0-9])([A-Z])/g
const IDENTIFIER_SEPARATOR_PATTERN = /[_.\-/\s]+/
const CONNECTOR_SUFFIX_SEPARATOR_PATTERN = /[/.]+/

function textForChunk(chunk: ChunkRecord, symbols: Record<string, SymbolRecord>) {
  const symbolText = chunk.symbolIds
    .flatMap((id) => {
      const symbol = symbols[id]
      return symbol ? [symbol.name, symbol.kind, symbol.filePath] : []
    })
    .join(" ")
  return [chunk.text, chunk.filePath, chunk.kind, ...chunk.nodeTypes, symbolText].join(" ")
}

function splitIdentifier(input: string) {
  return input
    .replace(CAMEL_CASE_BOUNDARY_PATTERN, "$1 $2")
    .split(IDENTIFIER_SEPARATOR_PATTERN)
    .map((part) => part.toLowerCase())
    .filter(Boolean)
}

function connectorSuffixes(input: string) {
  if (!CONNECTOR_SUFFIX_SEPARATOR_PATTERN.test(input)) {
    return []
  }

  const parts = input.split(CONNECTOR_SUFFIX_SEPARATOR_PATTERN).filter(Boolean)
  return parts.slice(1).map((_, index) => parts.slice(index + 1).join("."))
}

// biome-ignore lint/style/useConsistentTypeDefinitions: the public API explicitly exports this as a type alias.
export type RankedResult = { id: string; score: number }

export function tokenizeCodeText(input: string): string[] {
  const tokens: string[] = []
  for (const match of input.matchAll(TOKEN_PATTERN)) {
    const raw = match[0].toLowerCase()
    tokens.push(raw)
    tokens.push(...connectorSuffixes(raw))

    const parts = splitIdentifier(match[0])
    for (const part of parts) {
      tokens.push(part)
    }
    if (parts.length > 1) {
      tokens.push(parts.join(""))
    }
  }
  return tokens
}

export function buildLexicalIndex(
  chunks: Record<string, ChunkRecord>,
  symbols: Record<string, SymbolRecord>,
): { lexical: LexicalIndex; chunks: Record<string, ChunkRecord> } {
  const indexedChunks: Record<string, ChunkRecord> = {}
  const documentFrequencies: Record<string, number> = Object.create(null)
  let totalLength = 0

  for (const [id, chunk] of Object.entries(chunks)) {
    const terms = tokenizeCodeText(textForChunk(chunk, symbols))
    const termFrequencies: Record<string, number> = Object.create(null)
    for (const term of terms) {
      termFrequencies[term] = (termFrequencies[term] ?? 0) + 1
    }
    for (const term of new Set(terms)) {
      documentFrequencies[term] = (documentFrequencies[term] ?? 0) + 1
    }

    totalLength += terms.length
    indexedChunks[id] = {
      ...chunk,
      lexical: {
        length: terms.length,
        termFrequencies,
      },
    }
  }

  const documentCount = Object.keys(indexedChunks).length
  return {
    lexical: {
      documentCount,
      averageDocumentLength: documentCount === 0 ? 0 : totalLength / documentCount,
      documentFrequencies,
    },
    chunks: indexedChunks,
  }
}

export function bm25Search(
  query: string,
  chunks: ChunkRecord[],
  lexical: LexicalIndex | undefined,
  topK: number,
): RankedResult[] {
  if (
    !lexical ||
    lexical.documentCount === 0 ||
    lexical.averageDocumentLength === 0 ||
    chunks.length === 0 ||
    topK <= 0
  ) {
    return []
  }

  const queryTerms = new Set(tokenizeCodeText(query))
  const results = chunks.flatMap((chunk) => {
    if (!chunk.lexical || chunk.lexical.length === 0) {
      return []
    }

    let score = 0
    for (const term of queryTerms) {
      const frequency = chunk.lexical.termFrequencies[term] ?? 0
      if (frequency === 0) {
        continue
      }

      const documentFrequency = lexical.documentFrequencies[term] ?? 0
      const inverseDocumentFrequency = Math.log(
        1 + (lexical.documentCount - documentFrequency + BM25_IDF_SMOOTHING) / (documentFrequency + BM25_IDF_SMOOTHING),
      )
      const normalizedLength = 1 - BM25_B + BM25_B * (chunk.lexical.length / lexical.averageDocumentLength)
      score += inverseDocumentFrequency * ((frequency * (BM25_K1 + 1)) / (frequency + BM25_K1 * normalizedLength))
    }

    return score > 0 ? [{ id: chunk.id, score }] : []
  })

  return results.sort((a, b) => b.score - a.score || a.id.localeCompare(b.id)).slice(0, topK)
}

export function reciprocalRankFusion(input: {
  lists: { weight: number; results: RankedResult[] }[]
  rrfK: number
  topK: number
}): RankedResult[] {
  if (input.topK <= 0) {
    return []
  }

  const scores = new Map<string, number>()
  for (const list of input.lists) {
    list.results.forEach((result, index) => {
      scores.set(result.id, (scores.get(result.id) ?? 0) + list.weight / (input.rrfK + index + 1))
    })
  }

  return Array.from(scores, ([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))
    .slice(0, input.topK)
}
