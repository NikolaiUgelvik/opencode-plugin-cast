export type SourceRange = {
  byteStart: number
  byteEnd: number
  lineStart: number
  lineEnd: number
}

export type ChunkKind = "file" | "class" | "function" | "method" | "block" | "fallback"

export type HybridRetrievalMode = "parallel" | "bm25-prefilter" | "vector-prefilter"

export type HybridRetrievalOptions = {
  enabled: boolean
  mode: HybridRetrievalMode
  rrfK: number
  vectorCandidateMultiplier: number
  bm25CandidateMultiplier: number
  vectorWeight: number
  bm25Weight: number
}

export type LexicalIndex = {
  documentCount: number
  averageDocumentLength: number
  documentFrequencies: Record<string, number>
}

export type ChunkLexicalStats = {
  length: number
  termFrequencies: Record<string, number>
}

export type SearchResultRetrievalDetails = {
  mode: "vector" | "hybrid"
  hybridMode?: HybridRetrievalMode
  vectorRank?: number
  bm25Rank?: number
  bm25Score?: number
}

export type ChunkRecord = {
  id: string
  filePath: string
  language: string
  kind: ChunkKind
  range: SourceRange
  text: string
  nonWhitespaceChars: number
  nodeTypes: string[]
  symbolIds: string[]
  parentChunkId?: string
  childChunkIds: string[]
  previousSiblingChunkId?: string
  nextSiblingChunkId?: string
  embedding?: number[]
  embeddingError?: string
  lexical?: ChunkLexicalStats
}

export type SymbolRecord = {
  id: string
  name: string
  kind: "module" | "class" | "function" | "method" | "interface"
  filePath: string
  range: SourceRange
  parentSymbolId?: string
  childSymbolIds: string[]
}

export type FileRecord = {
  path: string
  language: string
  fingerprint: string
  chunkIds: string[]
  diagnostics: string[]
}

export type IndexMetadata = {
  schemaVersion: number
  projectId: string
  worktree: string
  cacheKey: string
  embeddingModel?: string
  embeddingDimensions?: number
  maxChunkNonWhitespaceChars: number
  updatedAt: number
  status: "empty" | "indexing" | "ready" | "stale" | "error"
  diagnostics: string[]
}

export type CastIndex = {
  metadata: IndexMetadata
  files: Record<string, FileRecord>
  chunks: Record<string, ChunkRecord>
  symbols: Record<string, SymbolRecord>
  lexical?: LexicalIndex
}

export type SearchInput = {
  query: string
  topK?: number
  maxContextChars?: number
  includeParents?: boolean
  refresh?: boolean
  paths?: string[]
}

export type TopologyNode = {
  id: string
  label: string
  range: string
}

export type SearchResultTopology = {
  chunk: TopologyNode
  parent?: TopologyNode
  children: TopologyNode[]
  previousSibling?: TopologyNode
  nextSibling?: TopologyNode
  symbols: string[]
}

export type ChunkLookupInput = {
  id: string
  includeParents?: boolean
  includeSiblings?: boolean
  includeChildren?: boolean
  maxContextChars?: number
}

export type ChunkLookupOutput = {
  status: IndexMetadata
  chunk?: {
    filePath: string
    language: string
    range: SourceRange
    kind: ChunkKind
    breadcrumbs: string[]
    text: string
    parentText?: string
    parentRange?: SourceRange
    topology: SearchResultTopology
    related: {
      parent?: ChunkLookupRelatedChunk
      previousSibling?: ChunkLookupRelatedChunk
      nextSibling?: ChunkLookupRelatedChunk
      children: ChunkLookupRelatedChunk[]
    }
  }
  diagnostics: string[]
}

export type ChunkLookupRelatedChunk = {
  id: string
  label: string
  range: string
  text?: string
}

export type SearchResult = {
  filePath: string
  language: string
  range: SourceRange
  score: number
  finalScore: number
  kind: ChunkKind
  breadcrumbs: string[]
  text: string
  parentText?: string
  parentRange?: SourceRange
  topology: SearchResultTopology
  retrieval?: SearchResultRetrievalDetails
}

export type SearchOutput = {
  status: IndexMetadata & { hydeUsed: boolean; bestScore?: number }
  results: SearchResult[]
  diagnostics: string[]
}
