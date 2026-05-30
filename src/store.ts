import { mkdir } from "node:fs/promises"
import path from "node:path"
import type { CastIndex } from "./types.js"

export const INDEX_SCHEMA_VERSION = 1

export function createEmptyIndex(input: {
  projectId: string
  worktree: string
  cacheKey: string
  maxChunkNonWhitespaceChars: number
  diagnostics?: string[]
}): CastIndex {
  return {
    metadata: {
      schemaVersion: INDEX_SCHEMA_VERSION,
      projectId: input.projectId,
      worktree: input.worktree,
      cacheKey: input.cacheKey,
      maxChunkNonWhitespaceChars: input.maxChunkNonWhitespaceChars,
      updatedAt: Date.now(),
      status: "empty",
      diagnostics: input.diagnostics ?? [],
    },
    files: {},
    chunks: {},
    symbols: {},
  }
}

export function createIndexStore(input: { cacheDir: string; cacheKey: string }) {
  const file = path.join(input.cacheDir, input.cacheKey, "index.json")
  return {
    async read() {
      if (!(await Bun.file(file).exists())) {
        return createEmptyIndex({
          projectId: input.cacheKey,
          worktree: "",
          cacheKey: input.cacheKey,
          maxChunkNonWhitespaceChars: 2000,
        })
      }
      try {
        const index = await Bun.file(file).json()
        if (isCastIndex(index)) {
          return index
        }
        return createEmptyIndex({
          projectId: input.cacheKey,
          worktree: "",
          cacheKey: input.cacheKey,
          maxChunkNonWhitespaceChars: 2000,
          diagnostics: ["rebuilding corrupt index"],
        })
      } catch {
        return createEmptyIndex({
          projectId: input.cacheKey,
          worktree: "",
          cacheKey: input.cacheKey,
          maxChunkNonWhitespaceChars: 2000,
          diagnostics: ["rebuilding corrupt index"],
        })
      }
    },
    async write(index: CastIndex) {
      await mkdir(path.dirname(file), { recursive: true })
      await Bun.write(file, JSON.stringify(index, null, 2))
    },
  }
}

export function cosineSimilarity(left: number[], right: number[]) {
  const dot = left.reduce((sum, value, index) => sum + value * (right[index] ?? 0), 0)
  const leftNorm = Math.sqrt(left.reduce((sum, value) => sum + value * value, 0))
  const rightNorm = Math.sqrt(right.reduce((sum, value) => sum + value * value, 0))
  return leftNorm && rightNorm ? dot / (leftNorm * rightNorm) : 0
}

export function searchVectors(query: number[], vectors: Array<{ id: string; vector: number[] }>, topK: number) {
  return vectors
    .map((vector) => ({ id: vector.id, score: cosineSimilarity(query, vector.vector) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(0, topK))
}

function isCastIndex(value: unknown): value is CastIndex {
  return (
    isObject(value) &&
    isIndexMetadata(value.metadata) &&
    isRecordMap(value.files, isFileRecord) &&
    isRecordMap(value.chunks, isChunkRecord) &&
    isRecordMap(value.symbols, isSymbolRecord) &&
    (value.lexical === undefined || isLexicalIndex(value.lexical))
  )
}

function isIndexMetadata(value: unknown) {
  if (!isObject(value)) {
    return false
  }
  return (
    value.schemaVersion === INDEX_SCHEMA_VERSION &&
    typeof value.projectId === "string" &&
    typeof value.worktree === "string" &&
    typeof value.cacheKey === "string" &&
    typeof value.maxChunkNonWhitespaceChars === "number" &&
    typeof value.updatedAt === "number" &&
    typeof value.status === "string" &&
    ["empty", "indexing", "ready", "stale", "error"].includes(value.status) &&
    isStringArray(value.diagnostics) &&
    isOptionalString(value.embeddingModel) &&
    (value.embeddingDimensions === undefined || typeof value.embeddingDimensions === "number")
  )
}

function isFileRecord(value: unknown) {
  return (
    isObject(value) &&
    typeof value.path === "string" &&
    typeof value.language === "string" &&
    typeof value.fingerprint === "string" &&
    isStringArray(value.chunkIds) &&
    isStringArray(value.diagnostics)
  )
}

function isChunkRecord(value: unknown) {
  return (
    isObject(value) &&
    typeof value.id === "string" &&
    typeof value.filePath === "string" &&
    typeof value.language === "string" &&
    typeof value.kind === "string" &&
    ["file", "class", "function", "method", "block", "fallback"].includes(value.kind) &&
    isSourceRange(value.range) &&
    typeof value.text === "string" &&
    typeof value.nonWhitespaceChars === "number" &&
    isStringArray(value.nodeTypes) &&
    isStringArray(value.symbolIds) &&
    isStringArray(value.childChunkIds) &&
    isOptionalString(value.parentChunkId) &&
    isOptionalString(value.previousSiblingChunkId) &&
    isOptionalString(value.nextSiblingChunkId) &&
    (value.embedding === undefined || isNumberArray(value.embedding)) &&
    isOptionalString(value.embeddingError) &&
    (value.lexical === undefined || isChunkLexicalStats(value.lexical))
  )
}

function isLexicalIndex(value: unknown) {
  return (
    isObject(value) &&
    isNonnegativeNumber(value.documentCount) &&
    isNonnegativeNumber(value.averageDocumentLength) &&
    isRecordMap(value.documentFrequencies, isNonnegativeNumber)
  )
}

function isChunkLexicalStats(value: unknown) {
  return isObject(value) && isNonnegativeNumber(value.length) && isRecordMap(value.termFrequencies, isNonnegativeNumber)
}

function isSymbolRecord(value: unknown) {
  return (
    isObject(value) &&
    typeof value.id === "string" &&
    typeof value.name === "string" &&
    typeof value.kind === "string" &&
    ["module", "class", "function", "method", "interface"].includes(value.kind) &&
    typeof value.filePath === "string" &&
    isSourceRange(value.range) &&
    isOptionalString(value.parentSymbolId) &&
    isStringArray(value.childSymbolIds)
  )
}

function isSourceRange(value: unknown) {
  return (
    isObject(value) &&
    typeof value.byteStart === "number" &&
    typeof value.byteEnd === "number" &&
    typeof value.lineStart === "number" &&
    typeof value.lineEnd === "number"
  )
}

function isRecordMap(value: unknown, isValue: (value: unknown) => boolean) {
  return isObject(value) && Object.values(value).every(isValue)
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function isStringArray(value: unknown) {
  return Array.isArray(value) && value.every((item) => typeof item === "string")
}

function isNumberArray(value: unknown) {
  return Array.isArray(value) && value.every((item) => typeof item === "number")
}

function isNonnegativeNumber(value: unknown) {
  return typeof value === "number" && value >= 0
}

function isOptionalString(value: unknown) {
  return value === undefined || typeof value === "string"
}
