import { Database } from "bun:sqlite"
import { mkdir } from "node:fs/promises"
import path from "node:path"
import { load as loadSqliteVec } from "sqlite-vec"
import type { CastIndex, ChunkingOptions, ChunkRecord, FileRecord, LexicalIndex, SymbolRecord } from "./types.js"

export const INDEX_SCHEMA_VERSION = 1
const SQLITE_SCHEMA_VERSION = 2
const GLOB_SYNTAX_PATTERN = /[*?[]/
const REGEXP_SPECIAL_CHAR_PATTERN = /[\\^$.*+?()[\]{}|]/
const CHARACTER_CLASS_SPECIAL_PATTERN = /[\\\]^]/
const RUN_ID_RANDOM_RADIX = 36
const globRegExpCache = new Map<string, RegExp>()

const DEFAULT_CHUNKING_OPTIONS: ChunkingOptions = {
  overlap: 0,
  expansion: false,
  minSemanticNonWhitespaceChars: 8,
}

class CorruptIndexError extends Error {
  constructor(cause?: unknown) {
    super("corrupt persisted index", { cause })
    this.name = "CorruptIndexError"
  }
}

export interface FileResult {
  file: FileRecord
  chunks: Record<string, ChunkRecord>
  symbols: Record<string, SymbolRecord>
}

export function createEmptyIndex(input: {
  projectId: string
  worktree: string
  cacheKey: string
  maxChunkNonWhitespaceChars: number
  chunking?: ChunkingOptions
  diagnostics?: string[]
}): CastIndex {
  return {
    metadata: {
      schemaVersion: INDEX_SCHEMA_VERSION,
      projectId: input.projectId,
      worktree: input.worktree,
      cacheKey: input.cacheKey,
      maxChunkNonWhitespaceChars: input.maxChunkNonWhitespaceChars,
      chunking: input.chunking ?? DEFAULT_CHUNKING_OPTIONS,
      updatedAt: Date.now(),
      status: "empty",
      diagnostics: input.diagnostics ?? [],
    },
    files: {},
    chunks: {},
    symbols: {},
  }
}

export function createIndexStore(input: { cacheDir: string; cacheKey: string; embeddingDimensions?: number }) {
  return createSqliteIndexStore(input.cacheDir, input.cacheKey, input.embeddingDimensions)
}

function createSqliteIndexStore(cacheDir: string, cacheKey: string, embeddingDimensions?: number) {
  const file = path.join(cacheDir, cacheKey, "index.sqlite")
  return {
    async read() {
      const db = await openSqliteIndex(file, embeddingDimensions)
      try {
        return readSqliteIndex(db, cacheKey, embeddingDimensions)
      } finally {
        db.close()
      }
    },
    async write(index: CastIndex) {
      const db = await openSqliteIndex(file, embeddingDimensions ?? inferEmbeddingDimensions(index))
      try {
        writeSqliteIndex(db, index)
      } finally {
        db.close()
      }
    },
    async searchVectorCandidates(queryEmbedding: number[], topK: number, paths?: string[]) {
      const db = await openSqliteIndex(file, embeddingDimensions ?? queryEmbedding.length)
      try {
        return searchSqliteVectorCandidates(db, queryEmbedding, topK, paths)
      } finally {
        db.close()
      }
    },
    async beginIndexRun(input: { configHash: string; metadata: CastIndex["metadata"] }) {
      const db = await openSqliteIndex(file, embeddingDimensions)
      try {
        return beginSqliteIndexRun(db, input.configHash, input.metadata)
      } finally {
        db.close()
      }
    },
    async getCompletedFile(runId: string, filePath: string, fingerprint: string) {
      const db = await openSqliteIndex(file, embeddingDimensions)
      try {
        return getCompletedSqliteFile(db, runId, filePath, fingerprint)
      } finally {
        db.close()
      }
    },
    async writeFileResult(runId: string, fileResult: FileResult) {
      const db = await openSqliteIndex(file, embeddingDimensions ?? inferFileResultEmbeddingDimensions(fileResult))
      try {
        writeSqliteFileResult(db, runId, fileResult)
      } finally {
        db.close()
      }
    },
    async activateRun(runId: string, index: CastIndex) {
      const db = await openSqliteIndex(file, embeddingDimensions ?? inferEmbeddingDimensions(index))
      try {
        activateSqliteRun(db, runId, index)
      } finally {
        db.close()
      }
    },
  }
}

async function openSqliteIndex(file: string, embeddingDimensions?: number) {
  await mkdir(path.dirname(file), { recursive: true })
  const db = new Database(file)
  try {
    loadSqliteVec(db)
    initializeSchema(db, embeddingDimensions)
    return db
  } catch (error) {
    db.close()
    throw error
  }
}

function readSqliteIndex(db: Database, cacheKey: string, embeddingDimensions?: number) {
  const activeRun = db.query("select value from meta where key = 'active_run_id'").get() as { value: string } | null
  if (!activeRun) {
    return createEmptySqliteIndex(cacheKey, embeddingDimensions)
  }

  try {
    const run = db.query("select metadata_json as metadataJson from runs where id = ?").get(activeRun.value) as {
      metadataJson: string
    } | null
    if (!run) {
      return createEmptySqliteIndex(cacheKey, embeddingDimensions, ["rebuilding corrupt index"])
    }

    const index: CastIndex = {
      metadata: parsePersistedJson(run.metadataJson),
      files: readFiles(db, activeRun.value),
      chunks: readChunks(db, activeRun.value, readVectors(db, activeRun.value)),
      symbols: readSymbols(db, activeRun.value),
    }
    const lexical = readLexical(db, activeRun.value)
    if (lexical) {
      index.lexical = lexical
    }

    return isCastIndex(index)
      ? index
      : createEmptySqliteIndex(cacheKey, embeddingDimensions, ["rebuilding corrupt index"])
  } catch (error) {
    if (!(error instanceof CorruptIndexError)) {
      throw error
    }
    return createEmptySqliteIndex(cacheKey, embeddingDimensions, ["rebuilding corrupt index"])
  }
}

function createEmptySqliteIndex(cacheKey: string, embeddingDimensions?: number, diagnostics?: string[]) {
  const index = createEmptyIndex({
    projectId: cacheKey,
    worktree: "",
    cacheKey,
    maxChunkNonWhitespaceChars: 2000,
    diagnostics,
  })
  if (embeddingDimensions !== undefined) {
    index.metadata.embeddingDimensions = embeddingDimensions
  }
  return index
}

function readFiles(db: Database, runId: string) {
  const records: Record<string, FileRecord> = {}
  const files = db
    .query(
      `select file_runs.path,
              coalesce(file_runs.language, files.language) as language,
              coalesce(file_runs.fingerprint, files.fingerprint) as fingerprint,
              coalesce(file_runs.diagnostics_json, files.diagnostics_json) as diagnosticsJson,
              file_runs.chunk_ids_json as chunkIdsJson
       from file_runs
       left join files on files.path = file_runs.path
       where file_runs.run_id = ?`,
    )
    .all(runId) as Array<{
    path: string
    language: string
    fingerprint: string
    diagnosticsJson: string
    chunkIdsJson: string
  }>
  for (const file of files) {
    records[file.path] = {
      path: file.path,
      language: file.language,
      fingerprint: file.fingerprint,
      chunkIds: parsePersistedJson(file.chunkIdsJson),
      diagnostics: parsePersistedJson(file.diagnosticsJson),
    }
  }
  return records
}

function readVectors(db: Database, runId: string) {
  const vectors = new Map<string, number[]>()
  if (!tableExists(db, "chunk_vectors")) {
    return vectors
  }
  const vectorRows = db
    .query(
      "select chunk_rowids.chunk_id as chunkId, vec_to_json(chunk_vectors.embedding) as embedding from chunk_rowids inner join chunk_vectors on chunk_vectors.rowid = chunk_rowids.rowid where chunk_rowids.run_id = ?",
    )
    .all(runId) as Array<{ chunkId: string; embedding: string }>
  for (const row of vectorRows) {
    vectors.set(row.chunkId, parsePersistedJson(row.embedding))
  }
  return vectors
}

function readChunks(db: Database, runId: string, vectors: Map<string, number[]>) {
  const records: Record<string, ChunkRecord> = {}
  const chunks = db.query("select id, record_json as recordJson from chunks where run_id = ?").all(runId) as Array<{
    id: string
    recordJson: string
  }>
  for (const chunk of chunks) {
    const record = parsePersistedJson<ChunkRecord>(chunk.recordJson)
    const embedding = vectors.get(chunk.id)
    if (embedding) {
      record.embedding = embedding
    }
    records[chunk.id] = record
  }
  return records
}

function readSymbols(db: Database, runId: string) {
  const records: Record<string, SymbolRecord> = {}
  const symbols = db.query("select id, record_json as recordJson from symbols where run_id = ?").all(runId) as Array<{
    id: string
    recordJson: string
  }>
  for (const symbol of symbols) {
    records[symbol.id] = parsePersistedJson(symbol.recordJson)
  }
  return records
}

function readLexical(db: Database, runId: string) {
  const lexical = db.query("select metadata_json as metadataJson from lexical where run_id = ?").get(runId) as {
    metadataJson: string
  } | null
  return lexical ? parsePersistedJson<LexicalIndex>(lexical.metadataJson) : undefined
}

function parsePersistedJson<T = unknown>(json: string): T {
  try {
    return JSON.parse(json) as T
  } catch (error) {
    throw new CorruptIndexError(error)
  }
}

function writeSqliteIndex(db: Database, index: CastIndex) {
  const runId = `ready-${Date.now()}`
  const write = db.transaction((castIndex: CastIndex) => {
    clearSqliteIndex(db)
    insertRun(db, runId, castIndex)

    for (const file of Object.values(castIndex.files)) {
      insertFile(db, runId, file)
    }

    let vectorRowid = 1
    for (const chunk of Object.values(castIndex.chunks)) {
      const { embedding: _embedding, ...storedChunk } = chunk
      db.run("insert into chunks (run_id, id, file_path, kind, record_json) values (?, ?, ?, ?, ?)", [
        runId,
        chunk.id,
        chunk.filePath,
        chunk.kind,
        JSON.stringify(storedChunk),
      ])
      if (chunk.embedding) {
        db.run("insert into chunk_vectors (rowid, embedding) values (?, ?)", [
          vectorRowid,
          JSON.stringify(chunk.embedding),
        ])
        db.run("insert into chunk_rowids (run_id, chunk_id, rowid) values (?, ?, ?)", [runId, chunk.id, vectorRowid])
        vectorRowid += 1
      }
    }

    for (const symbol of Object.values(castIndex.symbols)) {
      insertSymbol(db, runId, symbol)
    }

    if (castIndex.lexical) {
      insertLexical(db, runId, castIndex.lexical)
    }
    db.run("insert or replace into meta (key, value) values ('active_run_id', ?)", [runId])
  })

  write(index)
}

function beginSqliteIndexRun(db: Database, configHash: string, metadata: CastIndex["metadata"]) {
  const existing = db
    .query("select id from runs where status = 'indexing' and config_hash = ? order by started_at desc limit 1")
    .get(configHash) as { id: string } | null
  if (existing) {
    return { runId: existing.id }
  }

  const runId = `indexing-${Date.now()}-${Math.random().toString(RUN_ID_RANDOM_RADIX).slice(2)}`
  db.run(
    "insert into runs (id, status, config_hash, started_at, completed_at, metadata_json) values (?, ?, ?, ?, ?, ?)",
    [
      runId,
      "indexing",
      configHash,
      Date.now(),
      null,
      JSON.stringify({ ...metadata, status: "indexing", updatedAt: Date.now() }),
    ],
  )
  return { runId }
}

function getCompletedSqliteFile(
  db: Database,
  runId: string,
  filePath: string,
  fingerprint: string,
): FileResult | undefined {
  const file = db
    .query(
      `select path, language, fingerprint, diagnostics_json as diagnosticsJson, chunk_ids_json as chunkIdsJson
       from file_runs
       where run_id = ? and path = ? and fingerprint = ?`,
    )
    .get(runId, filePath, fingerprint) as {
    path: string
    language: string
    fingerprint: string
    diagnosticsJson: string
    chunkIdsJson: string
  } | null
  if (!file) {
    return
  }

  const record: FileRecord = {
    path: file.path,
    language: file.language,
    fingerprint: file.fingerprint,
    chunkIds: JSON.parse(file.chunkIdsJson),
    diagnostics: JSON.parse(file.diagnosticsJson),
  }
  const chunks = readChunks(db, runId, readVectors(db, runId))
  const symbols = readSymbols(db, runId)
  return {
    file: record,
    chunks: Object.fromEntries(record.chunkIds.map((id) => [id, chunks[id]]).filter((entry) => entry[1])),
    symbols: Object.fromEntries(Object.entries(symbols).filter((entry) => entry[1].filePath === filePath)),
  }
}

function writeSqliteFileResult(db: Database, runId: string, fileResult: FileResult) {
  const write = db.transaction((result: FileResult) => {
    deleteRunFile(db, runId, result.file.path)
    insertFile(db, runId, result.file, false)
    insertChunks(db, runId, Object.values(result.chunks))
    for (const symbol of Object.values(result.symbols)) {
      insertSymbol(db, runId, symbol)
    }
  })
  write(fileResult)
}

function activateSqliteRun(db: Database, runId: string, index: CastIndex) {
  const activate = db.transaction((castIndex: CastIndex) => {
    deleteRunRecords(db, runId)
    for (const file of Object.values(castIndex.files)) {
      insertFile(db, runId, file)
    }
    insertChunks(db, runId, Object.values(castIndex.chunks))
    for (const symbol of Object.values(castIndex.symbols)) {
      insertSymbol(db, runId, symbol)
    }
    if (castIndex.lexical) {
      insertLexical(db, runId, castIndex.lexical)
    }
    db.run("update runs set status = 'ready', completed_at = ?, metadata_json = ? where id = ?", [
      castIndex.metadata.updatedAt,
      JSON.stringify(castIndex.metadata),
      runId,
    ])
    db.run("insert or replace into meta (key, value) values ('active_run_id', ?)", [runId])
  })
  activate(index)
}

function searchSqliteVectorCandidates(db: Database, queryEmbedding: number[], topK: number, paths?: string[]) {
  const activeRun = db.query("select value from meta where key = 'active_run_id'").get() as { value: string } | null
  if (!activeRun || topK <= 0) {
    return []
  }

  return scoreSqliteVectorRows(queryEmbedding, readSqliteVectorRows(db, activeRun.value), paths).slice(0, topK)
}

interface SqliteVectorRow {
  id: string
  embedding: string
  filePath: string
}

function readSqliteVectorRows(db: Database, runId: string) {
  return db
    .query(
      `select chunk_rowids.chunk_id as id, vec_to_json(chunk_vectors.embedding) as embedding, chunks.file_path as filePath
       from chunk_rowids
       inner join chunk_vectors on chunk_vectors.rowid = chunk_rowids.rowid
       inner join chunks on chunks.run_id = chunk_rowids.run_id and chunks.id = chunk_rowids.chunk_id
       where chunk_rowids.run_id = ?`,
    )
    .all(runId) as SqliteVectorRow[]
}

function scoreSqliteVectorRows(queryEmbedding: number[], rows: SqliteVectorRow[], paths?: string[]) {
  return rows
    .filter((row) => matchesPaths(row.filePath, paths))
    .map((row) => ({ id: row.id, score: cosineSimilarity(queryEmbedding, JSON.parse(row.embedding)) }))
    .sort((left, right) => bScoreThenId(left, right))
}

function bScoreThenId(left: { id: string; score: number }, right: { id: string; score: number }) {
  return right.score - left.score || left.id.localeCompare(right.id)
}

function clearSqliteIndex(db: Database) {
  if (tableExists(db, "chunk_vectors")) {
    db.run("delete from chunk_vectors")
  }
  db.run("delete from chunk_rowids")
  db.run("delete from lexical")
  db.run("delete from symbols")
  db.run("delete from chunks")
  db.run("delete from file_runs")
  db.run("delete from files")
  db.run("delete from runs")
}

function insertRun(db: Database, runId: string, index: CastIndex) {
  db.run(
    "insert into runs (id, status, config_hash, started_at, completed_at, metadata_json) values (?, ?, ?, ?, ?, ?)",
    [
      runId,
      index.metadata.status,
      JSON.stringify({
        embeddingModel: index.metadata.embeddingModel,
        embeddingDimensions: index.metadata.embeddingDimensions,
        maxChunkNonWhitespaceChars: index.metadata.maxChunkNonWhitespaceChars,
        chunking: index.metadata.chunking,
      }),
      index.metadata.updatedAt,
      index.metadata.updatedAt,
      JSON.stringify(index.metadata),
    ],
  )
}

function insertFile(db: Database, runId: string, file: FileRecord, updateGlobalFile = true) {
  if (updateGlobalFile) {
    db.run("insert or replace into files (path, language, fingerprint, diagnostics_json) values (?, ?, ?, ?)", [
      file.path,
      file.language,
      file.fingerprint,
      JSON.stringify(file.diagnostics),
    ])
  }
  db.run(
    "insert or replace into file_runs (run_id, path, language, fingerprint, diagnostics_json, chunk_ids_json) values (?, ?, ?, ?, ?, ?)",
    [
      runId,
      file.path,
      file.language,
      file.fingerprint,
      JSON.stringify(file.diagnostics),
      JSON.stringify(file.chunkIds),
    ],
  )
}

function insertChunks(db: Database, runId: string, chunks: ChunkRecord[]) {
  let vectorRowid = nextVectorRowid(db)
  for (const chunk of chunks) {
    const { embedding: _embedding, ...storedChunk } = chunk
    db.run("insert into chunks (run_id, id, file_path, kind, record_json) values (?, ?, ?, ?, ?)", [
      runId,
      chunk.id,
      chunk.filePath,
      chunk.kind,
      JSON.stringify(storedChunk),
    ])
    if (chunk.embedding) {
      db.run("insert into chunk_vectors (rowid, embedding) values (?, ?)", [
        vectorRowid,
        JSON.stringify(chunk.embedding),
      ])
      db.run("insert into chunk_rowids (run_id, chunk_id, rowid) values (?, ?, ?)", [runId, chunk.id, vectorRowid])
      vectorRowid += 1
    }
  }
}

function nextVectorRowid(db: Database) {
  const row = db.query("select coalesce(max(rowid), 0) + 1 as rowid from chunk_rowids").get() as { rowid: number }
  return row.rowid
}

function deleteRunFile(db: Database, runId: string, filePath: string) {
  const rows = db
    .query(
      "select chunk_rowids.rowid from chunk_rowids inner join chunks on chunks.run_id = chunk_rowids.run_id and chunks.id = chunk_rowids.chunk_id where chunks.run_id = ? and chunks.file_path = ?",
    )
    .all(runId, filePath) as Array<{ rowid: number }>
  if (tableExists(db, "chunk_vectors")) {
    for (const row of rows) {
      db.run("delete from chunk_vectors where rowid = ?", [row.rowid])
    }
  }
  db.run(
    "delete from chunk_rowids where run_id = ? and chunk_id in (select id from chunks where run_id = ? and file_path = ?)",
    [runId, runId, filePath],
  )
  db.run("delete from chunks where run_id = ? and file_path = ?", [runId, filePath])
  db.run("delete from symbols where run_id = ? and file_path = ?", [runId, filePath])
  db.run("delete from file_runs where run_id = ? and path = ?", [runId, filePath])
}

function deleteRunRecords(db: Database, runId: string) {
  const rows = db.query("select rowid from chunk_rowids where run_id = ?").all(runId) as Array<{ rowid: number }>
  if (tableExists(db, "chunk_vectors")) {
    for (const row of rows) {
      db.run("delete from chunk_vectors where rowid = ?", [row.rowid])
    }
  }
  db.run("delete from chunk_rowids where run_id = ?", [runId])
  db.run("delete from lexical where run_id = ?", [runId])
  db.run("delete from symbols where run_id = ?", [runId])
  db.run("delete from chunks where run_id = ?", [runId])
  db.run("delete from file_runs where run_id = ?", [runId])
}

function insertSymbol(db: Database, runId: string, symbol: SymbolRecord) {
  db.run("insert into symbols (run_id, id, file_path, kind, record_json) values (?, ?, ?, ?, ?)", [
    runId,
    symbol.id,
    symbol.filePath,
    symbol.kind,
    JSON.stringify(symbol),
  ])
}

function insertLexical(db: Database, runId: string, lexical: LexicalIndex) {
  db.run("insert into lexical (run_id, metadata_json) values (?, ?)", [runId, JSON.stringify(lexical)])
}

function initializeSchema(db: Database, embeddingDimensions?: number) {
  db.run("create table if not exists meta (key text primary key, value text not null)")
  db.run("insert or replace into meta (key, value) values ('schema_version', ?)", [String(SQLITE_SCHEMA_VERSION)])
  db.run(
    "create table if not exists runs (id text primary key, status text not null, config_hash text not null, started_at integer not null, completed_at integer, metadata_json text not null)",
  )
  db.run(
    "create table if not exists files (path text primary key, language text not null, fingerprint text not null, diagnostics_json text not null)",
  )
  db.run(
    "create table if not exists file_runs (run_id text not null, path text not null, chunk_ids_json text not null, primary key (run_id, path))",
  )
  addColumnIfMissing(db, "file_runs", "language", "text")
  addColumnIfMissing(db, "file_runs", "fingerprint", "text")
  addColumnIfMissing(db, "file_runs", "diagnostics_json", "text")
  db.run(
    "create table if not exists chunks (run_id text not null, id text not null, file_path text not null, kind text not null, record_json text not null, primary key (run_id, id))",
  )
  db.run(
    "create table if not exists symbols (run_id text not null, id text not null, file_path text not null, kind text not null, record_json text not null, primary key (run_id, id))",
  )
  db.run("create table if not exists lexical (run_id text primary key, metadata_json text not null)")
  db.run(
    "create table if not exists chunk_rowids (run_id text not null, chunk_id text not null, rowid integer not null, primary key (run_id, chunk_id))",
  )
  if (embeddingDimensions !== undefined) {
    db.run(`create virtual table if not exists chunk_vectors using vec0(embedding float[${embeddingDimensions}])`)
  }
}

function tableExists(db: Database, table: string) {
  return Boolean(db.query("select name from sqlite_master where type = 'table' and name = ?").get(table))
}

function inferEmbeddingDimensions(index: CastIndex) {
  return (
    index.metadata.embeddingDimensions ??
    Object.values(index.chunks).find((chunk) => chunk.embedding)?.embedding?.length
  )
}

function inferFileResultEmbeddingDimensions(fileResult: FileResult) {
  return Object.values(fileResult.chunks).find((chunk) => chunk.embedding)?.embedding?.length
}

function addColumnIfMissing(db: Database, table: string, column: string, definition: string) {
  const columns = db.query(`pragma table_info(${table})`).all() as Array<{ name: string }>
  if (!columns.some((existing) => existing.name === column)) {
    db.run(`alter table ${table} add column ${column} ${definition}`)
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
    isChunkingOptions(value.chunking) &&
    typeof value.updatedAt === "number" &&
    typeof value.status === "string" &&
    ["empty", "indexing", "ready", "stale", "error"].includes(value.status) &&
    isStringArray(value.diagnostics) &&
    isOptionalString(value.embeddingModel) &&
    (value.embeddingDimensions === undefined || typeof value.embeddingDimensions === "number")
  )
}

function isChunkingOptions(value: unknown): value is ChunkingOptions {
  return (
    isObject(value) &&
    isNonnegativeNumber(value.overlap) &&
    Number.isInteger(value.overlap) &&
    typeof value.expansion === "boolean" &&
    typeof value.minSemanticNonWhitespaceChars === "number" &&
    Number.isInteger(value.minSemanticNonWhitespaceChars) &&
    value.minSemanticNonWhitespaceChars > 0
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
