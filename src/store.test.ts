import { Database } from "bun:sqlite"
import { describe, expect, test } from "bun:test"
import { mkdir, mkdtemp, rm } from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import { cosineSimilarity, createEmptyIndex, createIndexStore, searchVectors } from "./store.js"
import type { CastIndex, ChunkRecord, FileRecord, SymbolRecord } from "./types.js"

type VectorSearchStore = ReturnType<typeof createIndexStore> & {
  searchVectorCandidates?: (
    queryEmbedding: number[],
    topK: number,
    paths?: string[],
  ) => Promise<Array<{ id: string; score: number }>>
}

type ResumableStore = ReturnType<typeof createIndexStore> & {
  beginIndexRun(input: { configHash: string; metadata: CastIndex["metadata"] }): Promise<{ runId: string }>
  getCompletedFile(
    runId: string,
    path: string,
    fingerprint: string,
  ): Promise<
    | {
        file: FileRecord
        chunks: Record<string, ChunkRecord>
        symbols: Record<string, SymbolRecord>
      }
    | undefined
  >
  writeFileResult(
    runId: string,
    fileResult: { file: FileRecord; chunks: Record<string, ChunkRecord>; symbols: Record<string, SymbolRecord> },
  ): Promise<void>
  activateRun(runId: string, index: CastIndex): Promise<void>
}

const MISSING_CHUNK_RECORD_COLUMN_PATTERN = /record_json|no such column/

function chunk(id: string, filePath: string, embedding: number[]): ChunkRecord {
  return {
    id,
    filePath,
    language: "typescript",
    kind: "function",
    range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
    text: `function ${id}() {}`,
    nonWhitespaceChars: 10,
    nodeTypes: [],
    symbolIds: [],
    childChunkIds: [],
    embedding,
  }
}

async function testFingerprint(filePath: string) {
  const bytes = new Uint8Array(await Bun.file(filePath).arrayBuffer())
  const hash = new Bun.CryptoHasher("sha256")
  hash.update(bytes)
  return hash.digest("hex")
}

describe("index store", () => {
  test("does not persist chunk source text in SQLite record JSON", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const worktree = path.join(dir, "worktree")
      await mkdir(worktree, { recursive: true })
      const sourcePath = path.join(worktree, "src.ts")
      await Bun.write(sourcePath, "function hello() {}\n")
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 2 })
      const index = createEmptyIndex({
        projectId: "p",
        worktree,
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "ready"
      index.metadata.embeddingDimensions = 2
      index.files["src.ts"] = {
        path: "src.ts",
        language: "typescript",
        fingerprint: await testFingerprint(sourcePath),
        chunkIds: ["hello"],
        diagnostics: [],
      }
      index.chunks.hello = {
        id: "hello",
        filePath: "src.ts",
        language: "typescript",
        kind: "function",
        range: { byteStart: 0, byteEnd: 19, lineStart: 1, lineEnd: 1 },
        text: "function hello() {}",
        nonWhitespaceChars: 17,
        nodeTypes: [],
        symbolIds: [],
        childChunkIds: [],
        embedding: [1, 0],
      }
      await store.write(index)

      const db = new Database(path.join(dir, "project", "index.sqlite"))
      try {
        const row = db.query("select record_json as recordJson from chunks").get() as { recordJson: string }
        const record = JSON.parse(row.recordJson) as Record<string, unknown>

        expect(row.recordJson).not.toContain("function hello")
        expect(Object.hasOwn(record, "text")).toBe(false)
        expect(Object.hasOwn(record, "embedding")).toBe(false)
      } finally {
        db.close()
      }
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("returns empty chunk text and diagnostic when source fingerprint mismatches", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const worktree = path.join(dir, "worktree")
      const sourcePath = path.join(worktree, "src.ts")
      await mkdir(worktree, { recursive: true })
      await Bun.write(sourcePath, "function hello() {}\n")
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 2 })
      const index = createEmptyIndex({
        projectId: "p",
        worktree,
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "ready"
      index.metadata.embeddingDimensions = 2
      index.files["src.ts"] = {
        path: "src.ts",
        language: "typescript",
        fingerprint: await testFingerprint(sourcePath),
        chunkIds: ["hello"],
        diagnostics: [],
      }
      index.chunks.hello = {
        id: "hello",
        filePath: "src.ts",
        language: "typescript",
        kind: "function",
        range: { byteStart: 0, byteEnd: 19, lineStart: 1, lineEnd: 1 },
        text: "function hello() {}",
        nonWhitespaceChars: 17,
        nodeTypes: [],
        symbolIds: [],
        childChunkIds: [],
        embedding: [1, 0],
      }
      await store.write(index)
      await Bun.write(sourcePath, "function changed() {}\n")

      const cached = await store.read()

      expect(cached.chunks.hello.text).toBe("")
      expect(cached.metadata.diagnostics).toContain("source fingerprint mismatch for src.ts; chunk text unavailable")
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("returns empty chunk text and diagnostic when source read failed", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const worktree = path.join(dir, "worktree")
      const sourcePath = path.join(worktree, "src.ts")
      await mkdir(worktree, { recursive: true })
      await Bun.write(sourcePath, "function hello() {}\n")
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 2 })
      const index = createEmptyIndex({
        projectId: "p",
        worktree,
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "ready"
      index.metadata.embeddingDimensions = 2
      index.files["src.ts"] = {
        path: "src.ts",
        language: "typescript",
        fingerprint: await testFingerprint(sourcePath),
        chunkIds: ["hello"],
        diagnostics: [],
      }
      index.chunks.hello = {
        id: "hello",
        filePath: "src.ts",
        language: "typescript",
        kind: "function",
        range: { byteStart: 0, byteEnd: 19, lineStart: 1, lineEnd: 1 },
        text: "function hello() {}",
        nonWhitespaceChars: 17,
        nodeTypes: [],
        symbolIds: [],
        childChunkIds: [],
        embedding: [1, 0],
      }
      await store.write(index)
      await rm(sourcePath)

      const cached = await store.read()

      expect(cached.chunks.hello.text).toBe("")
      expect(cached.metadata.diagnostics).toContain("source read failed for src.ts; chunk text unavailable")
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("returns empty chunk text and diagnostic when source range invalid", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const worktree = path.join(dir, "worktree")
      const sourcePath = path.join(worktree, "src.ts")
      await mkdir(worktree, { recursive: true })
      await Bun.write(sourcePath, "function hello() {}\n")
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 2 })
      const index = createEmptyIndex({
        projectId: "p",
        worktree,
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "ready"
      index.metadata.embeddingDimensions = 2
      index.files["src.ts"] = {
        path: "src.ts",
        language: "typescript",
        fingerprint: await testFingerprint(sourcePath),
        chunkIds: ["hello"],
        diagnostics: [],
      }
      index.chunks.hello = {
        id: "hello",
        filePath: "src.ts",
        language: "typescript",
        kind: "function",
        range: { byteStart: 0, byteEnd: 200, lineStart: 1, lineEnd: 1 },
        text: "function hello() {}",
        nonWhitespaceChars: 17,
        nodeTypes: [],
        symbolIds: [],
        childChunkIds: [],
        embedding: [1, 0],
      }
      await store.write(index)

      const cached = await store.read()

      expect(cached.chunks.hello.text).toBe("")
      expect(cached.metadata.diagnostics).toContain("source range invalid for src.ts:hello; chunk text unavailable")
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("resumes the latest indexing run with the same config hash", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 2 }) as ResumableStore
      const index = createEmptyIndex({
        projectId: "p",
        worktree: "/repo",
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "indexing"
      index.metadata.embeddingDimensions = 2

      const first = await store.beginIndexRun({ configHash: "same", metadata: index.metadata })
      const resumed = await store.beginIndexRun({ configHash: "same", metadata: index.metadata })
      const different = await store.beginIndexRun({ configHash: "different", metadata: index.metadata })

      expect(resumed.runId).toBe(first.runId)
      expect(different.runId).not.toBe(first.runId)
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("keeps the previous active SQLite run readable until a new run is activated", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 2 }) as ResumableStore
      const oldIndex = createEmptyIndex({
        projectId: "p",
        worktree: "/repo",
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      oldIndex.metadata.status = "ready"
      oldIndex.metadata.embeddingDimensions = 2
      oldIndex.files["old.ts"] = {
        path: "old.ts",
        language: "typescript",
        fingerprint: "old",
        chunkIds: ["old"],
        diagnostics: [],
      }
      oldIndex.chunks.old = chunk("old", "old.ts", [1, 0])
      await store.write(oldIndex)

      const metadata = { ...oldIndex.metadata, status: "indexing" as const, updatedAt: Date.now() }
      const { runId } = await store.beginIndexRun({ configHash: "refresh", metadata })
      await store.writeFileResult(runId, {
        file: { path: "old.ts", language: "typescript", fingerprint: "new", chunkIds: ["new"], diagnostics: [] },
        chunks: { new: chunk("new", "old.ts", [0, 1]) },
        symbols: {},
      })

      const cached = await store.read()

      expect(Object.keys(cached.files)).toEqual(["old.ts"])
      expect(cached.files["old.ts"].fingerprint).toBe("old")
      expect(cached.chunks.old?.embedding).toEqual([1, 0])
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("does not let in-progress global file metadata change a legacy active run read", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 2 }) as ResumableStore
      const oldIndex = createEmptyIndex({
        projectId: "p",
        worktree: "/repo",
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      oldIndex.metadata.status = "ready"
      oldIndex.metadata.embeddingDimensions = 2
      oldIndex.files["old.ts"] = {
        path: "old.ts",
        language: "typescript",
        fingerprint: "old",
        chunkIds: ["old"],
        diagnostics: ["old diagnostic"],
      }
      oldIndex.chunks.old = chunk("old", "old.ts", [1, 0])
      await store.write(oldIndex)
      const db = new Database(path.join(dir, "project", "index.sqlite"))
      try {
        db.run("update file_runs set language = null, fingerprint = null, diagnostics_json = null")
      } finally {
        db.close()
      }

      const metadata = { ...oldIndex.metadata, status: "indexing" as const, updatedAt: Date.now() }
      const { runId } = await store.beginIndexRun({ configHash: "refresh", metadata })
      await store.writeFileResult(runId, {
        file: { path: "old.ts", language: "typescript", fingerprint: "new", chunkIds: ["new"], diagnostics: [] },
        chunks: { new: chunk("new", "old.ts", [0, 1]) },
        symbols: {},
      })

      const cached = await store.read()

      expect(cached.files["old.ts"]).toEqual(oldIndex.files["old.ts"])
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("hydrates a completed file from an indexing run by matching fingerprint", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const worktree = path.join(dir, "worktree")
      const sourcePath = path.join(worktree, "src/a.ts")
      await mkdir(path.dirname(sourcePath), { recursive: true })
      await Bun.write(sourcePath, "function a() {}\n")
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 2 }) as ResumableStore
      const index = createEmptyIndex({
        projectId: "p",
        worktree,
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "indexing"
      index.metadata.embeddingDimensions = 2
      const { runId } = await store.beginIndexRun({ configHash: "same", metadata: index.metadata })
      const file = {
        path: "src/a.ts",
        language: "typescript",
        fingerprint: await testFingerprint(sourcePath),
        chunkIds: ["a"],
        diagnostics: [],
      }
      await store.writeFileResult(runId, { file, chunks: { a: chunk("a", "src/a.ts", [1, 0]) }, symbols: {} })

      const completed = await store.getCompletedFile(runId, "src/a.ts", file.fingerprint)
      const stale = await store.getCompletedFile(runId, "src/a.ts", "changed")

      expect(completed?.file).toEqual(file)
      expect(completed?.chunks.a.embedding).toEqual([1, 0])
      expect(stale).toBeUndefined()
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("hydrates completed file chunk text from the run worktree source", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const worktree = path.join(dir, "worktree")
      const sourcePath = path.join(worktree, "src/a.ts")
      await mkdir(path.dirname(sourcePath), { recursive: true })
      await Bun.write(sourcePath, "const before = 1\nfunction alpha() {}\n")
      const sourceText = "function alpha() {}"
      const byteStart = "const before = 1\n".length
      const byteEnd = byteStart + sourceText.length
      const fingerprint = await testFingerprint(sourcePath)
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 2 }) as ResumableStore
      const index = createEmptyIndex({
        projectId: "p",
        worktree,
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "indexing"
      index.metadata.embeddingDimensions = 2
      const { runId } = await store.beginIndexRun({ configHash: "same", metadata: index.metadata })
      const file = { path: "src/a.ts", language: "typescript", fingerprint, chunkIds: ["alpha"], diagnostics: [] }
      await store.writeFileResult(runId, {
        file,
        chunks: {
          alpha: {
            id: "alpha",
            filePath: "src/a.ts",
            language: "typescript",
            kind: "function",
            range: { byteStart, byteEnd, lineStart: 2, lineEnd: 2 },
            text: sourceText,
            nonWhitespaceChars: 17,
            nodeTypes: [],
            symbolIds: [],
            childChunkIds: [],
            embedding: [1, 0],
          },
        },
        symbols: {},
      })

      const completed = await store.getCompletedFile(runId, "src/a.ts", fingerprint)

      expect(completed?.chunks.alpha.text).toBe(sourceText)
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("hydrates completed file without reading unrelated chunks", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const worktree = path.join(dir, "worktree")
      const firstPath = path.join(worktree, "src/a.ts")
      const secondPath = path.join(worktree, "src/b.ts")
      await mkdir(path.dirname(firstPath), { recursive: true })
      await Bun.write(firstPath, "function alpha() {}\n")
      await Bun.write(secondPath, "function beta() {}\n")
      const firstFingerprint = await testFingerprint(firstPath)
      const secondFingerprint = await testFingerprint(secondPath)
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 2 }) as ResumableStore
      const index = createEmptyIndex({
        projectId: "p",
        worktree,
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "indexing"
      index.metadata.embeddingDimensions = 2
      const { runId } = await store.beginIndexRun({ configHash: "same", metadata: index.metadata })
      await store.writeFileResult(runId, {
        file: {
          path: "src/a.ts",
          language: "typescript",
          fingerprint: firstFingerprint,
          chunkIds: ["alpha"],
          diagnostics: [],
        },
        chunks: {
          alpha: {
            id: "alpha",
            filePath: "src/a.ts",
            language: "typescript",
            kind: "function",
            range: { byteStart: 0, byteEnd: 19, lineStart: 1, lineEnd: 1 },
            text: "function alpha() {}",
            nonWhitespaceChars: 17,
            nodeTypes: [],
            symbolIds: [],
            childChunkIds: [],
            embedding: [1, 0],
          },
        },
        symbols: {},
      })
      await store.writeFileResult(runId, {
        file: {
          path: "src/b.ts",
          language: "typescript",
          fingerprint: secondFingerprint,
          chunkIds: ["beta"],
          diagnostics: [],
        },
        chunks: {
          beta: {
            id: "beta",
            filePath: "src/b.ts",
            language: "typescript",
            kind: "function",
            range: { byteStart: 0, byteEnd: 18, lineStart: 1, lineEnd: 1 },
            text: "function beta() {}",
            nonWhitespaceChars: 16,
            nodeTypes: [],
            symbolIds: [],
            childChunkIds: [],
            embedding: [0, 1],
          },
        },
        symbols: {},
      })
      const db = new Database(path.join(dir, "project", "index.sqlite"))
      try {
        db.run("update chunks set record_json = ? where run_id = ? and id = ?", ["{", runId, "beta"])
      } finally {
        db.close()
      }

      const completed = await store.getCompletedFile(runId, "src/a.ts", firstFingerprint)

      expect(completed?.chunks.alpha.text).toBe("function alpha() {}")
      expect(completed?.chunks.alpha.embedding).toEqual([1, 0])
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("does not return completed file when source fingerprint mismatches", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const worktree = path.join(dir, "worktree")
      const sourcePath = path.join(worktree, "src/a.ts")
      await mkdir(path.dirname(sourcePath), { recursive: true })
      await Bun.write(sourcePath, "function alpha() {}\n")
      const fingerprint = await testFingerprint(sourcePath)
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 2 }) as ResumableStore
      const index = createEmptyIndex({
        projectId: "p",
        worktree,
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "indexing"
      index.metadata.embeddingDimensions = 2
      const { runId } = await store.beginIndexRun({ configHash: "same", metadata: index.metadata })
      await store.writeFileResult(runId, {
        file: { path: "src/a.ts", language: "typescript", fingerprint, chunkIds: ["alpha"], diagnostics: [] },
        chunks: {
          alpha: {
            id: "alpha",
            filePath: "src/a.ts",
            language: "typescript",
            kind: "function",
            range: { byteStart: 0, byteEnd: 19, lineStart: 1, lineEnd: 1 },
            text: "function alpha() {}",
            nonWhitespaceChars: 17,
            nodeTypes: [],
            symbolIds: [],
            childChunkIds: [],
            embedding: [1, 0],
          },
        },
        symbols: {},
      })
      await Bun.write(sourcePath, "function changed() {}\n")

      const completed = await store.getCompletedFile(runId, "src/a.ts", fingerprint)

      expect(completed).toBeUndefined()
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("does not return completed file when source read failed", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const worktree = path.join(dir, "worktree")
      const sourcePath = path.join(worktree, "src/a.ts")
      await mkdir(path.dirname(sourcePath), { recursive: true })
      await Bun.write(sourcePath, "function alpha() {}\n")
      const fingerprint = await testFingerprint(sourcePath)
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 2 }) as ResumableStore
      const index = createEmptyIndex({
        projectId: "p",
        worktree,
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "indexing"
      index.metadata.embeddingDimensions = 2
      const { runId } = await store.beginIndexRun({ configHash: "same", metadata: index.metadata })
      await store.writeFileResult(runId, {
        file: { path: "src/a.ts", language: "typescript", fingerprint, chunkIds: ["alpha"], diagnostics: [] },
        chunks: {
          alpha: {
            id: "alpha",
            filePath: "src/a.ts",
            language: "typescript",
            kind: "function",
            range: { byteStart: 0, byteEnd: 19, lineStart: 1, lineEnd: 1 },
            text: "function alpha() {}",
            nonWhitespaceChars: 17,
            nodeTypes: [],
            symbolIds: [],
            childChunkIds: [],
            embedding: [1, 0],
          },
        },
        symbols: {},
      })
      await rm(sourcePath)

      const completed = await store.getCompletedFile(runId, "src/a.ts", fingerprint)

      expect(completed).toBeUndefined()
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("does not return completed file when source range is invalid", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const worktree = path.join(dir, "worktree")
      const sourcePath = path.join(worktree, "src/a.ts")
      await mkdir(path.dirname(sourcePath), { recursive: true })
      await Bun.write(sourcePath, "function alpha() {}\n")
      const fingerprint = await testFingerprint(sourcePath)
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 2 }) as ResumableStore
      const index = createEmptyIndex({
        projectId: "p",
        worktree,
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "indexing"
      index.metadata.embeddingDimensions = 2
      const { runId } = await store.beginIndexRun({ configHash: "same", metadata: index.metadata })
      await store.writeFileResult(runId, {
        file: { path: "src/a.ts", language: "typescript", fingerprint, chunkIds: ["alpha"], diagnostics: [] },
        chunks: {
          alpha: {
            id: "alpha",
            filePath: "src/a.ts",
            language: "typescript",
            kind: "function",
            range: { byteStart: 0, byteEnd: 200, lineStart: 1, lineEnd: 1 },
            text: "function alpha() {}",
            nonWhitespaceChars: 17,
            nodeTypes: [],
            symbolIds: [],
            childChunkIds: [],
            embedding: [1, 0],
          },
        },
        symbols: {},
      })

      const completed = await store.getCompletedFile(runId, "src/a.ts", fingerprint)

      expect(completed).toBeUndefined()
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("searches vectors with sqlite-vec", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 4 })
      const index = createEmptyIndex({
        projectId: "p",
        worktree: "/repo",
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "ready"
      index.metadata.embeddingDimensions = 4
      index.chunks.near = {
        id: "near",
        filePath: "src/near.ts",
        language: "typescript",
        kind: "function",
        range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
        text: "function near() {}",
        nonWhitespaceChars: 16,
        nodeTypes: [],
        symbolIds: [],
        childChunkIds: [],
        embedding: [1, 0, 0, 0],
      }
      index.chunks.far = {
        id: "far",
        filePath: "test/far.ts",
        language: "typescript",
        kind: "function",
        range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
        text: "function far() {}",
        nonWhitespaceChars: 15,
        nodeTypes: [],
        symbolIds: [],
        childChunkIds: [],
        embedding: [0, 1, 0, 0],
      }
      await store.write(index)

      const results = await (store as VectorSearchStore).searchVectorCandidates?.([1, 0, 0, 0], 2)

      expect(results?.map((result) => result.id)).toEqual(["near", "far"])
      expect(results?.every((result) => Number.isFinite(result.score))).toBe(true)
      expect(results?.[0].score).toBeGreaterThanOrEqual(results?.[1].score ?? Number.POSITIVE_INFINITY)
      expect(await (store as VectorSearchStore).searchVectorCandidates?.([1, 0, 0, 0], 2, ["test/"])).toEqual([
        { id: "far", score: results?.[1].score },
      ])
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("returns cosine scores for orthogonal sqlite-vec candidates", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 4 })
      const index = createEmptyIndex({
        projectId: "p",
        worktree: "/repo",
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "ready"
      index.metadata.embeddingDimensions = 4
      index.chunks.orthogonal = chunk("orthogonal", "src/orthogonal.ts", [0, 1, 0, 0])
      await store.write(index)

      const results = await (store as VectorSearchStore).searchVectorCandidates?.([1, 0, 0, 0], 1)

      expect(results?.[0]).toEqual({ id: "orthogonal", score: 0 })
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("returns cosine-best sqlite vector even when distance-best differs", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 2 })
      const index = createEmptyIndex({
        projectId: "p",
        worktree: "/repo",
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "ready"
      index.metadata.embeddingDimensions = 2
      index.chunks.distanceBest = chunk("distanceBest", "src/distance-best.ts", [0.9, 0.1])
      index.chunks.cosineBest = chunk("cosineBest", "src/cosine-best.ts", [100, 0])
      await store.write(index)

      const results = await (store as VectorSearchStore).searchVectorCandidates?.([1, 0], 1)

      expect(results).toEqual([{ id: "cosineBest", score: 1 }])
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("searches path-filtered sqlite vectors after applying path filters", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 4 })
      const index = createEmptyIndex({
        projectId: "p",
        worktree: "/repo",
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "ready"
      index.metadata.embeddingDimensions = 4
      index.chunks.outside = chunk("outside", "vendor/outside.ts", [1, 0, 0, 0])
      index.chunks.inside = chunk("inside", "src/inside.ts", [0.5, 0.5, 0, 0])
      await store.write(index)

      const results = await (store as VectorSearchStore).searchVectorCandidates?.([1, 0, 0, 0], 1, ["src/"])

      expect(results?.map((result) => result.id)).toEqual(["inside"])
      expect(results?.[0].score).toBeCloseTo(cosineSimilarity([1, 0, 0, 0], [0.5, 0.5, 0, 0]))
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("preserves mixed path filter OR semantics for sqlite vector candidates", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 4 })
      const index = createEmptyIndex({
        projectId: "p",
        worktree: "/repo",
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "ready"
      index.metadata.embeddingDimensions = 4
      index.chunks.src = chunk("src", "src/a.ts", [1, 0, 0, 0])
      index.chunks.test = chunk("test", "test/b.ts", [0.8, 0.2, 0, 0])
      index.chunks.other = chunk("other", "lib/c.ts", [0.9, 0.1, 0, 0])
      await store.write(index)

      const results = await (store as VectorSearchStore).searchVectorCandidates?.([1, 0, 0, 0], 2, [
        "src/*.ts",
        "test/",
      ])

      expect(results?.map((result) => result.id)).toEqual(["src", "test"])
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("matches bracket globs for sqlite vector path filters", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 4 })
      const index = createEmptyIndex({
        projectId: "p",
        worktree: "/repo",
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "ready"
      index.metadata.embeddingDimensions = 4
      index.chunks.a = chunk("a", "src/a.ts", [1, 0, 0, 0])
      index.chunks.b = chunk("b", "src/b.ts", [0.9, 0.1, 0, 0])
      index.chunks.c = chunk("c", "src/c.ts", [0.8, 0.2, 0, 0])
      await store.write(index)

      const results = await (store as VectorSearchStore).searchVectorCandidates?.([1, 0, 0, 0], 3, ["src/[ab].ts"])

      expect(results?.map((result) => result.id)).toEqual(["a", "b"])
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("creates a SQLite index database instead of index.json", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 4 })

      const index = await store.read()

      expect(index.metadata.status).toBe("empty")
      expect(index.files).toEqual({})
      expect(index.chunks).toEqual({})
      expect(index.symbols).toEqual({})
      expect(await Bun.file(path.join(dir, "project", "index.sqlite")).exists()).toBe(true)
      expect(await Bun.file(path.join(dir, "project", "index.json")).exists()).toBe(false)
      const db = new Database(path.join(dir, "project", "index.sqlite"))
      try {
        expect(db.query("select value from meta where key = 'schema_version'").get()).toEqual({ value: "3" })
      } finally {
        db.close()
      }
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("ignores old index.json caches", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const oldIndex = createEmptyIndex({
        projectId: "p",
        worktree: "/repo",
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      oldIndex.metadata.status = "ready"
      oldIndex.files["old.ts"] = {
        path: "old.ts",
        language: "typescript",
        fingerprint: "old",
        chunkIds: ["old"],
        diagnostics: [],
      }
      oldIndex.chunks.old = chunk("old", "old.ts", [1, 0, 0, 0])
      await mkdir(path.join(dir, "project"), { recursive: true })
      await Bun.write(path.join(dir, "project", "index.json"), JSON.stringify(oldIndex))

      const cached = await createIndexStore({ cacheDir: dir, cacheKey: "project" }).read()

      expect(cached.metadata.status).toBe("empty")
      expect(cached.files).toEqual({})
      expect(cached.chunks).toEqual({})
      expect(await Bun.file(path.join(dir, "project", "index.sqlite")).exists()).toBe(true)
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("persists and hydrates the active run index", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const worktree = path.join(dir, "worktree")
      const sourcePath = path.join(worktree, "src/a.ts")
      await mkdir(path.dirname(sourcePath), { recursive: true })
      await Bun.write(sourcePath, "\nfunction alpha() {}\n")
      const index: CastIndex = createEmptyIndex({
        projectId: "p",
        worktree,
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "ready"
      index.metadata.updatedAt = 1234
      index.metadata.embeddingModel = "test-model"
      index.metadata.embeddingDimensions = 4
      index.files["src/a.ts"] = {
        path: "src/a.ts",
        language: "typescript",
        fingerprint: await testFingerprint(sourcePath),
        chunkIds: ["chunk-1"],
        diagnostics: ["file diagnostic"],
      }
      index.chunks["chunk-1"] = {
        id: "chunk-1",
        filePath: "src/a.ts",
        language: "typescript",
        kind: "function",
        range: { byteStart: 1, byteEnd: 20, lineStart: 2, lineEnd: 4 },
        text: "function alpha() {}",
        nonWhitespaceChars: 17,
        nodeTypes: ["function_declaration"],
        symbolIds: ["symbol-1"],
        childChunkIds: [],
        embedding: [0.1, 0.2, 0.3, 0.4],
        lexical: { length: 3, termFrequencies: { alpha: 1, function: 1 } },
      }
      index.symbols["symbol-1"] = {
        id: "symbol-1",
        name: "alpha",
        kind: "function",
        filePath: "src/a.ts",
        range: { byteStart: 1, byteEnd: 20, lineStart: 2, lineEnd: 4 },
        childSymbolIds: [],
      }
      index.lexical = {
        documentCount: 1,
        averageDocumentLength: 3,
        documentFrequencies: { alpha: 1, function: 1 },
      }

      await createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 4 }).write(index)
      const cached = await createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 4 }).read()

      expect(cached.metadata.status).toBe("ready")
      expect(cached.metadata.updatedAt).toBe(1234)
      expect(cached.metadata.embeddingModel).toBe("test-model")
      expect(cached.metadata.embeddingDimensions).toBe(4)
      expect(cached.files["src/a.ts"]).toEqual(index.files["src/a.ts"])
      expect(cached.chunks["chunk-1"]).toEqual(index.chunks["chunk-1"])
      expect(cached.symbols["symbol-1"]).toEqual(index.symbols["symbol-1"])
      expect(cached.lexical).toEqual(index.lexical)
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("returns empty index for corrupt SQLite persisted JSON", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 4 })
      const index = createEmptyIndex({
        projectId: "p",
        worktree: "/repo",
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "ready"

      await store.write(index)
      const db = new Database(path.join(dir, "project", "index.sqlite"))
      try {
        db.run("update runs set metadata_json = '{bad json'")
      } finally {
        db.close()
      }

      const cached = await createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 4 }).read()

      expect(cached.metadata.status).toBe("empty")
      expect(cached.metadata.diagnostics[0]).toContain("rebuilding corrupt index")
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("rethrows operational SQLite failures while reading the active run", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 4 })
      const index = createEmptyIndex({
        projectId: "p",
        worktree: "/repo",
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "ready"
      index.metadata.embeddingDimensions = 4
      index.chunks.c = chunk("c", "src/a.ts", [1, 0, 0, 0])
      await store.write(index)

      const db = new Database(path.join(dir, "project", "index.sqlite"))
      try {
        db.run("alter table chunks rename to chunks_old")
        db.run("create table chunks (run_id text not null, id text not null)")
      } finally {
        db.close()
      }

      await expect(
        createIndexStore({ cacheDir: dir, cacheKey: "project", embeddingDimensions: 4 }).read(),
      ).rejects.toThrow(MISSING_CHUNK_RECORD_COLUMN_PATTERN)
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("writes and reads an index", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project" })
      const index = createEmptyIndex({
        projectId: "p",
        worktree: "/repo",
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "ready"
      await store.write(index)

      expect((await store.read()).metadata.status).toBe("ready")
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("reads valid lexical cache data", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project" })
      const index = createEmptyIndex({
        projectId: "p",
        worktree: "/repo",
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      index.metadata.status = "ready"
      index.lexical = {
        documentCount: 1,
        averageDocumentLength: 2,
        documentFrequencies: { alpha: 1 },
      }
      index.chunks.c = {
        id: "c",
        filePath: "src/a.ts",
        language: "typescript",
        kind: "function",
        range: { byteStart: 0, byteEnd: 10, lineStart: 1, lineEnd: 1 },
        text: "alpha alpha",
        nonWhitespaceChars: 10,
        nodeTypes: [],
        symbolIds: [],
        childChunkIds: [],
        lexical: { length: 2, termFrequencies: { alpha: 2 } },
      }
      await store.write(index)

      const cached = await store.read()

      expect(cached.metadata.status).toBe("ready")
      expect(cached.lexical?.documentFrequencies.alpha).toBe(1)
      expect(cached.chunks.c.lexical?.termFrequencies.alpha).toBe(2)
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("returns empty index without diagnostics for missing files", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const index = await createIndexStore({ cacheDir: dir, cacheKey: "project" }).read()

      expect(index.metadata.status).toBe("empty")
      expect(index.metadata.diagnostics).toEqual([])
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("scores vectors by cosine similarity", () => {
    expect(cosineSimilarity([1, 0], [1, 0])).toBe(1)
    expect(cosineSimilarity([1, 0], [0, 1])).toBe(0)
    expect(
      searchVectors(
        [1, 0],
        [
          { id: "a", vector: [0, 1] },
          { id: "b", vector: [1, 0] },
        ],
        1,
      ),
    ).toEqual([{ id: "b", score: 1 }])
    expect(
      searchVectors(
        [1, 0],
        [
          { id: "a", vector: [0, 1] },
          { id: "b", vector: [1, 0] },
        ],
        -1,
      ),
    ).toEqual([])
  })
})
