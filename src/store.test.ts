import { describe, expect, test } from "bun:test"
import { mkdir, mkdtemp, rm } from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import { cosineSimilarity, createEmptyIndex, createIndexStore, searchVectors } from "./store.js"

describe("index store", () => {
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

  test("returns empty index for corrupt files", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project" })
      await mkdir(path.join(dir, "project"), { recursive: true })
      await Bun.write(path.join(dir, "project", "index.json"), "not json")

      const index = await store.read()

      expect(index.metadata.status).toBe("empty")
      expect(index.metadata.diagnostics[0]).toContain("rebuilding corrupt index")
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("returns empty index for valid JSON with invalid index shape", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project" })
      await mkdir(path.join(dir, "project"), { recursive: true })

      await Bun.write(path.join(dir, "project", "index.json"), JSON.stringify({}))
      expect((await store.read()).metadata.status).toBe("empty")

      await Bun.write(
        path.join(dir, "project", "index.json"),
        JSON.stringify({
          metadata: { schemaVersion: 0 },
          files: {},
          chunks: {},
          symbols: {},
        }),
      )
      const index = await store.read()

      expect(index.metadata.status).toBe("empty")
      expect(index.metadata.diagnostics[0]).toContain("rebuilding corrupt index")
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("returns empty index for current-version JSON with invalid metadata", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project" })
      await mkdir(path.join(dir, "project"), { recursive: true })
      await Bun.write(
        path.join(dir, "project", "index.json"),
        JSON.stringify({
          metadata: { schemaVersion: 1 },
          files: {},
          chunks: {},
          symbols: {},
        }),
      )

      const index = await store.read()

      expect(index.metadata.status).toBe("empty")
      expect(index.metadata.diagnostics[0]).toContain("rebuilding corrupt index")
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("returns empty index for current-version JSON with invalid optional metadata", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project" })
      const index = createEmptyIndex({
        projectId: "p",
        worktree: "/repo",
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      await mkdir(path.join(dir, "project"), { recursive: true })

      await Bun.write(
        path.join(dir, "project", "index.json"),
        JSON.stringify({ ...index, metadata: { ...index.metadata, embeddingModel: 1 } }),
      )
      expect((await store.read()).metadata.diagnostics[0]).toContain("rebuilding corrupt index")

      await Bun.write(
        path.join(dir, "project", "index.json"),
        JSON.stringify({ ...index, metadata: { ...index.metadata, embeddingDimensions: "bad" } }),
      )
      expect((await store.read()).metadata.diagnostics[0]).toContain("rebuilding corrupt index")
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("returns empty index for current-version JSON with invalid records", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-store-"))
    try {
      const store = createIndexStore({ cacheDir: dir, cacheKey: "project" })
      const index = createEmptyIndex({
        projectId: "p",
        worktree: "/repo",
        cacheKey: "project",
        maxChunkNonWhitespaceChars: 2000,
      })
      await mkdir(path.join(dir, "project"), { recursive: true })

      await Bun.write(path.join(dir, "project", "index.json"), JSON.stringify({ ...index, files: { "a.ts": {} } }))
      expect((await store.read()).metadata.diagnostics[0]).toContain("rebuilding corrupt index")

      await Bun.write(path.join(dir, "project", "index.json"), JSON.stringify({ ...index, chunks: { c: {} } }))
      expect((await store.read()).metadata.diagnostics[0]).toContain("rebuilding corrupt index")

      await Bun.write(path.join(dir, "project", "index.json"), JSON.stringify({ ...index, symbols: { s: {} } }))
      expect((await store.read()).metadata.diagnostics[0]).toContain("rebuilding corrupt index")
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
