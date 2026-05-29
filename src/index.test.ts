import { describe, expect, test } from "bun:test"
import { mkdir, mkdtemp, rm, symlink } from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import castPlugin from "./index.js"
import { createCastPluginForTest } from "./plugin.js"
import { createEmptyIndex } from "./store.js"

const input = {
  project: { id: "p", worktree: "/repo", time: { created: 0 } },
  directory: "/repo",
  worktree: "/repo",
}

describe("cast plugin", () => {
  test("root module exposes only the default plugin", async () => {
    const entrypoint = await import("./index.js")

    expect(Object.keys(entrypoint)).toEqual(["default"])
    expect(entrypoint.default).toBeFunction()
  })

  test("registers semantic_search_code", async () => {
    const hooks = await castPlugin(input as never, {
      embedding: { baseURL: "https://example.test/v1", apiKey: "key", model: "embed" },
    })

    expect(Object.keys(hooks.tool ?? {})).toEqual(["semantic_search_code"])
    await hooks.dispose?.()
  })

  test("semantic_search_code returns configuration error when embeddings are missing", async () => {
    const hooks = await castPlugin(input as never, {})
    const result = await semanticSearchTool(hooks).execute({ query: "session" }, {
      worktree: "/repo",
      directory: "/repo",
    } as never)

    expect(typeof result).toBe("object")
    if (typeof result === "string") {
      throw new Error("expected object tool result")
    }
    expect(result.title).toBe("Semantic code search is not configured")
    expect(result.output).toContain("embedding.model is required")
    expect(result.metadata).toEqual({ configured: false })
  })

  test("configured semantic_search_code executes retrieval pipeline and returns result metadata", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-plugin-"))
    const fetchCalls: Array<{ url: string; body: unknown }> = []
    try {
      await Bun.write(path.join(dir, "code.ts"), "export function session() { return 'ok' }\n")
      const plugin = createCastPluginForTest({
        fetch: async (url, init) => {
          const body = JSON.parse(String(init.body))
          fetchCalls.push({ url, body })
          if (url.endsWith("/chat/completions")) {
            return Response.json({ choices: [{ message: { content: "session function" } }] })
          }
          return Response.json({
            data: [
              {
                embedding:
                  body.input === "session function" || String(body.input).includes("export function session")
                    ? [0, 1]
                    : [1, 0],
              },
            ],
          })
        },
      })
      const hooks = await plugin({ ...input, directory: dir, worktree: dir } as never, {
        cacheDir: path.join(dir, ".cache"),
        embedding: { baseURL: "https://example.test/v1", apiKey: "key", model: "embed" },
        hyde: { model: "hyde", threshold: 1, enabled: true },
      })

      const result = await semanticSearchTool(hooks).execute(
        { query: "session", topK: 1, includeParents: false, paths: ["code.ts"] },
        { worktree: dir, directory: dir } as never,
      )

      expect(typeof result).toBe("object")
      if (typeof result === "string") {
        throw new Error("expected object tool result")
      }
      expect(result.title).toBe("Semantic code search: session")
      expect(result.metadata).toEqual({ hydeUsed: true, resultCount: 1 })
      expect(JSON.parse(result.output).results[0].filePath).toBe("code.ts")
      expect(
        fetchCalls.some(
          (call) =>
            call.url.endsWith("/embeddings") &&
            (call.body as { input: string }).input.includes("export function session"),
        ),
      ).toBe(true)
      expect(
        fetchCalls.some(
          (call) => call.url.endsWith("/embeddings") && (call.body as { input: string }).input === "session",
        ),
      ).toBe(true)
      expect(
        fetchCalls.some(
          (call) => call.url.endsWith("/chat/completions") && (call.body as { model: string }).model === "hyde",
        ),
      ).toBe(true)
      await hooks.dispose?.()
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("refresh true forces an additional refresh before searching", async () => {
    const refreshes: string[] = []
    const plugin = createCastPluginForTest({
      createIndexer: () => ({
        refresh: async () => {
          refreshes.push("refresh")
          return emptyReadyIndex()
        },
      }),
      createStore: () => ({ read: async () => emptyReadyIndex(), write: async () => undefined }),
      retrieve: async () => ({
        status: { ...emptyReadyIndex().metadata, hydeUsed: false },
        results: [],
        diagnostics: [],
      }),
    })
    const hooks = await plugin(input as never, {
      embedding: { baseURL: "https://example.test/v1", apiKey: "key", model: "embed" },
    })

    await semanticSearchTool(hooks).execute({ query: "session", refresh: true }, {
      worktree: "/repo",
      directory: "/repo",
    } as never)

    expect(refreshes).toHaveLength(2)
  })

  test("refresh true waits for startup refresh before forcing another refresh", async () => {
    let resolveStartupRefresh: (() => void) | undefined
    const events: string[] = []
    const plugin = createCastPluginForTest({
      createIndexer: () => ({
        refresh: () =>
          events.length === 0
            ? new Promise((resolve) => {
                events.push("startup started")
                resolveStartupRefresh = () => {
                  events.push("startup finished")
                  resolve(emptyReadyIndex())
                }
              })
            : Promise.resolve(recordEventAndReturnIndex(events, "forced started")),
      }),
      createStore: () => ({ read: async () => emptyReadyIndex(), write: async () => undefined }),
      retrieve: async () => ({
        status: { ...emptyReadyIndex().metadata, hydeUsed: false },
        results: [],
        diagnostics: [],
      }),
    })
    const hooks = await plugin(input as never, {
      embedding: { baseURL: "https://example.test/v1", apiKey: "key", model: "embed" },
    })

    const executing = semanticSearchTool(hooks).execute({ query: "session", refresh: true }, {
      worktree: "/repo",
      directory: "/repo",
    } as never)
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(events).toEqual(["startup started"])
    resolveStartupRefresh?.()
    await executing

    expect(events).toEqual(["startup started", "startup finished", "forced started"])
  })

  test("concurrent forced refreshes run one at a time after startup refresh", async () => {
    let resolveStartupRefresh: (() => void) | undefined
    let resolveFirstForcedRefresh: (() => void) | undefined
    const events: string[] = []
    const plugin = createCastPluginForTest({
      createIndexer: () => ({
        refresh: () => {
          if (events.length === 0) {
            events.push("startup started")
            return new Promise((resolve) => {
              resolveStartupRefresh = () => {
                events.push("startup finished")
                resolve(emptyReadyIndex())
              }
            })
          }
          if (!events.includes("forced 1 started")) {
            events.push("forced 1 started")
            return new Promise((resolve) => {
              resolveFirstForcedRefresh = () => {
                events.push("forced 1 finished")
                resolve(emptyReadyIndex())
              }
            })
          }
          events.push("forced 2 started")
          events.push("forced 2 finished")
          return Promise.resolve(emptyReadyIndex())
        },
      }),
      createStore: () => ({ read: async () => emptyReadyIndex(), write: async () => undefined }),
      retrieve: async () => ({
        status: { ...emptyReadyIndex().metadata, hydeUsed: false },
        results: [],
        diagnostics: [],
      }),
    })
    const hooks = await plugin(input as never, {
      embedding: { baseURL: "https://example.test/v1", apiKey: "key", model: "embed" },
    })

    const first = semanticSearchTool(hooks).execute({ query: "first", refresh: true }, {
      worktree: "/repo",
      directory: "/repo",
    } as never)
    const second = semanticSearchTool(hooks).execute({ query: "second", refresh: true }, {
      worktree: "/repo",
      directory: "/repo",
    } as never)
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(events).toEqual(["startup started"])
    resolveStartupRefresh?.()
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(events).toEqual(["startup started", "startup finished", "forced 1 started"])
    resolveFirstForcedRefresh?.()
    await Promise.all([first, second])

    expect(events).toEqual([
      "startup started",
      "startup finished",
      "forced 1 started",
      "forced 1 finished",
      "forced 2 started",
      "forced 2 finished",
    ])
  })

  test("background refresh failure does not reject initialization or configured search", async () => {
    const plugin = createCastPluginForTest({
      createIndexer: () => ({
        refresh: async () => {
          throw new Error("refresh failed")
        },
      }),
      createStore: () => ({ read: async () => emptyReadyIndex(), write: async () => undefined }),
      retrieve: async () => ({
        status: { ...emptyReadyIndex().metadata, hydeUsed: false },
        results: [],
        diagnostics: [],
      }),
    })

    const hooks = await plugin(input as never, {
      embedding: { baseURL: "https://example.test/v1", apiKey: "key", model: "embed" },
    })
    const result = await semanticSearchTool(hooks).execute({ query: "session" }, {
      worktree: "/repo",
      directory: "/repo",
    } as never)

    expect(typeof result).toBe("object")
    if (typeof result === "string") {
      throw new Error("expected object tool result")
    }
    expect(result.metadata).toEqual({ hydeUsed: false, resultCount: 0 })
  })

  test("retrieve wiring passes store index, args, OpenAI calls, and worktree source reader", async () => {
    const index = emptyReadyIndex()
    const seen: { input?: unknown; index?: unknown; embed?: number[]; hyde?: string; source?: string } = {}
    const plugin = createCastPluginForTest({
      fetch: async (url, init) => {
        const body = JSON.parse(String(init.body))
        if (url.endsWith("/chat/completions")) {
          return Response.json({ choices: [{ message: { content: `hyde:${body.messages[1].content}` } }] })
        }
        return Response.json({ data: [{ embedding: String(body.input).startsWith("hyde:") ? [2] : [1] }] })
      },
      createIndexer: () => ({ refresh: async () => emptyReadyIndex() }),
      createStore: () => ({ read: async () => index, write: async () => undefined }),
      retrieve: async (input) => {
        seen.index = input.index
        seen.input = input.input
        seen.embed = await input.embed("query text")
        seen.hyde = await input.generateHyde("query text")
        seen.source = await input.readSource("nested/source.ts")
        return {
          status: { ...index.metadata, hydeUsed: true },
          results: [{ filePath: "nested/source.ts" } as never],
          diagnostics: [],
        }
      },
    })
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-plugin-"))
    try {
      await mkdir(path.join(dir, "nested"))
      await Bun.write(path.join(dir, "nested", "source.ts"), "source text")
      const hooks = await plugin({ ...input, directory: dir, worktree: dir } as never, {
        embedding: { baseURL: "https://example.test/v1", apiKey: "key", model: "embed" },
        hyde: { model: "hyde", enabled: true },
      })
      await semanticSearchTool(hooks).execute(
        { query: "query text", topK: 3, maxContextChars: 50, includeParents: true, paths: ["nested"] },
        { worktree: dir, directory: dir } as never,
      )

      expect(seen.index).toBe(index)
      expect(seen.input).toEqual({
        query: "query text",
        topK: 3,
        maxContextChars: 50,
        includeParents: true,
        paths: ["nested"],
      })
      expect(seen.embed).toEqual([1])
      expect(seen.hyde).toBe("hyde:query text")
      expect(seen.source).toBe("source text")
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  test("source reader rejects paths outside the plugin worktree", async () => {
    let rejected = false
    let outsidePath = ""
    let outsideRelativePath = ""
    const plugin = createCastPluginForTest({
      createIndexer: () => ({ refresh: async () => emptyReadyIndex() }),
      createStore: () => ({ read: async () => emptyReadyIndex(), write: async () => undefined }),
      retrieve: async (input) => {
        await input.readSource(outsideRelativePath).catch(() => {
          rejected = true
        })
        return { status: { ...emptyReadyIndex().metadata, hydeUsed: false }, results: [], diagnostics: [] }
      },
    })
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-plugin-"))
    try {
      outsidePath = path.join(dir, "..", `${path.basename(dir)}-outside.ts`)
      outsideRelativePath = `../${path.basename(outsidePath)}`
      await Bun.write(outsidePath, "outside")
      const hooks = await plugin({ ...input, directory: dir, worktree: dir } as never, {
        embedding: { baseURL: "https://example.test/v1", apiKey: "key", model: "embed" },
      })
      await semanticSearchTool(hooks).execute({ query: "session" }, { worktree: dir, directory: dir } as never)

      expect(rejected).toBe(true)
    } finally {
      await rm(dir, { recursive: true, force: true })
      if (outsidePath) {
        await rm(outsidePath, { force: true })
      }
    }
  })

  test("source reader rejects in-worktree symlinks that resolve outside the plugin worktree", async () => {
    let rejected = false
    const plugin = createCastPluginForTest({
      createIndexer: () => ({ refresh: async () => emptyReadyIndex() }),
      createStore: () => ({ read: async () => emptyReadyIndex(), write: async () => undefined }),
      retrieve: async (input) => {
        await input.readSource("secret-link.ts").catch(() => {
          rejected = true
        })
        return { status: { ...emptyReadyIndex().metadata, hydeUsed: false }, results: [], diagnostics: [] }
      },
    })
    const dir = await mkdtemp(path.join(os.tmpdir(), "cast-plugin-"))
    const outside = await mkdtemp(path.join(os.tmpdir(), "cast-plugin-outside-"))
    try {
      await Bun.write(path.join(outside, "secret.ts"), "outside")
      await symlink(path.join(outside, "secret.ts"), path.join(dir, "secret-link.ts"))
      const hooks = await plugin({ ...input, directory: dir, worktree: dir } as never, {
        embedding: { baseURL: "https://example.test/v1", apiKey: "key", model: "embed" },
      })
      await semanticSearchTool(hooks).execute({ query: "session" }, { worktree: dir, directory: dir } as never)

      expect(rejected).toBe(true)
    } finally {
      await rm(dir, { recursive: true, force: true })
      await rm(outside, { recursive: true, force: true })
    }
  })

  test("cache key is stable and separates embedding space settings and chunk size", async () => {
    const keys: string[] = []
    const plugin = createCastPluginForTest({
      createStore: (input) => {
        keys.push(input.cacheKey)
        return { read: async () => emptyReadyIndex(), write: async () => undefined }
      },
      createIndexer: () => ({ refresh: async () => emptyReadyIndex() }),
      retrieve: async () => ({
        status: { ...emptyReadyIndex().metadata, hydeUsed: false },
        results: [],
        diagnostics: [],
      }),
    })
    const baseOptions = {
      cacheDir: "/cache",
      embedding: { baseURL: "https://example.test/v1", apiKey: "key", model: "embed" },
    }

    await plugin(input as never, baseOptions)
    await plugin(input as never, baseOptions)
    await plugin(input as never, {
      ...baseOptions,
      embedding: { ...baseOptions.embedding, baseURL: "https://other.test/v1" },
    })
    await plugin(input as never, { ...baseOptions, embedding: { ...baseOptions.embedding, model: "other" } })
    await plugin(input as never, { ...baseOptions, embedding: { ...baseOptions.embedding, dimensions: 1024 } })
    await plugin(input as never, { ...baseOptions, maxChunkNonWhitespaceChars: 1000 })

    expect(keys[0]).toBe(keys[1])
    expect(keys[2]).not.toBe(keys[0])
    expect(keys[3]).not.toBe(keys[0])
    expect(keys[4]).not.toBe(keys[0])
    expect(keys[5]).not.toBe(keys[0])
  })

  test("dispose clears old refresh promise so later execution does not await it", async () => {
    let resolveRefresh: (() => void) | undefined
    const plugin = createCastPluginForTest({
      createIndexer: () => ({
        refresh: () =>
          new Promise((resolve) => {
            resolveRefresh = () => resolve(emptyReadyIndex())
          }),
      }),
      createStore: () => ({ read: async () => emptyReadyIndex(), write: async () => undefined }),
      retrieve: async () => ({
        status: { ...emptyReadyIndex().metadata, hydeUsed: false },
        results: [],
        diagnostics: [],
      }),
    })
    const hooks = await plugin(input as never, {
      embedding: { baseURL: "https://example.test/v1", apiKey: "key", model: "embed" },
    })

    await hooks.dispose?.()
    const result = await Promise.race([
      semanticSearchTool(hooks).execute({ query: "session" }, {
        worktree: "/repo",
        directory: "/repo",
      } as never),
      new Promise((resolve) => setTimeout(() => resolve("timed out"), 50)),
    ])

    resolveRefresh?.()
    expect(result).not.toBe("timed out")
  })
})

function emptyReadyIndex() {
  const index = createEmptyIndex({
    projectId: "p",
    worktree: "/repo",
    cacheKey: "key",
    maxChunkNonWhitespaceChars: 2000,
  })
  index.metadata.status = "ready"
  return index
}

function recordEventAndReturnIndex(events: string[], event: string) {
  events.push(event)
  return emptyReadyIndex()
}

function semanticSearchTool(hooks: Awaited<ReturnType<typeof castPlugin>>) {
  const semanticSearchCode = hooks.tool?.semantic_search_code
  if (!semanticSearchCode) {
    throw new Error("semantic_search_code tool was not registered")
  }
  return semanticSearchCode
}
