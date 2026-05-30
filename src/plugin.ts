import { createHash } from "node:crypto"
import { realpath } from "node:fs/promises"
import path from "node:path"
import { type Plugin, type ToolContext, tool } from "@opencode-ai/plugin"
import { getChunkById } from "./chunk-lookup.js"
import { parseSource } from "./language.js"
import { createOpenAIClient, type FetchLike } from "./openai.js"
import { parseOptions } from "./options.js"
import { retrieve } from "./retriever.js"
import { createIndexer } from "./scanner.js"
import { createIndexStore } from "./store.js"

interface VectorCandidateStore {
  searchVectorCandidates(
    queryEmbedding: number[],
    topK: number,
    paths?: string[],
  ): Promise<Array<{ id: string; score: number }>>
}

type IndexingStore = Parameters<typeof createIndexer>[0]["store"]
type WrappedIndexingStore = IndexingStore & Partial<VectorCandidateStore>

interface OpenCodeHydeClient {
  session: {
    create(parameters: { body?: { parentID?: string; title?: string }; query?: { directory?: string } }): Promise<{
      data?: { id: string }
      error?: unknown
    }>
    prompt(parameters: {
      path: { id: string }
      query?: { directory?: string }
      body?: {
        model?: { providerID: string; modelID: string }
        tools?: Record<string, boolean>
        system?: string
        parts: Array<{ type: string; text?: string }>
      }
    }): Promise<{
      data?: { parts: Array<{ type: string; text?: string }> }
      error?: unknown
    }>
    delete(parameters: { path: { id: string }; query?: { directory?: string } }): Promise<{
      data?: boolean
      error?: unknown
    }>
  }
}

class IndexUnavailableError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "IndexUnavailableError"
  }
}

export function createCastPluginForTest(
  dependencies: {
    fetch?: FetchLike
    createStore?: typeof createIndexStore
    createIndexer?: typeof createIndexer
    retrieve?: typeof retrieve
  } = {},
): Plugin {
  return async (input, rawOptions) => {
    const options = parseOptions(rawOptions)
    const storeInput = {
      cacheDir: options.cacheDir,
      cacheKey: createHash("sha256")
        .update(
          JSON.stringify({
            projectId: input.project.id,
            worktree: input.worktree,
            embedding: options.embedding
              ? {
                  baseURL: options.embedding.baseURL,
                  model: options.embedding.model,
                  dimensions: options.embedding.dimensions,
                }
              : "missing",
            maxChunkNonWhitespaceChars: options.maxChunkNonWhitespaceChars,
          }),
        )
        .digest("hex"),
      embeddingDimensions: options.embedding?.dimensions,
    }
    let store: ReturnType<typeof createIndexStore> | undefined
    let storeError: string | undefined
    try {
      store = (dependencies.createStore ?? createIndexStore)(storeInput)
    } catch (error) {
      storeError = formatThrownError(error)
    }
    const client = createOpenAIClient(dependencies.fetch ? { fetch: dependencies.fetch } : {})
    const sessionModels = new Map<string, { providerID: string; modelID: string }>()
    let refresh: Promise<unknown> | undefined
    let refreshTail = Promise.resolve()

    const queueRefresh = (refreshInput: { background?: boolean } = {}) => {
      const embedding = options.embedding
      if (!(embedding && store) || storeError) {
        return Promise.resolve()
      }
      const indexStore = store
      refresh = refreshTail
        .then(() => {
          if (storeError) {
            return
          }
          const indexingStore = wrapIndexingStore(indexStore)
          return (dependencies.createIndexer ?? createIndexer)({
            worktree: input.worktree,
            options: {
              maxChunkNonWhitespaceChars: options.maxChunkNonWhitespaceChars,
              maxFileBytes: options.maxFileBytes,
              includeGlobs: options.includeGlobs,
              excludeGlobs: options.excludeGlobs,
              topK: options.topK,
              maxContextChars: options.maxContextChars,
              chunking: options.chunking,
            },
            store: indexingStore,
            parse: parseSource,
            embed: (text) => client.embed({ ...embedding, input: text }),
          }).refresh()
        })
        .catch((error) => {
          if (error instanceof IndexUnavailableError) {
            storeError = error.message
            return
          }
          if (!refreshInput.background) {
            throw error
          }
          return
        })
      refreshTail = refresh.then(
        () => undefined,
        () => undefined,
      )
      return refresh
    }

    const recordStoreUnavailable = (error: unknown) => {
      if (!isStoreUnavailableError(error)) {
        return false
      }
      storeError = formatThrownError(error)
      return true
    }

    const readIndex = async () => {
      if (!store) {
        throw new Error(storeError ?? "index unavailable")
      }
      try {
        return await store.read()
      } catch (error) {
        if (!recordStoreUnavailable(error)) {
          throw error
        }
        throw new IndexUnavailableError(storeError ?? formatThrownError(error))
      }
    }

    const wrapStoreOperation = async <T>(operation: () => Promise<T>) => {
      try {
        return await operation()
      } catch (error) {
        if (!recordStoreUnavailable(error)) {
          throw error
        }
        throw new IndexUnavailableError(storeError ?? formatThrownError(error))
      }
    }

    const wrapIndexingStore = (indexStore: typeof store): IndexingStore => {
      if (!indexStore) {
        throw new IndexUnavailableError(storeError ?? "index unavailable")
      }
      const wrapped: WrappedIndexingStore = {
        read: () => wrapStoreOperation(() => indexStore.read()),
        write: (index) => wrapStoreOperation(() => indexStore.write(index)),
      }
      const maybeRunStore = indexStore as Partial<IndexingStore>
      if (typeof maybeRunStore.beginIndexRun === "function") {
        wrapped.beginIndexRun = (input) =>
          wrapStoreOperation(() => maybeRunStore.beginIndexRun?.(input) as Promise<{ runId: string }>)
      }
      if (typeof maybeRunStore.getCompletedFile === "function") {
        wrapped.getCompletedFile = (runId, filePath, fingerprint) =>
          wrapStoreOperation(
            () =>
              maybeRunStore.getCompletedFile?.(runId, filePath, fingerprint) as ReturnType<
                NonNullable<IndexingStore["getCompletedFile"]>
              >,
          )
      }
      if (typeof maybeRunStore.writeFileResult === "function") {
        wrapped.writeFileResult = (runId, fileResult) =>
          wrapStoreOperation(() => maybeRunStore.writeFileResult?.(runId, fileResult) as Promise<void>)
      }
      if (typeof maybeRunStore.activateRun === "function") {
        wrapped.activateRun = (runId, index) =>
          wrapStoreOperation(() => maybeRunStore.activateRun?.(runId, index) as Promise<void>)
      }
      if (hasVectorCandidateStore(indexStore)) {
        wrapped.searchVectorCandidates = (queryEmbedding: number[], topK: number, paths?: string[]) =>
          wrapStoreOperation(() => indexStore.searchVectorCandidates(queryEmbedding, topK, paths))
      }
      return wrapped
    }

    const vectorCandidateStore = (): VectorCandidateStore | undefined => {
      if (!hasVectorCandidateStore(store)) {
        return
      }
      return {
        searchVectorCandidates: async (queryEmbedding, topK, paths) => {
          try {
            return await store.searchVectorCandidates(queryEmbedding, topK, paths)
          } catch (error) {
            if (!recordStoreUnavailable(error)) {
              throw error
            }
            throw new IndexUnavailableError(storeError ?? formatThrownError(error))
          }
        },
      }
    }

    if (options.embedding && options.diagnostics.length === 0) {
      queueRefresh({ background: true })
    }

    const generateOpenCodeHyde = async (query: string, context: ToolContext) => {
      const model = sessionModels.get(context.sessionID)
      if (!model) {
        throw new Error(`No opencode model is tracked for session ${context.sessionID}`)
      }
      const opencodeClient = input.client as unknown as OpenCodeHydeClient | undefined
      if (!opencodeClient?.session) {
        throw new Error("OpenCode client is not available for HyDE generation")
      }

      const created = await opencodeClient.session.create({
        body: { parentID: context.sessionID, title: "OpenCode Cast HyDE" },
        query: { directory: context.directory },
      })
      if (created.error) {
        throw new Error(`OpenCode HyDE session create failed: ${formatSdkError(created.error)}`)
      }

      const hydeSessionID = created.data?.id
      if (!hydeSessionID) {
        throw new Error("OpenCode HyDE session create returned no session id")
      }
      try {
        const prompted = await opencodeClient.session.prompt({
          path: { id: hydeSessionID },
          query: { directory: context.directory },
          body: {
            model,
            tools: {},
            system:
              "Write a concise hypothetical code or documentation excerpt that would satisfy the search query. Return only useful search text.",
            parts: [{ type: "text", text: query }],
          },
        })
        if (prompted.error) {
          throw new Error(`OpenCode HyDE prompt failed: ${formatSdkError(prompted.error)}`)
        }
        if (!prompted.data) {
          throw new Error("OpenCode HyDE prompt returned no response")
        }

        const text = prompted.data.parts
          .filter((part) => part.type === "text" && typeof part.text === "string")
          .map((part) => part.text)
          .join("\n")
          .trim()
        if (!text) {
          throw new Error("OpenCode HyDE prompt returned no text")
        }
        return text
      } finally {
        await opencodeClient.session
          .delete({ path: { id: hydeSessionID }, query: { directory: context.directory } })
          .catch(() => undefined)
      }
    }

    return {
      "chat.message": async (event) => {
        if (event.model) {
          sessionModels.set(event.sessionID, event.model)
        }
      },
      tool: {
        semantic_search_code: tool({
          description: `
Find relevant code in the current repository by meaning instead of exact text, symbol, or implementation intent.

Use this as the default first tool for code discovery in this repository, including when the user asks how something works, where behavior, features, APIs, errors, data flow, or relevant code lives, or asks about a known class, function, method, type, test, or feature name. Prefer this before grep/glob/read because it returns ranked, syntax-aware matches with surrounding implementation context and file/line references.

This tool searches syntax-aware code chunks such as functions, classes, methods, and nearby context where parser support is available. Use grep only when you need exhaustive literal matching, occurrence counts, mechanical text replacement preparation, or matches in files that are not meaningfully represented as code chunks. Use read after this tool returns candidates when you need larger surrounding context or exact verification. Use paths to restrict the search area. Use refresh if files may have changed since the index was built.
`,
          args: {
            query: tool.schema.string(),
            topK: tool.schema.number().int().positive().optional(),
            minFinalScore: tool.schema.number().optional(),
            maxContextChars: tool.schema.number().int().positive().optional(),
            includeParents: tool.schema.boolean().optional(),
            refresh: tool.schema.boolean().optional(),
            paths: tool.schema.array(tool.schema.string()).optional(),
          },
          async execute(args, context) {
            const embedding = options.embedding
            if (!embedding || options.diagnostics.length > 0) {
              return {
                title: "Semantic code search is not configured",
                output: options.diagnostics.join("\n"),
                metadata: { configured: false },
              }
            }
            if (!store) {
              return unavailableToolResult("Semantic code search index unavailable", storeError)
            }

            if (args.refresh) {
              await queueRefresh()
            }
            await refresh
            if (storeError) {
              return unavailableToolResult("Semantic code search index unavailable", storeError)
            }
            let output: Awaited<ReturnType<typeof retrieve>>
            try {
              const indexStore = vectorCandidateStore()
              output = await (dependencies.retrieve ?? retrieve)({
                index: await readIndex(),
                input: args,
                options: { ...options, hybrid: options.retrieval.hybrid, rerank: options.rerank },
                embed: (text) => client.embed({ ...embedding, input: text }),
                generateHyde: (query) =>
                  options.hyde.mode === "openai-compatible" && options.hyde.baseURL && options.hyde.model
                    ? client.generateHyde({
                        baseURL: options.hyde.baseURL,
                        apiKey: options.hyde.apiKey,
                        model: options.hyde.model,
                        query,
                      })
                    : generateOpenCodeHyde(query, context),
                rerank: (query, documents) =>
                  options.rerank
                    ? client.rerank({
                        baseURL: options.rerank.baseURL,
                        apiKey: options.rerank.apiKey,
                        model: options.rerank.model,
                        query,
                        documents,
                      })
                    : Promise.reject(new Error("Rerank is not configured")),
                readSource: async (filePath) => Bun.file(await resolveWorktreePath(input.worktree, filePath)).text(),
                indexStore,
              })
            } catch (error) {
              if (!(error instanceof IndexUnavailableError)) {
                throw error
              }
              return unavailableToolResult("Semantic code search index unavailable", storeError)
            }

            return {
              title: `Semantic code search: ${args.query}`,
              output: JSON.stringify(output, null, 2),
              metadata: {
                hydeUsed: output.status.hydeUsed,
                rerankUsed: output.status.rerankUsed,
                resultCount: output.results.length,
                minFinalScore: output.status.minFinalScore,
                filteredCount: output.status.filteredCount,
              },
            }
          },
        }),
        semantic_get_chunk: tool({
          description: `
Fetch an indexed semantic code chunk by ID returned from semantic_search_code.

Use this after semantic_search_code when you need the exact cached chunk, its parent context, or nearby topology such as parents, siblings, and children.
`,
          args: {
            id: tool.schema.string(),
            includeParents: tool.schema.boolean().optional(),
            includeSiblings: tool.schema.boolean().optional(),
            includeChildren: tool.schema.boolean().optional(),
            maxContextChars: tool.schema.number().int().positive().optional(),
          },
          async execute(args) {
            if (!options.embedding || options.diagnostics.length > 0) {
              return {
                title: "Semantic chunk lookup is not configured",
                output: options.diagnostics.join("\n"),
                metadata: { configured: false },
              }
            }
            if (!store) {
              return unavailableToolResult("Semantic chunk lookup index unavailable", storeError)
            }

            await refresh
            if (storeError) {
              return unavailableToolResult("Semantic chunk lookup index unavailable", storeError)
            }
            let output: Awaited<ReturnType<typeof getChunkById>>
            try {
              output = await getChunkById({
                index: await readIndex(),
                input: args,
                readSource: async (filePath) => Bun.file(await resolveWorktreePath(input.worktree, filePath)).text(),
              })
            } catch (error) {
              if (!(error instanceof IndexUnavailableError)) {
                throw error
              }
              return unavailableToolResult("Semantic chunk lookup index unavailable", storeError)
            }

            return {
              title: `Semantic chunk lookup: ${args.id}`,
              output: JSON.stringify(output, null, 2),
              metadata: { found: Boolean(output.chunk) },
            }
          },
        }),
      },
      async dispose() {
        sessionModels.clear()
        refresh = undefined
        refreshTail = Promise.resolve()
      },
    }
  }
}

function hasVectorCandidateStore(value: unknown): value is VectorCandidateStore {
  return (
    typeof value === "object" &&
    value !== null &&
    "searchVectorCandidates" in value &&
    typeof value.searchVectorCandidates === "function"
  )
}

function unavailableToolResult(title: string, message: string | undefined) {
  return {
    title,
    output: `index unavailable${message ? `: ${message}` : ""}`,
    metadata: { configured: false },
  }
}

function isStoreUnavailableError(error: unknown) {
  const message = formatThrownError(error).toLowerCase()
  return (
    message.includes("sqlite") ||
    message.includes("database") ||
    message.includes("index unavailable") ||
    message.includes("failed to open") ||
    message.includes("unable to open")
  )
}

function formatThrownError(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}

export const castPlugin = createCastPluginForTest()

function formatSdkError(error: unknown) {
  if (error instanceof Error) {
    return error.message
  }
  if (typeof error === "string") {
    return error
  }
  try {
    return JSON.stringify(error)
  } catch {
    return String(error)
  }
}

async function resolveWorktreePath(worktree: string, filePath: string) {
  const root = path.resolve(worktree)
  const resolved = path.resolve(root, filePath)
  const relative = path.relative(root, resolved)
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`source path escapes worktree: ${filePath}`)
  }
  const realRoot = await realpath(root)
  const realResolved = await realpath(resolved)
  const realRelative = path.relative(realRoot, realResolved)
  if (realRelative.startsWith("..") || path.isAbsolute(realRelative)) {
    throw new Error(`source path escapes worktree: ${filePath}`)
  }
  return resolved
}
