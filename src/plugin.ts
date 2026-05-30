import { createHash } from "node:crypto"
import { realpath } from "node:fs/promises"
import path from "node:path"
import { type Plugin, tool } from "@opencode-ai/plugin"
import { getChunkById } from "./chunk-lookup.js"
import { parseSource } from "./language.js"
import { createOpenAIClient, type FetchLike } from "./openai.js"
import { parseOptions } from "./options.js"
import { retrieve } from "./retriever.js"
import { createIndexer } from "./scanner.js"
import { createIndexStore } from "./store.js"

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
    const store = (dependencies.createStore ?? createIndexStore)({
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
    })
    const client = createOpenAIClient(dependencies.fetch ? { fetch: dependencies.fetch } : {})
    let refresh: Promise<unknown> | undefined
    let refreshTail = Promise.resolve()

    const queueRefresh = () => {
      const embedding = options.embedding
      if (!embedding) {
        return Promise.resolve()
      }
      refresh = refreshTail
        .then(() =>
          (dependencies.createIndexer ?? createIndexer)({
            worktree: input.worktree,
            options,
            store,
            parse: parseSource,
            embed: (text) => client.embed({ ...embedding, input: text }),
          }).refresh(),
        )
        .catch(() => undefined)
      refreshTail = refresh.then(() => undefined)
      return refresh
    }

    if (options.embedding && options.diagnostics.length === 0) {
      queueRefresh()
    }

    return {
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
            maxContextChars: tool.schema.number().int().positive().optional(),
            includeParents: tool.schema.boolean().optional(),
            refresh: tool.schema.boolean().optional(),
            paths: tool.schema.array(tool.schema.string()).optional(),
          },
          async execute(args) {
            const embedding = options.embedding
            if (!embedding || options.diagnostics.length > 0) {
              return {
                title: "Semantic code search is not configured",
                output: options.diagnostics.join("\n"),
                metadata: { configured: false },
              }
            }

            if (args.refresh) {
              await queueRefresh()
            }
            await refresh
            const output = await (dependencies.retrieve ?? retrieve)({
              index: await store.read(),
              input: args,
              options: { ...options, hybrid: options.retrieval.hybrid, rerank: options.rerank },
              embed: (text) => client.embed({ ...embedding, input: text }),
              generateHyde: (query) =>
                options.hyde.baseURL && options.hyde.model
                  ? client.generateHyde({
                      baseURL: options.hyde.baseURL,
                      apiKey: options.hyde.apiKey,
                      model: options.hyde.model,
                      query,
                    })
                  : Promise.reject(new Error("HyDE is not configured")),
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
            })

            return {
              title: `Semantic code search: ${args.query}`,
              output: JSON.stringify(output, null, 2),
              metadata: {
                hydeUsed: output.status.hydeUsed,
                rerankUsed: output.status.rerankUsed,
                resultCount: output.results.length,
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

            await refresh
            const output = await getChunkById({
              index: await store.read(),
              input: args,
              readSource: async (filePath) => Bun.file(await resolveWorktreePath(input.worktree, filePath)).text(),
            })

            return {
              title: `Semantic chunk lookup: ${args.id}`,
              output: JSON.stringify(output, null, 2),
              metadata: { found: Boolean(output.chunk) },
            }
          },
        }),
      },
      async dispose() {
        refresh = undefined
        refreshTail = Promise.resolve()
      },
    }
  }
}

export const castPlugin = createCastPluginForTest()

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
