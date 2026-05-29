import { createHash } from "node:crypto"
import { realpath } from "node:fs/promises"
import path from "node:path"
import { type Plugin, tool } from "@opencode-ai/plugin"
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
          description: "Semantically search code in the current worktree with cAST retrieval and parent context.",
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
              options,
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
              readSource: async (filePath) => Bun.file(await resolveWorktreePath(input.worktree, filePath)).text(),
            })

            return {
              title: `Semantic code search: ${args.query}`,
              output: JSON.stringify(output, null, 2),
              metadata: { hydeUsed: output.status.hydeUsed, resultCount: output.results.length },
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
