# opencode-plugin-cast

Unofficial cAST semantic code search plugin for opencode. Code Retrieval-Augmented Generation with structural chunking via abstract syntax tree.

This is an unofficial community plugin. It is not affiliated with, endorsed by, sponsored by, or maintained by opencode, opencode.ai, or the `@opencode-ai` npm organization. It depends on `@opencode-ai/plugin` only for the public plugin API.

The plugin registers `semantic_search_code`, a repository search tool that chunks code with Tree-sitter AST structure where supported, and `semantic_get_chunk`, which fetches exact chunks by IDs returned in search topology.

## Installation

Add the plugin to your opencode config:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    [
      "opencode-plugin-cast",
      {
        "embedding": {
          "baseURL": "https://api.openai.com/v1",
          "apiKeyEnv": "OPENAI_API_KEY",
          "model": "text-embedding-3-small"
        },
        "hyde": {
          "model": "gpt-4o-mini",
          "threshold": 0.35
        }
      }
    ]
  ]
}
```

Restart opencode after changing plugin configuration. Plugins are loaded when opencode starts and are not hot-reloaded.

## OpenAI-Compatible Providers

`embedding.baseURL` must point at an OpenAI-compatible API root. `embedding.model` is required. Provide credentials with either `embedding.apiKey` or `embedding.apiKeyEnv`.

```json
{
  "embedding": {
    "baseURL": "https://api.openai.com/v1",
    "apiKeyEnv": "OPENAI_API_KEY",
    "model": "text-embedding-3-small",
    "dimensions": 1536
  }
}
```

`dimensions` is optional and is forwarded to providers that support embedding dimension selection.

## OpenRouter HyDE Example

OpenRouter can be used as the HyDE chat provider while embeddings come from a verified OpenAI-compatible embeddings provider. `embedding.baseURL` must point to a provider and model that supports the OpenAI-compatible embeddings endpoint; only use OpenRouter for embeddings if you have verified the selected OpenRouter model supports embeddings.

```json
{
  "plugin": [
    [
      "opencode-plugin-cast",
      {
        "embedding": {
          "baseURL": "https://api.openai.com/v1",
          "apiKeyEnv": "OPENAI_API_KEY",
          "model": "text-embedding-3-small"
        },
        "hyde": {
          "baseURL": "https://openrouter.ai/api/v1",
          "apiKeyEnv": "OPENROUTER_API_KEY",
          "model": "openai/gpt-4o-mini",
          "threshold": 0.35
        }
      }
    ]
  ]
}
```

## OpenRouter Rerank Example

OpenRouter's rerank endpoint can be used to reorder the candidates found by vector and hybrid retrieval. Reranking is enabled by default when `rerank.baseURL` and `rerank.model` are configured. If the rerank request fails, `semantic_search_code` falls back to the current retrieval order and includes a diagnostic.

```json
{
  "plugin": [
    [
      "opencode-plugin-cast",
      {
        "embedding": {
          "baseURL": "https://api.openai.com/v1",
          "apiKeyEnv": "OPENAI_API_KEY",
          "model": "text-embedding-3-small"
        },
        "rerank": {
          "baseURL": "https://openrouter.ai/api/v1",
          "apiKeyEnv": "OPENROUTER_API_KEY",
          "model": "cohere/rerank-4-fast",
          "candidateMultiplier": 4
        }
      }
    ]
  ]
}
```

`candidateMultiplier` controls how many already-ranked candidates are sent to the reranker: `topK * candidateMultiplier`. The default is `4`.

## Tool

`semantic_search_code` inputs:

- `query`: natural-language repository search query.
- `topK`: optional number of final results. Defaults to `5`.
- `maxContextChars`: optional context budget. Defaults to `12000`.
- `includeParents`: optional parent context toggle. Defaults to `true`.
- `refresh`: optional forced index refresh before searching.
- `paths`: optional path filters. Entries can be exact file paths or directory prefixes.

Each search result includes labeled topology entries that keep chunk IDs actionable while making parent, child, sibling, and symbol relationships readable:

```json
{
  "topology": {
    "chunk": {
      "id": "chunk_7f3a9c2e4b1d",
      "label": "function semanticSearchCode",
      "range": "src/plugin.ts:120-188"
    },
    "parent": {
      "id": "chunk_15d8e0a6c934",
      "label": "file src/plugin.ts",
      "range": "src/plugin.ts:20-220"
    },
    "children": [
      {
        "id": "chunk_b8a41f0c72de",
        "label": "block chunk",
        "range": "src/plugin.ts:142-166"
      }
    ],
    "previousSibling": {
      "id": "chunk_029d64f1a83c",
      "label": "function resolveWorktreePath",
      "range": "src/plugin.ts:90-118"
    },
    "nextSibling": {
      "id": "chunk_e61c5b9a0d2f",
      "label": "function semanticGetChunk",
      "range": "src/plugin.ts:190-218"
    },
    "symbols": ["function semanticSearchCode"]
  }
}
```

`semantic_get_chunk` inputs:

- `id`: chunk ID returned from `semantic_search_code` topology.
- `includeParents`: optional parent context toggle. Defaults to `true`.
- `includeSiblings`: optional sibling context toggle. Defaults to `true`.
- `includeChildren`: optional child context toggle. Defaults to `true`.
- `maxContextChars`: optional limit for parent and related chunk text. When omitted, lookup returns full stored related chunk text and full fitting parent context.

`semantic_search_code` returns a JSON payload with ranked results, diagnostics, and retrieval status metadata. `semantic_get_chunk` returns the requested chunk, labeled topology, related chunks, diagnostics, and status metadata. When embedding config is missing or invalid, opencode still starts and the tools return a clear configuration error.

## Retrieval

`semantic_search_code` uses hybrid semantic vector and BM25 lexical retrieval by default. BM25 helps with exact identifiers, paths, error strings, and config keys; vectors help with semantic intent and related code. Results are fused with reciprocal-rank fusion (RRF), so exact lexical matches and semantic matches can both contribute without requiring an external BM25 or search service.

Configure hybrid retrieval under `retrieval.hybrid`:

```json
{
  "retrieval": {
    "hybrid": {
      "enabled": true,
      "mode": "parallel",
      "rrfK": 60,
      "vectorCandidateMultiplier": 8,
      "bm25CandidateMultiplier": 8,
      "vectorWeight": 1,
      "bm25Weight": 1
    }
  }
}
```

`mode` can be `parallel`, `bm25-prefilter`, or `vector-prefilter`. `parallel` searches both candidate sets before RRF. `bm25-prefilter` keeps BM25 candidates in fusion while limiting vector-side contributions to overlapping candidates, and `vector-prefilter` ranks BM25 within the vector candidate pool. When HyDE is triggered, it remains on the semantic/vector side while BM25 still participates in fusion.

When reranking is configured, the plugin first finds candidates through the normal vector, HyDE, and hybrid pipeline, then sends a larger candidate set to the reranker before returning the final `topK` results. Reranker documents include the chunk path, line range, kind, and chunk text, but not expanded parent context.

Lexical stats are persisted in the index cache. Older or missing cache data degrades to vector-only retrieval with a diagnostic until the index is refreshed or rebuilt.

## Cache

The index is stored outside the repository. The default cache directory is `${XDG_CACHE_HOME:-~/.cache}/opencode/cast`.

Override the cache location with plugin config:

```json
{
  "cacheDir": "/absolute/path/to/cache"
}
```

Or with an environment variable:

```bash
export OPENCODE_CAST_CACHE_DIR=/absolute/path/to/cache
```

The cache key includes the opencode project id, worktree path, embedding model settings, and chunk-size setting.

## Language Coverage

The bundled grammar set targets TypeScript/TSX, JavaScript/JSX, Python, Go, Rust, Java, Ruby, and Bash. Files without a supported parser are indexed with fallback text chunks.

## HyDE

The plugin runs normal embedding search first. If the best similarity is below `hyde.threshold`, it asks the configured chat model to produce a concise hypothetical code-search target, embeds that text, and reranks candidates.

HyDE is enabled by default when `hyde.model` is configured. Set `hyde.enabled` to `false` to disable it:

```json
{
  "hyde": {
    "enabled": false
  }
}
```

HyDE defaults to the embedding provider base URL and API key when `hyde.baseURL`, `hyde.apiKey`, and `hyde.apiKeyEnv` are omitted. HyDE failures fall back to normal embedding results.

## Local Development

Install dependencies, run tests, typecheck, and build:

```bash
npm_config_nodedir=/usr npm exec --yes bun@1.3.14 -- install
npm exec --yes bun@1.3.14 -- test --timeout 30000
npm exec --yes bun@1.3.14 -- run typecheck
npm exec --yes bun@1.3.14 -- run build
```

To test a local build in opencode, point config at the built file URL:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    [
      "file:///absolute/path/to/opencode-plugin-cast/dist/index.js",
      {
        "embedding": {
          "baseURL": "https://api.openai.com/v1",
          "apiKeyEnv": "OPENAI_API_KEY",
          "model": "text-embedding-3-small"
        }
      }
    ]
  ]
}
```

Restart opencode after rebuilding the plugin or changing plugin config. The package root intentionally exports only the default plugin function so opencode does not confuse internal helper functions for plugin exports.

## Acknowledgements

This implementation is based on ideas from:

Zhang et al., “cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree,” arXiv:2506.15655, 2025.  
https://arxiv.org/abs/2506.15655

This repository is an independent implementation and is not affiliated with the original authors.
