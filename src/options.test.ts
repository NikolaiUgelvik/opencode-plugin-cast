import { describe, expect, test } from "bun:test"
import { parseOptions } from "./options.js"

describe("parseOptions", () => {
  test("applies defaults and resolves api keys from env", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKeyEnv: "CAST_TEST_KEY",
          model: "text-embedding-3-small",
        },
        hyde: {
          model: "gpt-4o-mini",
          threshold: 0.42,
        },
      },
      { CAST_TEST_KEY: "secret" },
    )

    expect(options.embedding).toEqual({
      baseURL: "https://example.test/v1",
      apiKey: "secret",
      model: "text-embedding-3-small",
      dimensions: undefined,
    })
    expect(options.hyde).toEqual({
      mode: "opencode",
      baseURL: undefined,
      apiKey: undefined,
      model: undefined,
      threshold: 0.42,
      enabled: true,
    })
    expect(options.maxChunkNonWhitespaceChars).toBe(2000)
    expect(options.maxContextChars).toBe(12_000)
    expect(options.maxFileBytes).toBe(2 * 1024 * 1024)
    expect(options.topK).toBe(5)
    expect(options.includeGlobs).toEqual(["**/*"])
    expect(options.excludeGlobs).toEqual(
      expect.arrayContaining([
        "**/*.{png,jpg,jpeg,gif,webp,ico,pdf,zip,gz,tgz,tar,7z,mp4,mov,mp3,woff,woff2,ttf,eot}",
        "**/bun.lock",
      ]),
    )
    expect(options.retrieval.hybrid).toEqual({
      enabled: true,
      mode: "parallel",
      rrfK: 60,
      vectorCandidateMultiplier: 8,
      bm25CandidateMultiplier: 8,
      vectorWeight: 1,
      bm25Weight: 1,
    })
  })

  test("parses configured hybrid retrieval options", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "literal",
          model: "text-embedding-3-small",
        },
        retrieval: {
          hybrid: {
            enabled: false,
            mode: "bm25-prefilter",
            rrfK: 40,
            vectorCandidateMultiplier: 5,
            bm25CandidateMultiplier: 9,
            vectorWeight: 0.75,
            bm25Weight: 1.25,
          },
        },
      },
      {},
    )

    expect(options.retrieval.hybrid).toEqual({
      enabled: false,
      mode: "bm25-prefilter",
      rrfK: 40,
      vectorCandidateMultiplier: 5,
      bm25CandidateMultiplier: 9,
      vectorWeight: 0.75,
      bm25Weight: 1.25,
    })
  })

  test("parses configured rerank options and resolves api key from env", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "embedding-key",
          model: "text-embedding-3-small",
        },
        rerank: {
          baseURL: "https://openrouter.ai/api/v1",
          apiKeyEnv: "OPENROUTER_API_KEY",
          model: "cohere/rerank-4-fast",
          candidateMultiplier: 6,
        },
      },
      { OPENROUTER_API_KEY: "rerank-key" },
    )

    expect(options.rerank).toEqual({
      baseURL: "https://openrouter.ai/api/v1",
      apiKey: "rerank-key",
      model: "cohere/rerank-4-fast",
      candidateMultiplier: 6,
    })
  })

  test("defaults rerank candidate multiplier when rerank is configured", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "embedding-key",
          model: "text-embedding-3-small",
        },
        rerank: {
          baseURL: "https://openrouter.ai/api/v1",
          apiKey: "rerank-key",
          model: "cohere/rerank-4-fast",
        },
      },
      {},
    )

    expect(options.rerank).toEqual({
      baseURL: "https://openrouter.ai/api/v1",
      apiKey: "rerank-key",
      model: "cohere/rerank-4-fast",
      candidateMultiplier: 4,
    })
  })

  test("reports invalid rerank fields without disabling valid embedding config", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "embedding-key",
          model: "text-embedding-3-small",
        },
        rerank: {
          baseURL: "not a url",
          model: 42,
          candidateMultiplier: 0,
        },
      },
      {},
    )

    expect(options.embedding).toEqual({
      baseURL: "https://example.test/v1",
      apiKey: "embedding-key",
      model: "text-embedding-3-small",
      dimensions: undefined,
    })
    expect(options.rerank).toBeUndefined()
    expect(options.diagnostics.some((diagnostic) => diagnostic.startsWith("rerank.baseURL:"))).toBe(true)
    expect(options.diagnostics.some((diagnostic) => diagnostic.startsWith("rerank.model:"))).toBe(true)
    expect(options.diagnostics.some((diagnostic) => diagnostic.startsWith("rerank.candidateMultiplier:"))).toBe(true)
  })

  test("reports invalid hybrid retrieval options", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "literal",
          model: "text-embedding-3-small",
        },
        retrieval: {
          hybrid: {
            mode: "bad",
            rrfK: 0,
          },
        },
      },
      {},
    )

    expect(options.diagnostics.some((diagnostic) => diagnostic.startsWith("retrieval.hybrid.mode:"))).toBe(true)
    expect(options.diagnostics.some((diagnostic) => diagnostic.startsWith("retrieval.hybrid.rrfK:"))).toBe(true)
  })

  test("applies chunking defaults", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "literal",
          model: "text-embedding-3-small",
        },
      },
      {},
    )

    expect(options.chunking).toEqual({
      overlap: 0,
      expansion: false,
      minSemanticNonWhitespaceChars: 8,
    })
  })

  test("parses configured chunking options", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "literal",
          model: "text-embedding-3-small",
        },
        chunking: {
          overlap: 1,
          expansion: true,
          minSemanticNonWhitespaceChars: 12,
        },
      },
      {},
    )

    expect(options.chunking).toEqual({
      overlap: 1,
      expansion: true,
      minSemanticNonWhitespaceChars: 12,
    })
  })

  test("reports invalid chunking options and falls back to defaults", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "literal",
          model: "text-embedding-3-small",
        },
        chunking: {
          overlap: -1,
          expansion: "yes",
          minSemanticNonWhitespaceChars: 0,
        },
      },
      {},
    )

    expect(options.chunking).toEqual({
      overlap: 0,
      expansion: false,
      minSemanticNonWhitespaceChars: 8,
    })
    expect(options.diagnostics.some((diagnostic) => diagnostic.startsWith("chunking.overlap:"))).toBe(true)
    expect(options.diagnostics.some((diagnostic) => diagnostic.startsWith("chunking.expansion:"))).toBe(true)
    expect(
      options.diagnostics.some((diagnostic) => diagnostic.startsWith("chunking.minSemanticNonWhitespaceChars:")),
    ).toBe(true)
  })

  test("preserves valid chunking fields when a sibling field is invalid", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "literal",
          model: "text-embedding-3-small",
        },
        chunking: {
          overlap: -1,
          expansion: true,
          minSemanticNonWhitespaceChars: 12,
        },
      },
      {},
    )

    expect(options.chunking).toEqual({
      overlap: 0,
      expansion: true,
      minSemanticNonWhitespaceChars: 12,
    })
    expect(options.diagnostics.some((diagnostic) => diagnostic.startsWith("chunking.overlap:"))).toBe(true)
  })

  test("returns missing embedding config instead of throwing", () => {
    const options = parseOptions({}, {})

    expect(options.embedding).toBeUndefined()
    expect(options.hyde.enabled).toBe(false)
    expect(options.diagnostics).toContain("embedding.model is required")
    expect(options.diagnostics).toContain("embedding.baseURL is required")
  })

  test("defaults hyde to opencode fallback when embedding is configured", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "literal",
          model: "text-embedding-3-small",
        },
      },
      {},
    )

    expect(options.hyde).toEqual({
      mode: "opencode",
      baseURL: undefined,
      apiKey: undefined,
      model: undefined,
      threshold: 0.35,
      enabled: true,
    })
  })

  test("keeps hyde disabled when explicitly disabled", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "literal",
          model: "text-embedding-3-small",
        },
        hyde: {
          enabled: false,
        },
      },
      {},
    )

    expect(options.hyde).toEqual({
      mode: "opencode",
      baseURL: undefined,
      apiKey: undefined,
      model: undefined,
      threshold: 0.35,
      enabled: false,
    })
  })

  test("uses opencode fallback for threshold-only hyde config", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "literal",
          model: "text-embedding-3-small",
        },
        hyde: {
          threshold: 0.6,
        },
      },
      {},
    )

    expect(options.hyde).toEqual({
      mode: "opencode",
      baseURL: undefined,
      apiKey: undefined,
      model: undefined,
      threshold: 0.6,
      enabled: true,
    })
  })

  test("uses opencode fallback for partial hyde config", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "literal",
          model: "text-embedding-3-small",
        },
        hyde: {
          model: "gpt-4o-mini",
          threshold: 0.2,
          enabled: true,
        },
      },
      {},
    )

    expect(options.hyde).toEqual({
      mode: "opencode",
      baseURL: undefined,
      apiKey: undefined,
      model: undefined,
      threshold: 0.2,
      enabled: true,
    })
  })

  test("uses openai-compatible hyde mode when base url and model are configured", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "embedding-key",
          model: "text-embedding-3-small",
        },
        hyde: {
          baseURL: "https://hyde.example.test/v1",
          apiKeyEnv: "HYDE_KEY",
          model: "gpt-4o-mini",
          threshold: 0.2,
        },
      },
      { HYDE_KEY: "hyde-secret" },
    )

    expect(options.hyde).toEqual({
      mode: "openai-compatible",
      baseURL: "https://hyde.example.test/v1",
      apiKey: "hyde-secret",
      model: "gpt-4o-mini",
      threshold: 0.2,
      enabled: true,
    })
  })

  test("uses OPENCODE_CAST_CACHE_DIR before xdg fallback", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "literal",
          model: "text-embedding-3-small",
        },
      },
      { OPENCODE_CAST_CACHE_DIR: "/tmp/cast-cache" },
    )

    expect(options.cacheDir).toBe("/tmp/cast-cache")
  })

  test("preserves valid embedding when optional fields are invalid", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "literal",
          model: "text-embedding-3-small",
        },
        topK: 0,
      },
      {},
    )

    expect(options.embedding).toEqual({
      baseURL: "https://example.test/v1",
      apiKey: "literal",
      model: "text-embedding-3-small",
      dimensions: undefined,
    })
    expect(options.topK).toBe(5)
    expect(options.diagnostics.some((diagnostic) => diagnostic.startsWith("topK:"))).toBe(true)
    expect(options.diagnostics).not.toContain("embedding.baseURL is required")
    expect(options.diagnostics).not.toContain("embedding.model is required")
  })

  test("parses configured file scanning limits", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "literal",
          model: "text-embedding-3-small",
        },
        maxFileBytes: 4096,
        excludeGlobs: ["**/*.generated.ts"],
      },
      {},
    )

    expect(options.maxFileBytes).toBe(4096)
    expect(options.excludeGlobs).toEqual(["**/*.generated.ts"])
  })

  test("reports invalid maxFileBytes and falls back to default", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "literal",
          model: "text-embedding-3-small",
        },
        maxFileBytes: 0,
      },
      {},
    )

    expect(options.maxFileBytes).toBe(2 * 1024 * 1024)
    expect(options.diagnostics.some((diagnostic) => diagnostic.startsWith("maxFileBytes:"))).toBe(true)
  })

  test("preserves valid embedding fields when dimensions is invalid", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "literal",
          model: "text-embedding-3-small",
          dimensions: 0,
        },
      },
      {},
    )

    expect(options.embedding).toEqual({
      baseURL: "https://example.test/v1",
      apiKey: "literal",
      model: "text-embedding-3-small",
      dimensions: undefined,
    })
    expect(options.diagnostics.some((diagnostic) => diagnostic.startsWith("embedding.dimensions:"))).toBe(true)
    expect(options.diagnostics).not.toContain("embedding.baseURL is required")
    expect(options.diagnostics).not.toContain("embedding.model is required")
  })

  test("reports invalid hyde fields while preserving opencode fallback", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "literal",
          model: "text-embedding-3-small",
        },
        hyde: {
          baseURL: "not a url",
          model: 42,
          threshold: 2,
        },
      },
      {},
    )

    expect(options.hyde).toEqual({
      mode: "opencode",
      baseURL: undefined,
      apiKey: undefined,
      model: undefined,
      threshold: 0.35,
      enabled: true,
    })
    expect(options.diagnostics.some((diagnostic) => diagnostic.startsWith("hyde.baseURL:"))).toBe(true)
    expect(options.diagnostics.some((diagnostic) => diagnostic.startsWith("hyde.model:"))).toBe(true)
    expect(options.diagnostics.some((diagnostic) => diagnostic.startsWith("hyde.threshold:"))).toBe(true)
    expect(options.diagnostics).not.toContain("embedding.baseURL is required")
    expect(options.diagnostics).not.toContain("embedding.model is required")
  })
})
