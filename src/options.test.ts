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
      baseURL: "https://example.test/v1",
      apiKey: "secret",
      model: "gpt-4o-mini",
      threshold: 0.42,
      enabled: true,
    })
    expect(options.maxChunkNonWhitespaceChars).toBe(2000)
    expect(options.maxContextChars).toBe(12_000)
    expect(options.topK).toBe(5)
  })

  test("returns missing embedding config instead of throwing", () => {
    const options = parseOptions({}, {})

    expect(options.embedding).toBeUndefined()
    expect(options.hyde.enabled).toBe(false)
    expect(options.diagnostics).toContain("embedding.model is required")
    expect(options.diagnostics).toContain("embedding.baseURL is required")
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

  test("preserves valid hyde fields when threshold is invalid", () => {
    const options = parseOptions(
      {
        embedding: {
          baseURL: "https://example.test/v1",
          apiKey: "literal",
          model: "text-embedding-3-small",
        },
        hyde: {
          model: "gpt-4o-mini",
          threshold: 2,
        },
      },
      {},
    )

    expect(options.hyde.model).toBe("gpt-4o-mini")
    expect(options.hyde.threshold).toBe(0.35)
    expect(options.hyde.enabled).toBe(true)
    expect(options.diagnostics.some((diagnostic) => diagnostic.startsWith("hyde.threshold:"))).toBe(true)
    expect(options.diagnostics).not.toContain("embedding.baseURL is required")
    expect(options.diagnostics).not.toContain("embedding.model is required")
  })
})
