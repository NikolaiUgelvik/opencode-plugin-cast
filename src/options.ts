import path from "node:path"
import process from "node:process"
import { z } from "zod"

const ApiConfig = z.object({
  baseURL: z.string().url().optional(),
  apiKey: z.string().optional(),
  apiKeyEnv: z.string().optional(),
  model: z.string().optional(),
  dimensions: z.number().int().positive().optional(),
})

const OptionsSchema = z.object({
  embedding: ApiConfig.optional(),
  hyde: ApiConfig.extend({
    enabled: z.boolean().optional(),
    threshold: z.number().min(-1).max(1).optional(),
  }).optional(),
  maxChunkNonWhitespaceChars: z.number().int().positive().optional(),
  maxContextChars: z.number().int().positive().optional(),
  topK: z.number().int().positive().optional(),
  cacheDir: z.string().optional(),
  includeGlobs: z.array(z.string()).optional(),
  excludeGlobs: z.array(z.string()).optional(),
})

const OptionFields = OptionsSchema.shape
const ApiFields = ApiConfig.shape
const HydeFields = OptionFields.hyde.unwrap().shape
const DEFAULT_HYDE_THRESHOLD = 0.35
const DEFAULT_MAX_CHUNK_NON_WHITESPACE_CHARS = 2000
const DEFAULT_MAX_CONTEXT_CHARS = 12_000
const DEFAULT_TOP_K = 5

export type CastPluginOptions = ReturnType<typeof parseOptions>

export function parseOptions(input: unknown, env: Record<string, string | undefined> = process.env) {
  const inputRecord = z.record(z.string(), z.unknown()).safeParse(input ?? {})
  const diagnostics = inputRecord.success
    ? []
    : inputRecord.error.issues.map((issue) => `${issue.path.join(".")}: ${issue.message}`)
  const parsed = {
    embedding: OptionFields.embedding.safeParse(inputRecord.success ? inputRecord.data.embedding : undefined),
    hyde: OptionFields.hyde.safeParse(inputRecord.success ? inputRecord.data.hyde : undefined),
    maxChunkNonWhitespaceChars: OptionFields.maxChunkNonWhitespaceChars.safeParse(
      inputRecord.success ? inputRecord.data.maxChunkNonWhitespaceChars : undefined,
    ),
    maxContextChars: OptionFields.maxContextChars.safeParse(
      inputRecord.success ? inputRecord.data.maxContextChars : undefined,
    ),
    topK: OptionFields.topK.safeParse(inputRecord.success ? inputRecord.data.topK : undefined),
    cacheDir: OptionFields.cacheDir.safeParse(inputRecord.success ? inputRecord.data.cacheDir : undefined),
    includeGlobs: OptionFields.includeGlobs.safeParse(inputRecord.success ? inputRecord.data.includeGlobs : undefined),
    excludeGlobs: OptionFields.excludeGlobs.safeParse(inputRecord.success ? inputRecord.data.excludeGlobs : undefined),
  }
  const raw = {
    embedding: parsed.embedding.success
      ? parsed.embedding.data
      : parseApiConfig(inputRecord.success ? inputRecord.data.embedding : undefined),
    hyde: parsed.hyde.success
      ? parsed.hyde.data
      : parseHydeConfig(inputRecord.success ? inputRecord.data.hyde : undefined),
    maxChunkNonWhitespaceChars: parsed.maxChunkNonWhitespaceChars.success
      ? parsed.maxChunkNonWhitespaceChars.data
      : undefined,
    maxContextChars: parsed.maxContextChars.success ? parsed.maxContextChars.data : undefined,
    topK: parsed.topK.success ? parsed.topK.data : undefined,
    cacheDir: parsed.cacheDir.success ? parsed.cacheDir.data : undefined,
    includeGlobs: parsed.includeGlobs.success ? parsed.includeGlobs.data : undefined,
    excludeGlobs: parsed.excludeGlobs.success ? parsed.excludeGlobs.data : undefined,
  }
  for (const [key, result] of Object.entries(parsed)) {
    if (result.success) {
      continue
    }
    diagnostics.push(
      ...result.error.issues.map((issue) => `${[key, ...issue.path].filter(Boolean).join(".")}: ${issue.message}`),
    )
  }
  const embeddingApiKey = raw.embedding?.apiKey ?? (raw.embedding?.apiKeyEnv ? env[raw.embedding.apiKeyEnv] : undefined)
  const hydeApiKey = raw.hyde?.apiKey ?? (raw.hyde?.apiKeyEnv ? env[raw.hyde.apiKeyEnv] : undefined) ?? embeddingApiKey
  const embedding =
    raw.embedding?.baseURL && raw.embedding.model
      ? {
          baseURL: raw.embedding.baseURL,
          apiKey: embeddingApiKey,
          model: raw.embedding.model,
          dimensions: raw.embedding.dimensions,
        }
      : undefined

  if (!raw.embedding?.baseURL) {
    diagnostics.push("embedding.baseURL is required")
  }
  if (!raw.embedding?.model) {
    diagnostics.push("embedding.model is required")
  }

  return {
    embedding,
    hyde: {
      baseURL: raw.hyde?.baseURL ?? raw.embedding?.baseURL,
      apiKey: hydeApiKey,
      model: raw.hyde?.model,
      threshold: raw.hyde?.threshold ?? DEFAULT_HYDE_THRESHOLD,
      enabled: raw.hyde?.enabled ?? Boolean(raw.hyde?.model),
    },
    maxChunkNonWhitespaceChars: raw.maxChunkNonWhitespaceChars ?? DEFAULT_MAX_CHUNK_NON_WHITESPACE_CHARS,
    maxContextChars: raw.maxContextChars ?? DEFAULT_MAX_CONTEXT_CHARS,
    topK: raw.topK ?? DEFAULT_TOP_K,
    cacheDir:
      raw.cacheDir ??
      env.OPENCODE_CAST_CACHE_DIR ??
      path.join(env.XDG_CACHE_HOME ?? path.join(env.HOME ?? process.cwd(), ".cache"), "opencode", "cast"),
    includeGlobs: raw.includeGlobs ?? ["**/*"],
    excludeGlobs: raw.excludeGlobs ?? [],
    diagnostics,
  }
}

function parseApiConfig(input: unknown) {
  const inputRecord = z.record(z.string(), z.unknown()).safeParse(input ?? {})
  if (!inputRecord.success) {
    return
  }

  const parsed = {
    baseURL: ApiFields.baseURL.safeParse(inputRecord.data.baseURL),
    apiKey: ApiFields.apiKey.safeParse(inputRecord.data.apiKey),
    apiKeyEnv: ApiFields.apiKeyEnv.safeParse(inputRecord.data.apiKeyEnv),
    model: ApiFields.model.safeParse(inputRecord.data.model),
    dimensions: ApiFields.dimensions.safeParse(inputRecord.data.dimensions),
  }

  return {
    baseURL: parsed.baseURL.success ? parsed.baseURL.data : undefined,
    apiKey: parsed.apiKey.success ? parsed.apiKey.data : undefined,
    apiKeyEnv: parsed.apiKeyEnv.success ? parsed.apiKeyEnv.data : undefined,
    model: parsed.model.success ? parsed.model.data : undefined,
    dimensions: parsed.dimensions.success ? parsed.dimensions.data : undefined,
  }
}

function parseHydeConfig(input: unknown) {
  const inputRecord = z.record(z.string(), z.unknown()).safeParse(input ?? {})
  if (!inputRecord.success) {
    return
  }

  const api = parseApiConfig(input)
  const parsed = {
    enabled: HydeFields.enabled.safeParse(inputRecord.data.enabled),
    threshold: HydeFields.threshold.safeParse(inputRecord.data.threshold),
  }

  return {
    baseURL: api?.baseURL,
    apiKey: api?.apiKey,
    apiKeyEnv: api?.apiKeyEnv,
    model: api?.model,
    dimensions: api?.dimensions,
    enabled: parsed.enabled.success ? parsed.enabled.data : undefined,
    threshold: parsed.threshold.success ? parsed.threshold.data : undefined,
  }
}
