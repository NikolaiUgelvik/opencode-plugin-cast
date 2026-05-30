export type FetchLike = (url: string, init: RequestInit) => Promise<Response>

const TRAILING_SLASHES_PATTERN = /\/+$/

export function createOpenAIClient(options: { fetch?: FetchLike } = {}) {
  const request = options.fetch ?? fetch

  return {
    embed: (input: { baseURL: string; apiKey?: string; model: string; dimensions?: number; input: string }) =>
      embed(request, input),

    generateHyde: (input: { baseURL: string; apiKey?: string; model: string; query: string }) =>
      generateHyde(request, input),

    rerank: (input: { baseURL: string; apiKey?: string; model: string; query: string; documents: string[] }) =>
      rerank(request, input),
  }
}

async function embed(
  request: FetchLike,
  input: { baseURL: string; apiKey?: string; model: string; dimensions?: number; input: string },
) {
  const response = await request(`${input.baseURL.replace(TRAILING_SLASHES_PATTERN, "")}/embeddings`, {
    method: "POST",
    headers: buildHeaders(input.apiKey),
    body: JSON.stringify({
      model: input.model,
      input: input.input,
      ...(input.dimensions === undefined ? {} : { dimensions: input.dimensions }),
    }),
  })

  if (!response.ok) {
    throw new Error(`Embedding request failed: ${response.status}`)
  }

  const body = await response.json().catch(() => undefined)
  const embedding =
    typeof body === "object" && body !== null && "data" in body && Array.isArray(body.data)
      ? body.data[0]?.embedding
      : undefined
  if (!Array.isArray(embedding) || embedding.length === 0 || !embedding.every((value) => typeof value === "number")) {
    throw new Error("Embedding response did not include data[0].embedding")
  }
  return embedding
}

async function generateHyde(
  request: FetchLike,
  input: { baseURL: string; apiKey?: string; model: string; query: string },
) {
  const response = await request(`${input.baseURL.replace(TRAILING_SLASHES_PATTERN, "")}/chat/completions`, {
    method: "POST",
    headers: buildHeaders(input.apiKey),
    body: JSON.stringify({
      model: input.model,
      messages: [
        {
          role: "system",
          content: "Produce a concise hypothetical code search target for the user's repository question.",
        },
        { role: "user", content: input.query },
      ],
      temperature: 0,
    }),
  })

  if (!response.ok) {
    throw new Error(`HyDE request failed: ${response.status}`)
  }

  const body = await response.json().catch(() => undefined)
  const content =
    typeof body === "object" && body !== null && "choices" in body && Array.isArray(body.choices)
      ? body.choices[0]?.message?.content
      : undefined
  if (typeof content !== "string" || content.trim() === "") {
    throw new Error("HyDE response did not include choices[0].message.content")
  }
  return content.trim()
}

async function rerank(
  request: FetchLike,
  input: { baseURL: string; apiKey?: string; model: string; query: string; documents: string[] },
) {
  const response = await request(`${input.baseURL.replace(TRAILING_SLASHES_PATTERN, "")}/rerank`, {
    method: "POST",
    headers: buildHeaders(input.apiKey),
    body: JSON.stringify({
      model: input.model,
      query: input.query,
      documents: input.documents,
    }),
  })

  if (!response.ok) {
    throw new Error(`Rerank request failed: ${response.status}`)
  }

  const body = await response.json().catch(() => undefined)
  const results =
    typeof body === "object" && body !== null && "results" in body && Array.isArray(body.results)
      ? body.results
      : undefined
  if (!results) {
    throw new Error("Rerank response did not include results")
  }

  return results.map((result: unknown) => {
    const index = typeof result === "object" && result !== null && "index" in result ? result.index : undefined
    const score =
      typeof result === "object" && result !== null && "relevance_score" in result ? result.relevance_score : undefined
    if (typeof index !== "number" || !Number.isInteger(index) || index < 0 || index >= input.documents.length) {
      throw new Error("Rerank response included invalid result index")
    }
    if (typeof score !== "number" || Number.isNaN(score)) {
      throw new Error("Rerank response included invalid relevance score")
    }
    return { index, score }
  })
}

function buildHeaders(apiKey: string | undefined): Record<string, string> {
  return apiKey
    ? {
        "content-type": "application/json",
        authorization: `Bearer ${apiKey}`,
      }
    : { "content-type": "application/json" }
}
