import { describe, expect, test } from "bun:test"
import { createOpenAIClient } from "./openai.js"

describe("createOpenAIClient", () => {
  test("embeds text with OpenAI-compatible request shape", async () => {
    const calls: unknown[] = []
    const client = createOpenAIClient({
      fetch: async (url, init) => {
        calls.push({ url, init })
        return Response.json({ data: [{ embedding: [0.1, 0.2, 0.3] }] })
      },
    })

    const embedding = await client.embed({
      baseURL: "https://example.test/v1/",
      apiKey: "key",
      model: "embed",
      input: "hello",
    })

    expect(embedding).toEqual([0.1, 0.2, 0.3])
    expect(calls).toHaveLength(1)
    expect((calls[0] as { url: string }).url).toBe("https://example.test/v1/embeddings")
    expect((calls[0] as { init: RequestInit }).init.method).toBe("POST")
    expect((calls[0] as { init: RequestInit }).init.headers).toEqual({
      "content-type": "application/json",
      authorization: "Bearer key",
    })
    expect(JSON.parse(String((calls[0] as { init: RequestInit }).init.body))).toEqual({
      model: "embed",
      input: "hello",
    })
  })

  test("passes optional embedding dimensions", async () => {
    const bodies: unknown[] = []
    const client = createOpenAIClient({
      fetch: async (_url, init) => {
        bodies.push(JSON.parse(String(init?.body)))
        return Response.json({ data: [{ embedding: [1] }] })
      },
    })

    await client.embed({
      baseURL: "https://example.test/v1",
      apiKey: "key",
      model: "embed",
      dimensions: 1,
      input: "hello",
    })

    expect(bodies).toEqual([{ model: "embed", input: "hello", dimensions: 1 }])
  })

  test("generates HyDE text with chat completions", async () => {
    const calls: unknown[] = []
    const client = createOpenAIClient({
      fetch: async (url, init) => {
        calls.push({ url, init })
        return Response.json({ choices: [{ message: { content: " look for class Session and method prompt " } }] })
      },
    })

    const text = await client.generateHyde({
      baseURL: "https://example.test/v1/",
      apiKey: "key",
      model: "chat",
      query: "where is prompt handled?",
    })

    expect(text).toBe("look for class Session and method prompt")
    expect((calls[0] as { url: string }).url).toBe("https://example.test/v1/chat/completions")
    expect(JSON.parse(String((calls[0] as { init: RequestInit }).init.body))).toEqual({
      model: "chat",
      messages: [
        {
          role: "system",
          content: "Produce a concise hypothetical code search target for the user's repository question.",
        },
        { role: "user", content: "where is prompt handled?" },
      ],
      temperature: 0,
    })
  })

  test("reranks documents with OpenRouter-compatible request shape", async () => {
    const calls: unknown[] = []
    const client = createOpenAIClient({
      fetch: async (url, init) => {
        calls.push({ url, init })
        return Response.json({
          results: [
            { index: 1, relevance_score: 0.91 },
            { index: 0, relevance_score: 0.12 },
          ],
        })
      },
    })

    const results = await client.rerank({
      baseURL: "https://openrouter.ai/api/v1/",
      apiKey: "key",
      model: "cohere/rerank-4-fast",
      query: "find parser errors",
      documents: ["a", "b"],
    })

    expect(results).toEqual([
      { index: 1, score: 0.91 },
      { index: 0, score: 0.12 },
    ])
    expect((calls[0] as { url: string }).url).toBe("https://openrouter.ai/api/v1/rerank")
    expect((calls[0] as { init: RequestInit }).init.method).toBe("POST")
    expect((calls[0] as { init: RequestInit }).init.headers).toEqual({
      "content-type": "application/json",
      authorization: "Bearer key",
    })
    expect(JSON.parse(String((calls[0] as { init: RequestInit }).init.body))).toEqual({
      model: "cohere/rerank-4-fast",
      query: "find parser errors",
      documents: ["a", "b"],
    })
  })

  test("throws clear errors for failed and malformed rerank responses", async () => {
    await expect(
      createOpenAIClient({ fetch: async () => new Response("nope", { status: 503 }) }).rerank({
        baseURL: "https://openrouter.ai/api/v1",
        model: "cohere/rerank-4-fast",
        query: "hello",
        documents: ["a"],
      }),
    ).rejects.toThrow("Rerank request failed: 503")

    await expect(
      createOpenAIClient({
        fetch: async () => Response.json({ results: [{ index: 2, relevance_score: 0.5 }] }),
      }).rerank({
        baseURL: "https://openrouter.ai/api/v1",
        model: "cohere/rerank-4-fast",
        query: "hello",
        documents: ["a"],
      }),
    ).rejects.toThrow("Rerank response included invalid result index")

    await expect(
      createOpenAIClient({
        fetch: async () => Response.json({ results: [{ index: 0, relevance_score: "bad" }] }),
      }).rerank({
        baseURL: "https://openrouter.ai/api/v1",
        model: "cohere/rerank-4-fast",
        query: "hello",
        documents: ["a"],
      }),
    ).rejects.toThrow("Rerank response included invalid relevance score")

    await expect(
      createOpenAIClient({ fetch: async () => new Response("not-json") }).rerank({
        baseURL: "https://openrouter.ai/api/v1",
        model: "cohere/rerank-4-fast",
        query: "hello",
        documents: ["a"],
      }),
    ).rejects.toThrow("Rerank response did not include results")
  })

  test("omits authorization header when api key is missing", async () => {
    const headers: unknown[] = []
    const client = createOpenAIClient({
      fetch: async (_url, init) => {
        headers.push(init?.headers)
        return Response.json({ data: [{ embedding: [1] }] })
      },
    })

    await client.embed({ baseURL: "https://example.test/v1", model: "embed", input: "hello" })

    expect(headers).toEqual([{ "content-type": "application/json" }])
  })

  test("throws clear errors for failed and malformed responses", async () => {
    await expect(
      createOpenAIClient({ fetch: async () => new Response("nope", { status: 500 }) }).embed({
        baseURL: "https://example.test/v1",
        model: "embed",
        input: "hello",
      }),
    ).rejects.toThrow("Embedding request failed: 500")

    await expect(
      createOpenAIClient({ fetch: async () => Response.json({ data: [] }) }).embed({
        baseURL: "https://example.test/v1",
        model: "embed",
        input: "hello",
      }),
    ).rejects.toThrow("Embedding response did not include data[0].embedding")

    await expect(
      createOpenAIClient({ fetch: async () => new Response("nope", { status: 429 }) }).generateHyde({
        baseURL: "https://example.test/v1",
        model: "chat",
        query: "hello",
      }),
    ).rejects.toThrow("HyDE request failed: 429")

    await expect(
      createOpenAIClient({
        fetch: async () => Response.json({ choices: [{ message: { content: "   " } }] }),
      }).generateHyde({
        baseURL: "https://example.test/v1",
        model: "chat",
        query: "hello",
      }),
    ).rejects.toThrow("HyDE response did not include choices[0].message.content")
  })

  test("rejects empty embedding arrays", async () => {
    await expect(
      createOpenAIClient({ fetch: async () => Response.json({ data: [{ embedding: [] }] }) }).embed({
        baseURL: "https://example.test/v1",
        model: "embed",
        input: "hello",
      }),
    ).rejects.toThrow("Embedding response did not include data[0].embedding")
  })

  test("maps invalid embedding JSON to malformed response error", async () => {
    await expect(
      createOpenAIClient({ fetch: async () => new Response("not-json") }).embed({
        baseURL: "https://example.test/v1",
        model: "embed",
        input: "hello",
      }),
    ).rejects.toThrow("Embedding response did not include data[0].embedding")
  })

  test("maps invalid HyDE JSON to malformed response error", async () => {
    await expect(
      createOpenAIClient({ fetch: async () => new Response("not-json") }).generateHyde({
        baseURL: "https://example.test/v1",
        model: "chat",
        query: "hello",
      }),
    ).rejects.toThrow("HyDE response did not include choices[0].message.content")
  })
})
