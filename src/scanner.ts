import { createHash } from "node:crypto"
import { readdir } from "node:fs/promises"
import path from "node:path"
import { minimatch } from "minimatch"
import { castChunks, type SyntaxNode } from "./cast.js"
import { fallbackChunks } from "./fallback.js"
import { assignSymbolsToChunks, attachTopology, extractSymbols } from "./topology.js"
import type { CastIndex } from "./types.js"

type Store = { read(): Promise<CastIndex>; write(index: CastIndex): Promise<void> }

export function createIndexer(input: {
  worktree: string
  options: {
    maxChunkNonWhitespaceChars: number
    includeGlobs: string[]
    excludeGlobs: string[]
    topK: number
    maxContextChars: number
  }
  store: Store
  parse(filePath: string, source: string): Promise<{ language: string; root?: SyntaxNode }>
  embed(text: string): Promise<number[]>
}) {
  return {
    async refresh() {
      const index = await input.store.read()
      index.metadata.status = "indexing"
      const canReuseExistingRecords =
        index.metadata.maxChunkNonWhitespaceChars === input.options.maxChunkNonWhitespaceChars
      const files = await scanFiles(input.worktree, input.options.includeGlobs, input.options.excludeGlobs)
      const nextFiles: CastIndex["files"] = {}
      const nextChunks: CastIndex["chunks"] = {}
      const nextSymbols: CastIndex["symbols"] = {}

      for (const relativePath of files) {
        const absolutePath = path.join(input.worktree, relativePath)
        const currentFingerprint = await fingerprint(absolutePath)
        const previousFile = index.files[relativePath]
        if (canReuseFile(index, previousFile, relativePath, currentFingerprint, canReuseExistingRecords)) {
          nextFiles[relativePath] = previousFile
          for (const chunkId of previousFile.chunkIds) {
            if (index.chunks[chunkId]) {
              nextChunks[chunkId] = index.chunks[chunkId]
            }
          }
          for (const symbol of Object.values(index.symbols).filter((symbol) => symbol.filePath === relativePath)) {
            nextSymbols[symbol.id] = symbol
          }
          continue
        }

        const text = await Bun.file(absolutePath).text()
        const parsed = await input.parse(absolutePath, text).catch((error) => ({
          language: "text",
          root: undefined,
          diagnostic: String(error),
        }))
        const rawChunks = parsed.root
          ? castChunks({
              filePath: relativePath,
              language: parsed.language,
              source: text,
              root: parsed.root,
              maxNonWhitespaceChars: input.options.maxChunkNonWhitespaceChars,
            })
          : fallbackChunks({
              filePath: relativePath,
              language: parsed.language,
              text,
              maxNonWhitespaceChars: input.options.maxChunkNonWhitespaceChars,
            })
        const symbols = parsed.root
          ? extractSymbols({ filePath: relativePath, source: text, nodes: parsed.root.children })
          : []
        const symbolsById = Object.fromEntries(symbols.map((symbol) => [symbol.id, symbol]))
        const chunks = attachTopology(assignSymbolsToChunks(rawChunks, symbolsById), symbolsById)
        const diagnostics = "diagnostic" in parsed ? [String(parsed.diagnostic)] : []

        for (const chunk of chunks) {
          const embedded = await input
            .embed(embeddingText(relativePath, parsed.language, chunk, symbolsById))
            .then((embedding) => ({ embedding }))
            .catch((error) => ({ embeddingError: error instanceof Error ? error.message : String(error) }))
          if ("embeddingError" in embedded) {
            diagnostics.push(`embedding failed: ${embedded.embeddingError}`)
          }
          nextChunks[chunk.id] = { ...chunk, ...embedded }
        }
        for (const symbol of symbols) {
          nextSymbols[symbol.id] = symbol
        }
        nextFiles[relativePath] = {
          path: relativePath,
          language: parsed.language,
          fingerprint: currentFingerprint,
          chunkIds: chunks.map((chunk) => chunk.id),
          diagnostics,
        }
      }

      index.files = nextFiles
      index.chunks = nextChunks
      index.symbols = nextSymbols
      index.metadata.worktree = input.worktree
      index.metadata.maxChunkNonWhitespaceChars = input.options.maxChunkNonWhitespaceChars
      index.metadata.status = "ready"
      index.metadata.updatedAt = Date.now()
      await input.store.write(index)
      return index
    },
  }
}

function canReuseFile(
  index: CastIndex,
  file: CastIndex["files"][string] | undefined,
  relativePath: string,
  fingerprint: string,
  canReuseExistingRecords: boolean,
) {
  if (!canReuseExistingRecords || file?.path !== relativePath || file.fingerprint !== fingerprint) {
    return false
  }
  const chunks = file.chunkIds.map((id) => ({ id, chunk: index.chunks[id] }))
  const chunkIds = new Set(file.chunkIds)
  if (chunks.some((entry) => !entry.chunk || entry.chunk.id !== entry.id)) {
    return false
  }
  if (
    chunks.some(
      (entry) =>
        entry.chunk.filePath !== relativePath ||
        entry.chunk.language !== file.language ||
        !entry.chunk.embedding ||
        entry.chunk.embeddingError ||
        entry.chunk.symbolIds.some((id) => index.symbols[id]?.id !== id || index.symbols[id]?.filePath !== file.path) ||
        hasDanglingChunkReference(index, entry.chunk, chunkIds),
    )
  ) {
    return false
  }
  return Object.values(index.symbols)
    .filter((symbol) => symbol.filePath === file.path)
    .every(
      (symbol) =>
        index.symbols[symbol.id]?.id === symbol.id &&
        (!symbol.parentSymbolId ||
          (index.symbols[symbol.parentSymbolId]?.id === symbol.parentSymbolId &&
            index.symbols[symbol.parentSymbolId]?.filePath === file.path)) &&
        symbol.childSymbolIds.every((id) => index.symbols[id]?.id === id && index.symbols[id]?.filePath === file.path),
    )
}

function hasDanglingChunkReference(index: CastIndex, chunk: CastIndex["chunks"][string], chunkIds: Set<string>) {
  return Boolean(
    (chunk.parentChunkId && !(chunkIds.has(chunk.parentChunkId) && index.chunks[chunk.parentChunkId])) ||
      (chunk.previousSiblingChunkId &&
        !(chunkIds.has(chunk.previousSiblingChunkId) && index.chunks[chunk.previousSiblingChunkId])) ||
      (chunk.nextSiblingChunkId &&
        !(chunkIds.has(chunk.nextSiblingChunkId) && index.chunks[chunk.nextSiblingChunkId])) ||
      chunk.childChunkIds.some((id) => !(chunkIds.has(id) && index.chunks[id])),
  )
}

function embeddingText(
  filePath: string,
  language: string,
  chunk: CastIndex["chunks"][string],
  symbols: CastIndex["symbols"],
) {
  return [
    `path: ${filePath}`,
    `language: ${language}`,
    `symbols:\n${chunk.symbolIds
      .map((id) => symbols[id])
      .filter((symbol) => symbol)
      .map((symbol) => `${symbol.kind} ${symbol.name}`)
      .join("\n")}`,
    `text:\n${chunk.text}`,
  ].join("\n")
}

async function scanFiles(root: string, includeGlobs: string[], excludeGlobs: string[]) {
  const files = await walk(root)
  return files.filter(
    (file) =>
      includeGlobs.some((pattern) => minimatch(file, pattern)) &&
      !excludeGlobs.some((pattern) => minimatch(file, pattern)),
  )
}

async function walk(root: string, prefix = ""): Promise<string[]> {
  const entries = await readdir(path.join(root, prefix), { withFileTypes: true })
  const ignored = new Set([".git", "node_modules", "dist", "build", ".cache"])
  const nested = await Promise.all(
    entries
      .filter((entry) => !(ignored.has(entry.name) || entry.isSymbolicLink()))
      .map((entry) => {
        const relative = path.join(prefix, entry.name)
        return entry.isDirectory() ? walk(root, relative) : Promise.resolve([relative])
      }),
  )
  return nested.flat()
}

async function fingerprint(filePath: string) {
  return createHash("sha256")
    .update(Buffer.from(await Bun.file(filePath).arrayBuffer()))
    .digest("hex")
}
