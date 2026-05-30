import { createHash } from "node:crypto"
import { readdir, readFile } from "node:fs/promises"
import path from "node:path"
import ignore, { type Ignore } from "ignore"
import { minimatch } from "minimatch"
import { castChunks, type SyntaxNode } from "./cast.js"
import { fallbackChunks } from "./fallback.js"
import { buildLexicalIndex } from "./lexical.js"
import { assignSymbolsToChunks, attachTopology, extractSymbols } from "./topology.js"
import type { CastIndex, ChunkingOptions, ChunkRecord, FileRecord, SymbolRecord } from "./types.js"

type FileResult = {
  file: FileRecord
  chunks: Record<string, ChunkRecord>
  symbols: Record<string, SymbolRecord>
}
type Store = {
  read(): Promise<CastIndex>
  write(index: CastIndex): Promise<void>
  beginIndexRun?(input: { configHash: string; metadata: CastIndex["metadata"] }): Promise<{ runId: string }>
  getCompletedFile?(runId: string, filePath: string, fingerprint: string): Promise<FileResult | undefined>
  writeFileResult?(runId: string, fileResult: FileResult): Promise<void>
  activateRun?(runId: string, index: CastIndex): Promise<void>
}
type GitignoreMatcher = { base: string; matcher: Ignore }
const BINARY_SAMPLE_BYTES = Number("16") * Number("1024")
const BYTE_NUL = 0
const BYTE_BACKSPACE = 8
const BYTE_TAB = 9
const BYTE_LINE_FEED = 10
const BYTE_FORM_FEED = 12
const BYTE_CARRIAGE_RETURN = 13
const CONTROL_BYTE_LIMIT = 32
const BINARY_CONTROL_RATIO = 0.3

export function createIndexer(input: {
  worktree: string
  options: {
    maxChunkNonWhitespaceChars: number
    maxFileBytes: number
    includeGlobs: string[]
    excludeGlobs: string[]
    topK: number
    maxContextChars: number
    chunking: ChunkingOptions
  }
  store: Store
  parse(filePath: string, source: string): Promise<{ language: string; root?: SyntaxNode }>
  embed(text: string): Promise<number[]>
}) {
  return {
    async refresh() {
      const store = input.store
      const index = await store.read()
      const initialStatus = index.metadata.status
      const canReuseExistingRecords =
        index.metadata.maxChunkNonWhitespaceChars === input.options.maxChunkNonWhitespaceChars &&
        sameChunkingOptions(index.metadata.chunking, input.options.chunking)
      const runConfigHash = indexRunConfigHash(index, input.options)
      const runStore = hasRunStore(store) && initialStatus !== "ready" ? store : undefined
      const files = await scanFiles(input.worktree, input.options.includeGlobs, input.options.excludeGlobs)
      const nextFiles: CastIndex["files"] = {}
      const nextChunks: CastIndex["chunks"] = {}
      const nextSymbols: CastIndex["symbols"] = {}
      const metadataDiagnostics: string[] = []
      let changed = false
      let run: { runId: string } | undefined

      const markIndexing = () => {
        index.metadata.status = "indexing"
        index.metadata.worktree = input.worktree
        index.metadata.maxChunkNonWhitespaceChars = input.options.maxChunkNonWhitespaceChars
        index.metadata.chunking = input.options.chunking
      }
      const ensureRun = async () => {
        if (!runStore) {
          return
        }
        if (!run) {
          markIndexing()
          run = await runStore.beginIndexRun({ configHash: runConfigHash, metadata: index.metadata })
        }
        return run
      }

      for (const relativePath of files) {
        const absolutePath = path.join(input.worktree, relativePath)
        const file = Bun.file(absolutePath)
        const skipDiagnostic = await skipFileDiagnostic(relativePath, file, input.options.maxFileBytes)
        if (skipDiagnostic) {
          metadataDiagnostics.push(skipDiagnostic)
          continue
        }
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
        changed = true
        const activeRun = await ensureRun()
        if (activeRun && runStore) {
          const completed = await runStore.getCompletedFile(activeRun.runId, relativePath, currentFingerprint)
          if (completed) {
            const completedIndex = {
              ...index,
              files: { [relativePath]: completed.file },
              chunks: completed.chunks,
              symbols: completed.symbols,
            }
            if (canReuseFile(completedIndex, completed.file, relativePath, currentFingerprint, true)) {
              nextFiles[relativePath] = completed.file
              Object.assign(nextChunks, completed.chunks)
              Object.assign(nextSymbols, completed.symbols)
              continue
            }
          }
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
              chunking: input.options.chunking,
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
        const fileDiagnostics = "diagnostic" in parsed ? [String(parsed.diagnostic)] : []

        const fileChunks: CastIndex["chunks"] = {}
        for (const chunk of chunks) {
          const embedded = await input
            .embed(embeddingText(relativePath, parsed.language, chunk, symbolsById, input.options.chunking.expansion))
            .then((embedding) => ({ embedding }))
            .catch((error) => ({ embeddingError: error instanceof Error ? error.message : String(error) }))
          if ("embeddingError" in embedded) {
            fileDiagnostics.push(`embedding failed: ${embedded.embeddingError}`)
          }
          fileChunks[chunk.id] = { ...chunk, ...embedded }
        }
        Object.assign(nextChunks, fileChunks)
        for (const symbol of symbols) {
          nextSymbols[symbol.id] = symbol
        }
        const fileRecord = {
          path: relativePath,
          language: parsed.language,
          fingerprint: currentFingerprint,
          chunkIds: chunks.map((chunk) => chunk.id),
          diagnostics: fileDiagnostics,
        }
        nextFiles[relativePath] = fileRecord
        if (run && runStore) {
          await runStore.writeFileResult(run.runId, {
            file: fileRecord,
            chunks: fileChunks,
            symbols: Object.fromEntries(symbols.map((symbol) => [symbol.id, symbol])),
          })
        }
      }

      const lexicalIndex = buildLexicalIndex(nextChunks, nextSymbols)
      const hasFileSetChange = !sameStringArray(Object.keys(index.files).sort(), Object.keys(nextFiles).sort())
      const hasDiagnosticsChange = !sameStringArray(index.metadata.diagnostics, metadataDiagnostics)
      if (
        index.metadata.status === "ready" &&
        !changed &&
        !hasFileSetChange &&
        !hasDiagnosticsChange &&
        index.metadata.worktree === input.worktree &&
        canReuseExistingRecords
      ) {
        return index
      }

      index.files = nextFiles
      index.chunks = lexicalIndex.chunks
      index.symbols = nextSymbols
      index.lexical = lexicalIndex.lexical
      index.metadata.worktree = input.worktree
      index.metadata.maxChunkNonWhitespaceChars = input.options.maxChunkNonWhitespaceChars
      index.metadata.chunking = input.options.chunking
      index.metadata.diagnostics = metadataDiagnostics
      index.metadata.status = "ready"
      index.metadata.updatedAt = Date.now()
      if (run && runStore) {
        await runStore.activateRun(run.runId, index)
      } else if (runStore) {
        const activeRun = await ensureRun()
        if (activeRun) {
          await runStore.activateRun(activeRun.runId, index)
        }
      } else {
        await store.write(index)
      }
      return index
    },
  }
}

function hasRunStore(
  store: Store,
): store is Store & Required<Pick<Store, "beginIndexRun" | "getCompletedFile" | "writeFileResult" | "activateRun">> {
  return Boolean(store.beginIndexRun && store.getCompletedFile && store.writeFileResult && store.activateRun)
}

function indexRunConfigHash(index: CastIndex, options: Parameters<typeof createIndexer>[0]["options"]) {
  return stableHash({
    schemaVersion: index.metadata.schemaVersion,
    embeddingModel: index.metadata.embeddingModel,
    embeddingDimensions: index.metadata.embeddingDimensions,
    includeGlobs: options.includeGlobs,
    excludeGlobs: options.excludeGlobs,
    maxFileBytes: options.maxFileBytes,
    maxChunkNonWhitespaceChars: options.maxChunkNonWhitespaceChars,
    chunking: options.chunking,
  })
}

function stableHash(value: unknown) {
  return createHash("sha256").update(stableStringify(value)).digest("hex")
}

function stableStringify(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(",")}]`
  }
  if (value && typeof value === "object") {
    return `{${Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => `${JSON.stringify(key)}:${stableStringify(item)}`)
      .join(",")}}`
  }
  return JSON.stringify(value)
}

async function skipFileDiagnostic(
  relativePath: string,
  file: { size: number; slice(start?: number, end?: number): Blob },
  maxFileBytes: number,
) {
  if (file.size > maxFileBytes) {
    return `${relativePath}: skipped file over maxFileBytes (${file.size} > ${maxFileBytes})`
  }
  const sample = new Uint8Array(await file.slice(0, Math.min(file.size, BINARY_SAMPLE_BYTES)).arrayBuffer())
  if (isProbablyBinary(sample)) {
    return `${relativePath}: skipped binary file`
  }
}

function isProbablyBinary(bytes: Uint8Array) {
  if (bytes.length === 0) {
    return false
  }
  let suspicious = 0
  for (const byte of bytes) {
    if (byte === BYTE_NUL) {
      return true
    }
    if (
      byte < CONTROL_BYTE_LIMIT &&
      byte !== BYTE_BACKSPACE &&
      byte !== BYTE_TAB &&
      byte !== BYTE_LINE_FEED &&
      byte !== BYTE_FORM_FEED &&
      byte !== BYTE_CARRIAGE_RETURN
    ) {
      suspicious++
    }
  }
  return suspicious / bytes.length > BINARY_CONTROL_RATIO
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
        entry.chunk.text.length === 0 ||
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

function sameChunkingOptions(left: ChunkingOptions | undefined, right: ChunkingOptions) {
  return (
    left?.overlap === right.overlap &&
    left.expansion === right.expansion &&
    left.minSemanticNonWhitespaceChars === right.minSemanticNonWhitespaceChars
  )
}

function sameStringArray(left: string[], right: string[]) {
  return left.length === right.length && left.every((value, index) => value === right[index])
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
  expansion: boolean,
) {
  const fields = [`path: ${filePath}`, `language: ${language}`]
  if (expansion) {
    const lineEnd = chunk.text.endsWith("\n")
      ? Math.max(chunk.range.lineStart, chunk.range.lineEnd - 1)
      : chunk.range.lineEnd
    fields.push(`chunk:\nkind: ${chunk.kind}\nrange: ${chunk.range.lineStart}-${lineEnd}`)
  }
  fields.push(
    `symbols:\n${chunk.symbolIds
      .map((id) => symbols[id])
      .filter((symbol) => symbol)
      .map((symbol) => `${symbol.kind} ${symbol.name}`)
      .join("\n")}`,
  )
  fields.push(`text:\n${chunk.text}`)
  return fields.join("\n")
}

async function scanFiles(root: string, includeGlobs: string[], excludeGlobs: string[]) {
  const files = await walk(root)
  return files.filter(
    (file) =>
      includeGlobs.some((pattern) => minimatch(file, pattern)) &&
      !excludeGlobs.some((pattern) => minimatch(file, pattern)),
  )
}

async function loadGitignore(root: string, prefix: string): Promise<GitignoreMatcher | undefined> {
  const matcher = ignore()
  try {
    matcher.add(await readFile(path.join(root, prefix, ".gitignore"), "utf8"))
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
      throw error
    }
    return
  }
  return { base: prefix, matcher }
}

async function walk(root: string, prefix = "", inheritedGitignores: GitignoreMatcher[] = []): Promise<string[]> {
  const entries = await readdir(path.join(root, prefix), { withFileTypes: true })
  const localGitignore = await loadGitignore(root, prefix)
  const gitignores = localGitignore ? [...inheritedGitignores, localGitignore] : inheritedGitignores
  const ignored = new Set([".git", "node_modules", "dist", "build", ".cache"])
  const nested = await Promise.all(
    entries
      .filter((entry) => {
        const relative = path.join(prefix, entry.name)
        return !(ignored.has(entry.name) || entry.isSymbolicLink() || isGitignored(relative, gitignores))
      })
      .map((entry) => {
        const relative = path.join(prefix, entry.name)
        return entry.isDirectory() ? walk(root, relative, gitignores) : Promise.resolve([relative])
      }),
  )
  return nested.flat()
}

function isGitignored(relativePath: string, gitignores: GitignoreMatcher[]) {
  return gitignores.some(({ base, matcher }) => {
    const relativeToBase = base ? path.relative(base, relativePath) : relativePath
    return relativeToBase && !relativeToBase.startsWith("..") && !path.isAbsolute(relativeToBase)
      ? matcher.ignores(toGitignorePath(relativeToBase))
      : false
  })
}

function toGitignorePath(relativePath: string) {
  return relativePath.split(path.sep).join("/")
}

async function fingerprint(filePath: string) {
  return createHash("sha256")
    .update(Buffer.from(await Bun.file(filePath).arrayBuffer()))
    .digest("hex")
}
