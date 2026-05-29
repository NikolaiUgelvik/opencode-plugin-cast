import { Language, type Node, Parser } from "web-tree-sitter"
import type { SyntaxNode } from "./cast.js"

type ParserEntry = {
  id: string
  extensions: string[]
  filenames?: string[]
  wasm: () => Promise<string>
}

const entries: ParserEntry[] = [
  {
    id: "bash",
    extensions: [".sh", ".bash"],
    filenames: [".bashrc", ".bash_profile"],
    wasm: async () =>
      (await import("tree-sitter-bash/tree-sitter-bash.wasm" as string, { with: { type: "wasm" } })).default,
  },
  {
    id: "go",
    extensions: [".go"],
    wasm: async () =>
      (await import("tree-sitter-go/tree-sitter-go.wasm" as string, { with: { type: "wasm" } })).default,
  },
  {
    id: "java",
    extensions: [".java"],
    wasm: async () =>
      (await import("tree-sitter-java/tree-sitter-java.wasm" as string, { with: { type: "wasm" } })).default,
  },
  {
    id: "javascript",
    extensions: [".js", ".jsx", ".mjs", ".cjs"],
    wasm: async () =>
      (await import("tree-sitter-javascript/tree-sitter-javascript.wasm" as string, { with: { type: "wasm" } }))
        .default,
  },
  {
    id: "python",
    extensions: [".py"],
    wasm: async () =>
      (await import("tree-sitter-python/tree-sitter-python.wasm" as string, { with: { type: "wasm" } })).default,
  },
  {
    id: "ruby",
    extensions: [".rb"],
    wasm: async () =>
      (await import("tree-sitter-ruby/tree-sitter-ruby.wasm" as string, { with: { type: "wasm" } })).default,
  },
  {
    id: "rust",
    extensions: [".rs"],
    wasm: async () =>
      (await import("tree-sitter-rust/tree-sitter-rust.wasm" as string, { with: { type: "wasm" } })).default,
  },
  {
    id: "typescript",
    extensions: [".ts"],
    wasm: async () =>
      (await import("tree-sitter-typescript/tree-sitter-typescript.wasm" as string, { with: { type: "wasm" } }))
        .default,
  },
  {
    id: "tsx",
    extensions: [".tsx"],
    wasm: async () =>
      (await import("tree-sitter-typescript/tree-sitter-tsx.wasm" as string, { with: { type: "wasm" } })).default,
  },
]

const PATH_SEPARATOR_PATTERN = /[\\/]/
let init: Promise<void> | undefined
const parsers = new Map<string, Promise<Parser>>()

export function languageForPath(filePath: string) {
  const filename = filePath.split(PATH_SEPARATOR_PATTERN).at(-1) ?? filePath
  const lowerFilePath = filePath.toLowerCase()
  return entries.find(
    (entry) =>
      entry.filenames?.includes(filename) || entry.extensions.some((extension) => lowerFilePath.endsWith(extension)),
  )
}

export async function parseSource(filePath: string, source: string) {
  const entry = languageForPath(filePath)
  if (!entry) {
    return { language: "text", root: undefined }
  }
  const parser = await parserFor(entry)
  const tree = parser.parse(source)
  if (!tree) {
    return { language: entry.id, root: undefined }
  }
  return { language: entry.id, root: adaptNode(tree.rootNode) }
}

function parserFor(entry: ParserEntry) {
  const cached = parsers.get(entry.id)
  if (cached) {
    return cached
  }
  const loading = createParser(entry).catch((error) => {
    parsers.delete(entry.id)
    throw error
  })
  parsers.set(entry.id, loading)
  return loading
}

function initializeParser() {
  if (!init) {
    init = import("web-tree-sitter/tree-sitter.wasm" as string, { with: { type: "wasm" } })
      .then((module) => Parser.init({ locateFile: () => resolveWasm(module.default) }))
      .catch((error) => {
        init = undefined
        throw error
      })
  }
  return init
}

async function createParser(entry: ParserEntry) {
  await initializeParser()
  const parser = new Parser()
  parser.setLanguage(await Language.load(resolveWasm(await entry.wasm())))
  return parser
}

function adaptNode(node: Node): SyntaxNode {
  return {
    type: node.type,
    startIndex: node.startIndex,
    endIndex: node.endIndex,
    children: node.children.filter((child): child is Node => child !== null).map(adaptNode),
  }
}

function resolveWasm(input: string) {
  return input.startsWith("file://") ? new URL(input).pathname : input
}
