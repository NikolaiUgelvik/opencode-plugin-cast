import { describe, expect, test } from "bun:test"
import { languageForPath, parseSource } from "./language.js"

describe("language registry", () => {
  test("matches curated parser entries by extension and filename", () => {
    expect(languageForPath("src/app.ts")?.id).toBe("typescript")
    expect(languageForPath("src/app.tsx")?.id).toBe("tsx")
    expect(languageForPath("bin/start.sh")?.id).toBe("bash")
    expect(languageForPath("public/index.html")?.id).toBe("html")
    expect(languageForPath("templates/page.htm")?.id).toBe("html")
    expect(languageForPath("public/index.php")?.id).toBe("php")
    expect(languageForPath("views/layout.phtml")?.id).toBe("php")
    expect(languageForPath("/home/me/.bashrc")?.id).toBe("bash")
    expect(languageForPath("README.md")).toBeUndefined()
  })

  test("matches extensions case-insensitively", () => {
    expect(languageForPath("src/App.TS")?.id).toBe("typescript")
    expect(languageForPath("src/App.TSX")?.id).toBe("tsx")
    expect(languageForPath("SCRIPT.SH")?.id).toBe("bash")
  })

  test("shares parser initialization across concurrent multi-language first parses", async () => {
    const parses = await Promise.all([
      parseSource("src/app.ts", "export const value = 1\n"),
      parseSource("src/app.js", "export const value = 1\n"),
      parseSource("src/app.py", "value = 1\n"),
      parseSource("src/app.rb", "value = 1\n"),
    ])

    expect(parses.map((parsed) => parsed.language)).toEqual(["typescript", "javascript", "python", "ruby"])
    expect(parses.every((parsed) => parsed.root?.startIndex === 0)).toBe(true)
  })

  test("shares parser initialization across concurrent first parses", async () => {
    const parses = await Promise.all(
      Array.from({ length: 12 }, (_, index) =>
        parseSource(`src/app-${index}.ts`, `export const value${index} = ${index}\n`),
      ),
    )

    expect(parses.every((parsed) => parsed.language === "typescript")).toBe(true)
    expect(parses.every((parsed) => parsed.root?.type === "program")).toBe(true)
  })

  test("parses TypeScript source into the shared syntax node shape", async () => {
    const parsed = await parseSource("src/app.ts", "export const value = 1\n")

    expect(parsed.language).toBe("typescript")
    expect(parsed.root?.type).toBe("program")
    expect(parsed.root?.startIndex).toBe(0)
    expect(parsed.root?.endIndex).toBe(23)
    expect(parsed.root?.children.length).toBeGreaterThan(0)
  })

  test("parses PHP source into the shared syntax node shape", async () => {
    const parsed = await parseSource("public/index.php", "<?php function value() { return 1; }\n")

    expect(parsed.language).toBe("php")
    expect(parsed.root?.type).toBe("program")
    expect(parsed.root?.startIndex).toBe(0)
    expect(parsed.root?.children.length).toBeGreaterThan(0)
  })

  test("parses HTML source into the shared syntax node shape", async () => {
    const parsed = await parseSource("public/index.html", "<!doctype html><html><body><h1>Title</h1></body></html>\n")

    expect(parsed.language).toBe("html")
    expect(parsed.root?.type).toBe("document")
    expect(parsed.root?.startIndex).toBe(0)
    expect(parsed.root?.children.length).toBeGreaterThan(0)
  })

  test("treats unsupported files as text without parsing", async () => {
    await expect(parseSource("README.md", "# title\n")).resolves.toEqual({ language: "text", root: undefined })
  })
})
