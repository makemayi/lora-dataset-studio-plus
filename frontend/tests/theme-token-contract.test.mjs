import assert from 'node:assert/strict'
import { readdir, readFile } from 'node:fs/promises'
import { dirname, extname, relative, resolve, sep } from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const TEST_DIR = dirname(fileURLToPath(import.meta.url))
const FRONTEND_DIR = resolve(TEST_DIR, '..')
const SRC_DIR = resolve(TEST_DIR, '../src')
const INDEX_HTML = resolve(FRONTEND_DIR, 'index.html')
const SOURCE_EXTENSIONS = new Set(['.js', '.jsx', '.ts', '.tsx'])

// These semantic tokens already bake their opacity into tailwind.config.js.
// Adding Tailwind's /NN modifier replaces that safe alpha with NN%, turning a
// dark surface or hairline border into an opaque white layer.
const INVALID_ALPHA_MODIFIER = /(?<![A-Za-z0-9_-])((?:bg-(?:surface|surface-raised)|border-(?:border|border-strong))\/(?:\[[^\]\s"'`]+\]|\d+))(?![A-Za-z0-9_/-])/g

async function sourceFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  const nested = await Promise.all(entries.map((entry) => {
    const path = resolve(dir, entry.name)
    if (entry.isDirectory()) return sourceFiles(path)
    return SOURCE_EXTENSIONS.has(extname(entry.name)) ? [path] : []
  }))
  return nested.flat()
}

async function guardedContentFiles() {
  return [INDEX_HTML, ...await sourceFiles(SRC_DIR)]
}

function invalidTokens(text) {
  return [...text.matchAll(INVALID_ALPHA_MODIFIER)].map((match) => match[1])
}

test('theme-token guard covers index.html and only Tailwind source inputs', async () => {
  const files = await guardedContentFiles()

  assert.ok(files.includes(INDEX_HTML), 'frontend/index.html is not guarded')
  assert.ok(
    files.every((file) => file === INDEX_HTML || file.startsWith(`${SRC_DIR}${sep}`)),
    'guard must not scan dist, node_modules, or unrelated frontend files',
  )
})

test('invalid token matcher accepts common punctuation as a token boundary', () => {
  const cases = [
    ['bg-surface/60.', 'bg-surface/60'],
    ['bg-surface-raised/50,', 'bg-surface-raised/50'],
    ['(border-border/40)', 'border-border/40'],
    ['border-border-strong/20;', 'border-border-strong/20'],
  ]

  for (const [text, expected] of cases) {
    assert.deepEqual(invalidTokens(text), [expected], text)
  }
})

test('invalid token matcher does not truncate longer class-like names', () => {
  assert.deepEqual(
    invalidTokens('x-bg-surface/60 bg-surface-overlay/60 border-border-extra/50'),
    [],
  )
})

test('alpha-embedded theme tokens never use Tailwind opacity modifiers', async () => {
  const violations = []

  for (const file of await guardedContentFiles()) {
    const lines = (await readFile(file, 'utf8')).split(/\r?\n/)
    lines.forEach((line, index) => {
      // Tailwind's content scanner also extracts class-looking tokens from
      // comments, so documentation must respect the same invariant.
      for (const token of invalidTokens(line)) {
        violations.push(`${relative(FRONTEND_DIR, file)}:${index + 1} ${token}`)
      }
    })
  }

  assert.equal(
    violations.length,
    0,
    `Opacity modifiers override the baked alpha of semantic theme tokens:\n${violations.join('\n')}`,
  )
})

/* The app flipped from dark to light on 2026-08-13/14, and the colour sweep that
   came with it missed 74 lines: `text-emerald-100` on `bg-emerald-50`, and
   friends. In the dark theme those were the BRIGHT end of a tinted alert box; on
   a light one they are the invisible end — emerald-100 on emerald-50 is about
   1.05:1, i.e. blank. Nothing failed, nothing logged; the text was simply gone.

   The pairing is what makes it decidable: a light text weight is correct ON A
   DARK SCRIM (a control over a photograph) and wrong on a light panel, so this
   only fires when both halves sit in one class string. */
test('no light text weight is painted onto a light background', async () => {
  const LIGHT_TEXT =
    /text-(slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-(50|100|200)\b/
  const LIGHT_BG =
    /bg-(slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-(50|100)\b|bg-white\b/
  // The scrim cases: a control painted ON an image is deliberately light-on-dark.
  const DARK_SCRIM = /bg-black|bg-white\/[1-6]?0\b/

  const violations = []
  for (const file of await guardedContentFiles()) {
    const lines = (await readFile(file, 'utf8')).split(/\r?\n/)
    lines.forEach((line, index) => {
      if (LIGHT_TEXT.test(line) && LIGHT_BG.test(line) && !DARK_SCRIM.test(line)) {
        violations.push(`${relative(FRONTEND_DIR, file)}:${index + 1}`)
      }
    })
  }

  assert.equal(
    violations.length,
    0,
    `Light text on a light background is unreadable, not subtle:\n${violations.join('\n')}`,
  )
})

/* The pairing test above only sees a class string that carries BOTH halves. Most
   of the damage was not shaped that way: the tinted background sat on the
   <section> and the invisible text three lines below it on a <p>, so 45 of the
   119 broken lines survived the first sweep.

   Whole-FILE decidability is what closes it. A component that never paints on a
   photograph has no dark ground anywhere in it, so a light text weight in that
   file cannot be correct — wherever its background is declared. Files that DO
   paint on images (lightboxes, the image tile, the canvas) are exempt, and are
   identified the same way a reader would: they say bg-black / bg-white/NN /
   text-white repeatedly, because that is the scrim treatment. */
test('a component that never paints on an image carries no light text weight', async () => {
  const LIGHT_TEXT =
    /text-(slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-(50|100|200)\b/
  const SCRIM = /bg-black|bg-white\/[1-6]?0\b|text-white/g

  const violations = []
  for (const file of await guardedContentFiles()) {
    if (!file.endsWith('.jsx')) continue
    const source = await readFile(file, 'utf8')
    // Three or more scrim markers: this component's job includes painting over
    // arbitrary image content, where a light weight is the correct choice.
    if ((source.match(SCRIM) || []).length >= 3) continue
    source.split(/\r?\n/).forEach((line, index) => {
      if (LIGHT_TEXT.test(line) && !/bg-black|bg-white\/[1-6]?0\b|text-white/.test(line)) {
        violations.push(`${relative(FRONTEND_DIR, file)}:${index + 1}`)
      }
    })
  }

  assert.equal(
    violations.length,
    0,
    `Light text weights outside an on-image surface are invisible:\n${violations.join('\n')}`,
  )
})

/* `${FLOAT_SHADOW}` inside a SINGLE-quoted string is not an interpolation — it
   is those literal characters, handed to Tailwind as a class name that does not
   exist. The element silently loses its shadow.

   Second time this shape has shipped: the 2026-08-13 sweep left 19
   `bg-surface shadow-[…]-raised` dead classes, and the 2026-08-14 token
   consolidation reintroduced it in 18 places — a ternary BRANCH inside a
   template literal is still a plain string:

     className={`base ${on ? `${SELECTED_PILL}` : 'quiet ${FLOAT_SHADOW}'}`}
                                                  ^ dead, and green in every
                                                    source-text test

   Scope, chosen after two wider attempts failed: SCREAMING_SNAKE only (that is
   what the shared tokens are named), single quotes only, and no backtick inside
   the span. A quote-state walker got 2077 false positives because an apostrophe
   in a comment desynchronises it, and allowing double quotes matched ordinary
   prose inside real templates ("… ${MAX_LABEL_LEN} chars …"). */
const DEAD_TOKEN_INTERPOLATION = /'[^'`\n]*\$\{[A-Z][A-Z0-9_]*\}[^'`\n]*'/g

test('token interpolations are inside template literals, not plain strings', async () => {
  const violations = []
  for (const file of await guardedContentFiles()) {
    const lines = (await readFile(file, 'utf8')).split(/\r?\n/)
    lines.forEach((line, index) => {
      for (const hit of line.match(DEAD_TOKEN_INTERPOLATION) || []) {
        violations.push(`${relative(FRONTEND_DIR, file)}:${index + 1} ${hit.slice(0, 60)}`)
      }
    })
  }
  assert.equal(
    violations.length,
    0,
    `A \${TOKEN} outside a backtick is a literal class name, not a value:\n${violations.join('\n')}`,
  )
})

test('the dead-interpolation matcher separates the bug from ordinary code', () => {
  const hit = (s) => DEAD_TOKEN_INTERPOLATION.test(s)
  DEAD_TOKEN_INTERPOLATION.lastIndex = 0
  assert.ok(hit("  : 'bg-white/60 ${FLOAT_SHADOW} hover:bg-white/80',"))
  DEAD_TOKEN_INTERPOLATION.lastIndex = 0
  assert.ok(!hit('className={`x ${FLOAT_SHADOW} y`}'))
  DEAD_TOKEN_INTERPOLATION.lastIndex = 0
  // Prose inside a REAL template, which the double-quoted variant mis-flagged.
  assert.ok(!hit('instructions: `a "b ${MAX_LABEL_LEN} c", d`'))
  DEAD_TOKEN_INTERPOLATION.lastIndex = 0
})
