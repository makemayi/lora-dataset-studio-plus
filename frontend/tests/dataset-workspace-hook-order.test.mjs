/**
 * Every hook in DatasetWorkspace runs BEFORE its `if (!d) return` early return.
 *
 * Why this file exists, twice over. The dataset page renders once with no `d`
 * (the fetch has not landed) and again with one. A hook placed after the early
 * return is skipped on the first render and appears on the second, and React
 * answers that with "Rendered more hooks than during the previous render" —
 * which the error boundary turns into a bare "an unexpected error occurred"
 * over the whole page. The withdrawn subject-trim wave shipped exactly this
 * bug (452f395b), and the rebuild shipped it AGAIN, in the same component, one
 * day later. Twice is a pattern, so it gets a test.
 *
 * A source-text test cannot see it — the code parses either way. This one
 * counts hook calls in source order against the early return's position, which
 * is cheap and needs no DOM.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const here = path.dirname(fileURLToPath(import.meta.url))
const SRC = path.join(here, '..', 'src', 'components', 'dataset', 'DatasetWorkspace.jsx')
const src = readFileSync(SRC, 'utf8')

// The component's own early return. Anchored on the exact line so a second,
// unrelated `if (!x) return` elsewhere cannot be mistaken for it.
const EARLY_RETURN = /^ {2}if \(!d\) return /m

test('the dataset page still has the early return this test is about', () => {
  assert.match(src, EARLY_RETURN,
    'if the early return moved or changed shape, re-point this test rather than deleting it')
})

test('no hook is called after the early return', () => {
  const cut = src.search(EARLY_RETURN)
  const tail = src.slice(cut)

  // Only top-level calls in DatasetWorkspace matter, and every one of them sits
  // at exactly two spaces of indentation — a hook inside a nested component
  // defined further down the file is that component's own business.
  const offenders = []
  const HOOK = /^ {2}(?:const|let)?\s*\[?[^\n]*\b(useState|useEffect|useCallback|useMemo|useRef|useLayoutEffect)\(/gm
  let m
  while ((m = HOOK.exec(tail)) !== null) {
    const line = tail.slice(0, m.index).split('\n').length
    offenders.push(`${m[1]} at +${line} lines past the early return`)
  }

  assert.deepEqual(offenders, [],
    'These hooks are skipped on the render where `d` is missing and appear on '
    + 'the next one. React throws "rendered more hooks than during the previous '
    + 'render" and the error boundary blanks the page. Move them above the '
    + `early return:\n  ${offenders.join('\n  ')}`)
})

test('the subject-trim hooks in particular are above it', () => {
  // Named on purpose: this is the pair that broke the page twice.
  const cut = src.search(EARLY_RETURN)
  for (const name of ['const [trimState', 'const [trimBusy', 'const startTrim']) {
    const at = src.indexOf(name)
    assert.notEqual(at, -1, `${name} is gone — has the feature been removed?`)
    assert.ok(at < cut, `${name} must be declared before the \`if (!d) return\``)
  }
})
