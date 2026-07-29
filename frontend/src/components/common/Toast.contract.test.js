/* The toast container must stay the TOP layer of the app.
 *
 * Why this test exists: the container shipped at z-[100] while every modal and
 * lightbox lived at 9990-9999, so a toast raised over an open dialog rendered
 * BEHIND it. Nothing failed, nothing logged — the app simply told the user
 * something they could not see, and callers learned to close the dialog first,
 * discarding whatever had been typed into it. It was found by accident, in a
 * screenshot, months after it shipped.
 *
 * A z-index is a global ordering expressed one component at a time, so no
 * component can be reviewed on its own. This test is the missing global view:
 * it reads every z-[N] in the source and fails when anything reaches the toast
 * layer. Raising a modal past the toasts is then a deliberate act that has to
 * come with an argument, not a silent regression.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const SRC = join(fileURLToPath(new URL('.', import.meta.url)), '..', '..')
const TOAST = join(SRC, 'components', 'common', 'Toast.jsx')

/* Arbitrary-value Tailwind z-index only: z-[1234]. The named scale (z-50 and
   friends) tops out at 50 and cannot reach the toast layer, so it is not a
   risk worth flagging. */
const Z_RE = /\bz-\[(\d+)\]/g

/* Comments must be stripped BEFORE matching, and this is not a nicety: the
   first version of this test read Toast.jsx's own explanatory comment — which
   names the layer in prose — and happily "passed" while the real className sat
   at the old, broken value. A guard that reads documentation instead of code
   guards nothing. Every file here is checked the same way, since the comment
   explaining a z-index is exactly where its number gets written down. */
function code(text) {
  return text.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/(^|[^:])\/\/[^\n]*/g, '$1')
}

function sourceFiles(dir, out = []) {
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name === 'dist') continue
    const full = join(dir, name)
    if (statSync(full).isDirectory()) sourceFiles(full, out)
    else if (/\.(jsx|js)$/.test(name) && !/\.test\.js$/.test(name)) out.push(full)
  }
  return out
}

function toastZ() {
  const found = [...code(readFileSync(TOAST, 'utf8')).matchAll(Z_RE)].map((m) => Number(m[1]))
  assert.ok(found.length, 'Toast.jsx no longer declares an explicit z-[N] layer')
  return Math.max(...found)
}

test('the toast container declares the highest z-index in the app', () => {
  const ceiling = toastZ()
  const offenders = []

  for (const file of sourceFiles(SRC)) {
    if (file === TOAST) continue
    for (const m of code(readFileSync(file, 'utf8')).matchAll(Z_RE)) {
      const z = Number(m[1])
      if (z >= ceiling) offenders.push(`${relative(SRC, file)} → z-[${z}]`)
    }
  }

  assert.deepEqual(offenders, [],
    `these overlays reach or exceed the toast layer (z-[${ceiling}]), so a toast raised
over them would be invisible. Either lower the overlay, or raise Toast.jsx on purpose:
${offenders.join('\n')}`)
})

test('the toast layer sits clear of every other overlay', () => {
  /* Not just "highest" but "highest by a margin": overlays cluster in the 9990s,
     and a new one landing exactly one below the toasts would be a coincidence,
     not a decision. */
  const ceiling = toastZ()
  let highestOther = 0

  for (const file of sourceFiles(SRC)) {
    if (file === TOAST) continue
    for (const m of code(readFileSync(file, 'utf8')).matchAll(Z_RE)) {
      highestOther = Math.max(highestOther, Number(m[1]))
    }
  }

  assert.ok(ceiling > highestOther,
    `toast layer z-[${ceiling}] does not clear the next overlay z-[${highestOther}]`)
})
