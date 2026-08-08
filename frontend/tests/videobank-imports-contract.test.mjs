import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

// 🎬 Every name a video-bank component imports must actually be exported.
//
// Written after a real miss: a helper call was added to PromoteVideoDialog while
// its import line was not. `vite build` passed — an undefined identifier is a
// RUNTIME ReferenceError in JavaScript, not a build error — and the whole
// promotion dialog would have thrown the moment a promotion succeeded. That is
// the same family as the TDZ break this project has already paid for: the tests
// were green, the build was green, and the screen was broken.
//
// Static on purpose: mounting every component needs a DOM and would only cover
// the paths a test happens to walk. This covers every import in the folder.
const DIR = fileURLToPath(new URL('../src/components/videobank/', import.meta.url))

const namedExportsOf = (file) => {
  const src = readFileSync(file, 'utf8')
  const names = new Set()
  for (const m of src.matchAll(/export\s+(?:async\s+)?function\s+(\w+)/g)) names.add(m[1])
  for (const m of src.matchAll(/export\s+(?:const|let|class)\s+(\w+)/g)) names.add(m[1])
  // `export { a, b }` re-exports.
  for (const m of src.matchAll(/export\s*\{([^}]+)\}/g)) {
    m[1].split(',').forEach((x) => {
      const n = x.trim().split(/\s+as\s+/).pop()
      if (n) names.add(n)
    })
  }
  return names
}

test('every named import from a sibling module is really exported by it', () => {
  const files = readdirSync(DIR).filter((f) => /\.(jsx?|js)$/.test(f)
    && !f.endsWith('.test.js'))
  const problems = []
  for (const file of files) {
    const src = readFileSync(path.join(DIR, file), 'utf8')
    for (const m of src.matchAll(/import\s*\{([^}]+)\}\s*from\s*'(\.\/[^']+)'/g)) {
      const targetName = m[2].replace(/^\.\//, '').replace(/\.js$/, '')
      const target = [`${targetName}.js`, `${targetName}.jsx`]
        .map((f) => path.join(DIR, f))
        .find((f) => files.includes(path.basename(f)))
      if (!target) continue          // not a sibling of this folder
      const exported = namedExportsOf(target)
      for (const raw of m[1].split(',')) {
        const name = raw.trim().split(/\s+as\s+/)[0].trim()
        if (name && !exported.has(name)) {
          problems.push(`${file} imports { ${name} } from ${m[2]}, which does not export it`)
        }
      }
    }
  }
  assert.deepEqual(problems, [])
})

test('every sibling helper a component CALLS is imported or defined locally', () => {
  // The other half of the same miss: the call was there, the import was not.
  const files = readdirSync(DIR).filter((f) => f.endsWith('.jsx'))
  const helpers = new Map()
  for (const f of readdirSync(DIR).filter((x) => x.endsWith('.js') && !x.endsWith('.test.js'))) {
    for (const name of namedExportsOf(path.join(DIR, f))) helpers.set(name, f)
  }
  const problems = []
  for (const file of files) {
    const src = readFileSync(path.join(DIR, file), 'utf8')
    const known = new Set()
    for (const m of src.matchAll(/import\s*\{([^}]+)\}\s*from/g)) {
      m[1].split(',').forEach((x) => known.add(x.trim().split(/\s+as\s+/).pop()))
    }
    for (const m of src.matchAll(/(?:const|let|function)\s+(\w+)/g)) known.add(m[1])
    for (const m of src.matchAll(/\b([a-zA-Z_]\w*)\s*\(/g)) {
      const name = m[1]
      // Only names this folder's helper modules actually export — anything else
      // is a method, a hook or a built-in and is not this test's business.
      if (helpers.has(name) && !known.has(name)) {
        problems.push(`${file} calls ${name}() (exported by ${helpers.get(name)}) without importing it`)
      }
    }
  }
  assert.deepEqual(problems, [])
})
