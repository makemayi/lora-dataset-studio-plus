/* Scratch harness: render the four ETA banner states into one static page so a
   headless browser can photograph them at phone width. Not a test — invoked by
   hand when a screenshot is wanted. */
import fs from 'node:fs'
import path from 'node:path'
import { renderToStaticMarkup, createElement } from './mountJsx.mjs'

const { ProgressBar } = await import('../../src/components/bank/BankWorkspace.jsx')

const cases = [
  ['🧠 Semantic index — the banner this was asked for', {
    kind: 'semantic_index', done: 12939, total: 37800, finished: false,
    eta_state: 'ready', eta_seconds: 5220, eta_scope: 'job',
    detail: 'loading siglip2-base-p16-224 on CUDA (local files only)',
  }],
  ['…while the estimate is still settling', {
    kind: 'semantic_index', done: 12939, total: 37800, finished: false,
    eta_state: 'estimating',
    detail: 'loading siglip2-base-p16-224 on CUDA (local files only)',
  }],
  ['✨ Score, write-back phase — scoped to the step', {
    kind: 'score', done: 4200, total: 21220, finished: false,
    eta_state: 'ready', eta_seconds: 1200, eta_scope: 'phase',
    detail: 'writing 21220 score(s) to the database…',
  }],
  ['✨ Score, style grouping — nothing countable, no estimate', {
    kind: 'score', done: 0, total: 0, finished: false, eta_state: 'none',
    detail: 'grouping styles over 23000 image(s) — the slow tail of this pass',
  }],
  ['🔎 Quality scan — longest possible clause', {
    kind: 'scan', done: 812, total: 50397, finished: false,
    eta_state: 'ready', eta_seconds: 3.5 * 3600, eta_scope: 'phase',
    detail: 'quality scan',
  }],
]

const cssFile = fs.readdirSync(path.resolve('dist/assets')).find((f) => f.endsWith('.css'))
const css = fs.readFileSync(path.resolve('dist/assets', cssFile), 'utf8')

const body = cases.map(([title, activity]) => `
  <p style="font:600 12px system-ui;color:#94a3b8;margin:18px 0 6px">${title}</p>
  ${renderToStaticMarkup(createElement(ProgressBar, { activity, onCancel: () => {} }))}`).join('')

fs.writeFileSync(process.argv[2], `<!doctype html><html class="dark"><head><meta charset="utf-8">
<style>${css}</style></head>
<body class="bg-surface p-3" style="width:400px">${body}</body></html>`)
console.log('wrote', process.argv[2])
