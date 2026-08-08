/* Build a standalone HTML page showing the model-file picker in its three
   meaningful states, using the app's REAL stylesheet, so a headless browser can
   measure it at 400 px. Not a test — the proof harness the report links to.

   Usage: node tests/support/renderPickerProof.mjs <absolute output .html> */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { renderToStaticMarkup, createElement } from './mountJsx.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const frontend = path.resolve(here, '..', '..')
const out = process.argv[2]
if (!out) throw new Error('usage: renderPickerProof.mjs <output.html>')

const ModelFilePicker = (await import('../../src/components/settings/ModelFilePicker.jsx')).default

const FILES = [
  'klein/flux-2-klein-9b-fp8.safetensors',
  'Krea/krea2_turbo_fp8_scaled.safetensors',
  'Krea/krea2_raw_fp8_scaled.safetensors',
]

const scan = { files: FILES, folder: 'ComfyUI’s models/unet', loading: false, error: false,
  rescan: () => {}, rescanning: false }

const card = (title, node) => `
  <section class="rounded-lg border border-border bg-surface p-3 mb-4">
    <h2 class="text-sm font-medium text-content mb-2">${title}</h2>
    ${node}
  </section>`

const pick = (props) => renderToStaticMarkup(createElement(ModelFilePicker, {
  id: 'krea-base-model', ariaLabel: 'Krea base model file', onChange: () => {},
  placeholder: 'auto — finds a Krea 2 Turbo/Raw build', ...scan, ...props,
}))

const cssName = fs.readdirSync(path.join(frontend, 'dist', 'assets')).find((f) => f.endsWith('.css'))
const css = fs.readFileSync(path.join(frontend, 'dist', 'assets', cssName), 'utf8')

fs.writeFileSync(out, `<!doctype html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>${css}</style></head>
<body class="font-[Inter] bg-app p-3">
${card('Empty — auto-detect', pick({ value: '' }))}
${card('A file that IS on disk', pick({ value: FILES[1] }))}
${card('PINNED but NOT on disk — kept, flagged, and the engine refuses',
  pick({ value: 'Krea/a_build_i_deleted_last_week_fp8_scaled.safetensors' }))}
</body></html>`)
console.log(out)
