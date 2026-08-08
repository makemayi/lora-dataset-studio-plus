/**
 * What the Setup install menu actually SAYS about an unreadable file, rendered.
 *
 * ── The bug, and why a green suite never saw it ──────────────────────────────
 * Klein's consistency LoRA is RECOMMENDED, not required: the backend's readiness
 * verdict never looks at it, so a corrupted one stops nothing. The install menu
 * asked one unfiltered question — "is any Klein asset blocking-invalid?" — and
 * painted the answer red: `⚠ On disk, unreadable`, the exact badge of a dead UNET,
 * on a screen the user opens to find out whether they are ready. Nothing was
 * blocked. A unit test on `installCatalog` would have been green either way,
 * because the defect was in what the badge MEANS, and only the rendered text
 * carries that.
 *
 * The mirror defect lived on the other screen: the ComfyUI step's download buttons
 * filtered on the REQUIRED list, so the very same unreadable file rendered
 * `✓ Installed` right above a button offering to download it.
 *
 * So these assertions render the real component and read the TEXT, in named states.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { default: InstallEverything } = await import(
  '../src/components/setup/InstallEverything.jsx')
const { ToastProvider } = await import('../src/components/common/Toast.jsx')
const { MemoryRouter } = await import('react-router')   // the help badges use useNavigate

const BROKEN = (asset, filename) => ({
  asset, filename, verdict: 'truncated_or_garbage', blocking: true,
  reason: `${filename} is shorter than its header declares`,
})

const caps = (klein_invalid) => ({
  engines: { klein: true, krea: false },
  face_scoring: true, masks: true, watermark_inpaint: true,
  ollama: { reachable: true, vision_model: 'qwen', vision_model_ready: true },
  comfyui: {
    reachable: true, dir_valid: true, klein_missing: [], krea_missing: [],
    krea_nodes_installed: true, klein_invalid,
  },
})

const html = (klein_invalid) => renderToStaticMarkup(
  createElement(MemoryRouter, null,
    createElement(ToastProvider, null,
      createElement(InstallEverything,
        { plan: [], caps: caps(klein_invalid), onDone: () => {} }))))

// Strip tags so an assertion reads the SENTENCE, not the markup around it.
const text = (h) => h.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ')

test('a broken OPTIONAL file is named, and says the engine still works', () => {
  const out = text(html([BROKEN('klein_lora', 'Flux2-Klein-9B-consistency-V2.safetensors')]))
  // Visible: not swept under a "✓ Installed" the file cannot honour.
  assert.match(out, /On disk, unreadable/)
  assert.match(out, /Flux2-Klein-9B-consistency-V2\.safetensors/)
  // …and honest about the stakes: nothing here is blocked. Matched on the badge
  // itself, not the loose word "optional" — the page says that about Krea too, so a
  // bare /optional/ would have been green before the fix.
  assert.match(out, /On disk, unreadable — optional/)
  assert.match(out, /still works without it/i)
  // The action is there, so "informative" does not mean "dead end".
  assert.match(out, /Download again/)
})

test('the optional row is amber, the required one red — a colour that means something', () => {
  // The badge colour is the whole point of the defect: it is what made a screen
  // read as blocked. Assert the class actually rendered, per row.
  const optional = html([BROKEN('klein_lora', 'lora.safetensors')])
  assert.match(optional, /text-amber-400[^"]*">\s*⚠ On disk, unreadable — optional/)
  assert.doesNotMatch(optional, /text-rose-300[^"]*">\s*⚠ On disk, unreadable</)

  const required = html([BROKEN('klein_model', 'unet.safetensors')])
  assert.match(required, /text-rose-300[^"]*">\s*⚠ On disk, unreadable</)
})

test('a required weight is still treated as the emergency it is', () => {
  const out = text(html([BROKEN('klein_text_encoder', 'te.safetensors')]))
  assert.match(out, /On disk, unreadable/)
  assert.match(out, /te\.safetensors/)
  // No softening on the file the engine cannot run without.
  assert.doesNotMatch(out, /still works without it/i)
})

test('an install with nothing broken says nothing about broken files', () => {
  const out = text(html([]))
  assert.doesNotMatch(out, /unreadable/i)
  assert.match(out, /Installed/)   // the ordinary green state still renders
})

/* The OTHER half — the ComfyUI step's one-click download buttons.
 *
 * Read as SOURCE, not rendered, and that limitation is stated rather than hidden:
 * SetupPage needs a capabilities context whose provider fetches, and the context
 * object itself is not exported, so a mount can only ever reach the empty-caps
 * state — the one state where this defect does not exist. Exporting the context
 * purely to be watched is a change to shipping code for a test's convenience, so
 * the honest trade is a narrower check with its blind spot named: this pins WHICH
 * list the buttons read and that the optional case has its own sentence; it cannot
 * prove what the browser paints. The rendered proof of the same rule lives above,
 * on the install menu, which shares `kleinAssetBlocks`. */
test('the ComfyUI step reads the UNFILTERED broken list, not the required-only one', async () => {
  const { readFileSync } = await import('node:fs')
  const src = readFileSync(new URL('../src/pages/SetupPage.jsx', import.meta.url), 'utf8')
  // The bug was the filtered list: klein_lora is not required, so it was absent —
  // and its button printed "✓ Installed" over a file that cannot load.
  assert.match(src, /step\.kleinBrokenAll/)
  // Severity comes from the asset, so the optional case is amber, not rose.
  assert.match(src, /kleinAssetBlocks\(action\)/)
  assert.match(src, /Klein still generates without it/)
})
