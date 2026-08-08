/**
 * What the Checkpoints panel actually SAYS about a Hugging Face repository —
 * rendered, against the state that produced it.
 *
 * ── The bug, and why a green suite never saw it ──────────────────────────────
 * `artifact_status` is stamped once, at delivery, and never revisited. The card
 * printed it verbatim and then, in the SAME sentence, appended a hard-coded
 * "— the model is there, not on this computer" whenever no file was on disk.
 * The status came from the payload; the sentence came from `rows.length`. So a
 * repository the backend had verified EMPTY rendered:
 *
 *     · missing — the model is there, not on this computer.
 *
 * A test asserting that `status: 'missing'` reaches the payload would have been
 * green. A test asserting the sentence appears for a run with no local files
 * would have been green. Both were true; the screen still contradicted itself,
 * because nothing ever held the two against each other.
 *
 * So every assertion below renders the real component and reads the TEXT, in a
 * named state. `renderToStaticMarkup` never runs effects, which is exactly why
 * the panel takes `hubPresenceOverride`: the states worth pinning (a repository
 * measured gone, one measured present, one we failed to reach) are the ones a
 * mount can otherwise never reach.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { default: DenseModelsPanel } = await import(
  '../src/components/dataset/DenseModelsPanel.jsx')
// The panel carries a HelpBadge and a "Test in Studio" Link, so it only renders
// inside a router. Mounting the WHOLE panel is the point: a sentence proven in
// isolation and never reached on screen is the failure mode this file is about.
const { MemoryRouter } = await import('react-router')

/* A run that delivered to Hugging Face and left nothing here — the shape every
   pre-local-delivery dense run is in, and the one that carried the sentence. */
const hubOnly = (hub = {}) => ({
  run_id: 90, dataset_id: 3, train_type: 'krea', variant: 'Raw', steps: 3000,
  active: false, master: null, fp8: null, can_quantize: false, can_delete: false,
  hub: {
    repo_id: 'acme/dense-90', url: 'https://huggingface.co/acme/dense-90',
    status: 'available', weight_filename: 'Krea_full_y.safetensors', ...hub,
  },
})

const render = (models, presence = null) => renderToStaticMarkup(
  createElement(MemoryRouter, null, createElement(DenseModelsPanel, {
    datasetId: 3, models, hubPresenceOverride: presence,
  })))

/* The rendered markup as readable prose: entity-decoded and tag-stripped, so an
   assertion reads the sentence a user reads and not an attribute soup. */
const text = (html) => html
  .replace(/<[^>]+>/g, ' ')
  .replace(/&#x27;|&#39;/g, "'").replace(/&amp;/g, '&')
  .replace(/&quot;/g, '"').replace(/&mdash;/g, '—')
  .replace(/\s+/g, ' ').trim()

test('THE regression: a verified-missing repository never says the model is there', () => {
  const body = text(render([hubOnly({ status: 'missing' })]))
  assert.doesNotMatch(body, /the model is there/,
    'the sentence came back — it must be chosen from the status, not from rows.length')
  assert.match(body, /The last check found no model in this repository/)
  assert.match(body, /Nothing from this run is on this computer/)
})

test('nothing has asked the Hub yet, so the card speaks about the delivery, dated', () => {
  const body = text(render([hubOnly({ checked_at: '2026-07-11T09:12:33' })]))
  assert.match(body, /Delivered and verified on 2026-07-11 — not re-checked since/)
  assert.match(body, /Delivered to Hugging Face/)          // the chip, past tense too
  assert.doesNotMatch(body, /the model is there|still holds/)
})

test('a repository measured gone says so, and says what is left to do', () => {
  const body = text(render([hubOnly()], { 90: { state: 'gone' } }))
  assert.match(body, /Hugging Face no longer returns this repository/)
  assert.match(body, /no recoverable model left/)
  assert.match(body, /No copy left/)                       // the chip agrees
  // And the button that promised "quantizing fetches it first" is refused HERE,
  // with its reason, rather than failing after the click.
  assert.match(body, /the repository it would be downloaded from is gone/)
  assert.match(render([hubOnly()], { 90: { state: 'gone' } }),
    /Quantize to fp8<\/button>/)
  assert.match(render([hubOnly()], { 90: { state: 'gone' } }), /disabled=""/)
})

test('a repository measured present is the only case allowed the present tense', () => {
  const body = text(render([hubOnly()], { 90: { state: 'present' } }))
  assert.match(body, /checked just now/)
  assert.match(body, /The repository still holds this model/)
  assert.match(body, /On Hugging Face/)
  assert.doesNotMatch(body, /not re-checked since/)
})

test('a check that FAILED reads as our failure, never as a deletion', () => {
  const body = text(render([hubOnly()], { 90: {
    state: 'unknown',
    detail: 'Hugging Face could not be reached, so the repository was not checked.',
  } }))
  assert.match(body, /could not check/)
  assert.match(body, /Hugging Face could not be reached/)
  assert.doesNotMatch(body, /deleted|no longer|gone/)
  // The quantize button survives an unreachable Hub: the repository is very
  // probably fine and the user is merely offline.
  assert.doesNotMatch(render([hubOnly()], { 90: { state: 'unknown' } }), /disabled=""/)
})

test('a run whose master is on disk is not told it lost anything', () => {
  const withMaster = {
    ...hubOnly(),
    master: { filename: 'Krea_full_y.safetensors', path: '/store/Krea_full_y.safetensors',
      size_bytes: 26e9, step: null, is_final: true, total_candidates: 1, others: [] },
    can_delete: true,
  }
  const body = text(render([withMaster], { 90: { state: 'gone' } }))
  assert.match(body, /full-precision master is still on this computer, so nothing is lost/)
  assert.match(body, /On this computer/)                   // the chip answers from disk
  assert.doesNotMatch(body, /no recoverable model left/)
})

test('the line wraps instead of pushing the card sideways at 400 px', () => {
  // A repository id plus a two-line sentence inside a fixed-width column is a
  // horizontal scrollbar on a phone — and this line is the one thing the card
  // exists to make readable.
  const html = render([hubOnly()], { 90: { state: 'gone' } })
  const start = html.indexOf('Backup:')
  const paragraph = html.slice(html.lastIndexOf('<p', start), html.indexOf('</p>', start))
  assert.match(paragraph, /leading-snug/)
  assert.match(paragraph, /break-all/)
  // The sentence is its own block, so it never shares a line with the repo id.
  assert.match(paragraph, /<span class="block">/)
})
