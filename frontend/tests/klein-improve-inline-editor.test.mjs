/**
 * The improve instruction is editable FROM the improve button — proved in the
 * DOM, not in a regex over the source.
 *
 * The pure decisions (what is read, what is written, when) are covered by
 * src/components/dataset/kleinImproveEditor.test.js. What that file cannot see
 * is whether the panel actually RENDERS: KleinImproveNote pulls in
 * PromptOverrideField, KleinModelSetting and SettingsLink, and the editor branch
 * is a branch no user click can reach inside `node --test`. mountJsx executes
 * the component, so a ReferenceError or a bad prop in that branch becomes a
 * failing test here instead of a blank rail on someone's screen.
 *
 * ⚠️ Effects never run under mountJsx and nothing fetches, so the loaded state
 * is reached by seeding the module cache (`_seedKleinImproveNoteCache`) exactly
 * as a settled /api/settings would have. Reset it between tests: it is shared.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { createElement, renderToStaticMarkup } from './support/mountJsx.mjs'

const { default: KleinImproveNote, _resetKleinImproveNoteCache, _seedKleinImproveNoteCache } =
  await import('../src/components/dataset/KleinImproveNote.jsx')
const { IMPROVE_SCOPE_NOTE, IMPROVE_OFF_NOTE } =
  await import('../src/components/dataset/kleinImproveEditor.js')

const SHIPPED = 'add detailed texture, add sharp details, add candid shot, add soft focus effect'

const payload = (identityPrompts = {}) => ({
  config: { identity_prompts: identityPrompts },
  identity_prompt_defaults: { klein_improve: SHIPPED },
})

/** Render the note as if /api/settings had answered with `identityPrompts`. */
const render = (identityPrompts, props = {}) => {
  if (identityPrompts === null) _resetKleinImproveNoteCache()
  else _seedKleinImproveNoteCache(payload(identityPrompts))
  const html = renderToStaticMarkup(
    createElement(KleinImproveNote, { subjectType: 'human', ...props }))
  _resetKleinImproveNoteCache()
  return html
}

/** The markup is HTML-escaped; assertions are about the TEXT the user reads. */
const text = (html) => html
  .replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&#39;/g, "'")
  .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&#x2F;/g, '/')
  .replace(/&amp;/g, '&')

/** What a <textarea> actually holds, decoded. */
const textareaValue = (html) => {
  const m = html.match(/<textarea[^>]*>([\s\S]*?)<\/textarea>/)
  return m ? text(m[1]) : null
}

test('closed by default, but the way in is on screen and named', () => {
  const html = render({})
  assert.match(html, /data-testid="klein-improve-edit-toggle"/,
    'the editor has to be reachable from the note itself')
  assert.doesNotMatch(html, /data-testid="klein-improve-editor"/,
    'a note is a line, not a form: the box stays out of the rail until asked for')
  // Settings stays reachable: the per-subject identity prompts and the four
  // strength knobs are not in this panel and never will be.
  assert.match(html, /focus=identity-prompt-klein-improve/)
  assert.match(html, /focus=klein-improve-strength/)
})

test('the box holds the SHIPPED text when nothing is overridden', () => {
  const html = render({}, { defaultEditorOpen: true })
  assert.match(html, /data-testid="klein-improve-editor"/)
  assert.equal(textareaValue(html), SHIPPED,
    'an empty box would be indistinguishable from "no instruction"')
  // Following the default is not an override, so there is nothing to reset TO.
  assert.doesNotMatch(text(html), /Reset to default/,
    'the reset button is itself the "you changed this" marker')
  assert.match(text(html), /Following the built-in default/)
})

test('the box holds the OVERRIDE when there is one, and offers the way back', () => {
  const html = render({ klein_improve: 'keep it a flat drawing, no skin texture' },
    { defaultEditorOpen: true })
  assert.equal(textareaValue(html), 'keep it a flat drawing, no skin texture')
  assert.match(text(html), /Reset to default/)
  assert.match(text(html), /Custom override/)
})

test('turning the instruction off stays reachable here', () => {
  // Half of what the old "Edit or turn off this instruction →" link promised.
  const on = render({}, { defaultEditorOpen: true })
  assert.match(on, /type="checkbox"[^>]*checked/)
  assert.doesNotMatch(text(on), new RegExp(IMPROVE_OFF_NOTE.slice(0, 30)))

  const off = render({ klein_improve_enabled: false }, { defaultEditorOpen: true })
  assert.doesNotMatch(off, /type="checkbox"[^>]*checked/)
  assert.match(text(off), new RegExp(IMPROVE_OFF_NOTE.slice(0, 30)),
    'and it says what "off" actually does — upscale, nothing else')
  assert.match(off, /<textarea[^>]*disabled/,
    'editing a text that is not being sent would be a lie about the outcome')
})

test('the panel states, in the markup, that the change is app-wide', () => {
  // Not a hover and not a tooltip: a global control inside a dataset screen
  // reads as per-dataset until something says otherwise.
  const html = text(render({}, { defaultEditorOpen: true }))
  assert.ok(html.includes(IMPROVE_SCOPE_NOTE), 'the scope sentence must be rendered')
  assert.match(IMPROVE_SCOPE_NOTE, /every dataset/i)
})

test('the quoted line above the box shows the same text the box holds', () => {
  // Quoting one sentence while editing another would be worse than the link
  // this panel replaced.
  const html = text(render({ klein_improve: 'keep it a flat drawing' },
    { defaultEditorOpen: true }))
  assert.ok(html.includes('keep it a flat drawing'))
  assert.ok(html.includes('Improve asks Klein to:'))
  assert.equal(textareaValue(render({ klein_improve: 'keep it a flat drawing' },
    { defaultEditorOpen: true })), 'keep it a flat drawing')
})

test('with no settings answer yet, the panel refuses to open', () => {
  const html = render(null, { defaultEditorOpen: true })
  assert.doesNotMatch(html, /data-testid="klein-improve-editor"/,
    'an editor rendered before /api/settings answers would show a box whose '
    + 'content is a guess')
  assert.match(html, /data-testid="klein-improve-edit-toggle"[^>]*disabled/,
    'and the way in is disabled rather than opening onto nothing')
})

test('the note renders on every host that mounts it, open and closed', () => {
  // Four mount sites (dataset lightbox rail + bottom bar, bulk toolbar,
  // generated-image lightbox) and two of them pass no datasetId.
  for (const props of [
    {}, { datasetId: 7 }, { subjectType: 'anime' }, { subjectType: 'anime', datasetId: 7 },
    { className: 'w-full border-t' },
  ]) {
    for (const open of [false, true]) {
      assert.doesNotThrow(() => render({}, { ...props, defaultEditorOpen: open }),
        `threw for ${JSON.stringify(props)} open=${open}`)
    }
  }
})

test('the anime caution survives the editor being open', () => {
  const html = text(render({}, { subjectType: 'anime', defaultEditorOpen: true }))
  assert.match(html, /subject type is set to anime/,
    'the warning that explains WHY the box needs editing must stay above it')
})
