/* The SeedVR2 surfaces must SAY where the thing comes from.

   Three links, asked for by a user on the Setup card, each answering a question
   the card raises and could not answer:
     * the node pack — the card tells you to install it through ComfyUI-Manager,
       and someone who would rather clone it by hand had no repo to go to;
     * the weights repo — a button downloads 3.9 GB, and where from is not a
       detail;
     * the original project — SeedVR2 is ByteDance-Seed's work, and the app
       leaning on it should say so.

   Pinned as a contract because links rot silently: a card that names a source it
   no longer links, or links a URL nobody checked, is worse than one that says
   nothing. All four URLs were verified 200 on 2026-08-02.
*/
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const read = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8')
const setup = read('./SeedVr2InstallCard.jsx')
const settings = read('../settings/EnginesSection.jsx')

const PACK = 'https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler'
const WEIGHTS = 'https://huggingface.co/numz/SeedVR2_comfyUI'
const PROJECT = 'https://github.com/ByteDance-Seed/SeedVR'

test('the Setup card links the pack, the weights and the original project', () => {
  for (const url of [PACK, WEIGHTS, PROJECT]) {
    assert.ok(setup.includes(url), `Setup card must link ${url}`)
  }
  // Real anchors, opened safely — not bare text a user has to retype.
  assert.match(setup, /<a href=\{PACK_URL\} target="_blank" rel="noreferrer"/)
  assert.match(setup, /<a href=\{WEIGHTS_URL\} target="_blank" rel="noreferrer"/)
  assert.match(setup, /<a href=\{PROJECT_URL\} target="_blank" rel="noreferrer"/)
})

test('the pack link is offered AT the moment the card asks you to install it', () => {
  // The "install it from ComfyUI-Manager" warning used to end on a bare URL in
  // a <span>: the one place the link is actionable was the one place it was not
  // a link.
  assert.match(setup, /Open the node pack on GitHub/)
  assert.doesNotMatch(setup, /Source: <span className="break-all">\{PACK_URL\}<\/span>/)
})

test('the Settings card carries the same three sources', () => {
  for (const url of [PACK, WEIGHTS, PROJECT]) {
    assert.ok(settings.includes(url), `Settings card must link ${url}`)
  }
})

test('both cards state the licence, because that is why these are safe to ship', () => {
  // The repo has refused a dependency over its licence before; naming Apache-2.0
  // next to the sources is what makes that check visible rather than folklore.
  for (const [name, src] of [['Setup', setup], ['Settings', settings]]) {
    assert.match(src, /Apache-2\.0/, `${name} card must state the licence`)
  }
})
