/* 🚩 Bank watermark mask — wiring contract (node --test cannot parse JSX, so the
 * properties that would silently un-ship the feature are asserted as text).
 *
 * Reported by Qeeyana (Reddit). The risk this file guards is NOT "the dialog
 * renders": it is the Bank quietly growing its own second mask editor, and the
 * mask being drawn in a coordinate space the cleaner does not use.
 */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import { WHATS_NEW } from '../src/whatsNew.js'
import { getHelpTopic } from '../src/help/helpRegistry.js'

const here = path.dirname(fileURLToPath(import.meta.url))
const frontend = path.resolve(here, '..')
const read = (rel) => fs.readFileSync(path.join(frontend, rel), 'utf8')
const dialog = read('src/components/bank/BankWatermarkMaskDialog.jsx')
const lightbox = read('src/components/bank/BankReviewLightbox.jsx')

test('the Bank mounts the DATASET editor — it never grows a second one', () => {
  assert.match(dialog, /import WatermarkRegionEditor from '\.\.\/dataset\/WatermarkRegionEditor'/)
  assert.match(dialog, /<WatermarkRegionEditor/)
  // The geometry helpers come from the one shared module too.
  assert.match(dialog, /from '\.\.\/\.\.\/utils\/watermarkRegions\.js'/)
  // No hand-rolled drag/normalise maths sneaking back in on the bank side.
  assert.doesNotMatch(dialog, /getBoundingClientRect|onPointerMove/)
})

test('the mask is drawn on the SOURCE pixels the cleaner works on', () => {
  // The bank's watermark lane (scan, crop, inpaint) reads abs_image_path — the
  // unrotated source. Drawing on the rotated view would put every zone in the
  // wrong space, silently.
  assert.match(dialog, /\/api\/bank\/\$\{bankId\}\/file\/\$\{image\.id\}\?original=1/)
})

test('every edit is persisted to the bank route, and a failure says so', () => {
  assert.match(dialog, /putJson\(`\/api\/bank\/\$\{bankId\}\/image\/\$\{image\.id\}\/watermark-regions`/)
  const failure = dialog.match(/\.catch\(\(e\) => \{([\s\S]*?)\}\)/)[1]
  assert.match(failure, /setSave\(\{ status: 'failed'/)
  assert.doesNotMatch(failure, /onSaved/)          // a failed save is not a save
  assert.match(dialog, /is NOT saved/)             // and the user is told
})

test('the review lightbox offers the editor on flagged images only', () => {
  assert.match(lightbox, /canEditMask\(img\) && \(/)
  assert.match(lightbox, /🚩 Edit mask/)
  assert.match(lightbox, /k === 'm' && canEditMask\(img\)/)
  assert.match(lightbox, /M watermark mask/)       // printed, not folklore
})

test('the keyboard cannot decide on an image while its mask is open', () => {
  // K/R/S are one keystroke from a decision; the editor must own the keyboard.
  assert.match(lightbox, /if \(maskId != null\) return/)
  assert.match(lightbox, /useFocusTrap\(dialogRef, maskId == null\)/)
})

test('the feature is announced and documented', () => {
  const entry = WHATS_NEW.find((e) => e.id === '2026-07-28-bank-watermark-mask-editing')
  assert.ok(entry, "What's new entry for the bank mask editor is missing")
  assert.equal(entry.to, '/bank')
  assert.match(entry.blurb, /Qeeyana/)             // credit where it is due
  const topic = getHelpTopic('bank-edit-watermark-mask')
  assert.ok(topic, 'help topic bank-edit-watermark-mask is missing')
  assert.equal(topic.guide.anchor, 'fix-a-watermark-mask-in-a-bank')
})

test('the dialog is usable with a thumb at 400 px', () => {
  // 44 px targets and a wrapping control row — drawing a rectangle by hand on a
  // phone is the hard case this feature has to survive.
  assert.match(dialog, /min-h-11/)
  assert.match(dialog, /flex-wrap/)
  assert.match(dialog, /\[container-type:size\]/)  // the photo caps to its cell
})
