/**
 * PASTING AN IMAGE IMPORTS IT.
 *
 * A clipboard image has no filename (Chrome says `image.png` for every
 * screenshot), no drop target and no focus, so all three of the existing import
 * gestures miss it. These pin the parts that decide what actually lands in the
 * dataset: which payloads count as images, what they get called, and the one
 * place a paste must be left alone — a text field, where pasting is a text
 * gesture and the caption box lives.
 */
import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'

const { clipboardImageBlobs, extensionFor, isEditableTarget, pastedFileName } =
  await import('../src/components/dataset/clipboardImages.js')

const blob = (type) => ({ type })
const clipboard = ({ items = [], files = [] } = {}) => ({ items, files })
const fileItem = (b) => ({ kind: 'file', getAsFile: () => b })

test('an image on the clipboard is picked up from items OR from files', () => {
  const png = blob('image/png')
  assert.deepEqual(clipboardImageBlobs(clipboard({ items: [fileItem(png)] })),
    [{ blob: png, type: 'image/png' }])
  assert.deepEqual(clipboardImageBlobs(clipboard({ files: [png] })),
    [{ blob: png, type: 'image/png' }])
})

test('a payload exposing the same blob twice imports it once', () => {
  /* Chrome fills BOTH items and files on some builds — reading either alone
     loses pastes on somebody's machine, reading both naively doubles them. */
  const png = blob('image/png')
  assert.equal(clipboardImageBlobs(clipboard({ items: [fileItem(png)], files: [png] })).length, 1)
})

test('only formats the server accepts survive the clipboard', () => {
  const payload = clipboard({ files: [blob('image/png'), blob('image/svg+xml'),
                                      blob('image/heic'), blob('text/plain'),
                                      blob('image/webp')] })
  assert.deepEqual(clipboardImageBlobs(payload).map((e) => e.type),
    ['image/png', 'image/webp'])
})

test('copied TEXT is not an import', () => {
  assert.deepEqual(clipboardImageBlobs(clipboard({ items: [{ kind: 'string' }] })), [])
  assert.deepEqual(clipboardImageBlobs(null), [])
})

test('a pasted image is named from the paste time and its position', () => {
  const when = new Date(2026, 7, 11, 4, 5, 6, 70)
  assert.equal(pastedFileName('image/png', 0, when), 'pasted-20260811-040506-070.png')
  assert.equal(pastedFileName('image/jpeg', 1, when), 'pasted-20260811-040506-070-2.jpg')
  // An unknown type never yields an extensionless name the server would refuse.
  assert.equal(pastedFileName('image/tiff', 0, when), 'pasted-20260811-040506-070.png')
  assert.equal(extensionFor('image/PNG; charset=x'), 'png')
})

test('a paste into a text field is left to the text field', () => {
  assert.equal(isEditableTarget({ tagName: 'TEXTAREA' }), true)
  assert.equal(isEditableTarget({ tagName: 'INPUT' }), true)
  assert.equal(isEditableTarget({ isContentEditable: true }), true)
  assert.equal(isEditableTarget({ tagName: 'DIV' }), false)
  assert.equal(isEditableTarget(null), false)
})

test('the dropzone listens for paste, and says so where the other gestures are named', () => {
  const src = readFileSync(
    new URL('../src/components/dataset/ImportDropzone.jsx', import.meta.url), 'utf8')
  assert.match(src, /window\.addEventListener\('paste', onPaste\)/)
  assert.match(src, /window\.removeEventListener\('paste', onPaste\)/)
  assert.match(src, /if \(isEditableTarget\(event\.target\)\) return/)
  assert.match(src, /if \(busy\) return/)
  // The gesture is invisible unless it is written next to the others.
  assert.match(src, /drag, drop, click or paste \(Ctrl\+V\)/)
})
