/**
 * 🖐 ARRANGING THE BOARD TAKES THE VIEW OVER — the ◉ LoRA Canvas stops
 * re-framing itself the moment the user places something.
 *
 * The board fits itself automatically until it is "touched", which used to mean
 * zoomed or panned only. Free placement made that gap expensive: a picture
 * dropped far from its lane GROWS the board, the fit signature changes, and the
 * effect re-framed the whole plateau the instant the hand let go. Carrying a
 * render up beside another lane to compare two characters — the gesture free
 * placement exists for — therefore ended by throwing away the zoom you had
 * chosen for it. Suppressing the fit only DURING the gesture (which is what
 * shipped) fixed the board sliding under the finger and left the jump at the
 * drop, where it is most visible.
 *
 * So a drop is now a take-over: images AND run cards, one rule for the board.
 * ✦ Fit stays the way back, one click, on purpose — the fix is that the frame
 * is offered rather than imposed.
 *
 * Read as source text: the rule lives in an effect and in two pointer handlers,
 * none of which `renderToStaticMarkup` can run (mountJsx.mjs says so plainly —
 * effects never run and no event ever fires). What a text test CAN do is hold on
 * to the seams, so this file slices the two handlers and asserts inside them
 * rather than grepping the file for a word that could be anywhere. The measured
 * proof — scale and offset identical across a drop — is the disposable-instance
 * run in the commit message.
 */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const canvas = readFileSync(
  new URL('../src/components/canvas/LineageCanvas.jsx', import.meta.url), 'utf8')

/** The body of one `const NAME = useCallback(` … `}, [deps]);` in the source. */
function callbackBody(name) {
  const start = canvas.indexOf(`const ${name} = useCallback(`)
  assert.ok(start > 0, `${name} is still a useCallback`)
  const end = canvas.indexOf('\n  }, [', start)
  assert.ok(end > start, `${name} still ends on a dependency array`)
  return canvas.slice(start, end)
}

test('a board nobody has arranged still frames itself on arrival', () => {
  // The whole point of the flag is that it starts false: opening the canvas on
  // a board you have never touched must show all of it, exactly as before.
  assert.match(canvas, /const touched = useRef\(false\)/)
  assert.match(canvas, /setView\(initialView\(world, viewport\)\)/)
})

test('the automatic fit gives up as soon as the view is taken over', () => {
  // Three guards, one line, in the effect that fits: taken over, mid-gesture, or
  // already fitted for this exact board/viewport pair.
  assert.match(canvas,
    /if \(touched\.current \|\| gesturing \|\| lastFit\.current === fitSignature\) return/)
})

test('take-over is ONE decision, named once', () => {
  assert.match(canvas, /const takeOverView = useCallback\(\(\) => \{ touched\.current = true; \}, \[\]\)/)
})

test('dropping a PICTURE takes the view over — but only if it moved', () => {
  const body = callbackBody('endPointer')
  const imageBranch = body.slice(body.indexOf('const gi = imgRef.current'),
    body.indexOf('const d = dragRef.current'))
  assert.match(imageBranch, /if \(gi\.moved\) takeOverView\(\);/,
    'the drop that placed a picture must claim the view')
  // A tap that never travelled is not an arrangement: it must leave a fresh
  // board free to keep fitting itself.
  assert.doesNotMatch(imageBranch, /^\s*takeOverView\(\);/m)
})

test('dropping a run CARD takes it over too — one rule for the whole board', () => {
  /* Left out at first, out of caution. It is the same gesture on the same
     surface: a board that holds still when you move a picture and jumps when you
     move a card is a board whose behaviour cannot be learned. */
  const body = callbackBody('endPointer')
  const cardBranch = body.slice(body.indexOf('const d = dragRef.current'))
  assert.match(cardBranch, /if \(d\.moved\) \{[^]{0,400}takeOverView\(\);/)
  // The guard is the SAME one that decides whether the move is persisted at all,
  // so a plain click on a card can never freeze the fit behind the user's back.
  assert.doesNotMatch(cardBranch.slice(cardBranch.indexOf('setDrag(null)')), /takeOverView/)
})

test('✦ Fit is still the way back, and still resets the flag', () => {
  const body = callbackBody('fitNow')
  assert.match(body, /touched\.current = false/)
  assert.match(body, /lastFit\.current = ''/)
  assert.match(body, /setView\(fitView\(world, viewport\)\)/)
})

test('zooming and panning still count as taking over', () => {
  // The original half of the rule, unchanged: applyView is what every wheel,
  // pinch and pan goes through.
  const body = callbackBody('applyView')
  assert.match(body, /touched\.current = true/)
})
