/**
 * The Bank's Stop button, as it actually reaches the screen.
 *
 * Reported from a live 36 925-image bank, mid ✨ Score:
 *
 *   Scoring pass running — 2200 / 36925 · writing 36925 score(s) to the
 *   database… — a few minutes on a bank this size          [ Stop ]
 *
 *   "the Stop button at that point works very badly."
 *
 * Measured on that instance while the pass was writing: `POST /cancel` answered
 * in 79 ms — it touches no database at all — while `GET /api/bank/<id>`, the
 * banner's own source, took 2 745 ms. So the click landed instantly and the
 * screen could not show it for three seconds. The log holds SEVEN cancel POSTs
 * inside 20 ms.
 *
 * The unit tests next to `passStop.js` pin the decisions; these pin what a
 * reader SEES, which no source-text assertion can reach: the button's own
 * label, the `disabled` attribute, and which of the two sentences is on screen.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { ProgressBar } = await import('../src/components/bank/BankWorkspace.jsx')

const text = (markup) => markup.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim()
const html = (activity) => renderToStaticMarkup(
  createElement(ProgressBar, { activity, onCancel: () => {} }))

/* The banner from the report, with the two phase sentences the write-back now
   publishes (see image_bank_service._apply_score_results). */
const WRITING = {
  kind: 'score', done: 2200, total: 36925, finished: false, cancelled: false,
  started_at: 1_754_400_000,
  detail: 'writing 36925 score(s) to the database… — a few minutes on a bank this size',
  eta_state: 'none',
  stop_cost: 'Scores already written stay. The style grouping is not written yet '
    + 'and only re-runs whole, so it needs another full pass.',
  stop_wait: 'Stopping — finishing the current batch of 200 rows, then saving.',
}

test('before the click the bar says what a Stop KEEPS, next to the button', () => {
  const seen = text(html(WRITING))
  // The right to know is before the press, not after it.
  assert.match(seen, /Scores already written stay/)
  assert.match(seen, /needs another full pass/)
  assert.match(html(WRITING), />Stop<\/button>/)
  // Nothing about stopping is claimed while nothing has been asked.
  assert.doesNotMatch(seen, /Stopping/)
})

test('the button is live and unpressed until it is used', () => {
  // `disabled=""` the ATTRIBUTE — the class list carries `disabled:opacity-60`
  // at every moment and a bare /disabled/ would pass on a live button forever.
  assert.doesNotMatch(html(WRITING), /\sdisabled=/)
})

test('once the stop is registered the button says so AND stops taking clicks', () => {
  // `cancelled` is the server half of the same state the click sets locally —
  // it is what a stop asked from another tab, or before this bar mounted,
  // looks like. Both reach the same two lines of markup.
  const markup = html({ ...WRITING, cancelled: true })
  assert.match(markup, /\sdisabled=""/)
  const seen = text(markup)
  assert.match(seen, /Stopping…/)
  // The price line is replaced by what the pass is finishing: at this phase the
  // flag is read once per commit batch, which is the honest reason for the wait.
  assert.match(seen, /finishing the current batch of 200 rows/)
  assert.doesNotMatch(seen, /Scores already written stay/)
})

test('the wait names a condition, never a duration', () => {
  // "a few seconds" would be a number this layer cannot keep: the batch is 200
  // rows, and how long 200 rows take is not knowable here.
  const seen = text(html({ ...WRITING, cancelled: true }))
  assert.doesNotMatch(seen, /\b\d+ (seconds?|minutes?)\b/)
  assert.doesNotMatch(seen, /soon|shortly|almost/i)
})

test('a pass whose phase promises nothing keeps the bar exactly as it was', () => {
  // The bar serves EVERY pass. Only phases that published a promise get a line;
  // guessing one from `kind` would be a confident sentence with no author.
  const seen = text(html({
    kind: 'caption', done: 12, total: 40, finished: false,
    started_at: 1, detail: 'captioning', eta_state: 'none',
  }))
  assert.equal(seen, '⏳ Captioning running — 12 / 40 · captioning Stop')
})

test('a promised phase adds ONE line, not a second banner', () => {
  // The bar already carries label + counter + time left + detail + button; at
  // 400 px every extra block costs a screenful. The promise is one <p>.
  const markup = html(WRITING)
  assert.equal((markup.match(/<p /g) || []).length, 1)
  assert.match(markup, /aria-live="polite"/)
})

test('the style-grouping phase no longer explains Stop inside its running label', () => {
  // It used to: "writing the style grouping over 23000 image(s) — this step
  // finishes even if you Stop, because a half-written grouping would mix two
  // numberings". True, and read by everyone at all times, wrapping the counter
  // off its row at 400 px. It belongs to the button.
  const grouping = {
    kind: 'score', done: 0, total: 0, finished: false, started_at: 2,
    detail: 'writing the style grouping over 23000 image(s)', eta_state: 'none',
    stop_cost: 'This step finishes even if you Stop — a half-written grouping '
      + 'would mix two numberings. Nothing already written is lost.',
    stop_wait: 'Stopping — the style grouping is written whole first.',
  }
  const seen = text(html(grouping))
  assert.match(seen, /^⏳ Scoring pass running — writing the style grouping over 23000 image\(s\) /)
  assert.match(seen, /finishes even if you Stop/)
})
