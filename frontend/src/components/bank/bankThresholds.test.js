import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
  BANK_SECTION, BANK_THRESHOLDS, THRESHOLD_GROUPS, APPLIES, PASS_RERUN,
  coerceValue, customisedCountInGroup, customisedFields, dirtyFields,
  directionHint, directionSummary, effectLine, isValidValue, previewableFields,
  rerunFor, resetAllEdits, thresholdByField, thresholdsInGroup,
} from './bankThresholds.js';

const ws = fs.readFileSync(new URL('./BankWorkspace.jsx', import.meta.url), 'utf8');
const panel = fs.readFileSync(new URL('./BankThresholdsPanel.jsx', import.meta.url), 'utf8');
const captioning = fs.readFileSync(
  new URL('../settings/CaptioningSection.jsx', import.meta.url), 'utf8');
const configPy = fs.readFileSync(
  new URL('../../../../backend/app/config.py', import.meta.url), 'utf8');

/** The keys of the `'bank'` block of backend/app/config.py — the ONE place the
 *  shipped defaults live. Parsed rather than copied on purpose: a copy would
 *  agree with itself forever while the real defaults moved. */
function bankKeysFromConfigPy() {
  const start = configPy.indexOf("'bank': {");
  assert.ok(start > 0, "found the 'bank' block in config.py");
  // Walk to the matching brace so trailing comment lines cannot widen the slice.
  let depth = 0;
  let end = start;
  for (let i = configPy.indexOf('{', start); i < configPy.length; i += 1) {
    if (configPy[i] === '{') depth += 1;
    else if (configPy[i] === '}') {
      depth -= 1;
      if (depth === 0) { end = i; break; }
    }
  }
  const block = configPy.slice(start, end);
  // Only KEYS, i.e. quoted names followed by a colon at the start of an entry —
  // commented prose inside the block mentions the same names without a colon.
  return [...block.matchAll(/'([a-z_]+)':/g)].map((m) => m[1])
    .filter((k) => k !== 'bank');
}

// ---------------------------------------------------------------------------
// The central assertion: every threshold is reachable from the Bank.
// ---------------------------------------------------------------------------

test('every bank threshold in config.py is exposed by the Bank panel', () => {
  const shipped = bankKeysFromConfigPy();
  assert.equal(shipped.length, 12, 'config.py still ships twelve bank thresholds');
  const exposed = BANK_THRESHOLDS.map((t) => t.field);
  assert.deepEqual([...exposed].sort(), [...shipped].sort());
});

test('the Bank workspace actually renders the threshold panel', () => {
  // Before this feature the twelve knobs existed ONLY in Settings ▸ Captioning:
  // this assertion is what was red.
  assert.match(ws, /BankThresholdsPanel/);
  assert.match(ws, /import BankThresholdsPanel from '\.\/BankThresholdsPanel\.jsx'/);
});

test('the Bank and Settings edit the SAME keys — no third source of truth', () => {
  // Settings ▸ Captioning writes setField('bank', '<key>', ...) for each one.
  const inSettings = new Set(
    [...captioning.matchAll(/setField\('bank',\s*'([a-z_]+)'/g)].map((m) => m[1]));
  for (const t of BANK_THRESHOLDS) {
    assert.ok(inSettings.has(t.field),
      `${t.field} is still editable in Settings ▸ Captioning`);
  }
  assert.equal(inSettings.size, BANK_THRESHOLDS.length,
    'Settings exposes exactly the same set, no more');
});

test('the panel writes through the shared settings API, into the bank section', () => {
  assert.match(panel, /\/api\/settings/);
  assert.match(panel, /config:\s*\{\s*\[BANK_SECTION\]/);
  assert.equal(BANK_SECTION, 'bank');
});

// ---------------------------------------------------------------------------
// Defaults come from the server, never from a copy.
// ---------------------------------------------------------------------------

test('no default value is typed into the frontend', () => {
  const src = fs.readFileSync(new URL('./bankThresholds.js', import.meta.url), 'utf8');
  // The shipped numbers, straight from config.py. None of them may appear as a
  // "default" anywhere in our module — the reset button reads config_defaults.
  const defaults = [...configPy.matchAll(/'(sharpness_min|noise_max|uniformity_min|dup_distance|min_side|face_threshold|aesthetic_min|nsfw_max|style_threshold|semantic_dup_threshold|detail_min|bars_max)':\s*([0-9.]+)/g)];
  assert.equal(defaults.length, 12, 'read all twelve shipped defaults');
  assert.doesNotMatch(src, /\bdefault(Value)?\s*:/,
    'bankThresholds.js declares no default — the server sends them');
  assert.doesNotMatch(panel, /\bconst\s+\w*DEFAULTS?\b/,
    'the panel holds no defaults table either');
});

test('every exposed threshold gets a reset, sourced from config_defaults', () => {
  // One shared ResetToDefault per field, section="bank", reading configDefaults.
  assert.match(panel, /import ResetToDefault from '\.\.\/settings\/ResetToDefault\.jsx'/);
  assert.match(panel, /<ResetToDefault[\s\S]*?section=\{BANK_SECTION\}/);
  assert.match(panel, /field=\{t\.field\}/);
  assert.match(panel, /configDefaults=\{configDefaults\}/);
});

test('resetAllEdits restores exactly the fields the server knows a default for', () => {
  const serverDefaults = { bank: { sharpness_min: 100.0, dup_distance: 8, nsfw_max: 0.5 } };
  assert.deepEqual(resetAllEdits(serverDefaults),
    { sharpness_min: 100.0, dup_distance: 8, nsfw_max: 0.5 });
  // An older backend that sends nothing gets no reset rather than a wrong one.
  assert.deepEqual(resetAllEdits({}), {});
  assert.deepEqual(resetAllEdits(undefined), {});
});

test('customised fields are the ones differing from the server defaults', () => {
  const defaults = { bank: { sharpness_min: 100, noise_max: 15, dup_distance: 8 } };
  const saved = { sharpness_min: 140, noise_max: 15, dup_distance: 8 };
  assert.deepEqual(customisedFields(saved, defaults), ['sharpness_min']);
  // A hand-edited config.json holds strings; "100" is not a customisation.
  assert.deepEqual(customisedFields({ ...saved, sharpness_min: '100' }, defaults), []);
  assert.equal(customisedCountInGroup('quality', saved, defaults), 1);
  assert.equal(customisedCountInGroup('duplicates', saved, defaults), 0);
});

// ---------------------------------------------------------------------------
// The direction trap — the guard-rail that stops a knob being set backwards.
// ---------------------------------------------------------------------------

test('the two duplicate thresholds are declared as mirrors of each other', () => {
  // dup_distance is a DISTANCE, semantic_dup_threshold a SIMILARITY. Both mean
  // "catch more near-duplicates", and they move in OPPOSITE directions. This is
  // the assertion that keeps the UI from teaching the wrong gesture.
  assert.equal(thresholdByField('dup_distance').catchesMoreWhen, 'raised');
  assert.equal(thresholdByField('semantic_dup_threshold').catchesMoreWhen, 'lowered');
  assert.equal(thresholdByField('dup_distance').group, 'duplicates');
  assert.equal(thresholdByField('semantic_dup_threshold').group, 'duplicates');
  assert.match(directionHint(thresholdByField('dup_distance')), /^▲ Raise/);
  assert.match(directionHint(thresholdByField('semantic_dup_threshold')), /^▼ Lower/);
});

test('every threshold declares which way catches more, and says it in words', () => {
  for (const t of BANK_THRESHOLDS) {
    assert.ok(['raised', 'lowered'].includes(t.catchesMoreWhen), t.field);
    assert.match(directionHint(t), /catch more images/);
    assert.match(directionSummary(t), /catch more images\.$/);
  }
});

test('the direction of every threshold matches what the backend does with it', () => {
  // Mirrors backend/app/services/image_bank_service.py:_flag_filter — a minimum
  // catches more when raised, a maximum when lowered; a similarity used to GROUP
  // catches more when lowered, a distance when raised.
  const expected = {
    sharpness_min: 'raised', noise_max: 'lowered', uniformity_min: 'raised',
    min_side: 'raised', detail_min: 'raised', bars_max: 'lowered',
    dup_distance: 'raised', semantic_dup_threshold: 'lowered',
    face_threshold: 'lowered', aesthetic_min: 'raised', nsfw_max: 'lowered',
    style_threshold: 'lowered',
  };
  for (const t of BANK_THRESHOLDS) {
    assert.equal(t.catchesMoreWhen, expected[t.field], t.field);
  }
});

test('the panel prints the direction as visible text, not only a tooltip', () => {
  assert.match(panel, /directionHint\(t\)/);
  // ...and it is rendered, not just passed to a title=.
  assert.doesNotMatch(panel, /title=\{directionHint\(t\)\}/);
});

// ---------------------------------------------------------------------------
// Grouping, structure, accessibility.
// ---------------------------------------------------------------------------

test('every threshold belongs to a declared group and every group has members', () => {
  const ids = new Set(THRESHOLD_GROUPS.map((g) => g.id));
  for (const t of BANK_THRESHOLDS) assert.ok(ids.has(t.group), t.field);
  for (const g of THRESHOLD_GROUPS) {
    assert.ok(thresholdsInGroup(g.id).length > 0, `${g.id} is not empty`);
  }
  // Opening the parent "Filter thresholds" panel reveals the group headings,
  // never all their controls at once. Every inner group starts folded.
  assert.equal(THRESHOLD_GROUPS.filter((g) => g.defaultOpen).length, 0,
    'opening Filter thresholds must leave every inner group folded');
});

test('every threshold says when it takes effect, using a declared phrase', () => {
  for (const t of BANK_THRESHOLDS) {
    assert.ok(APPLIES[t.applies], `${t.field} has a known applies key`);
  }
  // The four grouping thresholds are produced by a pass, so none may claim to
  // apply instantly — that promise is what a live count would be lying about.
  for (const f of ['dup_distance', 'semantic_dup_threshold', 'face_threshold', 'style_threshold']) {
    assert.notEqual(thresholdByField(f).applies, 'instant', f);
  }
});

test('every threshold has a real label, a unit and a hint', () => {
  for (const t of BANK_THRESHOLDS) {
    assert.ok(t.label && t.label.length > 3, t.field);
    assert.ok(t.unit && t.unit.length > 0, t.field);
    assert.ok(t.hint && t.hint.length > 20, t.field);
  }
});

test('each input is labelled and described, not placeholder-labelled', () => {
  assert.match(panel, /<label htmlFor=\{`bank-th-\$\{t\.field\}`\}/);
  assert.match(panel, /id=\{`bank-th-\$\{t\.field\}`\}/);
  assert.match(panel, /aria-describedby=\{`bank-th-\$\{t\.field\}-hint /);
  assert.match(panel, /id=\{`bank-th-\$\{t\.field\}-hint`\}/);
  assert.doesNotMatch(panel, /placeholder=/);
});

test('the effect counter is reachable by assistive tech without re-announcing', () => {
  // The live NUMBER is aria-hidden and the same fact sits in the input's
  // description, so nudging a value does not fire one announcement per
  // keystroke. The only live region carries a two-state sentence.
  assert.match(panel, /className="mt-1 min-h-\[1rem\][^"]*"\s+aria-hidden="true"/);
  assert.match(panel, /aria-describedby=\{`bank-th-\$\{t\.field\}-hint bank-th-\$\{t\.field\}-effect`\}/);
  assert.match(panel, /id=\{`bank-th-\$\{t\.field\}-effect`\}\s+className="sr-only"/);
  const live = [...panel.matchAll(/aria-live="polite"/g)];
  assert.equal(live.length, 1, 'exactly one live region in the panel');
  assert.match(panel, /You have unsaved threshold changes/);
});

test('a non-instant threshold offers to run the pass that would apply it', () => {
  // "Applies at the next pass" is half an answer if the pass is elsewhere.
  for (const t of BANK_THRESHOLDS) {
    const r = rerunFor(t);
    if (t.applies === 'instant') assert.equal(r, null, t.field);
    else assert.ok(r && r.endpoint && r.label && r.note, t.field);
  }
  assert.equal(rerunFor(thresholdByField('dup_distance')).endpoint, 'scan');
  assert.equal(rerunFor(thresholdByField('semantic_dup_threshold')).endpoint, 'semantic-dedup');
  assert.equal(rerunFor(thresholdByField('face_threshold')).endpoint, 'faces');
  assert.equal(rerunFor(thresholdByField('style_threshold')).endpoint, 'score');
  // Every endpoint named here is a real bank route.
  const routes = fs.readFileSync(
    new URL('../../../../backend/app/routes/bank.py', import.meta.url), 'utf8');
  for (const r of Object.values(PASS_RERUN)) {
    assert.match(routes, new RegExp(`@bp\\.post\\('/bank/<int:bank_id>/${r.endpoint}'\\)`),
      `${r.endpoint} is a real route`);
  }
  // The click goes through runPass(), which brackets onRunPass with the two
  // things a 202 cannot provide on its own: the "starting/running" state and
  // the figures the pass produced.
  assert.match(panel, /onRun=\{\(\) => runPass\(rerun\.endpoint, rerun\.body\)\}/);
  assert.match(panel, /await onRunPass\?\.\(endpoint, body\)/);
  assert.match(ws, /onRunPass=\{\(endpoint, body\)/);
  // And the INTENT travels with it. "↻ Re-group duplicates" posts to the scan
  // route, whose own pool is empty on an already-scanned bank — the quality scan
  // only re-groups when the hashes it stored actually moved (re-grouping 50 000
  // unchanged hashes at the tail of a scan that had 2 images to look at is what
  // took the app away for two minutes). Drop this body and the button becomes a
  // no-op that reports success.
  assert.deepEqual(PASS_RERUN.scan.body, { regroup: true });
  assert.match(
    fs.readFileSync(new URL('../../../../backend/app/routes/bank.py', import.meta.url), 'utf8'),
    /regroup=bool\(data\.get\('regroup'\)\)/,
    'the scan route must read the regroup intent out of the body');
});

test('a re-run button cannot be pressed while a pass owns the bank', () => {
  // A bank runs ONE job at a time, so this click could only ever have produced
  // the server's "a scan job is already running on this bank". The gating and
  // its wording live in bankPassRun.js (executable); this pins the WIRING —
  // that the panel is actually fed the live job and actually disables on it.
  assert.match(panel, /import \{[^}]*passButtonState[^}]*\} from '\.\/bankPassRun\.js'/s);
  assert.match(panel, /disabled=\{state\.disabled\}/);
  // A greyed-out control must SAY why, not merely be grey.
  assert.match(panel, /aria-describedby=\{describedBy \|\| undefined\}/);
  assert.match(panel, /title=\{why \|\| rerun\.note\}/);
  // The workspace hands over the same snapshot its progress bar reads — no
  // second poll, no fourth progress mechanism.
  assert.match(ws, /activity=\{payload\?\.activity\}\s+offline=\{!connection\.online\}/);
});

test('an occupied-bank refusal is reworded ONCE, for every bank action', () => {
  // act() is the single funnel for every mutating Bank click (the ✨ passes, the
  // ↻ re-runs, 🗑 Delete rejected, ⬆ Promote, 🚀 Launch all). Rewording the 409
  // there is what stops the server sentence reaching a toast anywhere in the
  // bank — a per-button fix would have left eleven other buttons raw.
  assert.match(ws, /busyRefusal\(\{ kind, activity: payload\?\.activity \}\)/);
  assert.match(ws, /e\?\.status === 409 && kind/);
  // The route has to label the refusal for that to be possible: the 409 often
  // lands before the first progress poll, so its body is the only thing that
  // knows which pass is in the way.
  const routes = fs.readFileSync(
    new URL('../../../../backend/app/routes/bank.py', import.meta.url), 'utf8');
  assert.match(routes, /'busy_kind': e\.kind/);
  assert.ok(!/jsonify\(\{'error': str\(e\)\}\), 409/.test(routes),
    'every occupied-bank 409 must go through the labelled _busy() helper');
});

test('a finished re-run reports its figures, announced exactly once', () => {
  assert.match(panel, /passOutcome\(/);
  // The counts come from the payload summaries the duplicate chips already use.
  assert.match(ws, /dupSummary=\{payload\?\.dup\}\s+semanticDupSummary=\{payload\?\.semantic_dup\}/);
  // role="status" (not aria-live) on a node that appears once per run: announced
  // when it lands, never re-announced as progress ticks.
  assert.match(panel, /id=\{outcomeId\} role="status"/);
  const live = [...panel.matchAll(/aria-live=/g)];
  assert.equal(live.length, 1, 'still exactly one aria-live region in the panel');
});

test('the live preview never raises a notification of its own', () => {
  // It fires while a value is being nudged, so it must be a background call:
  // an unreachable server would otherwise toast once per keystroke.
  const previews = [...panel.matchAll(/flag-preview[\s\S]{0,160}?background: true/g)];
  assert.equal(previews.length, 2, 'both preview calls (baseline + candidate) are background');
});

test('the preview endpoint exists and does not save anything', () => {
  const routes = fs.readFileSync(
    new URL('../../../../backend/app/routes/bank.py', import.meta.url), 'utf8');
  assert.match(routes, /@bp\.post\('\/bank\/<int:bank_id>\/flag-preview'\)/);
  const svc = fs.readFileSync(
    new URL('../../../../backend/app/services/image_bank_service.py', import.meta.url), 'utf8');
  const body = svc.slice(svc.indexOf('def flag_preview'), svc.indexOf('def _load_pipeline_report'));
  assert.ok(body.length > 100, 'found flag_preview');
  assert.doesNotMatch(body, /db\.session\.(commit|add|delete)|save_config/,
    'the preview writes nothing');
});

test('groups are collapsible with a real expanded state', () => {
  assert.match(panel, /aria-expanded=\{/);
  assert.match(panel, /aria-controls=/);
});

// ---------------------------------------------------------------------------
// Editing logic.
// ---------------------------------------------------------------------------

test('typed values are coerced to the type the config stores', () => {
  assert.equal(coerceValue(thresholdByField('min_side'), '768.7'), 768);
  assert.equal(coerceValue(thresholdByField('detail_min'), '0.72'), 0.72);
  // A field being cleared stays blank instead of snapping to 0 under the cursor.
  assert.equal(coerceValue(thresholdByField('detail_min'), ''), '');
  assert.equal(coerceValue(thresholdByField('detail_min'), 'abc'), '');
});

test('out-of-range and blank values are not savable', () => {
  const bars = thresholdByField('bars_max');
  assert.equal(isValidValue(bars, 0.04), true);
  assert.equal(isValidValue(bars, 4), false);      // 0–1, not a percentage
  assert.equal(isValidValue(bars, -1), false);
  assert.equal(isValidValue(bars, ''), false);
  assert.equal(isValidValue(thresholdByField('sharpness_min'), 500), true);  // no ceiling
});

test('dirty fields ignore invalid edits and no-op retypes', () => {
  const saved = { sharpness_min: 100, bars_max: 0.04 };
  assert.deepEqual(dirtyFields(saved, { sharpness_min: 140 }), ['sharpness_min']);
  assert.deepEqual(dirtyFields(saved, { sharpness_min: 100 }), []);
  assert.deepEqual(dirtyFields(saved, { sharpness_min: '100' }), []);
  assert.deepEqual(dirtyFields(saved, { bars_max: 4 }), []);       // out of range
  assert.deepEqual(dirtyFields(saved, { bars_max: '' }), []);      // mid-typing
  assert.deepEqual(dirtyFields(saved, { not_a_field: 3 }), []);
});

// ---------------------------------------------------------------------------
// Showing the effect.
// ---------------------------------------------------------------------------

test('only the read-time thresholds offer a live count', () => {
  assert.deepEqual(previewableFields().sort(), [
    'aesthetic_min', 'bars_max', 'detail_min', 'min_side', 'noise_max',
    'nsfw_max', 'sharpness_min', 'uniformity_min',
  ]);
});

test('the effect line compares candidate to current, and stays silent when it cannot', () => {
  const blur = thresholdByField('sharpness_min');
  // Thousands are grouped with the READER's separator (a French browser writes
  // "1 240", a German one "1.240"), so the assertion normalises it away instead
  // of pinning a locale. These are flag counts, never fractional, so a literal
  // '.' is always a grouping separator here, not a decimal point.
  const SEPS = [',', '.', 160, 8239, 8201].map((c) => (typeof c === 'number' ? String.fromCharCode(c) : c));
  const plain = (s) => (s === null ? null : [...s].filter((c) => !SEPS.includes(c)).join(''));
  assert.equal(plain(effectLine(blur, { blur: 3019 }, { blur: 1240 })), '1240 → 3019 images flagged ↑');
  assert.equal(plain(effectLine(blur, { blur: 900 }, { blur: 1240 })), '1240 → 900 images flagged ↓');
  assert.equal(plain(effectLine(blur, { blur: 1240 }, { blur: 1240 })), '1240 images flagged');
  assert.equal(effectLine(blur, { blur: 12 }, null), '12 images flagged');
  // No preview yet, or a threshold with no flag: say nothing rather than 0.
  assert.equal(effectLine(blur, null, { blur: 1240 }), null);
  assert.equal(effectLine(thresholdByField('dup_distance'), { blur: 1 }, null), null);
});
