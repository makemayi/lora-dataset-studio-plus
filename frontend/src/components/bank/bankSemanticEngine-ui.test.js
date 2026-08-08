import assert from 'node:assert/strict'
import fs from 'node:fs'
import test from 'node:test'

import { BANK_PASSES, bankPass, passScopeRows,
  passSelectionAvailability } from './bankPasses.js'

const ws = fs.readFileSync(new URL('./BankWorkspace.jsx', import.meta.url), 'utf8')
const panel = fs.readFileSync(new URL('./BankSemanticEngine.jsx', import.meta.url), 'utf8')
const dialog = fs.readFileSync(new URL('./LaunchAllDialog.jsx', import.meta.url), 'utf8')

test('the accessible Bank selector persists only the requested engine with PATCH', () => {
  assert.match(panel, /<fieldset/)
  assert.match(panel, /<legend[^>]*>[\s\S]*Semantic engine/)
  assert.match(panel, /type="radio"/)
  assert.match(panel, /name="bank-semantic-engine"/)
  assert.match(ws, /patchJson\(`\/api\/bank\/\$\{bankId\}\/semantic-engine`,\s*semanticEnginePatchBody\(engine\)\)/)
  const change = ws.slice(ws.indexOf('const changeSemanticEngine'),
    ws.indexOf('const batchStatus'))
  assert.doesNotMatch(change, /semantic-index/,
    'switching engines must not automatically build an index')
  assert.doesNotMatch(change, /delete|del\(/i,
    'switching engines must not delete either cache')
})

test('the selector keeps the ownership and cache-conservation copy visible', () => {
  assert.match(panel, /semanticPurposeSentence\(state\.engine\)/)
  assert.match(panel, /SCORE_STAYS_CLIP_SENTENCE/)
  assert.match(panel, /SEMANTIC_CACHE_SENTENCE/)
  assert.match(panel, /Produced by ✨ Score/)
})

test('switching and unmount release the text encoder that actually owns memory', () => {
  const change = ws.slice(ws.indexOf('const changeSemanticEngine'),
    ws.indexOf('const batchStatus'))
  assert.match(change, /const previousEngine = semanticState\.engine/)
  assert.match(change, /if \(d\) \{[\s\S]*releaseTextEncoder\(previousEngine\)/)
  assert.match(ws, /const semanticEngineRef = useRef\('clip'\)/)
  assert.match(ws, /semanticEngineRef\.current = semanticState\.engine/)
  assert.match(ws, /semanticEnginePatchBody\(semanticEngineRef\.current\)/)
  assert.doesNotMatch(ws, /~2\.4 GB/,
    'memory copy must not claim CLIP-specific weight for every engine')
})

test('switching engines exits engine-specific views and cannot race semantic actions', () => {
  const change = ws.slice(ws.indexOf('const changeSemanticEngine'),
    ws.indexOf('const batchStatus'))
  assert.match(change, /semanticOperationBusy/)
  assert.match(change, /exitSelectionView\(\)/)
  assert.match(change, /setOffset\(0\)/)
  assert.match(change, /refreshImages\(filter, 0, \{ on: false \}\)/)
  assert.match(ws, /const \[similarBusy, setSimilarBusy\] = useState\(false\)/)
  assert.match(ws, /switching=\{semanticSwitching\} disabled=\{semanticOperationBusy\}/)
  assert.match(ws, /semanticEngineRef\.current !== requestEngine/)
  assert.match(ws, /semanticPayloadMatches\(d, requestEngine, requestModelKey\)/)
  assert.match(panel, /\{switching && <span[^>]*>Saving choice…<\/span>\}/)
  assert.doesNotMatch(panel, /\{disabled && <span[^>]*>Saving choice…<\/span>\}/)
})

test('coverage is request-scoped and labelled by the engine that produced it', () => {
  assert.match(ws, /const coverageRequestRef = useRef\(0\)/)
  assert.match(ws, /requestId !== coverageRequestRef\.current/)
  assert.match(ws, /semanticPayloadMatches\(next, expectedEngine, expectedModelKey\)/)
  assert.match(ws, /semanticEngine=\{coverage\?\.engine \|\| semanticState\.engine\}/)
  assert.match(ws, /semanticLabel=\{semanticEngineLabel\(coverage\?\.engine \|\| semanticState\.engine\)\}/)
  assert.match(ws, /const textStatusRequestRef = useRef\(0\)/)
  assert.match(ws, /requestId === textStatusRequestRef\.current/)
})

test('missing SigLIP2 links to Setup and never exposes a build button', () => {
  assert.match(panel, /state\.engine === 'siglip2' && !capsLoading && !state\.installed/)
  assert.match(panel, /href="#\/setup\?step=quality"/)
  assert.match(panel, /Open Setup ▸ Quality tools/)
  assert.match(panel, /state\.engine === 'siglip2' && !capsLoading && action/)
})

test('semantic-ready is rendered separately from the aesthetic scored count', () => {
  assert.match(ws, /const scored = counts\?\.scored \|\| 0/)
  assert.match(ws, /const semanticReady = semanticState\.ready/)
  assert.match(ws, /label=\{`\$\{semanticState\.label\} semantic-ready`\}/)
  assert.match(ws, /semanticIndexed/)
})

test('every semantic action gates on semanticReady, while Medium stays CLIP/Score-owned', () => {
  const semanticControls = ws.slice(ws.indexOf("setPassOpen('semantic_dedup')"),
    ws.indexOf('{coverageOpen &&'))
  assert.match(semanticControls, /disabled=\{live \|\| !semanticReady\}/)
  assert.match(semanticControls, /disabled=\{live \|\| !semanticReady \|\| diverseBusy\}/)
  assert.match(semanticControls,
    /disabled=\{live \|\| !semanticReady \|\| selected\.size !== 1 \|\| similarBusy\}/)
  assert.match(semanticControls, /disabled=\{live \|\| !semanticReady\}[\s\S]*openTextSearch/)

  const medium = ws.slice(ws.indexOf("setPassOpen('medium')"),
    ws.indexOf("setPassOpen('framing')"))
  assert.match(medium, /!caps\.bank_scoring/)
  assert.match(medium, /CLIP embeddings ✨ Score/)
})

test('semantic index is whole-Bank and always posts an explicit rescan boolean', () => {
  const spec = BANK_PASSES.semantic_index
  assert.equal(spec.endpoint, 'semantic-index')
  assert.equal(spec.countable, false)
  assert.equal(spec.redo.key, 'rescan')
  assert.equal(spec.redo.explicit, true)
  assert.match(String(spec.scopes), /WHOLE Bank/)
  assert.equal(passScopeRows('semantic_index').every((row) => !row.ok), true)
  assert.equal(passSelectionAvailability('semantic_index').ok, false)

  const body = ws.slice(ws.indexOf('const passBody'), ws.indexOf('const runPass'))
  assert.match(body, /spec\?\.redo\?\.explicit/)
  assert.match(body, /\{ \[spec\.redo\.key\]: !!redo \}/)
})

test('pass copy names the selected engine instead of always claiming Score reuse', () => {
  const clip = bankPass('semantic_dedup', { semanticEngine: 'clip' })
  const siglip = bankPass('semantic_dedup', { semanticEngine: 'siglip2' })
  assert.match(clip.what, /CLIP semantic index/)
  assert.match(siglip.what, /SigLIP 2 semantic index/)
  assert.doesNotMatch(siglip.what, /✨ Score/)
  assert.match(siglip.settings.map((setting) => setting.name).join(' '),
    /bank_semantic\.siglip2_semantic_dup_threshold/)
  assert.doesNotMatch(siglip.settings.map((setting) => setting.name).join(' '),
    /style_threshold/)
  assert.match(siglip.notHere.join(' '), /never reuses.*CLIP style clusters|CLIP style clusters.*never reuses/i)
  assert.match(clip.settings.map((setting) => setting.note || '').join(' '),
    /CLIP-only comparison-blocking optimisation/)
  assert.match(bankPass('semantic_index', { semanticEngine: 'siglip2' }).label,
    /SigLIP 2/)
})

test('Launch all uses the selected engine and owns the conditional index definition', () => {
  assert.match(dialog, /semanticEngine = 'clip'/)
  assert.match(dialog, /pipelineStepKeys\(engine\)/)
  assert.match(dialog, /semantic_index: engine === 'siglip2' && !!caps\?\.bank_siglip2/)
  assert.match(dialog, /semantic_dedup: engine === 'siglip2'/)
})
