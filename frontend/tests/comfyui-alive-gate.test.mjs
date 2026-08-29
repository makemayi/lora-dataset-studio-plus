/**
 * "我现在开着 comfyui,但还是会 Test Studio needs ComfyUI".
 *
 * Every ComfyUI-gated surface used to read `caps.studio_visible` off
 * /api/capabilities — 37 s cold on a real install, cached 30 s, and starting
 * from an all-false placeholder. So the Studio page told a user to go and
 * configure ComfyUI while ComfyUI was busy rendering that user's test.
 *
 * The rule this file pins: only a FRESH, cheap answer may say "not there", and
 * "not asked yet" is never a no.
 */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const read = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8')

test('the hook reports an unknown as null, never as false', () => {
  const src = read('../src/hooks/useComfyuiAlive.js')
  assert.match(src, /useState\(null\)/, 'unknown is its own state')
  // A failed poll must keep the last answer instead of inventing "gone".
  const failed = src.slice(src.indexOf('} catch {'))
  assert.doesNotMatch(failed.slice(0, failed.indexOf('}\n')), /setAlive/,
    'a failed poll may not publish a verdict')
})

test('the Studio page claims ComfyUI is missing only on a fresh false', () => {
  const src = read('../src/pages/StudioPage.jsx')
  assert.match(src, /if \(comfyuiAlive === false\) \{/)
  assert.doesNotMatch(src, /if \(!caps\.studio_visible\)/,
    'the slow, cached, placeholder-first flag no longer gates this page')
  assert.match(src, /Test Studio needs ComfyUI/, 'the real empty state stays')
})

test('the header item follows the same fast answer', () => {
  const src = read('../src/App.jsx')
  assert.match(src, /available=\{comfyuiAlive !== false\}/)
  assert.match(src, /const comfyuiAlive = useComfyuiAlive\(\)/)
})
