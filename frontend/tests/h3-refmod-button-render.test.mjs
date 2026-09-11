/**
 * The H3 RefMod button, RENDERED. The whole point of this component is the
 * waiting state (the encode takes a minute-plus server-side), so both labels —
 * idle and "Encoding RefMod…" — must stay mounted and flip with `hidden`: a
 * ternary that unmounts one is what Chrome auto-translate turns into a
 * removeChild crash. Zero kept images renders nothing at all.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { default: H3RefModButton } = await import(
  '../src/components/dataset/H3RefModButton.jsx')

const btn = (props) => renderToStaticMarkup(createElement(H3RefModButton, props))

test('idle: the idle label is visible, the busy label mounted but hidden', () => {
  const html = btn({ onClick: () => {}, count: 12 })
  const idle = html.match(/<span [^>]*data-label="idle"[^>]*>/)[0]
  const busy = html.match(/<span [^>]*data-label="busy"[^>]*>/)[0]
  assert.ok(!idle.includes('hidden'), 'idle span must be visible while idle')
  assert.ok(busy.includes('hidden'), 'busy span must be hidden while idle')
  assert.match(html, /Generate H3 RefMod/)
  assert.match(html, /Encoding RefMod…/, 'the busy label stays mounted while idle')
})

test('busy: the labels flip and the button disables', () => {
  const html = btn({ onClick: () => {}, count: 12, busy: true })
  const idle = html.match(/<span [^>]*data-label="idle"[^>]*>/)[0]
  const busy = html.match(/<span [^>]*data-label="busy"[^>]*>/)[0]
  assert.ok(idle.includes('hidden'), 'idle span must hide while busy')
  assert.ok(!busy.includes('hidden'), 'busy span must be visible while busy')
  assert.match(html, /disabled/, 'a running pass disables the button')
})

test('a dataset with no kept images renders nothing at all', () => {
  assert.equal(btn({ onClick: () => {}, count: 0 }), '')
})
