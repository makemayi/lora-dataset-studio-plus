/**
 * The plan panel, RENDERED. Reuse is a decision with a cost — "50 of these 90
 * come from a picture used more than once" — so the sentence that says it has to
 * survive being drawn, in both states. Both variants stay mounted and flip with
 * `hidden`: a ternary that unmounts one is what Chrome auto-translate turns into
 * a removeChild crash.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { BuildPlanPanel } = await import(
  '../src/components/bank/PromoteDialog.jsx')

const panel = (plan) => renderToStaticMarkup(createElement(BuildPlanPanel, { plan }))

test('a plan that needs no reuse says so, and the reuse line stays mounted', () => {
  const html = panel({ usable: 120, with_face_box: 120, unranked: 0,
    counts: { face: 30, half: 30, full: 30 }, total: 90, reused: 0, shortfall: {} })
  assert.match(html, /90/)
  assert.match(html, /120 pictures/)
  assert.match(html, /hidden=""[^>]*>[^<]*used more than once/)
})

test('a plan that reuses pictures names how many', () => {
  const html = panel({ usable: 40, with_face_box: 34, unranked: 0,
    counts: { face: 30, half: 30, full: 30 }, total: 90, reused: 40, shortfall: {} })
  assert.match(html, /40[^<]*used more than once/)
})

test('a shortfall is stated before the run, not discovered after it', () => {
  const html = panel({ usable: 10, with_face_box: 10, unranked: 0,
    counts: { face: 10, half: 10, full: 10 }, total: 30, reused: 10,
    shortfall: { face: 20, half: 20, full: 20 } })
  assert.match(html, /30/)
  assert.match(html, /short/i)
})

test('no plan yet does not throw', () => {
  assert.equal(typeof panel(null), 'string')
})
