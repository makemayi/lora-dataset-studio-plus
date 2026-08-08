/**
 * Le panneau « Datasets » du ◉ LoRA Canvas s'ouvre REPLIÉ.
 *
 * Il s'ouvrait déplié sur écran large. Sa liste de cases (une par dataset, plus
 * les modèles, les statuts, la recherche) poussait alors le board — ce qu'on
 * vient regarder — sous la ligne de flottaison, à CHAQUE chargement, alors qu'un
 * filtre ne se consulte pas à l'arrivée.
 *
 * Ce qui est épinglé : l'état INITIAL (replié, sans localStorage), le respect du
 * choix mémorisé, et l'absence de la logique de largeur d'écran qui rouvrait le
 * panneau à tout redimensionnement — elle aurait annulé un pli délibéré.
 */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

import {
  CANVAS_FILTER_OPEN_KEY, readCanvasFilterOpen, writeCanvasFilterOpen,
} from '../src/utils/canvasFamilyFilter.js'

const SOURCE = readFileSync(
  new URL('../src/components/canvas/CanvasDatasetFilter.jsx', import.meta.url), 'utf8')

/** Un localStorage de test, injectable comme le reste des helpers du canvas. */
const store = (initial = {}) => {
  const map = new Map(Object.entries(initial))
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    dump: () => Object.fromEntries(map),
  }
}

test('with nothing stored the filter is FOLDED', () => {
  assert.equal(readCanvasFilterOpen(store()), false)
})

test('an explicit unfold survives the reload, an explicit fold too', () => {
  const s = store()
  writeCanvasFilterOpen(s, true)
  assert.equal(s.dump()[CANVAS_FILTER_OPEN_KEY], '1')
  assert.equal(readCanvasFilterOpen(s), true)
  writeCanvasFilterOpen(s, false)
  assert.equal(readCanvasFilterOpen(s), false)
})

test('a junk or absent store never throws and reads as folded', () => {
  assert.equal(readCanvasFilterOpen(null), false)
  assert.equal(readCanvasFilterOpen(undefined), false)
  assert.equal(readCanvasFilterOpen(store({ [CANVAS_FILTER_OPEN_KEY]: 'yes' })), false)
  const hostile = { getItem() { throw new Error('private mode') } }
  assert.equal(readCanvasFilterOpen(hostile), false)
  // …et écrire dans un store qui refuse ne casse pas le clic.
  assert.equal(writeCanvasFilterOpen({ setItem() { throw new Error('quota') } }, true), false)
})

test('the component reads its initial state from the store, not from the viewport', () => {
  assert.match(SOURCE, /useState\(\s*\(\)\s*=>\s*readCanvasFilterOpen\(/)
  assert.match(SOURCE, /writeCanvasFilterOpen\(/)
  // La largeur d'écran ne décide plus de l'état initial ni des suivants : un
  // agrandissement de fenêtre rouvrait le panneau qu'on venait de replier.
  assert.doesNotMatch(SOURCE, /matchMedia/)
  assert.doesNotMatch(SOURCE, /min-width: 640px/)
  assert.doesNotMatch(SOURCE, /useState\(wide\)/)
})
