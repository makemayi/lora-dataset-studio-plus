import test from 'node:test'
import assert from 'node:assert/strict'

import {
  videoClipThumbUrl, videoSourceMediaUrl, videoBankUrl, videoClipsUrl,
  videoPassUrl, videoDatasetUrl,
  videoClipBoundsUrl, videoClipSplitUrl, videoSourceClipsUrl, videoSearchUrl,
} from './videoBankApi.js'

test('thumb and media URLs address a clip and its SOURCE', () => {
  assert.equal(videoClipThumbUrl(3, 41), '/api/video-bank/3/clip/41/thumb')
  // Per-source, not per-clip: no clip file exists until promotion.
  assert.equal(videoSourceMediaUrl(3, 7), '/api/video-bank/3/source/7/media')
})

test('the poll URL only asks for a folder re-walk when told to', () => {
  assert.equal(videoBankUrl(9), '/api/video-bank/9')
  assert.equal(videoBankUrl(9, { refresh: true }), '/api/video-bank/9?refresh=1')
})

test('an "all statuses" filter sends NO status parameter', () => {
  // The server filters on `status in TRIAGE_STATUSES`; `status=all` would be
  // ignored and return everything while the UI claimed to have filtered.
  assert.equal(videoClipsUrl(1, { status: 'all' }), '/api/video-bank/1/clips?limit=200')
  assert.equal(videoClipsUrl(1, { status: null }), '/api/video-bank/1/clips?limit=200')
  assert.equal(videoClipsUrl(1, { status: 'keep' }),
    '/api/video-bank/1/clips?status=keep&limit=200')
})

test('a source filter is dropped when it is falsy', () => {
  assert.equal(videoClipsUrl(1, { sourceId: null }), '/api/video-bank/1/clips?limit=200')
  assert.equal(videoClipsUrl(1, { sourceId: 0 }), '/api/video-bank/1/clips?limit=200')
  assert.equal(videoClipsUrl(1, { sourceId: 12 }),
    '/api/video-bank/1/clips?source_id=12&limit=200')
})

test('paging rides along, and offset 0 is left implicit', () => {
  assert.equal(videoClipsUrl(1, { offset: 0, limit: 60 }), '/api/video-bank/1/clips?limit=60')
  assert.equal(videoClipsUrl(1, { offset: 60, limit: 60 }),
    '/api/video-bank/1/clips?offset=60&limit=60')
})

test('ids_only drops the paging it would contradict', () => {
  // The server answers the WHOLE filter for ids_only; carrying limit=200 next to
  // it reads as a paged answer and is how "select all" silently selects 200.
  const url = videoClipsUrl(4, { status: 'pending', offset: 200, limit: 50, idsOnly: true })
  assert.equal(url, '/api/video-bank/4/clips?status=pending&ids_only=1')
  assert.ok(!url.includes('limit'))
  assert.ok(!url.includes('offset'))
})

test('pass and dataset URLs', () => {
  assert.equal(videoPassUrl(2, 'pipeline'), '/api/video-bank/2/pipeline')
  assert.equal(videoPassUrl(2, 'cancel'), '/api/video-bank/2/cancel')
  assert.equal(videoDatasetUrl(5), '/api/video-dataset/5')
})

test('a search carries its phrase encoded, punctuation and all', () => {
  // "-hat" and "&" are part of the query grammar and of ordinary English; a
  // hand-built query string mangles both.
  assert.equal(videoSearchUrl(3, { q: 'a red car -hat' }),
    '/api/video-bank/3/search?q=a+red+car+-hat&n=60')
  assert.ok(videoSearchUrl(3, { q: 'rock & roll' }).includes('rock+%26+roll'))
})

test('a search inside one triage bucket sends the bucket, and "all" sends none', () => {
  assert.ok(videoSearchUrl(3, { q: 'car', status: 'keep' }).includes('status=keep'))
  assert.ok(!videoSearchUrl(3, { q: 'car', status: 'all' }).includes('status'))
  assert.ok(!videoSearchUrl(3, { q: 'car', status: null }).includes('status'))
})

test('the retouch URLs hang off the right thing', () => {
  // A new shot is created under the SOURCE and not under a clip: the whole point
  // is that there is no existing shot to hang it off — the detector missed it.
  assert.equal(videoClipBoundsUrl(3, 77), '/api/video-bank/3/clip/77/bounds')
  assert.equal(videoClipSplitUrl(3, 77), '/api/video-bank/3/clip/77/split')
  assert.equal(videoSourceClipsUrl(3, 12), '/api/video-bank/3/source/12/clips')
})
