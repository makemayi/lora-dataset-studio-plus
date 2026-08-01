import assert from 'node:assert/strict';
import test from 'node:test';

import {
  boundedCanvasSize, containRect, nextTimelineIndex, orderTimelineFrames,
  orderTimelineSeries, pickWebMMimeType, timelineEndpoints, timelineFrameLabel,
  timelineGifError, timelineGifUrl, timelineLimitMessage, timelineListUrl, timelineSeriesLabel,
  timelineStepLabel,
  webmSourceLimit, withinWebMByteBudget, TIMELINE_PLAYBACK_MODES,
} from './checkpointTimeline.js';

test('timeline endpoints stay inside one run and encode identifiers', () => {
  const seriesId = 'a'.repeat(64);
  assert.equal(timelineListUrl(42), '/api/train/run/42/timeline');
  assert.equal(timelineGifUrl(42, seriesId),
    `/api/train/run/42/timeline/${seriesId}/gif`);
  assert.deepEqual(timelineEndpoints(42, seriesId), {
    list: '/api/train/run/42/timeline',
    gif: `/api/train/run/42/timeline/${seriesId}/gif`,
  });
  assert.equal(timelineEndpoints(null), null);
  assert.equal(timelineGifUrl(42, null), null);
  assert.equal(timelineListUrl(0), null);
  assert.equal(timelineListUrl('9223372036854775808'), null);
  assert.equal(timelineGifUrl(42, 'not-a-digest'), null);
});

test('GIF failures are actionable and preserve a safe server explanation', () => {
  assert.equal(timelineGifError(429),
    'The GIF renderer is busy. Wait a moment and try again.');
  assert.equal(timelineGifError(413), 'This timeline is too large to export as a GIF.');
  assert.equal(timelineGifError(404),
    'This timeline is no longer available. Refresh it and try again.');
  assert.equal(timelineGifError(500), 'Could not export this timeline as a GIF (HTTP 500).');
  assert.equal(timelineGifError(429, 'Please retry in one second.'),
    'Please retry in one second.');
});

test('frames order by numeric step with stable, honest unknowns at the end', () => {
  const ordered = orderTimelineFrames([
    { id: 9, step: null, created_at: '2026-02-03T00:00:00Z' },
    { id: 4, step: 2000, created_at: '2026-02-02T00:00:00Z' },
    { id: 2, step: 500, created_at: '2026-02-01T00:00:00Z' },
    { id: 3, step: '500', created_at: '2026-02-02T00:00:00Z' },
  ]);
  assert.deepEqual(ordered.map((frame) => frame.id), [2, 3, 4, 9]);
  assert.deepEqual(orderTimelineFrames(null), []);
});

test('series order newest first and derive missing metadata from frames', () => {
  const series = orderTimelineSeries([
    { id: 1, frames: [{ id: 1, step: 1000, created_at: '2026-02-01T00:00:00Z' }] },
    { id: 2, frames: [
      { id: 3, step: 2000, created_at: '2026-03-02T00:00:00Z' },
      { id: 2, step: 500, created_at: '2026-03-01T00:00:00Z' },
    ] },
  ]);
  assert.deepEqual(series.map((item) => item.id), [2, 1]);
  assert.deepEqual(series[0].steps, [500, 2000]);
  assert.equal(series[0].created_at, '2026-03-01T00:00:00Z');
  assert.equal(series[0].frame_count, 2);
  assert.deepEqual(orderTimelineSeries({ series }).map((item) => item.id), [2, 1]);
});

test('top-level timeline caps are reported instead of looking complete', () => {
  assert.equal(timelineLimitMessage({ truncated: false, count: 30, shown: 20 }), null);
  assert.equal(timelineLimitMessage({
    truncated: true,
    count: 30,
    shown: 20,
    candidate_count: 2500,
    candidates_scanned: 2000,
    frame_count: 2100,
    frames_shown: 1200,
  }), '20 of 30 preview series shown. 2000 of 2500 candidate frames scanned. '
    + '1200 of 2100 comparable frames shown.');
  assert.equal(timelineLimitMessage({ truncated: true }),
    'Some timeline results were omitted by server safety limits.');
});

test('loop wraps while ping-pong reverses at both ends', () => {
  assert.deepEqual(nextTimelineIndex(2, 3, 1, TIMELINE_PLAYBACK_MODES.LOOP),
    { index: 0, direction: 1 });
  assert.deepEqual(nextTimelineIndex(0, 3, -1, TIMELINE_PLAYBACK_MODES.LOOP),
    { index: 2, direction: -1 });
  assert.deepEqual(nextTimelineIndex(2, 3, 1, TIMELINE_PLAYBACK_MODES.PING_PONG),
    { index: 1, direction: -1 });
  assert.deepEqual(nextTimelineIndex(0, 3, -1, TIMELINE_PLAYBACK_MODES.PING_PONG),
    { index: 1, direction: 1 });
  assert.deepEqual(nextTimelineIndex(9, 1), { index: 0, direction: 1 });
});

test('labels state the step, counter, conditions and frame count', () => {
  assert.equal(timelineStepLabel(1250), 'Step 1,250');
  assert.equal(timelineStepLabel(null), 'Step unknown');
  assert.equal(timelineFrameLabel({ step: 1250 }, 1, 4), 'Step 1,250 · 2 of 4');
  assert.equal(timelineSeriesLabel({
    conditions: { seed: 123, strength: 0.85, aspect: '4:3', prompt: 'a very long secret prompt',
      sampler: 'this must not make the option longer' },
    frame_count: 4,
  }), 'Series 1 · Seed 123 · strength 0.85 · 4:3 · 4 frames');
  assert.doesNotMatch(timelineSeriesLabel({ conditions: { prompt: 'portrait' }, frame_count: 2 }),
    /portrait/);
  assert.equal(timelineSeriesLabel({ frames: [{}] }, 2), 'Series 3 · 1 frame');
  assert.notEqual(
    timelineSeriesLabel({ conditions: { seed: 1, strength: 1 }, frame_count: 2 }, 0),
    timelineSeriesLabel({ conditions: { seed: 1, strength: 1 }, frame_count: 2 }, 1),
  );
});

test('WebM mime selection prefers VP9, falls back to VP8, and never offers MP4', () => {
  class VP8Recorder {
    static isTypeSupported(mime) { return mime.includes('vp8'); }
  }
  assert.equal(pickWebMMimeType(VP8Recorder), 'video/webm;codecs=vp8');
  assert.equal(pickWebMMimeType(undefined), null);
  class UnsupportedRecorder {
    static isTypeSupported() { return false; }
  }
  assert.equal(pickWebMMimeType(UnsupportedRecorder), null);
});

test('contain geometry letterboxes without cropping and canvas size is bounded', () => {
  assert.deepEqual(containRect(1600, 800, 1000, 1000),
    { x: 0, y: 250, width: 1000, height: 500 });
  assert.deepEqual(containRect(800, 1600, 1000, 500),
    { x: 375, y: 0, width: 250, height: 500 });
  assert.deepEqual(containRect(0, 100, 500, 500),
    { x: 0, y: 0, width: 0, height: 0 });
  assert.deepEqual(boundedCanvasSize(2400, 1200, 1200), { width: 1200, height: 600 });
  assert.deepEqual(boundedCanvasSize(640, 480, 1200), { width: 640, height: 480 });
});

test('WebM source count obeys both decoded-memory and capture-frame budgets', () => {
  assert.equal(webmSourceLimit(60, 2, 2, 600, 16), 16);
  assert.equal(webmSourceLimit(60, 10, 8, 100, 16), 6);
  assert.equal(webmSourceLimit(8, 10, 8, 600, 16), 8);
  assert.equal(webmSourceLimit(0, 10, 8, 600, 16), 0);
});

test('WebM chunks stop at the aggregate byte budget', () => {
  assert.equal(withinWebMByteBudget(60, 4, 64), true);
  assert.equal(withinWebMByteBudget(60, 5, 64), false);
  assert.equal(withinWebMByteBudget(-1, 1, 64), false);
});
