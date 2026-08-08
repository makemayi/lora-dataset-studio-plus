import assert from 'node:assert/strict';
import test from 'node:test';

import {
  closedRoads, formatCost, formatDuration, formatSize, initialTransport, rateNote,
  transportBlockedReason, transportOption, transportSummary,
} from './podTransportChoice.js';

const hubOpen = {
  transport: 'hub', available: true, bytes: 26e9, seconds: 1130,
  gpu_cost: 0.44, rate_mbps: 200, rate_source: 'floor', rate_samples: 0,
};
const directOpen = {
  transport: 'direct', available: true, bytes: 26e9, seconds: 11040,
  gpu_cost: 4.29, rate_mbps: 20, rate_source: 'measured', rate_samples: 3,
};
const hubClosed = {
  transport: 'hub', available: false,
  reason: 'this run was delivered to this computer only, so no Hugging Face copy of it was ever made.',
};

test('both roads are always addressable, even the one that is missing', () => {
  // A road absent from the UI is a trade-off the user never learns exists.
  const plan = { options: [directOpen] };
  assert.equal(transportOption(plan.options, 'direct'), directOpen);
  assert.deepEqual(transportOption(plan.options, 'hub'),
    { transport: 'hub', available: false, reason: null });
  assert.deepEqual(transportOption(null, 'hub'),
    { transport: 'hub', available: false, reason: null });
});

test('the dialog opens on the backend default when it is usable', () => {
  assert.equal(initialTransport(
    { default_transport: 'hub', options: [hubOpen, directOpen] }), 'hub');
});

test('the dialog never opens on a road it would have to disable', () => {
  // Opening on a disabled option with a disabled button reads as broken software.
  assert.equal(initialTransport(
    { default_transport: 'hub', options: [hubClosed, directOpen] }), 'direct');
});

test('with nothing usable at all it still names a road rather than crashing', () => {
  assert.equal(initialTransport({ default_transport: 'hub', options: [hubClosed] }), 'hub');
  assert.equal(initialTransport(null), 'hub');
});

test('durations round UP, because this number carries a price', () => {
  assert.equal(formatDuration(3540), '59 min');
  assert.equal(formatDuration(3541), '60 min', '59.01 min must not read as 59');
  assert.equal(formatDuration(5400), '1.5 h');
  assert.equal(formatDuration(45), '45 s');
  assert.equal(formatDuration(0), '1 s');
});

test('sizes read the way a person says them', () => {
  assert.equal(formatSize(26e9), '26.0 GB');
  assert.equal(formatSize(85e6), '85 MB');
  assert.equal(formatSize(0), '', 'an unknown size says nothing, never "0 GB"');
});

test('a measured rate says it was measured, and on how much evidence', () => {
  assert.match(rateNote(directOpen), /measured at 20 Mbit\/s on your last 3 transfers/);
  assert.match(rateNote({ ...directOpen, rate_samples: 1 }), /1 transfer\b/,
    'one sample is not "1 transfers"');
});

test('an unmeasured rate ADMITS it and names its assumption', () => {
  // A forecast labelled "measured" that was a guess is worth less than no
  // forecast, because it will be believed.
  const note = rateNote({ rate_mbps: 50, rate_source: 'assumed', rate_samples: 0 });
  assert.match(note, /estimated/);
  assert.match(note, /50 Mbit\/s/);
  // It may say the word (it says nothing HAS been measured yet); what it must
  // never do is CLAIM the number was.
  assert.doesNotMatch(note, /measured at/);
});

test('the pod side is never described as measured on this machine', () => {
  assert.match(rateNote(hubOpen), /a rented pod is required to have/);
  assert.doesNotMatch(rateNote(hubOpen), /measured at/);
});

test('a configured speed is used but not passed off as evidence', () => {
  const note = rateNote({ rate_mbps: 500, rate_source: 'configured' });
  assert.match(note, /set in Settings/);
  assert.doesNotMatch(note, /measured at/);
});

test('the summary states size, duration AND the GPU billed while it waits', () => {
  const line = transportSummary(directOpen);
  assert.match(line, /26\.0 GB/);
  assert.match(line, /3\.1 h/);
  assert.match(line, /\$4\.29 of rented GPU while it waits/);
});

test('the summary of an unavailable or unmeasured road says nothing at all', () => {
  // A row of zeroes reads as a real forecast of zero.
  assert.equal(transportSummary(hubClosed), '');
  assert.equal(transportSummary({ transport: 'hub', available: true, bytes: 0, seconds: 0 }), '');
});

test('a free pod shows no price rather than "$0.00"', () => {
  assert.equal(formatCost(0), '');
  const line = transportSummary({ ...hubOpen, gpu_cost: 0 });
  assert.match(line, /26\.0 GB/);
  assert.doesNotMatch(line, /\$/);
});

test('a blocked road hands back the backend reason verbatim', () => {
  const plan = { options: [hubClosed, directOpen] };
  assert.equal(transportBlockedReason(plan, 'hub'), hubClosed.reason);
  assert.equal(transportBlockedReason(plan, 'direct'), null);
});

test('a blocked road with no stated reason still says something', () => {
  const plan = { options: [{ transport: 'hub', available: false }] };
  assert.match(transportBlockedReason(plan, 'hub'), /Hugging Face/);
});

test('no plan at all blocks nothing — the dialog behaves as it always did', () => {
  assert.equal(transportBlockedReason(null, 'hub'), null);
});

test('a closed road explains itself on screen, selected or not', () => {
  // The whole complaint this feature answers is an invisible trade-off. A
  // `title` tooltip would reproduce it one level down.
  const rows = closedRoads({ options: [hubClosed, directOpen] });
  assert.equal(rows.length, 1);
  assert.equal(rows[0].transport, 'hub');
  assert.equal(rows[0].label, '☁ Hugging Face');
  assert.equal(rows[0].reason, hubClosed.reason);
});

test('a road with both roads open explains nothing, and a plan-less dialog neither', () => {
  assert.deepEqual(closedRoads({ options: [hubOpen, directOpen] }), []);
  assert.deepEqual(closedRoads(null), []);
  // A closed road with no stated reason is not rendered as an empty warning.
  assert.deepEqual(closedRoads({ options: [{ transport: 'hub', available: false }] }), []);
});
