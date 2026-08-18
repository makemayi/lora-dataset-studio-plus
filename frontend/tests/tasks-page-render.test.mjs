import { test } from 'node:test'
import assert from 'node:assert/strict'
import { render } from './support/mountJsx.mjs'

// The JSX loader is registered while mountJsx is evaluated, so these imports
// have to stay dynamic (see support/mountJsx.mjs).
const { TaskRow } = await import('../src/components/tasks/TaskRow.jsx')
const { StatusStrip } = await import('../src/components/tasks/StatusStrip.jsx')

test('TaskRow renders every status branch — paused, failed, running, queued', () => {
  const cases = [
    { status: 'awaiting_comfyui', expect: 'Paused · waiting for ComfyUI' },
    { status: 'failed', expect: 'Failed' },
    { status: 'processing', expect: 'Running' },
    { status: 'pending', expect: 'Queued' },
  ]
  for (const c of cases) {
    const html = render(TaskRow, { task: { job_id: 'x', title: 'klein_edit',
      status: c.status, actions: [], resource: { type: 'dataset', dataset_id: 1 } } })
    assert.ok(html.includes(c.expect), `${c.status}: ${c.expect} missing`)
  }
})

test('TaskRow shows retry on failed and cancel on active', () => {
  const failed = render(TaskRow, { task: { job_id: 'a', title: 'x', status: 'failed',
    actions: ['retry'], resource: null } })
  assert.ok(failed.includes('Retry'))
  const active = render(TaskRow, { task: { job_id: 'b', title: 'x',
    status: 'awaiting_comfyui', actions: ['cancel'], resource: null } })
  assert.ok(active.includes('Cancel'))
})

test('TaskRow deep-link chip names the image when the resource is an image', () => {
  const html = render(TaskRow, { task: { job_id: 'c', title: 'seedvr2_upscale',
    status: 'completed', actions: [], resource: { type: 'image', dataset_id: 7, image_id: 42 } } })
  assert.ok(html.includes('Image #42'), 'image resource chip must name the image')
})

test('StatusStrip renders offline ComfyUI with paused count', () => {
  const html = render(StatusStrip, { comfyui: { reachable: false },
    summary: { queued: 2, paused: 5, running: 1, today_done: 34, today_failed: 1 },
    training: false, vision: false })
  assert.ok(html.includes('Offline'))
  assert.ok(html.includes('5'))
})

test('StatusStrip shows training card when the GPU is busy training', () => {
  const html = render(StatusStrip, { comfyui: { reachable: true },
    summary: { queued: 0, paused: 0, running: 0, today_done: 0, today_failed: 0 },
    training: true, vision: false })
  assert.ok(html.includes('Training in progress'))
})
