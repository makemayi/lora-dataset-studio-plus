import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const app = readFileSync(new URL('../../App.jsx', import.meta.url), 'utf8')
const page = readFileSync(new URL('../../pages/BankPage.jsx', import.meta.url), 'utf8')
const workspace = readFileSync(new URL('./BankWorkspace.jsx', import.meta.url), 'utf8')
const overview = readFileSync(new URL('./BankOverview.jsx', import.meta.url), 'utf8')

test('/bank shares the wide 1800px shell with Canvas', () => {
  assert.match(app, /pathname === '\/canvas' \|\| pathname === '\/bank'/)
  assert.match(app, /max-w-\[1800px\]/)
})

test('bank list grows to three columns only at xl', () => {
  assert.match(page, /grid-cols-1 sm:grid-cols-2 xl:grid-cols-3/)
})

test('the four-megapixel resolution bucket is inclusive in the workspace too', () => {
  assert.match(workspace, /id: 'res_gt_4', label: '≥ 4 MP'/)
  assert.doesNotMatch(workspace, /id: 'res_gt_4', label: '> 4 MP'/)
})

test('workspace stays stacked on mobile and uses a twelve-column xl layout', () => {
  const grids = workspace.match(/grid gap-4 xl:grid-cols-12 xl:items-start/g) || []
  assert.equal(grids.length, 2)
  assert.match(workspace, /xl:col-span-7/)
  assert.match(workspace, /xl:col-span-5 xl:sticky xl:top-20/)
  assert.match(workspace, /xl:col-span-8/)
  assert.match(workspace, /xl:col-span-4 xl:sticky xl:top-20/)
  assert.match(workspace, /<BankOverview payload=\{payload\} \/>/)
})

test('overview never opens the expensive kept-only coverage endpoint', () => {
  const overviewMount = workspace.slice(workspace.indexOf('<BankOverview'),
    workspace.indexOf('<BankOverview') + 200)
  assert.doesNotMatch(overviewMount, /coverage/)
})

test('non-zero Bank segments use exact widths and remain physically visible', () => {
  assert.match(overview, /width: `\$\{row\.widthPercent\}%`/)
  assert.match(overview, /minWidth: row\.value > 0 \? '1px'/)
  assert.match(page, /width: `\$\{row\.widthPercent\}%`, minWidth: '1px'/)
  assert.doesNotMatch(page, /width: `\$\{row\.percent\}%`/)
})

test('the overview is open by default and folds without tying its state to live payload refreshes', () => {
  assert.match(overview, /const \[open, setOpen\] = useState\(true\)/)
  assert.match(overview, /onClick=\{\(\) => setOpen\(\(value\) => !value\)\}/)
  assert.match(overview, /aria-expanded=\{open\}/)
  assert.match(overview, /aria-controls=\{contentId\}/)
  assert.match(overview, /<div id=\{contentId\} hidden=\{!open\}/)
  assert.doesNotMatch(overview, /useState\([^)]*payload/)
})

test('the overview header and live total stay visible while its details fold', () => {
  const detailsAt = overview.indexOf('<div id={contentId}')
  assert.ok(detailsAt > 0)
  assert.ok(overview.indexOf('📊 Bank overview') < detailsAt)
  assert.ok(overview.indexOf('{totalText}') < detailsAt)
})

test('the overview has an explicit unavailable state before bank data arrives', () => {
  assert.match(overview, /!model\.available \? \(/)
  assert.match(overview, /Overview unavailable — waiting for bank data/)
  assert.match(overview, /'Total unavailable'/)
})
