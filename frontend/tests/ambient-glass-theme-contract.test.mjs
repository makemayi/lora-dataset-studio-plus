import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { CARD_SURFACE } from '../src/components/common/surfaces.js'
import tailwindConfig from '../tailwind.config.js'

const here = dirname(fileURLToPath(import.meta.url))
const frontend = join(here, '..')
const read = (rel) => readFileSync(join(frontend, rel), 'utf8')

test('the page ground carries a static ambient-light veil', () => {
  const css = read('src/index.css')
  assert.match(css, /body\s*\{[^}]*background-image:[^}]*radial-gradient/s)
  assert.match(css, /body\s*\{[^}]*background-attachment:\s*fixed/s)
  // static — no animation on the ground, so reduced-motion is satisfied by construction
  assert.doesNotMatch(css, /body\s*\{[^}]*animation:/s)
})

test('cards are solid white with a 16px radius and a soft diffuse shadow', () => {
  assert.match(CARD_SURFACE, /bg-surface/)
  assert.match(CARD_SURFACE, /rounded-2xl/)
  assert.match(CARD_SURFACE, /shadow-\[/)
})

test('the primary button gradient layers a top sheen over the base', () => {
  const g = tailwindConfig.theme.extend.backgroundImage['gradient-primary']
  assert.ok(g.includes('rgba(255 255 255 / 0.10)'), g)
  assert.equal((g.match(/linear-gradient/g) || []).length, 2)
})

test('the app header is translucent glass', () => {
  assert.match(read('src/App.jsx'), /bg-surface-overlay\/80\s+backdrop-blur-md/)
})

const GLASS = /bg-surface-overlay\/85\s+backdrop-blur-md/

// Floating overlays that must carry the glass recipe. Grown per task.
const OVERLAY_FILES = [
  'src/components/common/HeaderMenu.jsx',
  'src/components/common/WhatsNew.jsx',
  'src/components/common/FolderPicker.jsx',
  'src/components/bank/DeleteRejectedDialog.jsx',
  'src/components/bank/LaunchAllDialog.jsx',
  'src/components/bank/PassDialog.jsx',
  'src/components/bank/PersonPreflightDialog.jsx',
  'src/components/bank/PromoteDialog.jsx',
  'src/components/bank/RelocateBankDialog.jsx',
  'src/components/bank/ScoringPythonDialog.jsx',
  'src/components/dataset/DatasetSettingsModal.jsx',
  'src/components/dataset/DatasetToBankDialog.jsx',
  'src/components/dataset/PublishHfModal.jsx',
  'src/components/dataset/ReferenceEditModal.jsx',
  'src/components/dataset/PromptEditPopover.jsx',
  'src/components/dataset/CaptionOptionsPopover.jsx',
  'src/components/dataset/CheckpointActionsPopover.jsx',
  'src/components/dataset/DatasetGrid.jsx',
  'src/components/dataset/DatasetWorkspace.jsx',
  'src/components/dataset/FullBackupControls.jsx',
  'src/components/dataset/studio/DatasetCaptionControl.jsx',
  'src/components/dataset/studio/DescribeImageModal.jsx',
  'src/components/dataset/studio/ExportGridModal.jsx',
  'src/components/shared/RunDeleteSection.jsx',
  'src/components/settings/KleinLoraCombobox.jsx',
  'src/components/settings/ModelFilePicker.jsx',
  'src/components/shared/CheckpointGalleryPanel.jsx',
  'src/components/videobank/PromoteVideoDialog.jsx',
]

test('floating overlays use the glass recipe', () => {
  const offenders = OVERLAY_FILES.filter((rel) => !GLASS.test(read(rel)))
  assert.deepEqual(offenders, [],
    'floating overlays must use bg-surface-overlay/85 backdrop-blur-md:\n' + offenders.join('\n'))
})

test('multi-panel files migrate every floating panel', () => {
  const bank = read('src/components/bank/BankWorkspace.jsx')
  assert.equal(bank.split('bg-surface-overlay/85 backdrop-blur-md').length - 1, 5, bank)
  const training = read('src/components/dataset/TrainingPanel.jsx')
  assert.equal(training.split('bg-surface-overlay/85 backdrop-blur-md').length - 1, 2, training)
})
