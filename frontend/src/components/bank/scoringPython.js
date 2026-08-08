/* "Use a GPU Python you already have" — the decidable half, JSX-free so
 * `node --test` can import it.
 *
 * ✨ Score ships CPU-only PyTorch on purpose, and on a machine with a card that
 * costs hours. The fix is NOT to download a 2.5 GB CUDA wheel from inside the
 * app (wrong wheel index = a broken environment, and it is a big download for
 * people who may not need it). It is to reuse an interpreter this machine has
 * already proven — the one ComfyUI runs on, the one that trains LoRAs.
 *
 * The whole value is in being specific. "ai-toolkit: no" is useless; "ai-toolkit
 * has CUDA but is missing OpenCLIP, here is the command" is actionable. So every
 * row carries a per-dependency verdict and the backend refuses anything it could
 * not prove — a wrong pick would surface as an import error an hour into a pass.
 */

/** The two features that can borrow an interpreter, and the ONLY thing that
 *  differs between them: which endpoint answers, what the sentences call the
 *  pass, and what it needs. Same dialog, same rules — a second copy of this
 *  screen would be a second place to fix the next wording bug.
 *
 *  `key` is a route/profile id shared with the backend (`scoring_python.PROFILES`)
 *  and is NOT a display string: never rename it. */
export const PICKER_PROFILES = {
  scoring: {
    key: 'scoring',
    endpoint: '/api/scoring-python',
    feature: '✨ Score',
    needs: 'PyTorch, OpenCLIP and a couple of others',
    ariaLabel: 'Choose the Python that runs Score',
    inputId: 'scoring-python-path',
    extraNote: '',
  },
  semantic: {
    key: 'semantic',
    endpoint: '/api/semantic-python',
    feature: 'the SigLIP 2 index',
    needs: 'PyTorch, a SigLIP2-capable Transformers, NumPy and Pillow',
    ariaLabel: 'Choose the Python that runs the SigLIP 2 index',
    inputId: 'semantic-python-path',
    // The reason borrowing is cheap here and nowhere else: the checkpoint is not
    // in the interpreter. Someone who just paid 1.5 GB for it needs to be told
    // they will not pay it twice.
    extraNote: 'The pinned SigLIP 2 weights live in the app’s data folder, not '
      + 'in the interpreter — borrowing one downloads nothing and installs nothing. '
      + 'Setup ▸ Quality tools keeps installing into the app’s own environment.',
  },
}

export const DEFAULT_PICKER = PICKER_PROFILES.scoring

/** Accept a profile object, its key, or nothing (Score — the original caller). */
export function pickerProfile(profile) {
  if (profile && typeof profile === 'object') return profile
  return PICKER_PROFILES[profile] || DEFAULT_PICKER
}

/** Best first: a ready GPU interpreter, then a working CPU one, then the ones
 *  that need something, then the ones that did not answer. Ties keep the
 *  backend's order (the interpreter in use first, the app's own last). */
const RANK = { gpu_ready: 0, cpu_only: 1, incomplete: 2, unreachable: 3 }

export function sortInterpreters(rows) {
  return [...(rows || [])].sort(
    (a, b) => (RANK[a.status] ?? 9) - (RANK[b.status] ?? 9))
}

/** Badge wording + tone per status. 'ok' green, 'warn' amber, 'off' muted. */
export function statusBadge(status) {
  switch (status) {
    case 'gpu_ready': return { tone: 'ok', label: 'GPU ready' }
    case 'cpu_only': return { tone: 'warn', label: 'CPU only' }
    case 'incomplete': return { tone: 'warn', label: 'Missing packages' }
    default: return { tone: 'off', label: 'No answer' }
  }
}

/** The one interpreter worth suggesting: a GPU-ready one that isn't already the
 *  selected one. null when there is nothing better than what is in use. */
export function bestUpgrade(rows) {
  return (sortInterpreters(rows).find((r) => r.status === 'gpu_ready' && !r.selected)) || null
}

/** Can the user pick this row? Only interpreters proven able to run the whole
 *  pass — the backend enforces the same rule; this just greys the button. */
export function canSelect(row) {
  return Boolean(row && row.usable && !row.selected)
}

/** The names of what's missing, for a sentence like "missing OpenCLIP, timm". */
export function missingLabels(row) {
  return (row?.deps || []).filter((d) => !d.present).map((d) => d.label)
}

/** One honest line under the dialog title. Every machine gets the sentence that
 *  is TRUE FOR IT — that specificity is the whole feature; an opaque "no" is
 *  what makes someone give up. The five situations, in the order they matter:
 *
 *    1. no NVIDIA card at all — there is nothing to fix and nothing to sell;
 *       say so and stop. Never mention CUDA to a machine that has none.
 *    2. nothing to check yet — the honest state of a fresh install.
 *    3. one or more are ready — the good case, counted.
 *    4. one reaches the GPU but lacks packages — name the interpreter AND the
 *       packages, because that is the sentence someone can act on.
 *    5. none reaches the GPU — plainly, without inviting a hunt.
 *
 *  `nvidiaPresent` defaults to true so a caller that hasn't loaded it yet gets
 *  the old wording rather than wrongly telling someone they have no card. */
export function detectionSummary(rows, nvidiaPresent = true, profile = DEFAULT_PICKER) {
  const { feature } = pickerProfile(profile)
  const list = rows || []
  const usable = list.filter((r) => r.usable)
  if (!nvidiaPresent) {
    // A card-less machine can still SKIP the install by borrowing an
    // interpreter that already has the packages — worth offering, worth being
    // honest that it changes nothing about speed.
    const offer = usable.length
      ? ` ${usable.length} interpreter${usable.length === 1 ? '' : 's'} here already `
        + `${usable.length === 1 ? 'has' : 'have'} the packages, if you would rather not install them again.`
      : ''
    return `No NVIDIA card detected on this machine — ${feature} runs on the CPU either way.${offer}`
  }
  if (!list.length) return 'No Python interpreters found to check yet.'
  const ready = list.filter((r) => r.status === 'gpu_ready')
  if (ready.length) {
    return `${ready.length} of ${list.length} can run ${feature} on your GPU.`
  }
  const close = list.filter((r) => r.status === 'incomplete' && r.cuda)
  if (close.length) {
    const names = close.map((r) => `${r.label} (${missingLabels(r).join(', ')})`)
    return `None is ready yet. Reaches the GPU but needs packages: ${names.join('; ')}.`
  }
  return `None of these can reach the GPU — ${feature} stays on the CPU. `
    + 'If you have another Python with a CUDA PyTorch, enter its path below.'
}

/** The banner shown when the SEARCH itself broke, not when it found nothing.
 *
 *  Those two used to be the same screen: a crash returned an empty list, which
 *  is also the honest verdict "there is nothing to borrow on this machine".
 *  Someone reading "No Python interpreters found to check yet" has no reason to
 *  press ↻ Check again — and retrying is exactly what would fix a transient
 *  failure. null when the detection ran (whatever it found). */
export function detectionFailure(result) {
  if (!result?.detection_failed) return null
  return {
    title: 'Could not look for interpreters',
    text: 'Something went wrong while checking this machine, so the list below is '
      + 'empty because the search failed — not because there is nothing to find. '
      + '↻ Check again often clears it.',
    detail: (result.detection_error || '').trim(),
  }
}

/** Title + intro for the dialog, adapted to the machine. Same rule as above: a
 *  machine with no NVIDIA card is never shown a CUDA pitch. */
export function dialogCopy(nvidiaPresent = true, profile = DEFAULT_PICKER) {
  const { feature, needs } = pickerProfile(profile)
  if (!nvidiaPresent) {
    return {
      title: `⚡ Run ${feature} in a Python you already have`,
      intro: `${feature} needs ${needs}. If another `
        + 'Python on this machine already carries them, it can run the pass — no '
        + 'second install. Nothing is ever installed into those environments: '
        + 'they are checked, never changed.',
    }
  }
  return {
    title: `⚡ Run ${feature} on a GPU Python you already have`,
    intro: 'If this machine already has a working CUDA PyTorch — the one that '
      + `trains your LoRAs, or the one ComfyUI runs on — ${feature} can borrow it `
      + 'instead of downloading another. Nothing is ever installed into those '
      + 'environments: they are checked, never changed.',
  }
}

/** The label of the button that opens the picker. A machine with no NVIDIA card
 *  must not be offered "a GPU Python" — it would be a promise we cannot keep. */
export function openerLabel(nvidiaPresent = true) {
  return nvidiaPresent
    ? '⚡ Use a GPU Python I already have'
    : '⚡ Use a Python I already have'
}

/** What to tell someone who just typed a path and pressed Check it. Silence is
 *  the wrong answer for the route most installs depend on — and the sneaky case
 *  is a path that resolves onto an interpreter ALREADY in the list, where the
 *  screen would otherwise look unchanged. null when nothing was entered. */
export function enteredNote(result) {
  const status = result?.entered_status
  if (!status) return null
  if (status === 'no_interpreter') {
    return {
      tone: 'warn',
      text: 'That folder exists, but holds nothing that looks like a Python '
        + 'interpreter. Point at the interpreter itself, or at the environment '
        + 'folder that contains it.',
    }
  }
  const hit = (result.interpreters || []).find((r) => r.entered)
  if (!hit) return null
  if (hit.source !== 'manual') {
    return { tone: 'info', text: `That is the one already listed as “${hit.label}” — ${hit.detail}` }
  }
  return { tone: 'info', text: `Checked: ${hit.detail}` }
}

/** What the Score panel says about the current interpreter, once one has been
 *  chosen explicitly. null when the app default is in use (the CPU note already
 *  covers that case, and saying it twice is noise). */
export function selectionNote(result, profile = DEFAULT_PICKER) {
  const rows = result?.interpreters || []
  const current = rows.find((r) => r.selected)
  if (!current) return null
  return `${pickerProfile(profile).feature} runs in ${current.label} — ${current.detail}`
}
