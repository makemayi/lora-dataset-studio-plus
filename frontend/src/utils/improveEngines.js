/* Which engine runs the ✨ Upscale & improve pass, and how the UI says what the
   difference IS.

   The whole point of issue #32 (SurpassHR) is that the two passes are not two
   qualities of the same thing: Klein REWRITES (a diffusion edit that re-renders
   skin and micro-detail — it fixes a soft photo and it changes it), SeedVR2
   RESTORES (one-step super-resolution that leaves the content alone). Someone
   picking blind will pick wrong on a dataset built around an exact look, so the
   one-line positioning lives here, next to the ids, rather than being retyped in
   each surface where it could drift apart.

   Pure module, no JSX — the contract tests run under `node --test`. */

export const IMPROVE_ENGINES = [
  {
    /* The id stays 'klein' FOREVER: it is written into derivation_kind and
       improve_engine on rows already in people's databases (CLAUDE.md ▸ never
       rename a stored key without an alias path). Everything the user READS
       changed, because the pipeline behind it did.

       It has not run Flux.2 Klein 9B since the Krea2-Ostris swap. It is now
       Krea 2 Ostris Edit (a small tone/sharpness pass) -> ColorTransfer (the
       edit recoloured back to the ORIGINAL) -> SeedVR2 tiled restore — see
       backend/app/services/krea_hq_helper.py.

       The old copy said "skin and colour can shift", which was true of Klein
       and is precisely the defect this pipeline was built to remove: the
       ColorTransfer stage exists to put the original's tone back. So the label
       named a thing that no longer runs and warned about a fault that had been
       fixed — which is why the user, who wrote the workflow, could not find
       their own pipeline in the app. */
    id: 'klein',
    label: 'Krea 2 + SeedVR2',
    emoji: '✨',
    action: 'Improve via Krea 2 + SeedVR2',
    /* Says what it CHANGES and what it PROTECTS — the second half is the whole
       reason this pipeline replaced the Klein one. */
    summary: 'A small Krea 2 tone pass, recoloured back to the original, then '
      + 'SeedVR2 restores the detail. Sharper without the colour drift.',
    confirm: 'Create a separate Krea 2 + SeedVR2 improvement candidate',
  },
  {
    /* Added 2026-08-16. The Krea2-Ostris swap above took the 'klein' id with it,
       which left the app with no entrance to Flux.2 Klein 9B on this lane at
       all — the two remaining choices were one restorer and one Krea 2 rewrite.
       That is the gap this fills, and it is why the id is `klein_hq` rather
       than a reclaimed 'klein': the old id is spoken for by rows already in
       people's databases and can never mean something else. */
    id: 'klein_hq',
    label: 'Klein 9B',
    emoji: '🩹',
    action: 'Improve via Flux.2 Klein 9B',
    /* Names the trade in the same shape as the other two: what it CHANGES,
       and what that buys. A restorer cannot sharpen detail that was never
       captured; a rewrite can invent it, at the cost of no longer being the
       same pixels. */
    summary: 'Re-renders skin and micro-detail from the ✨ instruction. Reaches '
      + 'frames too soft for a restore to fix, and changes them in the process.',
    confirm: 'Create a separate Flux.2 Klein 9B improvement candidate',
  },
  {
    id: 'seedvr2',
    label: 'SeedVR2',
    emoji: '🔍',
    action: 'Upscale via SeedVR2',
    summary: 'Resolves detail at a higher resolution and keeps the original look.',
    confirm: 'Create a separate SeedVR2 upscale candidate',
  },
]

export const DEFAULT_IMPROVE_ENGINE = 'klein'

export function improveEngine(id) {
  return IMPROVE_ENGINES.find((e) => e.id === id)
    || IMPROVE_ENGINES.find((e) => e.id === DEFAULT_IMPROVE_ENGINE)
}

/** The engines this install can actually run right now, in display order.

    Klein is always listed: it is the historical pass, and when it is not ready
    its button carries the reason (an engine that vanishes from the toolbar
    teaches nobody why). SeedVR2 only appears once it is READY, because until
    then it is not a choice, it is a setup task that belongs in Setup. */
export function availableImproveEngines(caps) {
  const comfy = (caps && caps.comfyui) || {}
  return IMPROVE_ENGINES.filter((e) => e.id !== 'seedvr2' || comfy.seedvr2_ready === true)
}

/** Why an engine's bulk button is disabled, or null when it can run. */
export function improveEngineBlockedReason(engineId, { caps, engines, eligibleCount } = {}) {
  const engine = improveEngine(engineId)
  if (engineId === 'klein' && engines && engines.klein === false) {
    return 'Klein is not available in this setup'
  }
  /* `klein_hq` is the one that genuinely runs Flux.2 Klein 9B, so it is the one
     the Klein availability flag really describes. The check above is kept as-is
     rather than moved: 'klein' still needs SOME gate, and re-pointing it at the
     Krea 2 flag is a separate change with its own way of being wrong. */
  if (engineId === 'klein_hq' && engines && engines.klein === false) {
    return 'Flux.2 Klein 9B is not available in this setup'
  }
  if (engineId === 'seedvr2' && !((caps && caps.comfyui) || {}).seedvr2_ready) {
    return 'SeedVR2 is not installed yet — Setup ▸ ComfyUI can download it'
  }
  if (!eligibleCount) return 'No selected image is eligible.'
  return null
}

/** The confirm text for a bulk run. States the engine's trade in the sentence
    the user reads immediately before committing a long batch. */
export function improveConfirmMessage(engineId, { eligibleCount = 0, exclusionSummary = '',
  excludedCount = 0 } = {}) {
  const engine = improveEngine(engineId)
  const skipped = excludedCount
    ? `\n\n${excludedCount} selected image(s) will be skipped: ${exclusionSummary}.`
    : ''
  return `${engine.confirm} for ${eligibleCount} image(s)?\n\n${engine.summary}${skipped}`
    + '\n\nThey are queued a few at a time in the background — you can close this tab,'
    + ' and ⏹ Stop generation ends the batch.'
    + '\n\nOriginal images stay unchanged until you review the candidates.'
}

/** Toast wording for a launched batch: what the server took, what it dropped,
    and which engine actually ran (the server echoes it, so a stale tab cannot
    claim the wrong one). */
export function describeImproveLaunch({ queued = 0, skipped = 0, engine } = {}) {
  const name = improveEngine(engine).label
  const tail = skipped ? ` · ${skipped} not eligible and skipped` : ''
  return `${name}: processing ${queued} image(s) in the background${tail} — originals stay intact.`
    + ' You can close this tab; ⏹ Stop generation ends the batch.'
}

/** Live progress line for a bulk button, from the dataset's server activity.
    `null` when no improve batch is running. The engine comes from the activity
    (the server records it at begin()), so the label names the run that is
    ACTUALLY going, not whichever button you are hovering. */
export function improveBatchLabel(activity) {
  if (!activity || activity.kind !== 'improve') return null
  const engine = improveEngine(activity.engine)
  const total = Number(activity.total) || 0
  const done = Number(activity.done) || 0
  if (activity.cancelling) return `${engine.emoji} Stopping…`
  return total
    ? `${engine.emoji} ${engine.label} ${done}/${total}`
    : `${engine.emoji} ${engine.label}…`
}

/** The per-image ✨ buttons for ONE image in the lightbox: one entry per engine
    this install can run, in display order.

    The lightbox is an EXPLICIT per-image choice, so both engines are offered
    side by side and neither is decided by the `improve.engine` setting — that
    setting only governs surfaces with a single ✨ button. This is the same rule
    the bulk toolbar follows, expressed once.

    `{id, label, title, disabled, showKleinNote}` per engine:
      * `label` reflects the IMAGE's state (a candidate already waiting for
        review, one still rendering) before the engine's own name, because that
        state blocks every engine equally and is what the user needs to read.
      * `showKleinNote` is true for Klein alone. The amber anime/drawn warning is
        about Klein's INSTRUCTION ("detailed texture, sharp details") pulling
        drawn skin towards realism — SeedVR2 sends no instruction at all, so
        repeating the warning under it would be false and would push people away
        from the very pass that fixes their case.

    Pure: no JSX, no capabilities probing of its own — `node --test` covers it. */
export function lightboxImproveButtons({ caps, engines, improving = false,
  improvePending = false, improveReady = false, busy = false,
  // Which dataset pass is holding this image, when the caller knows. "Another
  // action is running" is true and useless; the named pass tells you how long.
  busyReason = null } = {}) {
  const active = improving || improvePending
  // Blocked for reasons that have nothing to do with WHICH engine: one
  // improvement per image at a time, and one waiting result must be reviewed
  // before another is made.
  const imageBlocked = busy || active || improveReady
    ? (improveReady
        ? 'A new version is waiting for validation.'
        : active
          ? 'An improvement is already running for this image.'
          : (busyReason || 'Another action is running on this image.'))
    : null
  return availableImproveEngines(caps).map((engine) => {
    // `eligibleCount: 1` — the lightbox always acts on exactly this image, so
    // the shared reason function is asked only about engine readiness.
    const engineBlocked = improveEngineBlockedReason(engine.id, {
      caps, engines, eligibleCount: 1,
    })
    const reason = imageBlocked || engineBlocked
    return {
      id: engine.id,
      label: improveReady
        ? '✓ Review improvement first'
        : active
          ? `${engine.emoji} Improving…`
          : `${engine.emoji} ${engine.action}`,
      title: reason
        ? `${reason} ${engine.summary}`
        : `${engine.summary} The original stays intact — a separate candidate is created for you to validate.`,
      disabled: !!reason,
      showKleinNote: engine.id === 'klein',
    }
  })
}
