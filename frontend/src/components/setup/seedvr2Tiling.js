/* The OPTIONAL high-resolution lane for SeedVR2, and how the UI talks about it.

   Contributed by SurpassHR (GitHub #32): upscaling a whole frame at once runs
   out of VRAM past a certain size — he hit a real CUDA OOM on an 11.6 GB card —
   and cutting the frame into overlapping tiles, upscaling each and blending
   them back gets past it on the same hardware.

   Two things this module exists to keep honest:

   * The lane is OPTIONAL and its absence is not a fault. Without the tiling node
     pack the ordinary lane still upscales; it is only capped. So the wording
     must never read like something is broken.
   * The cap must be said BEFORE a run, not discovered in a traceback. When the
     card's ceiling is unknown we say nothing at all rather than invent a number.

   Pure — `node --test` cannot parse JSX and this is the part with the cases. */

export const TTP_PACK = 'Comfyui_TTP_Toolset'
export const TTP_URL = 'https://github.com/TTPlanetPig/Comfyui_TTP_Toolset'

/** `{state, text}` for the tiling row on the Setup card.
    state: 'ready' | 'restart' | 'absent' | 'unknown'. */
export function tilingStatus(caps) {
  const c = (caps && caps.comfyui) || {}
  if (!c.reachable) {
    // The probe fails CLOSED for this lane, so an unreachable ComfyUI is
    // genuinely "we cannot tell" — not "you are missing something".
    return { state: 'unknown',
      text: 'Start ComfyUI and this page will tell you whether the tiling pack is there.' }
  }
  if (c.seedvr2_tiling_ready) {
    return { state: 'ready',
      text: 'Ready — large upscales are cut into tiles automatically, so they no longer '
        + 'have to fit on the card in one piece.' }
  }
  const missing = Array.isArray(c.seedvr2_tiling_nodes_missing)
    ? c.seedvr2_tiling_nodes_missing : []
  if (missing.length && missing.length < 2) {
    // Half the classes present = an old build of the pack, not an absent one.
    return { state: 'restart',
      text: `The ${TTP_PACK} pack is there but does not expose ${missing.join(', ')} — `
        + 'update it in ComfyUI-Manager and restart ComfyUI.' }
  }
  return { state: 'absent',
    text: `Optional. Without it, upscales still run — they are just limited to what this `
      + `GPU can hold in one pass. Install ${TTP_PACK} in ComfyUI-Manager and restart `
      + 'ComfyUI to lift that limit.' }
}

/** Which lane TODAY'S settings will send an upscale down, said in one sentence.

    WHY THIS EXISTS. The 'auto' crossover is STRICT — a target must be ABOVE it
    to be tiled — and the crossover is derived as 1.5x the tile size, so it lands
    exactly on round numbers people type: 1536 with the default 1024 tile, 768 if
    they drop the tile to 512 for an 8 GB card. Someone who asks for exactly 1536
    therefore runs full-frame, and until now NOTHING said so: no tiling, no
    warning, no line in this panel. The person who reported the tiling lane in the
    first place (SurpassHR, GitHub #32) hit precisely that and had no way to tell
    "it decided against tiling" from "my setting did nothing".

    The rule itself is left alone on purpose — 'above' is what the setting says
    and what a stored `tile_threshold` means — so the fix is to make the decision
    legible, with the three levers that change it. Returns null when there is no
    number to reason about rather than inventing one.

    @param mode 'auto' | 'always' | 'never'
    @param target short-edge target resolution, px
    @param crossover the crossover 'auto' will really use, px */
export function laneForTarget(mode, target, crossover) {
  const px = Number(target)
  const above = Number(crossover)
  if (!(px > 0)) return null
  if (mode === 'never') return `Nothing is tiled: your ${px} px target runs full-frame.`
  if (mode === 'always') {
    return `Your ${px} px target is tiled whenever it comes out bigger than one tile.`
  }
  if (!(above > 0)) return null
  if (px > above) {
    return `Your ${px} px target is above the ${above} px crossover, so it is tiled.`
  }
  // The equal case gets the extra half-sentence: "below" surprises nobody,
  // landing ON the number you were told about does.
  const where = px === above
    ? `exactly at the ${above} px crossover, so it runs full-frame — the crossover has to `
      + 'be passed, not just reached'
    : `below the ${above} px crossover, so it runs full-frame`
  return `Your ${px} px target is ${where}. To tile it: raise the target above ${above} px, `
    + 'lower “Start tiling above”, or pick “Always tile large frames”.'
}

/** The sentence naming this machine's full-frame ceiling, or null when the card
    is unknown. Never invents a number: an unseen GPU gets silence. */
export function ceilingLine(caps) {
  const mp = ((caps && caps.comfyui) || {}).seedvr2_ceiling_mp
  if (typeof mp !== 'number' || !(mp > 0)) return null
  const ready = ((caps && caps.comfyui) || {}).seedvr2_tiling_ready
  return ready
    ? `This GPU is good for roughly ${mp} MP in a single pass; anything larger is tiled.`
    : `This GPU is good for roughly ${mp} MP in a single pass. Past that an upscale may `
      + 'run out of memory — that is what the tiling pack below is for.'
}
