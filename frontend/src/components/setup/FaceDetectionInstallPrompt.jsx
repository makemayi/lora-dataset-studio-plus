/* Contextual install prompt for face detection (InsightFace), rendered WHERE the
   capability is used rather than in Setup.

   The bug this closes: the capability is named `face_scoring` everywhere, so the
   Mask faces option could only say "install the ML extras (Face-similarity
   scoring) from the Setup tab" — a dead end asking the user to guess that a face
   MASK needs a face SCORER. The install path itself already existed; what was
   missing was declaring the dependency where the decision is made, and making it
   one click.

   Deliberately an OFFER, never a requirement: nothing installs on its own, the
   cost is stated before the click, and declining leaves a fully usable app with
   this one option off. No second installer action is introduced — this reuses
   InstallRunner on the existing `face_scoring` action, so the Setup tile and this
   prompt drive the same queue and the same progress (a install started in either
   place is picked back up by the other on mount).

   Wording/state logic lives in utils/faceDetectionInstall.js so node --test can
   cover it (this file is JSX). */
import { useCapabilities } from '../../context/CapabilitiesContext'
import { faceDetectionInstallState } from '../../utils/faceDetectionInstall'
import InstallRunner from './InstallRunner'

/** @param compact  true inside a dense options panel (smaller type, no card chrome). */
export default function FaceDetectionInstallPrompt({ why, compact = false, onInstalled }) {
  const { caps, loading, refresh } = useCapabilities()
  const state = faceDetectionInstallState({
    capable: caps.face_scoring, capsLoading: loading, python: caps.python,
  })
  // Quiet while capabilities are in flight, and silent once it's there: nothing
  // to nag about on a machine that already has it (or has decided against it —
  // the prompt only ever appears next to the option the user just reached for).
  if (state.status === 'loading' || state.status === 'ready') return null

  const text = compact ? 'text-[0.6875rem]' : 'text-xs'
  return (
    <div
      role="status"
      aria-live="polite"
      className={`mt-1 flex flex-col gap-1.5 rounded-lg border border-amber-500/40
        bg-amber-500/5 p-2 ${text} leading-relaxed text-content-muted`}
    >
      <p className="font-semibold text-amber-700">⚠️ {state.headline}</p>
      {why && <p>{why}</p>}
      <p>{state.detail}</p>
      {state.canInstall
        ? (
          <InstallRunner
            action={state.action}
            buttonLabel={`Install ${state.label}`}
            onDone={() => { refresh(true); onInstalled?.() }}
          />
        )
        : (
          // No button that could only fail. The Setup page is still where the
          // interpreter override lives, so point at it — with a real link, not
          // an instruction to go find a tab.
          <a href="#/settings/local-tools"
            className="self-start rounded-md bg-surface px-2.5 py-1
              font-semibold text-content hover:bg-surface-raised">
            Open Settings ▸ Local tools
          </a>
        )}
    </div>
  )
}
