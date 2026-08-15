import { useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { INPUT_CLASS, PRIMARY_BUTTON } from '../common/surfaces'
import {
  framePromoteProblem, framePromotePayload, frameScopeLabel,
  frameCeilingHint, frameFaceNote,
  FRAMES_PER_CLIP_MAX, FRAMES_PER_CLIP_DEFAULT,
} from './videoFramePromote'

/** 🖼 Turn the shots you kept into an IMAGE training set.
 *
 * The second promotion target, next to 🎬. Same kept clips, but each one
 * contributes a handful of STILLS chosen for sharpness, exposure and — with the
 * face filter on — for actually showing a usable face.
 *
 * TWO THINGS THIS SCREEN HAS TO SAY OUT LOUD, because neither is visible in the
 * result:
 *
 *  · "frames per clip" is a CEILING. A clip whose every frame is over-exposed,
 *    or that never shows a usable face, contributes nothing and nothing pads to
 *    reach the number. Someone who asks for 3 × 40 and receives 61 images has to
 *    be able to read that as the filter working;
 *  · with the face filter OFF, frames are picked on sharpness and exposure
 *    alone — so a set can come back full of sharp pictures of the wrong person,
 *    or of nobody. That is a legitimate choice for a style dataset and a trap
 *    for a character one.
 */
export default function PromoteFramesDialog({
  bankId, keepCount, selectedIds, onClose, onDone,
}) {
  const toast = useToast()
  const [name, setName] = useState('')
  const [framesPerClip, setFramesPerClip] = useState(FRAMES_PER_CLIP_DEFAULT)
  const [totalLimit, setTotalLimit] = useState('')
  const [maxPerSource, setMaxPerSource] = useState('')
  const [requireFace, setRequireFace] = useState(true)
  // Datasets that actually HAVE a reference photo — the others cannot answer
  // "is this the right person", so offering them would be offering a failure.
  const [refDatasets, setRefDatasets] = useState([])
  const [refDatasetId, setRefDatasetId] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let alive = true
    apiFetch('/api/dataset/list')
      .then((d) => {
        if (!alive) return
        const withRef = (d?.datasets || []).filter((x) => x.ref_filename)
        setRefDatasets(withRef)
        if (withRef.length === 1) setRefDatasetId(String(withRef[0].id))
      })
      .catch(() => { /* the refusal below already says what is missing */ })
    return () => { alive = false }
  }, [])

  // NOT gated on ffmpeg, unlike the clip promotion beside it: extraction
  // decodes through PyAV, which carries its own libav. Borrowing the 'promote'
  // capability here would refuse a run that works, with a message about cutting
  // clips that this screen does not do.
  const problem = framePromoteProblem({ name, framesPerClip, requireFace, refDatasetId })
  const scope = frameScopeLabel((selectedIds || []).length, keepCount)
  const ceiling = frameCeilingHint({
    frames_ceiling: ((selectedIds || []).length || Number(keepCount) || 0)
      * (Number(framesPerClip) || 0),
  })

  const submit = async (e) => {
    e.preventDefault()
    if (busy || problem) return
    setBusy(true)
    setError(null)
    try {
      const d = await postJson(`/api/video-bank/${bankId}/promote-frames`,
        framePromotePayload({ name, framesPerClip, totalLimit,
          maxPerSource, requireFace, refDatasetId, ids: selectedIds }))
      toast.success(`Building “${d.name}” — ${d.clips} clip(s) being read for frames.`)
      // Said after the request lands, because "the filter is off" is invisible
      // in a folder of sharp pictures of the wrong person.
      const faceNote = frameFaceNote(d.composition)
      if (faceNote) toast.warning(faceNote)
      onDone?.(d)
      onClose?.()
    } catch (err) {
      // The server's 400 names what was wrong — it beats anything invented here.
      setError(err?.message || 'Could not start the extraction.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div role="dialog" aria-modal="true" aria-label="Build an image training set"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-3 sm:p-4">
      <form onSubmit={submit}
        className="w-full max-w-lg max-h-[90vh] space-y-4 overflow-y-auto rounded-2xl bg-surface-overlay/85 backdrop-blur-md p-4 shadow-2xl sm:p-5">
        <h2 className="text-base font-bold text-content">🖼 Build an image training set</h2>
        <p className="text-sm text-content-muted">
          Reads {scope} and keeps the sharpest usable still frames of each. Your
          source files are never touched, and the clips stay in the bank.
        </p>

        <div>
          <label htmlFor="frames-ds-name" className="block text-sm font-medium text-content">Name</label>
          <input id="frames-ds-name" value={name} onChange={(e) => setName(e.target.value)}
            placeholder="ada — stills from the interviews" required
            className={INPUT_CLASS} />
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <label htmlFor="frames-per-clip" className="block text-sm font-medium text-content">
              Frames per clip
            </label>
            <input id="frames-per-clip" type="number" min="1" step="1"
              max={FRAMES_PER_CLIP_MAX} value={framesPerClip}
              onChange={(e) => setFramesPerClip(Number(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-1 text-xs text-content-muted">{ceiling}</p>
          </div>
          <div>
            <label htmlFor="frames-total" className="block text-sm font-medium text-content">
              Stop after (optional)
            </label>
            <input id="frames-total" type="number" min="1" step="1"
              value={totalLimit} onChange={(e) => setTotalLimit(e.target.value)}
              placeholder="no limit" className={INPUT_CLASS} />
            <p className="mt-1 text-xs text-content-muted">
              A hard stop on the whole run, in images. Clips are read in order,
              so a low number means the later sources never get a turn.
            </p>
          </div>
        </div>

        <div>
          <label htmlFor="frames-max-source" className="block text-sm font-medium text-content">
            Max clips per source (optional)
          </label>
          <input id="frames-max-source" type="number" min="1" step="1"
            value={maxPerSource} onChange={(e) => setMaxPerSource(e.target.value)}
            placeholder="no cap" className={INPUT_CLASS} />
          <p className="mt-1 text-xs text-content-muted">
            Trims dominance, never scarcity: each source keeps its earliest clips.
            Without it one long video can supply most of the set — one lighting
            setup, one wardrobe — which is invisible once the images are in a folder.
          </p>
        </div>

        <div className="rounded-xl bg-surface p-3">
          <label className="flex items-start gap-2 text-sm text-content">
            <input type="checkbox" checked={requireFace} className="mt-0.5"
              onChange={(e) => setRequireFace(e.target.checked)} />
            <span>
              Only keep frames showing a usable face
              {/* Both halves stay MOUNTED and flip `hidden`: this app is read
                  through Chrome auto-translate, which rewrites text nodes and
                  makes a React swap crash the whole section. */}
              <span hidden={!requireFace} className="mt-1 block text-xs text-content-muted">
                Frames whose face is too small, turned too far away, or missing
                are dropped — even when they are the sharpest in the clip.
              </span>
              <span hidden={requireFace} className="mt-1 block text-xs text-content-muted">
                Off: frames are picked on sharpness and exposure alone. Fine for a
                style set; for a character set it can return sharp pictures of the
                wrong person, or of nobody.
              </span>
            </span>
          </label>
          <div hidden={!requireFace} className="mt-2">
            <label htmlFor="frames-ref-ds" className="block text-xs font-medium text-content">
              Compare against
            </label>
            <select id="frames-ref-ds" value={refDatasetId} className={INPUT_CLASS}
              onChange={(e) => setRefDatasetId(e.target.value)}>
              <option value="">Pick a dataset…</option>
              {refDatasets.map((d) => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
            {/* The list holds ONLY datasets with a reference photo, so an empty
                list is not "no datasets" — it is "none of them can answer this
                question", and saying which is the difference between a fixable
                message and a dead end. */}
            <p hidden={refDatasets.length > 0} className="mt-1 text-xs text-amber-700">
              None of your datasets has a reference photo yet. Set one on the
              dataset of the person you are collecting, or turn this filter off.
            </p>
            <p hidden={refDatasets.length === 0} className="mt-1 text-xs text-content-muted">
              Its reference photo (and any extra references) decide whether a
              frame shows the right person.
            </p>
          </div>
        </div>

        <p hidden={!error} className="text-sm text-red-600">{error}</p>
        <p hidden={!problem || !!error} className="text-sm text-amber-700">{problem}</p>

        <div className="flex justify-end gap-2">
          <button type="button" onClick={onClose}
            className="rounded-full px-3 py-1.5 text-sm text-content-muted hover:text-content">
            Cancel
          </button>
          <button type="submit" disabled={busy || !!problem} className={PRIMARY_BUTTON}>
            <span hidden={busy}>🖼 Extract from {scope}</span>
            <span hidden={!busy}>Starting…</span>
          </button>
        </div>
      </form>
    </div>
  )
}
