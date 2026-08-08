/* Concept face masking — the Advanced training option and its preview.
   Reported by shivdbz2010 (GitHub issue #15): a concept LoRA also learns the faces
   of its dataset, so combining it with a character LoRA makes the two fight over
   the identity.

   Everything here is DECISION support, placed where the decision is made (next to
   the checkbox, before training), not in the image grid where it would be a
   curiosity. Three things are visible before you commit:
     - what the mask would actually cover, drawn on your own photos;
     - what the head-coverage setting does, redrawn live as you drag it;
     - how many images got no face at all, because a PARTLY masked set is the bad
       case: the faces left unmasked are then the only ones carrying loss weight.

   The overlay idiom (normalised coords -> percentage-positioned absolute divs over
   a relative wrapper) is the one WatermarkRegionEditor already uses; the geometry
   lives in utils/faceMaskBox.js, mirrored from the Python that paints the real mask. */
import { useCallback, useEffect, useRef, useState } from 'react';
import { postJson } from '../../hooks/useDataset';
import { HelpBadge } from '../../help/HelpMode';
import FaceDetectionInstallPrompt from '../setup/FaceDetectionInstallPrompt';
import { boxStyle, coverageFraction, MAX_COVERAGE } from '../../utils/faceMaskBox';
import {
  previewError, previewPercent, previewProgressValue, previewRunning, previewStartLabel,
  previewStatusLabel, previewStopCost, previewStoppedNotice, previewStopLabel,
} from '../../utils/faceMaskProgress';

const previewUrl = (datasetId) => `/api/dataset/${datasetId}/train/face-mask-preview`;
// Fast enough that the count visibly moves, cheap enough to leave running: the
// endpoint reads an in-memory snapshot and stats the kept files.
const POLL_MS = 800;

/* The bar itself. `role="progressbar"` with live values while there is a count,
   and an aria-live status line otherwise — during the model load there is
   genuinely nothing to measure, and announcing a fake 0% would be a lie a screen
   reader cannot see through. The text alone must carry the whole story. */
function PreviewProgress({ job }) {
  const value = previewProgressValue(job);
  const percent = previewPercent(job);
  return (
    <div className="mt-1.5">
      <p aria-live="polite" className="text-[0.6875rem] text-content-muted">
        {previewStatusLabel(job)}
      </p>
      <div
        role="progressbar"
        aria-label="Face detection progress"
        {...(value
          ? { 'aria-valuenow': value.done, 'aria-valuemin': 0, 'aria-valuemax': value.total }
          : {})}
        className="mt-1 h-1.5 w-full max-w-xs overflow-hidden rounded-full bg-surface-raised"
      >
        {/* Indeterminate is drawn FULL WIDTH and dimmed-pulsing, never as a
            partial fill: a third-full bar during the model load would read as
            "33% done", which is the same lie as the frozen 0/N it replaces. */}
        <div
          className={value ? 'h-full rounded-full bg-indigo-500'
            : 'h-full w-full rounded-full bg-indigo-500/40 animate-pulse'}
          style={value ? { width: `${percent}%` } : undefined}
        />
      </div>
    </div>
  );
}

const imageUrl = (datasetId, filename) =>
  `/api/dataset/${datasetId}/img/${encodeURIComponent(filename)}`;

/* One sample with its mask drawn over it. `expand` is applied here, in the
   browser, so dragging the slider is instant — the server pass ran once and
   returned the raw detected boxes. */
function SamplePreview({ datasetId, sample, expand }) {
  const boxes = sample.boxes || [];
  const covered = coverageFraction(boxes, expand);
  const tooLarge = covered > MAX_COVERAGE;
  return (
    <figure className="min-w-0">
      <div className="relative overflow-hidden rounded-lg border border-border bg-black">
        <img src={imageUrl(datasetId, sample.filename)} alt=""
          className="block w-full h-auto select-none" loading="lazy" />
        {boxes.map((b, i) => (
          <div key={i} aria-hidden style={boxStyle(b, expand)}
            className={`absolute rounded-[50%] border-2 ${tooLarge
              ? 'border-amber-300 bg-amber-400/25'
              : 'border-sky-300 bg-sky-400/35'}`} />
        ))}
        {!boxes.length && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="rounded-lg bg-black/75 px-2 py-1 text-[0.6875rem] font-semibold text-amber-200">
              no face found
            </span>
          </div>
        )}
      </div>
      <figcaption className="mt-1 text-[0.625rem] leading-tight text-content-subtle">
        {!boxes.length
          ? 'Trains unmasked — nothing was detected here.'
          : tooLarge
            ? `Covers ${Math.round(covered * 100)}% of the frame — left unmasked.`
            : `${boxes.length} face${boxes.length > 1 ? 's' : ''} · ${Math.round(covered * 100)}% of the frame`}
      </figcaption>
    </figure>
  );
}

export default function ConceptFaceMaskField({
  datasetId, enabled, supported, conceptConflict, faceCapability, expandDefault, onToggle,
}) {
  const [preview, setPreview] = useState(null);
  const [job, setJob] = useState(null);
  // What a resume would be worth, as the SERVER sees it: the detections banked
  // by a stopped pass, already checked against the current kept set's
  // fingerprint. Null once that set moves — a resume onto changed images would
  // mix in boxes from photos that are no longer in the run.
  const [resume, setResume] = useState(null);
  const [expand, setExpand] = useState(expandDefault ?? 2);
  const [err, setErr] = useState(null);
  const expandTouched = useRef(false);

  /* Adopt a server snapshot. The detection is a SERVER-side job, so this is what
     makes leaving the page free: on mount we ask what is running and what was
     already computed, and the panel comes back exactly as it was instead of
     offering a fresh pass over the same images. */
  const adopt = useCallback((d) => {
    if (!d || !d.ok) return;
    setJob(d.job || null);
    setResume(d.resume || null);
    if (d.result) {
      setPreview(d.result);
      // The slider is the user's; only seed it from the server before they touch it.
      if (typeof d.result.expand === 'number' && !expandTouched.current) {
        setExpand(d.result.expand);
      }
    }
    setErr(previewError(d.job) || null);
  }, []);

  const running = previewRunning(job);

  // Mount + poll. One effect: the mount read and the poll read the same endpoint,
  // so rebinding to a pass started before this component existed is the same code
  // path as watching one it started itself.
  useEffect(() => {
    if (!supported || !datasetId) return undefined;
    let alive = true;
    const tick = async () => {
      try {
        const r = await fetch(previewUrl(datasetId), { credentials: 'include' });
        if (!r.ok || !alive) return;
        adopt(await r.json());
      } catch { /* a blip just means "ask again next tick" */ }
    };
    tick();
    if (!running) return () => { alive = false; };
    const id = setInterval(tick, POLL_MS);
    return () => { alive = false; clearInterval(id); };
  }, [supported, datasetId, running, adopt]);

  // The pass failed specifically because InsightFace is missing (server 409 with
  // reason:'face_scoring'). Reachable when the client's capabilities are stale —
  // and it is exactly the moment to offer the install rather than report a defeat.
  const [needsFaceDetection, setNeedsFaceDetection] = useState(false);

  // NOT a concept dataset. The server publishes `mask_faces_supported`
  // deliberately — "so the panel states the reason instead of hiding it"
  // (effective_train_settings) — and this component returned null anyway, so the
  // row simply vanished and the feature read as missing rather than as
  // inapplicable. Same contract as `masked_supported` and the dual-captions note
  // right above it in the panel: the control is shown, disabled, with the reason.
  if (!supported) {
    return (
      <div className="flex flex-col gap-0.5">
        <label className="flex items-center gap-2 flex-wrap">
          <span className="text-content-muted text-[0.75rem] w-28 shrink-0 inline-flex items-center gap-1">
            Mask faces<HelpBadge topic="training.mask_faces" />
          </span>
          <input type="checkbox" checked={false} disabled readOnly
            aria-label="Mask faces while training this concept"
            className="h-4 w-4 rounded border-border bg-surface accent-indigo-500 opacity-40" />
          <span className="text-content-subtle text-[0.75rem]">concept datasets only</span>
        </label>
        <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
          A character LoRA has to learn the face, and a style LoRA has to learn how faces
          are rendered — weighing faces down would amputate the very thing being trained.
          This lever only applies to a <b className="text-content-muted font-medium">concept</b> dataset,
          where the identities in your photos are incidental. Change the dataset type in
          Dataset settings if that is what you meant to train.
        </span>
      </div>
    );
  }

  const runPreview = async () => {
    setErr(null);
    setNeedsFaceDetection(false);
    // Optimistic, so the first frame after the click already says something —
    // the server answers in milliseconds but the phase it reports is the truth.
    setJob({ phase: 'starting', done: 0, total: 0, error: null, finished: false });
    const d = await postJson(previewUrl(datasetId), { limit: 6 });
    if (d && d.ok) {
      adopt(d);
    } else if (d && d.reason === 'face_scoring') {
      // Turn the diagnosis into an action instead of leaving the user with a
      // sentence about a module they have no way to name.
      setJob(null);
      setNeedsFaceDetection(true);
    } else {
      setJob(null);
      setErr((d && d.error) || 'preview failed');
    }
  };

  /* Ask the pass to wind up. Deliberately NOT optimistic about the outcome: the
     server answers "stopping", and the pass stays running until the child hands
     back what it found. Flipping the UI straight to "stopped" would claim the
     work was banked before it actually was. */
  const stopPreview = async () => {
    const d = await postJson(`${previewUrl(datasetId)}/stop`, {});
    if (d && d.ok) adopt(d);
  };

  const cov = preview?.coverage;
  const samples = preview?.samples || [];
  // A partially masked set is worse than a consistently unmasked one, so this is a
  // warning, not a statistic. Only meaningful once at least one face was found.
  const partial = cov && cov.masked > 0 && cov.masked < cov.total;

  return (
    <div className="flex flex-col gap-0.5">
      <label className="flex items-center gap-2 flex-wrap cursor-pointer">
        <span className="text-content text-[0.75rem] w-28 shrink-0 inline-flex items-center gap-1">
          Mask faces<HelpBadge topic="training.mask_faces" />
        </span>
        <input type="checkbox" checked={Boolean(enabled)} disabled={faceCapability === false}
          onChange={(e) => onToggle(e.target.checked)}
          aria-label="Mask faces while training this concept"
          className="h-4 w-4 rounded border-border bg-surface accent-indigo-500 disabled:opacity-40" />
        <span className="text-content-muted text-[0.75rem]">keep the act, drop the identities</span>
      </label>

      {/* The dependency is DECLARED where it is ticked, and installable from here.
          It used to point at "the ML extras (Face-similarity scoring) in the Setup
          tab" — correct, and useless: nobody ticking "Mask faces" would go install
          a face SCORER. Same install action underneath, named for what it does. */}
      {faceCapability === false && (
        <FaceDetectionInstallPrompt compact
          why="Faces have to be found before they can be weighted down, and that
            detection is an optional extra this install doesn't have yet." />
      )}

      <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
        <b className="text-content-muted font-medium">Why:</b> a concept LoRA also picks up the
        faces it was trained on, and then pulls against a character LoRA over whose face to
        render. This weighs the detected faces down in the training loss, so the concept
        binds to the act rather than to the people in your photos. Your images are never
        altered — nothing is blurred or painted over, only the loss is re-weighted.
        <b className="text-content-muted font-medium"> How:</b> turn it on, then preview it on
        your own shots. Adjust the head coverage in Settings ▸ Training if the mask sits too
        tight or too wide.
      </span>

      {/* The maintainers' answer, stated where the decision is taken. Without it we
          ship a patch people reach for INSTEAD of fixing the dataset. */}
      <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
        <b className="text-content-muted font-medium">Worth knowing:</b> the people who maintain
        these trainers consider dataset variety to matter more than masking here. A concept shown
        by ten different people already dilutes identity; with two, the faces are as constant as
        the concept itself and no mask fully makes up for it. Nobody has published a measured
        before/after of face masking on a concept LoRA — this gives you the lever, not a promise.
      </span>

      {conceptConflict && (
        <span className="text-amber-300 text-[0.6875rem] leading-relaxed">
          ⚠️ Your concept description mentions the face, mouth or gaze. If the face is where your
          concept actually happens, masking it can erase the thing you are teaching. Preview it
          before you train — you know your dataset, so this is a heads-up, not a block.
        </span>
      )}

      {enabled && faceCapability !== false && (
        <div className="mt-1">
          {/* 400px-first: the two buttons wrap instead of overflowing, and the
              cost sentence sits on its own line under them rather than being
              squeezed next to Stop. */}
          <div className="flex flex-wrap items-center gap-1.5">
            <button type="button" onClick={runPreview} disabled={running}
              className="min-h-8 rounded-lg border border-border bg-surface px-2.5 text-[0.6875rem] font-semibold text-content hover:bg-surface-raised disabled:opacity-50">
              {running ? 'Looking for faces…' : previewStartLabel(resume, Boolean(preview))}
            </button>
            {running && (
              <button type="button" onClick={stopPreview} disabled={Boolean(job && job.stopping)}
                className="min-h-8 rounded-lg border border-border bg-surface px-2.5 text-[0.6875rem] font-semibold text-amber-200 hover:bg-surface-raised disabled:opacity-50">
                {previewStopLabel(job)}
              </button>
            )}
          </div>
          {/* The price of stopping, stated where the button is — and it CHANGES
              while the pass runs, which is exactly why it cannot be a one-off
              confirmation dialog. */}
          {running && (
            <p className="mt-1 text-[0.625rem] leading-tight text-content-subtle">
              {previewStopCost(job)}
            </p>
          )}
          {!running && previewStoppedNotice(job, resume) && (
            <p role="status" className="mt-1 text-[0.6875rem] leading-relaxed text-content-muted">
              {previewStoppedNotice(job, resume)}
            </p>
          )}
          {running && <PreviewProgress job={job} />}
          {err && !running && (
            <p role="alert" className="mt-1 text-amber-300 text-[0.6875rem] leading-relaxed">
              ⚠️ {err}
            </p>
          )}
          {needsFaceDetection && !running && (
            <FaceDetectionInstallPrompt compact
              why="The preview couldn't run: finding faces needs this optional extra."
              onInstalled={() => { setNeedsFaceDetection(false); runPreview(); }} />
          )}

          {preview && (
            <div className="mt-2 rounded-lg border border-border bg-app/40 p-2">
              {/* A preview describes the exact kept set it was computed from. Once
                  that set moves, showing it as fresh would be worse than showing
                  nothing — the boxes would be drawn from photos that are no longer
                  in the run. So it is kept visible and clearly labelled, never
                  quietly. */}
              {preview.stale && (
                <p role="status" className="mb-1.5 rounded-md bg-amber-500/10 px-2 py-1 text-[0.6875rem] leading-relaxed text-amber-300">
                  ⚠️ Your kept images changed since this preview ran, so it no longer
                  describes what would be trained. Refresh it.
                </p>
              )}
              {cov && cov.total === 0 && (
                <p className="text-[0.6875rem] text-content-muted">
                  No kept images to look at yet — keep a few shots first, then preview.
                </p>
              )}
              {cov && cov.total > 0 && (
                <p className={`text-[0.6875rem] ${partial ? 'text-amber-300' : 'text-content-muted'}`}>
                  {partial && '⚠️ '}
                  Masked on {cov.masked} of {cov.total} image{cov.total > 1 ? 's' : ''}
                  {cov.no_face > 0 && ` · ${cov.no_face} with no face found`}
                  {cov.too_large > 0 && ` · ${cov.too_large} too close-up to mask`}
                  {cov.failed > 0 && ` · ${cov.failed} unreadable`}
                  {partial && ' — the unmasked faces are the only ones the LoRA still learns from, '
                    + 'so they end up over-represented.'}
                </p>
              )}
              {samples.length > 0 && (
              <label className="mt-2 flex items-center gap-2 flex-wrap text-[0.6875rem] text-content-muted">
                <span className="shrink-0">Head coverage</span>
                <input type="range" min="1" max="3" step="0.1" value={expand}
                  onChange={(e) => { expandTouched.current = true; setExpand(parseFloat(e.target.value)); }}
                  aria-label="Preview head coverage"
                  className="w-32 accent-indigo-500" />
                <span className="tabular-nums text-content">×{expand.toFixed(1)}</span>
                <span className="text-content-subtle">
                  preview only — save it in Settings ▸ Training
                </span>
              </label>
              )}
              {/* 400px-first: one column on a phone, more as the panel widens. */}
              {samples.length > 0 && (
                <>
                  <div className="mt-2 grid grid-cols-2 sm:grid-cols-3 gap-2">
                    {samples.map((s) => (
                      <SamplePreview key={s.image_id} datasetId={datasetId} sample={s} expand={expand} />
                    ))}
                  </div>
                  <p className="mt-1.5 text-[0.625rem] leading-tight text-content-subtle">
                    Images where no face was found are shown first — those are the ones worth looking at.
                  </p>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
