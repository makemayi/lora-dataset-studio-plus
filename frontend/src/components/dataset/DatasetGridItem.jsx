/** One curation tile: image + keep/reject + source/framing badges + caption + crop. */
import { improvementBadge } from './improveCandidates.js';
import { FLOAT_HOVER, GLASS_CAPSULE_DARK, ON_PHOTO_HOVER,
  TILE_SELECTED_GLOW, TILE_SHADOW, TILE_SURFACE } from '../common/surfaces.js';
import { TrashIcon } from '../common/icons.jsx';
import { FaceSwapIcon } from '../common/icons.jsx';
import { useEffect, useRef, useState } from 'react';
import { displayLabel } from '../../utils/labels';
// WHO wrote this tile's caption — the per-image half of the provenance the pass
// already reports in aggregate (utils/captionEngines.js).
import { captionIsAsserted, captionOriginInfo } from '../../utils/captionOrigin.js';
import { isSmallImageRescueRow } from '../../utils/smallImageRescue';
import CaptionEditorDialog from './CaptionEditorDialog';
import PromptEditPopover from './PromptEditPopover';
import SourceAttribution from './SourceAttribution';
import { ENGINE_ACCENTS, ENGINE_LABELS } from './engineSelection.js';
import { canRegenerateGeneric, improveRerunAffordance, isImageImproveRow } from './improveRerun.js';
import { rememberImageRatio } from './lightboxActionPlacement.js';
import { FACE_BADGE_CLASS, PROVENANCE_BADGE_CLASS, TILE_BADGE_STACK_CLASS,
  WATERMARK_BADGE_CLASS } from './tileBadgeLayout.js';

/* Provenance wording per STORED derivation kind. `klein_image_improve` is a
   legacy key: it predates the second engine and is written in user databases,
   so a SeedVR2 result carries it too and the badge must NOT claim Klein made
   it. The engine is named by the candidate's own label ('Klein upscale &
   improve' / 'SeedVR2 upscale'), which is where a per-row truth belongs; this
   badge says only what KIND of row it is, which is true for both. */
const DERIVATION_LABEL = {
  klein_small_image: 'Klein rescue',
  small_image_source: 'rescue original',
  klein_image_improve: 'upscale candidate',
};

/* The kept/rejected/pending state lives in a corner DOT (top-left, always
   visible), not a border — the tile's edge used to carry a status border AND
   a selection ring at once, two mechanisms competing for the same line, and
   the photo was already boxed inside a white card. The dot pairs colour with
   an accessible name (never colour alone, same rule as faceBadge below). */
const STATUS_DOT = {
  keep: { cls: 'bg-green-500', label: 'Kept' },
  reject: { cls: 'bg-red-500', label: 'Rejected' },
  pending: { cls: 'bg-amber-400', label: 'Undecided' },
  failed: { cls: 'bg-red-600', label: 'Failed' },
};

// Seuils calibres antelopev2 (test3) — face_score brut persiste -> ajustables dans
// Settings (face_scoring.green/orange) ; ces valeurs ne servent que de repli.
const GREY_LABEL = { no_face: 'no face detected', low_det: 'low detection',
  too_small: 'face too small', extreme_pose: 'profile — not scored',
  unreadable: 'unreadable', error: 'error' };

// Retourne {icon, cls, label} d'apres face_state/face_score, ou null si pas analysé.
// La couleur accompagne un GLYPHE (✓/~ /⚠ ou 👁) — jamais la couleur seule (WCAG 1.4.1) —
// et le grade en lettre (A/B/C/D) reste le label accessible.
// Score → letter grade: ≥0.9 A, ≥0.8 B, ≥0.7 C, <0.7 D.
function faceBadge(img) {
  if (img.face_state == null) return null;
  if (img.face_state !== 'scorable' || img.face_score == null) {
    return { icon: '👁', cls: 'text-gray-600', graded: false,
      label: GREY_LABEL[img.face_state] || 'not scored' };
  }
  const s = img.face_score;
  if (s >= 0.9) return { icon: '✓', cls: 'text-green-700', graded: true, label: 'A' };
  if (s >= 0.8) return { icon: '✓', cls: 'text-emerald-700', graded: true, label: 'B' };
  if (s >= 0.7) return { icon: '~', cls: 'text-amber-700', graded: true, label: 'C' };
  return { icon: '⚠', cls: 'text-red-600', graded: true, label: 'D' };
}

// Watermark V1 badge from watermark_state (🚩 detected / ⊘ dismissed / ✨ cleaned /
// ⚠ failed), or null when never scanned ('none' is also silent — nothing to show).
// `dismissed` shows a DISCREET grey ⊘ (not nothing): it confirms the user's "not a
// watermark" ruling took effect and explains why a re-scan won't re-flag it — silence
// would read as "did my dismiss work?". The tooltip names what Clean will do; when the
// payload carries the exact route (watermark_route, computed backend-side from the
// dims) the detected tooltip names the precise action, else it lists the possibilities.
const WATERMARK_ROUTE_HINT = {
  crop: 'Overlaid watermark on the border — Clean will crop it off',
  lama: 'Small off-center watermark — Clean will inpaint it (LaMa)',
  review: 'Watermark on the subject — Clean flags it for manual review (auto crop/inpaint would damage the photo); reject or crop manually',
};
const WATERMARK_BADGE = {
  detected: { icon: '🚩', cls: 'text-amber-700', text: 'watermark',
    label: 'Overlaid watermark detected — Clean will crop the border, inpaint a small mark, or flag it for manual review (V2 handles on-subject watermarks)' },
  dismissed: { icon: '⊘', cls: 'text-content-subtle', text: 'not a watermark',
    label: 'You marked this “not a watermark” — future 🧽 Find passes skip it' },
  cleaned: { icon: '✨', cls: 'text-emerald-700', text: 'watermark', label: 'Watermark removed (original kept as a .orig backup)' },
  failed: { icon: '⚠', cls: 'text-red-600', text: 'watermark', label: 'Watermark removal failed' },
};

/* READS ARE NOT WRITES.
 * `busy` means "a pass is running on this dataset, so nothing here may CHANGE
 * it". It used to switch off the whole tile — including the button that opens
 * the image and the tick that selects it, neither of which writes anything —
 * which is how "during a generation everything around the dataset is blocked"
 * became the app's most-reported frustration. Inspecting and ticking now ignore
 * `busy` entirely; every button below that touches pixels, status, captions or
 * files still refuses, and `busyReason` is the sentence it shows instead of
 * going quietly grey.
 */
export default function DatasetGridItem({ img, datasetId, onStatus, onCaption, onCrop, onDelete,
                                         onLockToggle, onMirror, mirrorBusy = false, busy = false,
                                         busyReason = null,
                                          onScoreFace, scoreFaceBusy = false, faceScoringBusy = false, faceScoringBlocked = null,
                                          onRegenerate, onReimprove, onFaceSwap, onUndoFaceSwap, swapBusy = false, hasRef = false, onView, nonce = 0, faceThresholds,
                                          selected = false, onToggleSelect, tileSize = 'M',
                                          datasetKind = 'character', dualCaptions = false,
                                          improvementState = undefined }) {
  const [cap, setCap] = useState(img.caption || '');
  const [captionEditorOpen, setCaptionEditorOpen] = useState(false);
  // ✏️ edit-prompt bubble open state (regenerate this tile with an edited prompt).
  const [editingPrompt, setEditingPrompt] = useState(false);
  // While the textarea has focus, a poll-driven refresh must never overwrite
  // the draft (C1) — the server value only syncs in when nobody is typing.
  const editingRef = useRef(false);
  // Sync the textarea when the server fills/updates the caption (e.g. after the
  // Qwen3-VL captioning pass) — useState's initial value alone would stay stale.
  useEffect(() => { if (!editingRef.current) setCap(img.caption || ''); }, [img.caption]);
  // `nonce` busts the browser cache after an in-place crop (same filename).
  const url = img.filename
    ? `/api/dataset/${datasetId}/img/${encodeURIComponent(img.filename)}${nonce ? `?v=${nonce}` : ''}`
    : null;
  // Regenerate applies to generated tiles that are not mid-generation —
  // finished AND failed ones (failure recovery path) (F2).
  const isRescueDerived = isSmallImageRescueRow(img);
  // A manual Klein improvement is derived from THIS image, not the dataset's
  // main reference. Sending it through the generic regenerate route would lose
  // that source and silently make an unrelated variation instead — so it gets
  // its OWN re-run below (same parent, current improve settings) instead.
  const isImageImproveCandidate = isImageImproveRow(img);
  const canRegenerate = canRegenerateGeneric(img, { isRescueDerived });
  const canFaceSwap = hasRef && !!img.filename;
  // A swap that came back empty restored this tile's own picture; the server
  // leaves its reason behind so the tile can say so. A row that IS failed shows
  // the reason in its placeholder instead, so this is the with-a-picture case.
  const swapRestoredNote = (img.filename && img.status !== 'failed' && img.fail_reason)
    ? img.fail_reason : null;
  const rerunImprove = onReimprove ? improveRerunAffordance(img) : null;
  const scoreFaceTitle = faceScoringBlocked
    || (scoreFaceBusy
      ? 'Scoring facial resemblance to the reference…'
      : faceScoringBusy
        ? 'Face scoring is already running for another image…'
        : busy
          ? (busyReason || 'Wait for the current dataset action to finish before scoring.')
          : 'Score facial resemblance to the reference');
  // Every refused write says WHICH pass holds it; idle, each keeps its own words.
  const refused = busy ? busyReason : null;

  const fb = faceBadge(img);
  const wb = WATERMARK_BADGE[img.watermark_state];
  // A big batch leaves every not-yet-done tile on the same amber "pending"
  // dot, with no way to tell which one a worker has actually claimed from the
  // ones still merely queued behind it. is_generating (see dataset_payload) is
  // true for ONLY that tile — the dot PULSES then (never colour alone, same
  // rule faceBadge follows below) so it reads even without colour.
  const generating = img.status === 'pending' && img.is_generating;
  const statusDot = STATUS_DOT[img.status] || STATUS_DOT.pending;
  // A rejected tile recedes (and its dot with it): the grid reads "rejects"
  // at a glance without needing to land on each one. The dot stays mounted —
  // the whole tile just steps back.
  const rejected = img.status === 'reject';
  // The tile stays a square (crop decisions need a stable grid), but at the L
  // size — fewer, bigger tiles, the whole point being to judge a composition
  // before deciding — a hard object-cover square crop hides exactly what you'd
  // need to see (is this shot portrait or landscape?). So L switches to
  // object-contain (letterboxed on the existing black tile background); S/M
  // stay object-cover so the dense overview grid reads as a clean tiled wall.
  // object-top keeps the top of the frame — where the face is — instead of
  // centering the crop and taking a bite out of the forehead.
  const imgFitCls = tileSize === 'L' ? 'object-contain' : 'object-cover object-top';
  // Provenance badge text. Kept as data (not inline JSX) so the SAME wording
  // can go into title/aria-label — the badge is clamped to two lines at the
  // bottom of a narrow tile, and a truncated engine name must stay readable
  // on hover and to a screen reader.
  // 'ready' | 'generating' | undefined, decided once for the whole grid
  // (DatasetGrid) so each tile is a lookup, not a scan of every sibling.
  const improvementBadgeInfo = improvementBadge(improvementState);
  const originText = DERIVATION_LABEL[img.derivation_kind]
    || (img.source === 'import' ? 'real' : 'generated');
  const engineLabel = ENGINE_ACCENTS[img.engine] ? ENGINE_LABELS[img.engine] : null;
  const provenanceTitle = [originText, img.framing, engineLabel && `made with ${engineLabel}`]
    .filter(Boolean).join(' · ');

  return (
    <div tabIndex={0} aria-label={`${displayLabel(img.variation_label) || 'Dataset image'} card`}
      /* The photo IS the tile: no white card, no inner margin, no status border
         (the state is the corner dot below) and no selection ring (the state is
         the outward glow). TILE_SURFACE carries the radius + clip; the resting
         shadow is swapped for the glow while selected — and a selected tile
         carries its OWN hover glow (TILE_SELECTED_GLOW includes the hover twin)
         instead of FLOAT_HOVER, whose hover shadow would replace the glow. */
      data-selected={selected || undefined}
      className={`dataset-grid-item ${TILE_SURFACE} ${selected ? TILE_SELECTED_GLOW : `${TILE_SHADOW} ${FLOAT_HOVER}`} ${rejected ? 'opacity-50' : ''}`}>
      <div className="relative aspect-square bg-black">
        {/* Fresh content nobody has looked at yet — a first-time generation
            OR a regenerate, not yet opened (unlike the in-progress emerald
            glow above, this one is meant to survive being buried back among
            the others once done). Top-left/top-right are both claimed (face
            badge, wide-tile provenance re-anchor, the hover-revealed action
            row) — top-right corner, just outside the action row's own inset,
            is the one spot nothing else occupies at rest. Always visible,
            not hover-gated: cleared server-side the moment the tile is
            opened (see onView below). */}
        {/* State DOT + lock, one row at the top-left, both always visible. The
            dot is the kept/rejected/undecided decision (mounted, data-status
            so a test can name it); the lock badge sits beside it — top-left is
            the corner nothing else claims at rest, same rule as the unseen dot
            top-right. */}
        <span data-status={img.status} role="img"
          title={generating ? 'Generating…' : statusDot.label}
          aria-label={generating ? 'Generating…' : statusDot.label}
          className={`absolute top-1.5 left-1.5 z-20 h-2.5 w-2.5 rounded-full ring-2 ring-black/25 ${statusDot.cls} ${generating ? 'animate-pulse' : ''}`} />
        {img.is_locked && (
          <span
            className="absolute top-1.5 left-5 z-20 grid place-items-center w-5 h-5 rounded-full bg-black/70 text-amber-700 text-[11px]"
            title="Locked — cannot be deleted until unlocked" aria-label="Locked, cannot be deleted">
            🔒
          </span>
        )}
        {img.unseen && (
          <span
            className="absolute top-1.5 right-1.5 z-20 w-2.5 h-2.5 rounded-full bg-red-500 ring-2 ring-black/60"
            title="New — not yet viewed" aria-label="New, not yet viewed" />
        )}
        {onToggleSelect && img.filename && (
          <label
            className={`dataset-grid-item__actions absolute bottom-1 left-1 z-10 flex items-center justify-center w-6 h-6 rounded-full ${ON_PHOTO_HOVER} cursor-pointer`}
            title="Select for bulk actions"
            onClick={(e) => e.stopPropagation()}>
            {/* NOT gated on `busy`: ticking changes nothing on the server, and
                a running pass was handed its id list at launch time — a
                selection made afterwards cannot shift what it is processing.
                What DOES gate this tick is a bulk action currently CONSUMING
                the selection, and that one is decided by the grid, which
                withholds `onToggleSelect` entirely. */}
            <input type="checkbox" checked={selected}
              onChange={() => onToggleSelect(img.id)}
              aria-label={`Select ${displayLabel(img.variation_label) || 'this image'} for bulk actions`}
              className="w-4 h-4 accent-indigo-500 cursor-pointer" />
          </label>
        )}
        {url ? (
          // A pure read: it opens the same bytes the tile is already showing.
          // No `disabled={busy}` — that is the whole point of this change.
          <button type="button" onClick={() => onView?.(img)}
            title="Inspect (zoom)"
            aria-label={`Inspect ${displayLabel(img.variation_label) || 'the image'} full screen`}
            className="block w-full h-full cursor-zoom-in disabled:cursor-not-allowed">
            {/* The tile and the lightbox request the SAME url, so the tile is
                where the intrinsic size is known FIRST. Recording it here is
                what lets the lightbox open with its actions already in the
                right place, instead of committing them once the image paints
                — see lightboxActionPlacement.js. */}
            <img src={url} alt={displayLabel(img.variation_label)} loading="lazy"
              onLoad={(e) => rememberImageRatio(
                img.id, e.currentTarget.naturalWidth, e.currentTarget.naturalHeight)}
              className={`w-full h-full ${imgFitCls}`} />
          </button>
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center gap-1 px-2 text-center"
            title={img.status === 'failed' ? (img.fail_reason || 'generation failed') : undefined}>
            {img.status === 'failed' ? (
              <>
                <span className="text-red-600 text-xs font-semibold">⚠ failed</span>
                {img.fail_reason && (
                  <span className="text-content-subtle text-[0.5625rem] leading-tight line-clamp-4 break-words">
                    {img.fail_reason}
                  </span>
                )}
                <span className="text-content-subtle text-[0.5625rem]">🔄 to retry</span>
              </>
            ) : generating ? (
              <span className="text-emerald-700 text-xs font-semibold animate-pulse">⚙ generating…</span>
            ) : (
              <span className="text-content-subtle text-xs">…</span>
            )}
          </div>
        )}
        {/* Bottom-anchored badge stack. A container query (index.css) lifts the
            provenance badge back to the top-left as soon as the TILE — not the
            window — is wide enough to hold it next to the action buttons. */}
        <div className={TILE_BADGE_STACK_CLASS}>
          {/* A face swap that ended without a picture — cancelled, or failed in
              ComfyUI — puts this tile's PREVIOUS image back. Without a badge the
              restore is completely silent: the tile shows the same photo it
              showed before, so nothing anywhere says the swap was attempted and
              lost. `fail_reason` on a row that has an image and is not `failed`
              is exactly that case, and only that case. */}
          {swapRestoredNote && (
            <span className={`${WATERMARK_BADGE_CLASS} bg-black/70 text-amber-700`}
              title={swapRestoredNote} aria-label={swapRestoredNote}>
              ↩ restored
            </span>
          )}
          {/* An upscale of THIS image is waiting (or still rendering). The
              candidate is a separate row that lands on its own tile elsewhere
              in the grid, so from here nothing looked like it had happened —
              and the pass got re-run on images that already had a result,
              paying GPU time for a duplicate. */}
          {improvementBadgeInfo && (
            <span className={`${WATERMARK_BADGE_CLASS} bg-black/70 ${
              improvementBadgeInfo.tone === 'ready'
                ? 'text-indigo-700' : 'text-white/70'}`}
              title={improvementBadgeInfo.title} aria-label={improvementBadgeInfo.title}>
              {improvementBadgeInfo.text}
            </span>
          )}
          {wb && (
            <span className={`${WATERMARK_BADGE_CLASS} bg-black/70 ${wb.cls}`}
              title={(img.watermark_state === 'detected' && WATERMARK_ROUTE_HINT[img.watermark_route]) || wb.label}>
              {wb.icon} {wb.text}
            </span>
          )}
          {/* Last child = closest to the bottom edge, and the engine pill names
              which engine made it — the only way a multi-engine batch is
              comparable ("this one came out of ChatGPT, that one out of
              Klein"). The server sends `engine` ONLY when it can tell for sure,
              so older rows show no pill rather than a made-up one. */}
          <span className={`${PROVENANCE_BADGE_CLASS} bg-black/60 text-white`}
            title={provenanceTitle} aria-label={provenanceTitle}>
            {originText}{img.framing ? ` · ${img.framing}` : ''}
            {engineLabel && (
              <span className={`dataset-tile-badge__engine ml-1 px-1 rounded ${ENGINE_ACCENTS[img.engine].pill}`}>
                {engineLabel}
              </span>
            )}
          </span>
        </div>
        {fb && (
          <span className={`${FACE_BADGE_CLASS} px-1.5 py-0.5 rounded bg-black/70 ${fb.cls}`}
            title={`Resemblance to the reference face — ${fb.label}`}>
            {fb.graded ? fb.label : fb.icon}
          </span>
        )}
        <div className={`dataset-grid-item__actions absolute top-1.5 right-1.5 z-10 flex max-w-[calc(100%_-_1rem)] flex-wrap justify-end gap-0.5 ${GLASS_CAPSULE_DARK} p-0.5`}>
          {fb && (
            <span className={`flex items-center gap-1 rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${fb.cls}`}
              title={`Resemblance to the reference face — ${fb.label}`}>
              {fb.graded ? fb.label : fb.icon}
            </span>
          )}
          {url && onScoreFace && ['keep', 'pending'].includes(img.status) && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); onScoreFace(img.id); }}
              disabled={busy || faceScoringBusy || !!faceScoringBlocked || scoreFaceBusy}
              aria-busy={scoreFaceBusy}
              title={scoreFaceTitle} aria-label={scoreFaceTitle}
              className={`grid min-h-5 min-w-5 place-items-center rounded-full ${ON_PHOTO_HOVER} text-[10px] text-white disabled:cursor-not-allowed disabled:opacity-45`}>
              <span aria-hidden="true" className={scoreFaceBusy ? 'animate-pulse' : ''}>{scoreFaceBusy ? '…' : '🎭'}</span>
            </button>
          )}
          {canRegenerate && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); onRegenerate?.(img.id); }}
              disabled={busy}
              title={refused || 'Regenerate this variation (new seed)'}
              aria-label={refused || 'Regenerate this variation (new seed)'}
              className={`grid min-h-5 min-w-5 place-items-center rounded-full ${ON_PHOTO_HOVER} text-white text-[10px] disabled:cursor-not-allowed disabled:opacity-45`}>🔄</button>
          )}
          {canRegenerate && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); setEditingPrompt(true); }}
              disabled={busy}
              title={refused || 'Edit the prompt, then regenerate this variation'}
              aria-label={refused || 'Edit the prompt, then regenerate this variation'}
              className={`grid min-h-5 min-w-5 place-items-center rounded-full ${ON_PHOTO_HOVER} text-white text-[10px] disabled:cursor-not-allowed disabled:opacity-45`}>✏️</button>
          )}
          {canFaceSwap && onFaceSwap && (
            /* Disabled WHILE THE REQUEST IS OUT, not merely while the row is
               pending: on the app-mask lane the server spends ~20 s masking
               before the row changes at all, and a button that still looks
               clickable through that gets clicked again — one tile, three
               swaps. `aria-busy` is what a screen reader hears meanwhile. */
            <button type="button"
              onClick={(e) => { e.stopPropagation(); onFaceSwap(img.id); }}
              disabled={busy || swapBusy}
              aria-busy={swapBusy}
              title={swapBusy ? 'Preparing the face swap…'
                : "Swap this tile's face with the reference image"}
              aria-label={swapBusy ? 'Preparing the face swap…'
                : "Swap this tile's face with the reference image"}
              className={`grid min-h-5 min-w-5 place-items-center rounded-full ${ON_PHOTO_HOVER} text-white disabled:cursor-not-allowed disabled:opacity-45`}>
              <span hidden={swapBusy}><FaceSwapIcon className="h-3.5 w-3.5" /></span>
              <span hidden={!swapBusy} className="animate-pulse text-[10px]">…</span>
            </button>
          )}
          {/* ↩ Undo a swap that already landed. Offered only while the server
              says the picture it replaced is still in the Trash, so the button
              never promises something emptying the Trash has taken away. */}
          {img.can_undo_swap && onUndoFaceSwap && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); onUndoFaceSwap(img.id); }}
              disabled={busy || swapBusy}
              title="Undo the face swap — bring back the image it replaced"
              aria-label="Undo the face swap — bring back the image it replaced"
              className={`grid min-h-5 min-w-5 place-items-center rounded-full ${ON_PHOTO_HOVER} text-white text-[10px] disabled:cursor-not-allowed disabled:opacity-45`}>↩🎭</button>
          )}
          {rerunImprove && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); onReimprove?.(img.id); }}
              disabled={busy || !rerunImprove.enabled}
              title={refused || rerunImprove.title}
              aria-label={refused || rerunImprove.title}
              className={`grid min-h-5 min-w-5 place-items-center rounded-full ${ON_PHOTO_HOVER} text-[10px] text-white disabled:cursor-not-allowed disabled:opacity-45`}>
              <span aria-hidden="true">🔄✨</span>
            </button>
          )}
          {url && onMirror && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); onMirror(img.id); }}
              disabled={busy || mirrorBusy}
              aria-busy={mirrorBusy}
              aria-label={refused || (mirrorBusy
                ? `Mirroring ${displayLabel(img.variation_label) || 'this image'} horizontally`
                : `Mirror ${displayLabel(img.variation_label) || 'this image'} horizontally`)}
              title={refused
                || (mirrorBusy ? 'Mirroring horizontally…' : 'Mirror horizontally (flip left and right)')}
              className={`grid min-h-5 min-w-5 place-items-center rounded-full ${ON_PHOTO_HOVER} text-[10px] text-white disabled:cursor-not-allowed disabled:opacity-45`}>
              <span aria-hidden="true">{mirrorBusy ? '…' : '⇆'}</span>
            </button>
          )}
          {url && (
            <button type="button" onClick={(e) => { e.stopPropagation(); onCrop(img); }}
              disabled={busy}
              title={refused || 'Crop'} aria-label={refused || 'Crop'}
              className={`grid min-h-5 min-w-5 place-items-center rounded-full ${ON_PHOTO_HOVER} text-white text-[10px] disabled:cursor-not-allowed disabled:opacity-45`}>✂</button>
          )}
          {img.status === 'keep' && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); setCaptionEditorOpen(true); }}
              disabled={busy}
              title={refused || 'Open a larger caption editor'}
              aria-label={refused || 'Expand caption editor'}
              className={`grid min-h-5 min-w-5 place-items-center rounded-full ${ON_PHOTO_HOVER} text-white text-[10px] disabled:cursor-not-allowed disabled:opacity-45`}>⛶</button>
          )}
          {img.status === 'keep' && cap && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); editingRef.current = false; setCap(''); onCaption(img.id, ''); }}
              disabled={busy}
              title={refused || 'Delete this image’s caption (then “Caption” regenerates it via JoyCaption)'}
              aria-label={refused || 'Delete this image’s caption'}
              className={`grid min-h-5 min-w-5 place-items-center rounded-full ${ON_PHOTO_HOVER} text-white text-[10px] disabled:cursor-not-allowed disabled:opacity-45`}>🗑</button>
          )}
          {onLockToggle && (
            <button type="button"
              onClick={(e) => { e.stopPropagation(); onLockToggle(img.id, !img.is_locked); }}
              title={img.is_locked ? 'Unlock (allow delete again)' : 'Lock (cannot be deleted)'}
              aria-label={img.is_locked ? 'Unlock this image' : 'Lock this image against deletion'}
              aria-pressed={!!img.is_locked}
              className={`grid min-h-5 min-w-5 place-items-center rounded-full text-[10px] ${img.is_locked ? 'bg-amber-400 text-black' : `${ON_PHOTO_HOVER} text-white`}`}>
              {img.is_locked ? '🔒' : '🔓'}
            </button>
          )}
          {!isRescueDerived && (
            <button type="button"
              onClick={(e) => {
                e.stopPropagation();
                if (img.is_locked || busy) return;
                if (window.confirm('Permanently delete this image?')) onDelete(img.id);
              }}
              disabled={busy || img.is_locked}
              title={img.is_locked ? 'Locked — unlock to delete' : (refused || 'Delete permanently')}
              aria-label={img.is_locked ? 'Locked — unlock to delete' : (refused || 'Delete permanently')}
              className={`grid min-h-5 min-w-5 place-items-center rounded-full text-red-300 ${ON_PHOTO_HOVER} disabled:cursor-not-allowed disabled:opacity-45`}>
              <TrashIcon className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        {editingPrompt && (
          <PromptEditPopover
            initialPrompt={img.variation_prompt || ''}
            onSubmit={(prompt) => onRegenerate?.(img.id, undefined, prompt, { silent: true })}
            onClose={() => setEditingPrompt(false)} />
        )}
        {/* Caption — a DARK glass capsule on the photo, revealed on hover like
            the toolbar, one line, truncated; the full caption lives in the
            Lightbox. Read-only here: it opens the editor on click. This is its
            OWN gate (.dataset-grid-item__caption), NOT .dataset-grid-item__actions:
            that one keeps controls visible on touch screens (they must be
            reachable), while a caption permanently visible on touch would
            resurrect the always-on bottom plate. Mounted, never unloaded —
            hover show/hide is the CSS opacity gate, because Chrome auto-
            translate rewrites text nodes and a React removal would crash
            (CLAUDE.md ▸ UI changes 5). */}
        {(cap || '').trim() && img.status === 'keep' && (
          <button type="button"
            onClick={(e) => { e.stopPropagation(); if (!busy) setCaptionEditorOpen(true); }}
            title={busy ? (refused || 'Wait for the running pass to finish') : 'Click to edit the caption'}
            aria-label="Edit the caption"
            className={`dataset-grid-item__caption absolute inset-x-2 bottom-14 z-20 flex items-center gap-1.5 ${GLASS_CAPSULE_DARK} px-2.5 py-1 text-left`}>
            <span className="min-w-0 flex-1 truncate text-[11px] leading-snug text-white">{cap}</span>
            {captionOriginInfo(img.caption_origin).known && (
              <span className={`shrink-0 text-[9px] leading-none ${captionIsAsserted(img.caption_origin) ? 'text-emerald-300' : 'text-white/60'}`}>
                {captionOriginInfo(img.caption_origin).chip}
              </span>
            )}
          </button>
        )}
        {/* Source credit — ALWAYS visible (it is a legal attribution), a tiny
            frosted pill just above the keep/reject row. */}
        <SourceAttribution metadata={img.source_metadata}
          className={`absolute bottom-9 left-2 z-20 max-w-[calc(100%_-_1rem)] ${GLASS_CAPSULE_DARK} px-1.5 py-0.5 text-[0.5625rem] leading-relaxed text-white/80`} />
        {/* Keep / reject — now ON the photo, above the caption capsule, flatter pills. */}
        {isRescueDerived ? (
          <span className={`absolute inset-x-2 bottom-2 z-20 rounded-full ${GLASS_CAPSULE_DARK} px-2 py-1 text-center text-[0.625rem] text-white/90`}
            title="This winner was chosen atomically with its provenance pair. Caption and crop remain available.">
            ✓ Chosen in Klein rescue review
          </span>
        ) : (
          <div className="dataset-grid-item__actions absolute inset-x-2 bottom-2 z-20 flex gap-1.5">
            <button type="button" onClick={() => onStatus(img.id, img.status === 'keep' ? 'pending' : 'keep')}
              disabled={busy}
              title={refused || 'Keep'} aria-label={refused || 'Keep'}
              aria-pressed={img.status === 'keep'}
              className={`flex-1 py-1 rounded-full text-[11px] disabled:cursor-not-allowed disabled:opacity-45 ${img.status === 'keep' ? 'bg-gradient-to-b from-green-400 to-green-700 text-white shadow-[0_2px_6px_rgba(22,163,74,0.35)]' : `${GLASS_CAPSULE_DARK} text-white/80`}`}>✓</button>
            <button type="button"
              onClick={() => {
                // Rejecting a GENERATED image offers an immediate retry of the same
                // variation (in place, new seed) so the composition stays on target.
                if (!isImageImproveCandidate && img.status !== 'reject'
                    && img.source === 'generated' && img.filename && onRegenerate
                    && window.confirm('Photo rejected — regenerate a new attempt of this variation?\n(OK = replace with a new attempt · Cancel = reject only)')) {
                  onRegenerate(img.id);
                  return;
                }
                onStatus(img.id, img.status === 'reject' ? 'pending' : 'reject');
              }}
              disabled={busy}
              title={refused || 'Reject (offers a regeneration)'} aria-label={refused || 'Reject'}
              aria-pressed={img.status === 'reject'}
              className={`flex-1 py-1 rounded-full text-[11px] ${GLASS_CAPSULE_DARK} text-white/80 disabled:cursor-not-allowed disabled:opacity-45`}>✕</button>
          </div>
        )}
      </div>
      {captionEditorOpen && (
        <CaptionEditorDialog initialCaption={cap} imageUrl={url}
          datasetId={datasetId} imageId={img.id}
          initialShortCaption={img.caption_short || ''} showShort={dualCaptions}
          captionOrigin={img.caption_origin} shortCaptionOrigin={img.caption_short_origin}
          imageLabel={displayLabel(img.variation_label)}
          onClose={() => setCaptionEditorOpen(false)}
          /* Awaited, and its answer is handed BACK: an unawaited handler cannot
             know it was refused, which is how a refused save used to close the
             editor and destroy the caption. `silent` because the dialog draws
             the refusal itself, next to the text it is about. The tile's own
             copy is only advanced on a success — otherwise a failed save would
             show the new text on a tile the server never accepted. */
          onSave={async (nextCaption, nextShort) => {
            if (busy) {
              return { ok: false,
                error: refused || 'Wait for the running dataset pass to finish before saving.' };
            }
            // Persist when either field changed; `nextShort` is undefined unless dual is on.
            const changed = nextCaption !== (img.caption || '')
              || (nextShort !== undefined && nextShort !== (img.caption_short || ''));
            if (!changed) return { ok: true };
            const outcome = await onCaption(img.id, nextCaption, nextShort, { silent: true });
            if (outcome?.ok) { editingRef.current = false; setCap(nextCaption); }
            return outcome;
          }} />
      )}
    </div>
  );
}
