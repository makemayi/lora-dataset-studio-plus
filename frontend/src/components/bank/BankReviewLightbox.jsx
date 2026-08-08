/**
 * ▶ Review — the Bank's fast-triage lightbox.
 *
 * One full-size image at a time with ✓ Keep / ✕ Reject / ⏭ Skip; every button
 * (and its keyboard shortcut) decides AND moves on, so a 3 000-image dump is
 * worked through without ever going back to the grid. "🎲 Random order"
 * shuffles what's left instead of walking the folder sequentially — on a dump
 * that means a representative sample straight away rather than 200 near-
 * identical frames in a row.
 *
 * All of the ordering logic (snapshot, shuffle, skip, end of pool) lives in the
 * JSX-free `bankReview.js` so `node --test` covers it. This file is the shell:
 * rendering, keyboard, and the one-POST-per-decision network lane.
 *
 * Decisions are sent IMMEDIATELY, one image per POST: closing after 50 calls
 * loses nothing. A POST that fails does NOT advance — the image stays under the
 * cursor with the error visible, so a decision is never silently dropped.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useFocusTrap } from '../../hooks/useFocusTrap'
import {
  createSession, currentId, isFinished, progress, back, decide, skip, setShuffle,
} from './bankReview.js'
import {
  DETAIL_CAVEAT, PROVENANCE_FLAG_LABEL, detailSummary, originHint, originLabel,
} from './bankProvenance.js'
import BankWatermarkMaskDialog from './BankWatermarkMaskDialog'
import { canEditMask } from './bankWatermarkMask.js'

// How many upcoming images we pull metadata for in one go (the decision helpers
// below the image). The grid page only holds the ids it rendered.
const META_WINDOW = 40

const FLAG_TEXT = {
  blur: '🌫 Blurry', noise: '📺 Noisy', uniform: '⬜ Flat', small: '📐 Small',
  unreadable: '❌ Unreadable', low_aesthetic: '💔 Low aesthetic', nsfw: '🔞 NSFW',
  watermark: '🚩 Watermark', ...PROVENANCE_FLAG_LABEL,
}

// Origin chip colours, one per state. 'unknown' is deliberately the quiet grey
// of a non-answer rather than a warning colour: it is an absence of evidence.
const ORIGIN_CLASS = {
  ai: 'bg-violet-500/25 text-violet-100',
  camera: 'bg-emerald-500/25 text-emerald-100',
  unknown: 'bg-white/10 text-white/60',
}

function Facts({ img }) {
  if (!img) return <span className="text-xs text-white/40">Reading image details…</span>
  const chip = (key, text, cls, title) => (
    <span key={key} title={title}
      className={`rounded px-1.5 py-px text-[11px] font-medium ${cls}`}>{text}</span>
  )
  const detail = detailSummary(img)
  const origin = originLabel(img)
  const hint = originHint(img)
  return (
    <div className="flex flex-wrap items-center justify-center gap-1.5">
      <span className="max-w-[22rem] truncate text-xs text-white/70" title={img.name}>{img.name}</span>
      {chip('res', `${img.width || '?'}×${img.height || '?'}`, 'bg-white/10 text-white/80')}
      {detail && chip('detail', detail.soft ? `🫧 ~${detail.real} px real` : '🫧 full detail',
        detail.soft ? 'bg-amber-500/20 text-amber-100' : 'bg-white/10 text-white/60',
        detail.soft ? `${detail.text}. ${DETAIL_CAVEAT}` : DETAIL_CAVEAT)}
      {origin && chip('origin', `${origin.icon} ${origin.label}`, ORIGIN_CLASS[origin.state],
        `${origin.detail}${hint ? ` ${hint}` : ''}`)}
      {img.aesthetic_score != null
        && chip('aes', `✨ ${img.aesthetic_score.toFixed(1)}`, 'bg-white/10 text-amber-200')}
      {img.nsfw_score != null
        && chip('nsfw', `🔞 ${Math.round(img.nsfw_score * 100)}%`, 'bg-white/10 text-rose-200')}
      {img.blur_score != null
        && chip('sharp', `sharpness ${Math.round(img.blur_score)}`, 'bg-white/10 text-white/70')}
      {img.face_cluster != null && chip('face', `👤 #${img.face_cluster}`, 'bg-white/10 text-sky-200')}
      {img.framing && chip('framing', `📐 ${img.framing}`, 'bg-white/10 text-teal-200')}
      {/* 🎨 the medium, with the margin BEHIND it in the tooltip: a verdict shown
          without its confidence is how a guess turns into a fact. 'unsure' is
          shown too — here, where the user is looking at the picture and can
          settle it, it is the most useful thing the classifier can say. */}
      {img.medium && chip('medium', `🎨 ${img.medium}`,
        img.medium === 'unsure' ? 'bg-white/10 text-white/50' : 'bg-white/10 text-lime-200',
        img.medium_margin != null
          ? `Zero-shot CLIP over the ✨ Score embedding — confidence gap ${img.medium_margin.toFixed(3)}.`
          : null)}
      {img.face_yaw != null && chip('yaw',
        `⤢ ${Math.round(Math.abs(img.face_yaw))}°`, 'bg-white/10 text-cyan-200',
        'How far the head is turned, measured by the 🎭 Faces pass.')}
      {img.dup_group != null && chip('dup', `≈ dup #${img.dup_group}`, 'bg-white/10 text-fuchsia-200')}
      {img.semantic_dup_group != null
        && chip('sdup', `✂ same shot #${img.semantic_dup_group}`, 'bg-white/10 text-orange-200')}
      {(img.flags || []).map((f) => chip(f, FLAG_TEXT[f] || f, 'bg-amber-500/20 text-amber-200'))}
      {img.status === 'keep' && chip('st', '✓ already kept', 'bg-emerald-500/25 text-emerald-200')}
      {img.status === 'reject' && chip('st', '✕ already rejected', 'bg-rose-500/25 text-rose-200')}
      {img.promoted_dataset_id != null && chip('pr', '⬆ promoted to a dataset', 'bg-indigo-500/25 text-indigo-200')}
      {img.promoted_bank_id != null && chip('prb', '⬆ promoted to another bank', 'bg-indigo-500/25 text-indigo-200')}
    </div>
  )
}

export default function BankReviewLightbox({
  bankId, ids, startId = null, seedImages = [], onDecided, onRotated, onClose,
}) {
  const [session, setSession] = useState(() => createSession(ids, { startId }))
  const [meta, setMeta] = useState(() => {
    const m = {}
    for (const img of seedImages || []) m[img.id] = img
    return m
  })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  // 🚩 Which image's watermark mask is open, if any. Review is where a flagged
  // image is actually LOOKED at, so it is where a wrong detection box gets
  // fixed — until now that was only possible inside a dataset (Qeeyana, Reddit).
  const [maskId, setMaskId] = useState(null)
  const dialogRef = useRef(null)
  const requested = useRef(new Set())

  // Hand the trap over while the mask editor is open — two live traps fight over
  // the focus and the editor's own controls become unreachable.
  useFocusTrap(dialogRef, maskId == null)

  const id = currentId(session)
  const done = isFinished(session)
  const p = progress(session)
  const img = id == null ? null : meta[id]

  // Pull metadata for the current image and the ones just behind it, in one
  // request. Ids already asked for are never re-asked (an image deleted from the
  // bank simply stays without facts instead of looping).
  useEffect(() => {
    if (id == null) return
    const window = session.order.slice(session.pos, session.pos + META_WINDOW)
      .filter((x) => meta[x] === undefined && !requested.current.has(x))
    if (!window.length) return
    window.forEach((x) => requested.current.add(x))
    apiFetch(`/api/bank/${bankId}/images?ids=${window.join(',')}&limit=${window.length}`)
      .then((d) => setMeta((prev) => {
        const next = { ...prev }
        for (const row of d.images || []) next[row.id] = row
        return next
      }))
      .catch(() => { /* facts are an aid, not a gate — the image still shows */ })
  }, [bankId, id, session.order, session.pos, meta])

  const sendDecision = useCallback(async (status) => {
    const target = currentId(session)
    if (target == null || busy) return
    setBusy(true)
    setError(null)
    try {
      await postJson(`/api/bank/${bankId}/images/status`, { ids: [target], status })
      setMeta((prev) => (prev[target] ? { ...prev, [target]: { ...prev[target], status } } : prev))
      setSession((s) => decide(s, status))
      onDecided?.(target, status)
    } catch (e) {
      // Stay on this image: a lost decision is worse than a stalled one.
      setError(e?.message || 'Could not save that decision — it was NOT recorded.')
    } finally {
      setBusy(false)
    }
  }, [bankId, busy, onDecided, session])

  // 🔄 Quarter turn of the image under the cursor (idea by 1Tomber, GitHub #17).
  // It does NOT advance — a sideways photo is fixed and then judged, which is the
  // whole point of noticing it here. The user's own file is never rewritten: the
  // angle is stored and the app rebuilds its view from the pristine source.
  const rotateCurrent = useCallback(async (degrees) => {
    const target = currentId(session)
    if (target == null || busy) return
    setBusy(true)
    setError(null)
    try {
      const d = await postJson(`/api/bank/${bankId}/rotate`, { ids: [target], degrees })
      const angle = d?.rotations?.[target] ?? d?.rotations?.[String(target)] ?? 0
      // A quarter turn transposes the frame; a half turn does not.
      const quarter = Math.abs(degrees) % 180 !== 0
      setMeta((prev) => (prev[target]
        ? { ...prev,
            [target]: { ...prev[target],
              rotation: angle,
              width: quarter ? prev[target].height : prev[target].width,
              height: quarter ? prev[target].width : prev[target].height } }
        : prev))
      onRotated?.(target, angle)
    } catch (e) {
      setError(e?.message || 'Could not rotate that image — it was NOT changed.')
    } finally {
      setBusy(false)
    }
  }, [bankId, busy, onRotated, session])

  // Moving forward without judging IS a skip: it stays undecided in the DB and
  // is not proposed again in this session (→ and ⏭ are deliberately the same).
  const doSkip = useCallback(() => { setError(null); setSession((s) => skip(s)) }, [])
  const goBack = useCallback(() => { setError(null); setSession((s) => back(s)) }, [])
  const toggleShuffle = useCallback(() => setSession((s) => setShuffle(s, !s.shuffle)), [])

  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      // The mask editor owns the keyboard while it is open: every letter here is
      // one keystroke away from a decision on the image being masked.
      if (maskId != null) return
      if (e.key === 'Escape') { onClose(); return }
      // Only TEXT entry swallows the shortcuts. A blanket "input" guard broke
      // the whole mode: the focus trap lands on the 🎲 checkbox when the
      // lightbox opens, so K/R/S did nothing until you clicked elsewhere.
      const el = e.target
      const tag = (el?.tagName || '').toLowerCase()
      const type = (el?.type || '').toLowerCase()
      const typing = tag === 'textarea' || tag === 'select' || el?.isContentEditable
        || (tag === 'input' && !['checkbox', 'radio', 'button', 'submit', 'range'].includes(type))
      if (typing) return
      const k = e.key.toLowerCase()
      if (k === 'k') { e.preventDefault(); sendDecision('keep') }
      else if (k === 'r') { e.preventDefault(); sendDecision('reject') }
      else if (k === 's' || e.key === 'ArrowRight') { e.preventDefault(); doSkip() }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); goBack() }
      // [ and ] turn the image without deciding anything. Deliberately NOT
      // letters: every free letter here is one keystroke away from a decision.
      else if (e.key === '[') { e.preventDefault(); rotateCurrent(-90) }
      else if (e.key === ']') { e.preventDefault(); rotateCurrent(90) }
      // M edits the watermark mask of the flagged image under the cursor.
      else if (k === 'm' && canEditMask(img)) { e.preventDefault(); setMaskId(id) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, sendDecision, doSkip, goBack, rotateCurrent, maskId, img, id])

  const shortcut = (label) => (
    <kbd className="ml-1 rounded border border-white/25 px-1 text-[10px] font-mono text-white/70">{label}</kbd>
  )

  const summary = useMemo(() => {
    const bits = [`${p.kept} kept`, `${p.rejected} rejected`]
    if (p.skipped) bits.push(`${p.skipped} skipped`)
    return bits.join(' · ')
  }, [p.kept, p.rejected, p.skipped])

  return (
    <div ref={dialogRef} role="dialog" aria-modal="true" aria-label="Review the bank image by image"
      className="fixed inset-0 z-[9996] flex flex-col bg-black/95">
      <div className="flex shrink-0 flex-wrap items-center gap-3 px-4 py-2 text-sm">
        <span className="font-semibold text-white">▶ Review</span>
        <span className="tabular-nums text-white/80">
          {done ? `${p.total} / ${p.total}` : `${p.position} / ${p.total}`}
        </span>
        <span className="text-white/50">{summary}</span>
        <label className="flex items-center gap-1.5 text-xs text-white/80"
          title="Walk what's left in random order instead of folder order — on a big dump that shows you a representative sample straight away instead of 200 near-identical shots. Nothing you have already seen comes back.">
          <input type="checkbox" checked={session.shuffle} onChange={toggleShuffle} />
          🎲 Random order
        </label>
        <button type="button" onClick={onClose} title="Close (Esc)" aria-label="Close review"
          className="ml-auto h-9 w-9 rounded-full bg-white/10 text-lg leading-none text-white hover:bg-white/20">✕</button>
      </div>

      {done ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-4 p-6 text-center">
          <p className="text-2xl font-bold text-white">🎉 All {p.total.toLocaleString()} image{p.total === 1 ? '' : 's'} reviewed</p>
          <p className="text-sm text-white/70">
            {p.kept} kept · {p.rejected} rejected
            {p.skipped ? ` · ${p.skipped} skipped (still undecided)` : ''}
          </p>
          <div className="flex flex-wrap items-center justify-center gap-2">
            {p.total > 0 && (
              <button type="button" onClick={goBack}
                className="rounded-lg border border-white/25 px-4 py-2 text-sm text-white hover:bg-white/10">
                ← Back to the last image
              </button>
            )}
            <button type="button" onClick={onClose}
              className="rounded-lg bg-gradient-primary px-5 py-2 text-sm font-semibold text-white">
              Back to the grid
            </button>
          </div>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 items-center justify-center p-3">
          {/* key= forces a fresh <img> per image so a slow load never shows the
              previous shot under the new one's buttons. */}
          {/* ?r= busts the browser cache after a turn — the bytes at this URL
              change while the URL itself does not. */}
          <img key={id} src={`/api/bank/${bankId}/file/${id}${img?.rotation ? `?r=${img.rotation}` : ''}`}
            alt={img?.name || `Bank image ${id}`}
            className="max-h-full max-w-full select-none object-contain" />
        </div>
      )}

      {!done && (
        <div className="shrink-0 space-y-2 bg-black/60 px-4 py-2.5">
          {error && (
            <p role="alert" className="text-center text-sm text-rose-300">
              ⚠️ {error} — try again, or close and check the app.
            </p>
          )}
          <Facts img={img} />
          <div className="flex flex-wrap items-center justify-center gap-2">
            <button type="button" onClick={goBack} disabled={session.pos === 0}
              title="Previous image (←) — navigation only, decides nothing"
              className="rounded-lg border border-white/20 px-3 py-2 text-sm text-white disabled:opacity-35 hover:bg-white/10">
              ←
            </button>
            <button type="button" onClick={() => rotateCurrent(-90)} disabled={busy}
              aria-label="Rotate this image 90 degrees left"
              title="Rotate 90° left ([) — decides nothing. Your own file is never modified: the turn is stored and applied to what you see and to what gets promoted."
              className="rounded-lg border border-white/20 px-3 py-2 text-sm text-white disabled:opacity-35 hover:bg-white/10">
              <span aria-hidden="true">↺</span><span className="sr-only">Rotate left</span>
            </button>
            <button type="button" onClick={() => rotateCurrent(90)} disabled={busy}
              aria-label="Rotate this image 90 degrees right"
              title="Rotate 90° right (]) — decides nothing. Your own file is never modified: the turn is stored and applied to what you see and to what gets promoted."
              className="rounded-lg border border-white/20 px-3 py-2 text-sm text-white disabled:opacity-35 hover:bg-white/10">
              <span aria-hidden="true">↻</span><span className="sr-only">Rotate right</span>
            </button>
            {canEditMask(img) && (
              <button type="button" onClick={() => setMaskId(id)} disabled={busy}
                title="Fix the watermark zones on this image (M) — decides nothing. 🧽 Inpaint then repaints exactly what you draw."
                className="rounded-lg border border-amber-400/60 bg-amber-500/20 px-4 py-2 text-sm font-semibold text-amber-100 disabled:opacity-50 hover:bg-amber-500/30">
                🚩 Edit mask{shortcut('M')}
              </button>
            )}
            <button type="button" onClick={() => sendDecision('keep')} disabled={busy}
              title="Keep this image and move on (K)"
              className="rounded-lg border border-emerald-400/60 bg-emerald-500/20 px-5 py-2 text-sm font-semibold text-emerald-100 disabled:opacity-50 hover:bg-emerald-500/30">
              ✓ Keep{shortcut('K')}
            </button>
            <button type="button" onClick={() => sendDecision('reject')} disabled={busy}
              title="Reject this image and move on (R) — reversible, nothing is deleted from disk"
              className="rounded-lg border border-rose-400/60 bg-rose-500/20 px-5 py-2 text-sm font-semibold text-rose-100 disabled:opacity-50 hover:bg-rose-500/30">
              ✕ Reject{shortcut('R')}
            </button>
            <button type="button" onClick={doSkip}
              title="Decide later (S) — stays undecided and is not shown again in this review"
              className="rounded-lg border border-white/25 px-5 py-2 text-sm font-semibold text-white disabled:opacity-50 hover:bg-white/10">
              ⏭ Skip{shortcut('S')}
            </button>
          </div>
          <p className="text-center text-[11px] text-white/45">
            K keep · R reject · S skip · [ ] rotate · M watermark mask · ← → move without
            deciding · Esc close. Decisions are saved one by one — closing loses nothing.
          </p>
        </div>
      )}

      {maskId != null && meta[maskId] && (
        <BankWatermarkMaskDialog bankId={bankId} image={meta[maskId]}
          onSaved={(next) => setMeta((prev) => ({ ...prev, [next.id]: next }))}
          onClose={() => setMaskId(null)} />
      )}
    </div>
  )
}
