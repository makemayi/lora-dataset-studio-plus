// react-frontend/src/components/dataset/studio/ResultTile.jsx
/**
 * Une vignette (image + 👍/👎) pour UNE génération (un seed). En batch, plusieurs
 * tuiles sont affichées côte à côte dans la même cellule (bande). Extrait 1:1 du
 * `renderTile` de l'ancien LoraTestStudio (behavior-preserving).
 *
 * La clé `key={cell.id}` est posée par le PARENT (ResultCell) lors du `.map`.
 */
export default function ResultTile({ cell, row, strength, variant, datasetId, onRate, onOpen, fmt }) {
  const isGenerating = cell.queue_status === 'generating';
  // Current rows remain `pending` while their queue is stalled so Cancel/Resume
  // continues to target the same run. Accept `status: stalled` too: older or
  // in-flight payloads must never fall back to a misleading spinning "queued" tile.
  const isStalled = cell.queue_status === 'stalled'
    && (cell.status === 'pending' || cell.status === 'stalled');
  // The backend supplies a redacted, paste-safe queue reason. Keep it visible
  // instead of hiding the same sentence in a hover-only tooltip.
  const stalledReason = typeof cell.queue_error === 'string' && cell.queue_error.trim()
    ? cell.queue_error.trim()
    : 'ComfyUI needs to be recovered before this tile can run.';
  return (
    <div className="flex flex-col gap-1 items-center">
      {isStalled && (
        <div
          role="status" aria-live="polite" aria-atomic="true"
          className="w-20 min-h-28 rounded-md border border-amber-500/50 bg-amber-500/10 flex flex-col gap-1 items-center justify-center px-1 py-1 text-center text-amber-200 text-[0.625rem]">
          <span aria-hidden className="text-sm">⏸</span>
          <span className="font-semibold">paused</span>
          <p className="m-0 break-words leading-tight select-text text-amber-200/80">{stalledReason}</p>
          <span className="leading-tight text-amber-200/90">Recover or restart ComfyUI, then cancel and resume.</span>
        </div>
      )}
      {cell.status === 'pending' && !isStalled && (
        <div className="w-20 h-28 rounded-md border border-border bg-surface flex flex-col gap-2 items-center justify-center"
          role="status" aria-label={isGenerating ? 'Generating' : 'Queued'}>
          <span className="inline-block w-5 h-5 border-2 border-purple-400/40 border-t-purple-400 rounded-full animate-spin" aria-hidden />
          <span className="text-content-muted text-[0.625rem]">
            {isGenerating ? 'generating' : 'queued'}
          </span>
        </div>
      )}
      {/* Failed tiles are no longer mute: the real reason (ComfyUI validation /
          node error / timeout, from the `error` column) shows on hover so the
          user knows WHY instead of relaunching blind (P0-b). */}
      {cell.status === 'failed' && (
        <div
          title={cell.error || 'Generation failed — see the 🪵 Server log in Settings for details.'}
          className="w-20 h-28 overflow-hidden rounded-md border border-red-500/50 bg-red-500/10 flex flex-col items-center justify-center gap-0.5 text-red-300 text-[0.625rem] cursor-help px-1 text-center">
          <span aria-hidden className="text-sm">⚠</span>
          <span>failed</span>
          {cell.error && <span className="text-red-300/70 leading-tight line-clamp-3">{cell.error}</span>}
        </div>
      )}
      {cell.status === 'cancelled' && (
        <div className="w-20 h-28 rounded-md border border-amber-500/40 bg-amber-500/10 flex flex-col items-center justify-center text-amber-300 text-[0.625rem] gap-0.5"><span aria-hidden>⏸</span> stopped</div>
      )}
      {cell.status === 'done' && cell.filename && (
        <button type="button" onClick={() => onOpen(cell)}
          title={`${row.label} @ ${fmt(strength)} (${variant.aspect || '—'}) seed ${cell.seed}${cell.batch_lora ? ` · + ${cell.batch_lora}` : ''} — open larger`}
          className="block p-0 m-0 border-0 bg-transparent cursor-pointer">
          <img src={`/api/dataset/${datasetId}/img/${encodeURIComponent(cell.filename)}`}
            alt={`${row.label} strength ${fmt(strength)} ${variant.aspect || ''} seed ${cell.seed}`} loading="lazy"
            className="w-20 h-28 object-cover rounded-md border border-border" />
        </button>
      )}
      {/* Score facial objectif (InsightFace vs référence) — mêmes seuils que le
          Dataset Maker : ≥0.50 vert, ≥0.45 orange, sinon rouge. */}
      {cell.face_score != null && (
        <span title={`Face similarity vs the dataset reference: ${cell.face_score.toFixed(3)}`}
          className={`px-1 py-px rounded border text-[0.5625rem] font-semibold tabular-nums ${cell.face_score >= 0.50
            ? 'border-emerald-400/50 bg-emerald-400/10 text-emerald-300'
            : cell.face_score >= 0.45
              ? 'border-amber-400/50 bg-amber-400/10 text-amber-300'
              : 'border-red-400/50 bg-red-400/10 text-red-300'}`}>
          🎯 {cell.face_score.toFixed(2)}
        </span>
      )}
      {/* Badge de l'axe ⚖ batch : distingue la cellule AVEC le LoRA testé de sa
          jumelle sans (même config, même seed). */}
      {cell.batch_lora && (
        <span className="max-w-[5rem] truncate px-1 py-px rounded border border-amber-400/50 bg-amber-400/15 text-amber-300 text-[0.5625rem] font-semibold"
          title={`Batch axis: with ${cell.batch_lora}`}>
          + {cell.batch_lora}
        </span>
      )}
      {/* Pile combinée : les LoRA chargés EN PLUS de celui de la colonne, avec leur
          poids — sans ce badge une image de pile serait indiscernable d'un run solo.
          Depuis le balayage 🧬, un même run contient PLUSIEURS combinaisons : la
          tuile doit donc dire LAQUELLE elle est, poids de tête compris. Sans ça,
          neuf images d'un balayage sont neuf tuiles identiques au badge près. */}
      {Array.isArray(cell.combined_loras) && cell.combined_loras.length > 0 && (
        <span className="max-w-[5rem] truncate px-1 py-px rounded border border-sky-400/50 bg-sky-400/15 text-sky-300 text-[0.5625rem] font-semibold tabular-nums"
          title={`Blend: ${[fmt(cell.strength), ...cell.combined_loras.map((e) => `${e.label} @ ${e.weight}`)].join(' × ')}`}>
          🧬 {fmt(cell.strength)}
          {cell.combined_loras.map((e) => ` × ${e.weight}`).join('')}
        </span>
      )}
      {/* Votes only make sense for a finished image — a failed/pending/stopped tile
          has nothing to judge and must not pollute the ranking (P0-b). */}
      {cell.status === 'done' && (
        <div className="flex items-center gap-1">
          <button type="button" aria-pressed={cell.rating === 1}
            aria-label={`Like ${row.label} @ ${fmt(strength)} (${variant.aspect || '—'}) seed ${cell.seed}`}
            onClick={() => onRate(cell.id, cell.rating === 1 ? 0 : 1)}
            className={`px-1.5 py-0.5 rounded text-[0.75rem] border ${cell.rating === 1 ? 'border-green-400/60 bg-green-500/20' : 'border-border bg-surface opacity-70'}`}>👍</button>
          <button type="button" aria-pressed={cell.rating === -1}
            aria-label={`Dislike ${row.label} @ ${fmt(strength)} (${variant.aspect || '—'}) seed ${cell.seed}`}
            onClick={() => onRate(cell.id, cell.rating === -1 ? 0 : -1)}
            className={`px-1.5 py-0.5 rounded text-[0.75rem] border ${cell.rating === -1 ? 'border-red-400/60 bg-red-500/20' : 'border-border bg-surface opacity-70'}`}>👎</button>
        </div>
      )}
    </div>
  );
}
