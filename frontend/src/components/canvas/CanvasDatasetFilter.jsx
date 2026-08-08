import { useState } from 'react';
import { selectionSummary } from '../../utils/canvasSelection';
import {
  familyLabel, readCanvasFilterOpen, writeCanvasFilterOpen,
} from '../../utils/canvasFamilyFilter';

/* Which datasets sit on the board.

   The canvas' whole promise is "all your datasets at once", so this is a
   SUBTRACTIVE control: everything is on the board until you take something off.

   It folds, and it opens FOLDED — at every width. It used to open expanded on a
   wide screen, "because there it costs one row"; it costs a row plus a wrapping
   checkbox list per dataset, and on a library of any size that pushed the board
   — the thing you came to look at — under the fold on every single load. A
   filter is not something you consult on arrival.
   Nothing is hidden by that: the button says what is currently shown ("3 of 7"),
   so the state is readable folded, never behind a mystery icon. And unfolding is
   REMEMBERED, so someone who works with the filter open keeps it open. */

export default function CanvasDatasetFilter({
  datasets, selected, onToggle, onAll, onNone,
  families, selectedFamilies, onToggleFamily, onAllFamilies, onNoFamilies,
  query, onQueryChange, statuses, selectedStatuses, onToggleStatus,
  showPinned, onTogglePinned, onResetFilters, visibleRuns,
}) {
  // Replié tant que l'utilisateur n'a pas dit le contraire — et son choix survit
  // au rechargement. Plus de largeur d'écran ici : l'ancienne version rouvrait
  // le panneau sur tout redimensionnement vers `sm`, ce qui aurait annulé un pli
  // délibéré à chaque fois que la fenêtre grandit.
  const [open, setOpen] = useState(
    () => readCanvasFilterOpen(typeof localStorage !== 'undefined' ? localStorage : null));
  const toggleOpen = () => setOpen((v) => {
    writeCanvasFilterOpen(typeof localStorage !== 'undefined' ? localStorage : null, !v);
    return !v;
  });

  const sel = new Set(selected || []);
  const familySel = new Set(selectedFamilies || []);
  const total = (datasets || []).length;

  return (
    <section className="mb-3 rounded-xl border border-border bg-surface-raised p-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <button type="button"
          onClick={toggleOpen}
          aria-expanded={open}
          className="flex h-9 items-center gap-1.5 rounded-md border border-border bg-app/60 px-2.5 text-content text-[0.75rem] font-semibold hover:border-indigo-400/50">
          <span aria-hidden>{open ? '▾' : '▸'}</span> Datasets
          <span className="font-normal text-content-muted">· {selectionSummary(sel.size, total)}</span>
          <span className="font-normal text-content-subtle">· {visibleRuns} runs shown</span>
        </button>
        {open && total > 0 && (
          <div className="flex items-center gap-1.5">
            <button type="button" onClick={onAll}
              className="flex h-9 items-center rounded-md border border-border bg-app/60 px-2.5 text-content-muted text-[0.6875rem] hover:text-content">
              Select all
            </button>
            <button type="button" onClick={onNone}
              className="flex h-9 items-center rounded-md border border-border bg-app/60 px-2.5 text-content-muted text-[0.6875rem] hover:text-content">
              Clear
            </button>
          </div>
        )}
      </div>
      {open && (
        total === 0 ? (
          <p className="mt-2 text-content-subtle text-[0.75rem]">
            No dataset has been trained yet — the canvas draws training runs, so it fills up
            after your first run finishes.
          </p>
        ) : (
          <>
          <div className="mt-2 grid gap-2 border-t border-border pt-2 sm:grid-cols-[minmax(12rem,1fr)_auto_auto]">
            <label className="sr-only" htmlFor="canvas-filter-search">Search canvas runs</label>
            <input id="canvas-filter-search" type="search" value={query}
              onChange={(e) => onQueryChange(e.target.value)}
              placeholder="Search dataset, run ID, model or variant…"
              className="h-9 min-w-0 rounded-md border border-border bg-app/60 px-3 text-content text-[0.75rem] placeholder:text-content-subtle" />
            <label className="flex min-h-9 cursor-pointer items-center gap-2 rounded-md border border-border bg-app/50 px-2 text-content text-[0.75rem]">
              <input type="checkbox" checked={showPinned} onChange={onTogglePinned}
                className="h-4 w-4 accent-indigo-500" />
              Pinned images
            </label>
            <button type="button" onClick={onResetFilters}
              className="h-9 rounded-md border border-border px-3 text-content-muted text-[0.75rem] hover:text-content">
              Reset filters
            </button>
          </div>
          {!!statuses?.length && (
            <fieldset className="mt-2 flex flex-wrap items-center gap-1.5">
              <legend className="sr-only">Filter by run status</legend>
              <span className="text-content-muted text-[0.6875rem] font-semibold">Status</span>
              {statuses.map((status) => (
                <label key={status}
                  className="flex min-h-9 cursor-pointer items-center gap-1.5 rounded-md border border-border bg-app/50 px-2 text-content text-[0.6875rem]">
                  <input type="checkbox" checked={selectedStatuses.includes(status)}
                    onChange={() => onToggleStatus(status)} className="h-4 w-4 accent-indigo-500" />
                  {status === 'active' ? 'Active' : status === 'completed' ? 'Completed'
                    : status === 'error' ? 'Errors' : 'Unknown'}
                </label>
              ))}
            </fieldset>
          )}
          <div className="mt-2 flex flex-wrap items-center gap-1.5 pt-2"
            aria-label="Filter canvas by model">
            <span className="mr-0.5 text-content-muted text-[0.6875rem] font-semibold">Models</span>
            {(families || []).map((family) => (
              <label key={family}
                className="flex min-h-9 cursor-pointer items-center gap-1.5 rounded-md border border-border bg-app/50 px-2 text-content text-[0.6875rem] hover:border-indigo-400/50">
                <input type="checkbox" checked={familySel.has(family)}
                  onChange={() => onToggleFamily(family)}
                  className="h-4 w-4 shrink-0 accent-indigo-500" />
                <span>{familyLabel(family)}</span>
              </label>
            ))}
            {(families || []).length > 0 && (
              <div className="flex items-center gap-1">
                <button type="button" onClick={onAllFamilies}
                  className="min-h-9 rounded-md px-2 text-content-muted text-[0.625rem] hover:text-content">
                  All models
                </button>
                <button type="button" onClick={onNoFamilies}
                  className="min-h-9 rounded-md px-2 text-content-muted text-[0.625rem] hover:text-content">
                  None
                </button>
              </div>
            )}
          </div>
          {familySel.size === 0 && (
            <p className="mt-2 text-amber-200/80 text-[0.6875rem]">
              No model selected — dataset choices are kept, but the board is empty.
            </p>
          )}
          <ul className="mt-2 grid max-h-56 list-none grid-cols-1 gap-1 overflow-y-auto p-0 sm:grid-cols-2 lg:grid-cols-3">
            {datasets.map((d) => (
              <li key={d.id}>
                <label className="flex min-h-[2.25rem] cursor-pointer items-center gap-2 rounded-md border border-transparent px-2 py-1 hover:border-border hover:bg-app/50">
                  <input type="checkbox" checked={sel.has(d.id)}
                    onChange={() => onToggle(d.id)}
                    className="h-4 w-4 shrink-0 accent-indigo-500" />
                  <span className="min-w-0 flex-1 truncate text-content text-[0.75rem]" title={d.name}>
                    {d.name}
                  </span>
                  <span className="shrink-0 text-content-subtle text-[0.625rem] tabular-nums"
                    title={(d.families || []).map(familyLabel).join(', ')}>
                    {d.runs}
                  </span>
                </label>
              </li>
            ))}
          </ul>
          </>
        )
      )}
    </section>
  );
}
