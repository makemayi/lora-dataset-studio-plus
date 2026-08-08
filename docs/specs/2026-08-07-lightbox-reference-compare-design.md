# Lightbox — compare an image with the dataset reference photo

*2026-08-07 · dataset lightbox only*

## The gap

`DatasetLightbox.jsx` already ships a complete side-by-side comparison mode:
state, a local `ComparePane`, a two-cell grid, and a `⧉ Compare with original`
button. It is gated by `utils/derivedCompare.js`, whose table has exactly two
entries — `klein_image_improve` and `klein_small_image`. Any other image returns
`null`, so a plainly generated variation gets no button at all.

That is the wrong axis for most of a character dataset. "Is this candidate
sharper than the shot it came from" only exists for the two derivation flows.
The question an ordinary generated shot raises is **"is this still the same
person?"**, and its answer — the dataset's reference photo — lives in another
panel and is therefore never on screen next to the image being judged.

## Design

**Two buttons, not a picker.** On a plain image only the new one appears; on an
improved image both do, because both questions are genuinely live there and a
selector would hide one behind the other.

**Exclusive modes.** Two pairs of panes at once is four thumbnails and shows
nothing. The lightbox's boolean `comparing` therefore becomes
`compareMode: 'none' | 'derived' | 'reference'` — one state, so entering either
comparison leaves the other with no teardown code. `lightboxActionPlacement`
gets `comparing: compareMode !== 'none'`: the rail-vs-bar rule only needs to
know that two panes want the width, not which pair.

**A second descriptor, not a second renderer.** `utils/referenceCompare.js`
returns exactly the shape `describeDerivedComparison` returns —
`{ beforeLabel, afterLabel, parent, available, reason }` — so `ComparePane` and
the `available`-alone availability guard are reused verbatim. Pure JS because
`node --test` cannot parse JSX; that is the repo's convention for decisions the
lightbox merely renders.

Its cases:

| input | result | why |
|---|---|---|
| no `ref_filename` | `null` | The reference panel already asks for one; a second nudge on a screen that cannot act on it is noise. **No button and no note.** |
| the image IS the reference | `null` | Comparing something with itself teaches nothing. |
| reference recorded, no usable filename | `available: false` + reason | A missing button next to a present one reads as a bug unless it explains itself. |
| otherwise | `available: true` | Every image qualifies — generated or imported. |

Labels: `Reference` / `This image`.

**Zero backend work.** The reference is served by the same endpoint as the
dataset images (`/api/dataset/<id>/img/<name>`, cf. `DatasetListPanel.jsx`), so
the lightbox's existing `fileUrl()` covers it. `DatasetWorkspace` passes
`refFilename` and `refNonce` (the reference's own cache buster — a crop or a
re-upload bumps it, and it is *not* the parent nonce).

**Each pane at its own scale — and the text has to change.** `ComparePane`
already produces this: its images are `h-full w-full object-contain` inside two
identical grid cells, so a square head reference and a full-body plan each fill
their own pane. No render code was written. What *was* wrong is the status
line: `same scale — exit comparison to zoom to 100 %` is a true and load-bearing
promise about the derived pair (an improve pass rescales to a megapixel budget
and keeps the ratio) and a plain falsehood about two unrelated crops. It is now
per mode; the reference mode says *different framings — each pane fits its own
image*.

## Out of scope

`studio/ResultLightbox.jsx` and `shared/GeneratedImageLightbox.jsx` are
untouched. The dataset lightbox is the only surface that has a reference photo
in context.

## Files

- `frontend/src/utils/referenceCompare.js` (+ `.test.js`) — new, pure.
- `frontend/src/components/dataset/DatasetLightbox.jsx` — `compareMode`, second
  button, per-mode hint, second unavailability note.
- `frontend/src/components/dataset/DatasetWorkspace.jsx` — `refFilename`,
  `refNonce`.
- `docs/guide/using-the-app.md` — new H2, backing the help topic
  `action-compare-with-reference`.
