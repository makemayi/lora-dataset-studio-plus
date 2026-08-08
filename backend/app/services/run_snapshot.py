"""The full FREEZE of a training launch — everything that defines the run, kept
so that comparing two runs tells the truth.

`TrainingRunRecord.manifest` already froze WHICH images and WHETHER their caption
changed (an id + two short hashes per image). That is enough to say "3 captions
edited" and nothing more: the diff could never show WHAT they said, which is
precisely where a comparison of two trainings is decided. And "same images" was
asserted from a `size:mtime` proxy, so a re-crop, a "Reset to auto", a rembg mask
or a scrubbed watermark could change the pixels while the manifest shrugged.

This module builds the companion blob (`TrainingRunRecord.snapshot`, JSON):

    {"v": 1,
     "captions": {"<image_id>": ["long caption", "short caption"|null]},
     "images":   {"<image_id>": {"c": <content sha1>, "e": engine,
                                 "s": source, "f": framing}},
     "env":      {...run_environment.capture()...},
     "dataset":  {"name", "kind", "subject_type", "fidelity",
                  "reference": {"filename", "c": <content sha1>}}}

Design constraints, all of them load-bearing:

* **The launch must not slow down.** Content hashes are CACHED on the image row
  (`content_sig` + the `size:mtime_ns` they were computed for): the first launch
  after the upgrade pays the read, every later launch pays one `stat` per image.
  `build()` does all of its file I/O OUTSIDE any transaction and returns the
  cache updates for the caller to flush in the SINGLE short commit that also
  writes the record — the launch path already lost three cloud runs to
  `database is locked`, it is not getting a second writer.

* **The fingerprint is untouched.** The content hash lives here, NOT in the
  manifest, because the manifest feeds `fingerprint_of()` and every dataset
  version in every existing database is keyed on it. Changing what goes into the
  fingerprint would bump every dataset to a new version on its next launch and
  announce a change that never happened.

* **Absence is a fact, not a failure.** A run recorded before this existed has
  `snapshot = NULL`; the diff says so. A probe that fails leaves its key out.
  Nothing here may ever raise into a launch.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

SNAPSHOT_VERSION = 1

def _content_sig(path):
    """sha1 of an image's BYTES (short hex), or None when it can't be read.

    This is the fact `size:mtime_ns` only approximated: restoring a backup changes
    the mtime without touching a pixel, and a re-crop that happens to land on the
    same byte count within the same second changes the pixels without touching
    either. Only the content answers "is this the same image?"."""
    try:
        size = os.path.getsize(path)
        h = hashlib.sha1(str(size).encode())
        with open(path, 'rb') as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return None


def _stat_key(path):
    try:
        st = os.stat(path)
        return f'{st.st_size}:{st.st_mtime_ns}'
    except OSError:
        return None


def content_sig_for(row, path, updates):
    """Cached content signature of one image row.

    Reuses `row.content_sig` when the file's size+mtime_ns still match the ones it
    was computed for; otherwise re-hashes and appends `(row, sig, stat)` to
    `updates` so the caller can persist it inside its own single commit. Never
    writes here — this runs before the transaction, on purpose."""
    stat = _stat_key(path)
    if stat is None:
        return None
    if getattr(row, 'content_sig_stat', None) == stat and getattr(row, 'content_sig', None):
        return row.content_sig
    sig = _content_sig(path)
    if sig:
        updates.append((row, sig, stat))
    return sig


def _dataset_facts(ds) -> dict:
    """The dataset-level knobs that steer captions and generation and are NOT in
    the settings snapshot — swapping a dataset from `character` to `concept`, or
    `face` to `body` fidelity, retrains a different thing entirely."""
    out = {}
    for attr, key in (('name', 'name'), ('kind', 'kind'),
                      ('subject_type', 'subject_type'), ('fidelity', 'fidelity'),
                      ('concept_desc', 'concept_desc')):
        value = getattr(ds, attr, None)
        if value:
            out[key] = str(value)
    return out


def build(ds, rows, images_dir, base_model=None) -> tuple:
    """`(snapshot_dict, cache_updates)` for one launch.

    `rows` are the KEPT images, exactly the ones the manifest is built from (they
    are passed in rather than re-queried so the two can never disagree about what
    was in the dataset at this instant). `cache_updates` is the list of
    `(row, sig, stat)` the caller must apply in its own commit.

    Never raises: a snapshot that cannot be built degrades to `({}, [])` and the
    launch proceeds with the manifest alone, exactly as before this feature."""
    updates: list = []
    try:
        captions, images = {}, {}
        for r in rows:
            key = str(r.id)
            long_c = (r.caption or '').strip()
            short_c = (getattr(r, 'caption_short', None) or '').strip()
            if long_c or short_c:
                captions[key] = [long_c, short_c or None]
            entry = {}
            if r.filename:
                sig = content_sig_for(r, os.path.join(images_dir, r.filename), updates)
                if sig:
                    entry['c'] = sig
            # Which ENGINE produced a generated image. `klein_model` carries the
            # local UNET for a Klein/Z-Image render and the engine id for the API
            # engines — five of them now, and "which engine made these" is a real
            # difference between two datasets that look identical otherwise.
            if getattr(r, 'klein_model', None):
                entry['e'] = str(r.klein_model)
            if getattr(r, 'source', None):
                entry['s'] = str(r.source)
            if getattr(r, 'framing', None):
                entry['f'] = str(r.framing)
            if entry:
                images[key] = entry

        snap = {'v': SNAPSHOT_VERSION}
        if captions:
            snap['captions'] = captions
        if images:
            snap['images'] = images
        facts = _dataset_facts(ds)

        # The reference photo of a GENERATED dataset is an input of the run as
        # surely as the images are: swap it and every future variation changes.
        ref = getattr(ds, 'ref_filename', None)
        if ref:
            ref_entry = {'filename': str(ref)}
            sig = _content_sig(os.path.join(images_dir, str(ref)))
            if sig:
                ref_entry['c'] = sig
            facts['reference'] = ref_entry
        if facts:
            snap['dataset'] = facts

        try:
            from . import run_environment
            env = run_environment.capture(base_model=base_model)
        except Exception:
            logger.debug('environment capture failed', exc_info=True)
            env = None
        if env:
            snap['env'] = env
        return snap, updates
    except Exception:
        logger.exception('run snapshot build failed (launch continues)')
        return {}, updates


def loads(rec):
    """The snapshot dict of a record, or None when it has none / is unreadable.
    None means "this run predates full snapshots" — the UI must SAY that rather
    than render an empty comparison."""
    raw = getattr(rec, 'snapshot', None)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def portable(raw):
    """The part of a snapshot that survives a BACKUP → RESTORE on another machine:
    the environment and the dataset-level facts.

    `captions` and `images` are keyed by image id, and a restore allocates FRESH
    ids for every image — carrying them over would attach one run's captions to
    another image, which is worse than not carrying them at all. The machine and
    the dataset's kind/fidelity/reference are id-free and stay true, so a restored
    run can still answer "which ai-toolkit trained this?". Returns a JSON string
    or None (nothing worth carrying)."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else None
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    kept = {k: data[k] for k in ('env', 'dataset') if isinstance(data.get(k), dict)}
    if not kept:
        return None
    return json.dumps({'v': data.get('v', SNAPSHOT_VERSION), 'restored': True,
                       **kept}, separators=(',', ':'))


def signatures(snap) -> set:
    """Every archived content hash this snapshot points at — one per training
    image plus the dataset's reference photo.

    This is the ONLY durable link between a run and the blobs `run_archive` kept
    for it, so it is also what "which archived images does this run own?" is
    answered with when a run is deleted. A run with no snapshot (legacy) yields
    an empty set: its blobs are unattributable and must therefore be left alone,
    never guessed at."""
    out = set()
    if not isinstance(snap, dict):
        return out
    for entry in (snap.get('images') or {}).values():
        if isinstance(entry, dict) and entry.get('c'):
            out.add(str(entry['c']))
    ref = (snap.get('dataset') or {}).get('reference')
    if isinstance(ref, dict) and ref.get('c'):
        out.add(str(ref['c']))
    return out


def signatures_of_raw(raw) -> set:
    """`signatures()` straight from the stored JSON text — lets a caller sweep
    every OTHER run's snapshot column without materialising ORM objects. An
    unreadable blob yields an empty set (never raises)."""
    if not raw:
        return set()
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return set()
    return signatures(data if isinstance(data, dict) else None)


def caption_of(snap, image_id):
    """`(long, short)` for one image id in a snapshot, `(None, None)` when the
    snapshot didn't record it (image had no caption, or run predates snapshots)."""
    entry = ((snap or {}).get('captions') or {}).get(str(image_id))
    if not isinstance(entry, list) or not entry:
        return None, None
    return (entry[0] or None), (entry[1] if len(entry) > 1 else None)


def content_of(snap, image_id):
    """Recorded content signature of one image id, or None."""
    entry = ((snap or {}).get('images') or {}).get(str(image_id))
    return entry.get('c') if isinstance(entry, dict) else None
