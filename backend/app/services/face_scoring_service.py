"""Face-similarity scoring for a dataset: comparing every kept image against
the dataset's reference with InsightFace (antelopev2, in a CPU subprocess) and
persisting the verdict, plus the counts the UI shows.

The heavy lifting lives in the scorer subprocess; this module owns the pass
around it -- what is eligible, what invalidates a previous score, and how the
progress indicator is raised and cleared.

Split out of face_dataset_service.py (2026-08, Phase 8 of a multi-phase file
split) -- pure move, no behavior change.
"""
import os
import re
import stat

from ..extensions import db
from ..models import FaceDatasetImage
from . import dataset_activity


# --- Face similarity scoring (InsightFace antelopev2, CPU subprocess) -------
# WHICH ROWS A FACE PASS SCORES.
#   'keep'    = the curated set. The original (and, until now, the only) scope.
#   'pending' = the TRIAGE PILE: images that have landed but carry no ✓/✕ yet —
#               i.e. exactly the freshly GENERATED variations. Those are the ones
#               whose identity nobody can judge by eye ("is this still her?" on a
#               grainy party photo is not an eyeball question), and 🎯 Auto-triage
#               (DatasetGrid.jsx) has ALWAYS selected on `status === 'pending' &&
#               scorable` — a set this pass could never produce while it filtered
#               on 'keep' alone. The bar was built against a scope that did not
#               exist; widening it here is the whole wiring.
# 'reject'/'failed' stay out: scoring an image the user already threw away, or one
# with no file, is GPU-free but not free — and it would re-arm auto-triage on rows
# it must never touch.
FACE_SCORING_STATUSES = ('keep', 'pending')


def _face_score_content_revision(path):
    """Return the current (content signature, stat) pair, or None on a race.

    The signature makes edits that happen to preserve byte length detectable;
    the second stat read rejects a file changed while it was being fingerprinted.
    """
    from . import run_snapshot

    stat_key = run_snapshot._stat_key(path)
    if stat_key is None:
        return None
    signature = run_snapshot._content_sig(path)
    if not signature or run_snapshot._stat_key(path) != stat_key:
        return None
    return signature, stat_key


def face_scoring_counts(imgs):
    """{'total', 'unscored'} over an ALREADY-LOADED image list — pure, no query,
    so `dataset_payload` pays nothing for it. `unscored` counts rows the pass has
    never written a verdict for (face_state is NULL), which is what the button
    label needs to promise honest work ("Analyze 42 faces") instead of a silent
    no-op on a dataset that is already fully scored."""
    rows = [i for i in (imgs or [])
            if i.filename and i.status in FACE_SCORING_STATUSES]
    return {'total': len(rows),
            'unscored': sum(1 for i in rows if i.face_state is None)}


def face_scoring_rows(dataset_id):
    """The rows a face pass would score, straight from the DB."""
    return (FaceDatasetImage.query
            .filter(FaceDatasetImage.dataset_id == dataset_id,
                    FaceDatasetImage.status.in_(FACE_SCORING_STATUSES),
                    FaceDatasetImage.filename.isnot(None))
            .all())


def analyze_faces(user_id, dataset_id) -> dict:
    """Score les images GARDEES **et la pile de triage** vs la reference
    (InsightFace antelopev2, CPU subprocess) — cf. FACE_SCORING_STATUSES.
    Persiste face_score (cosinus brut, None si non note) + face_state. AUCUNE
    suppression, aucune decision : la passe ecrit un chiffre, c'est 🎯 Auto-triage
    qui agit dessus. Tourne sur CPU -> pas de fenetre GPU. Retourne {state: count}."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    # Checked BEFORE the reference guard on purpose: an anime dataset with no
    # reference must hear the useful thing ("this tool can't read a drawn face"),
    # not "set a reference photo first" — which would send the user off to fix
    # something that would not have helped. Returned as a scoring_error rather
    # than raised so the existing toast path states the reason instead of the pass
    # disappearing silently — a refusal that does not explain itself is the very
    # failure mode this gate exists to remove.
    blocked = face_scoring_block_reason(ds)
    if blocked:
        return {}, {'kind': 'subject_not_photographic', 'detail': blocked}
    if not ds.ref_filename:
        raise ValueError('reference photo missing')
    ref_path = _ref_path(ds)
    if not os.path.exists(ref_path):
        raise ValueError('reference photo missing')
    rows = face_scoring_rows(dataset_id)
    by_path = {}
    for img in rows:
        p = _img_path(img)
        if os.path.exists(p):
            by_path[p] = img
    try:
        from .face_similarity import score_dataset_faces
    except ImportError:
        raise RuntimeError('face scoring service not configured/available yet')
    # scoring_error ({kind, detail} | None) remonte jusqu'au toast : un scorer
    # cassé doit dire POURQUOI, pas « 0 analyzed » en vert.
    # Persistent indicator (survives reload). The scoring is a single CPU
    # subprocess, but NOT an opaque one: it prints "[face] i/N" for every image it
    # finishes, and the service now streams those into this counter — the bar used
    # to sit at 0 for the whole (multi-minute) pass and then fill in one jump,
    # which is indistinguishable from a hung pass. try/finally clears the
    # indicator even if scoring raises.
    score_lock = _face_scoring_lock(ds.id)
    if not score_lock.acquire(blocking=False):
        return {}, _face_scoring_busy_error()

    # Stamp every eligible file before inference.  A crop/mirror/rotate clears
    # this pair, making the final per-row write below fail closed if pixels move.
    reserved_by_path = {}
    try:
        from sqlalchemy import update
        for p, img in by_path.items():
            revision = _face_score_content_revision(p)
            if revision is None:
                continue
            content_sig, content_sig_stat = revision
            reservation = db.session.execute(
                update(FaceDatasetImage)
                .where(FaceDatasetImage.id == img.id,
                       FaceDatasetImage.dataset_id == ds.id,
                       FaceDatasetImage.filename == img.filename,
                       FaceDatasetImage.status.in_(FACE_SCORING_STATUSES),
                       _nullable_equals(FaceDatasetImage.content_sig, img.content_sig),
                       _nullable_equals(FaceDatasetImage.content_sig_stat,
                                        img.content_sig_stat))
                .values(content_sig=content_sig, content_sig_stat=content_sig_stat)
                .execution_options(synchronize_session=False))
            if reservation.rowcount == 1:
                reserved_by_path[p] = (img.id, img.filename,
                                       content_sig, content_sig_stat)
        db.session.commit()
    except Exception:
        db.session.rollback()
        score_lock.release()
        raise
    if not reserved_by_path:
        score_lock.release()
        return {}, None

    try:
        token = dataset_activity.begin(dataset_id, 'analyze_faces', total=len(reserved_by_path))
    except Exception:
        score_lock.release()
        raise

    try:
        results, scoring_error = score_dataset_faces(
            ref_path, list(reserved_by_path.keys()),
            on_progress=lambda done, total: dataset_activity.progress(
                token, done=done, total=total),
            extra_ref_paths=_extra_ref_paths(ds))
        counts = {}
        # The counter is already at N: the persist loop below is a fraction of the
        # pass (no model load, no inference), so it does NOT bump — doing so would
        # count every image twice and take the bar past its own total.
        for p, (image_id, filename, content_sig, content_sig_stat) in reserved_by_path.items():
            r = results.get(p)
            if not r:
                continue
            if _face_score_content_revision(p) != (content_sig, content_sig_stat):
                continue
            write = db.session.execute(
                update(FaceDatasetImage)
                .where(FaceDatasetImage.id == image_id,
                       FaceDatasetImage.dataset_id == ds.id,
                       FaceDatasetImage.filename == filename,
                       FaceDatasetImage.status.in_(FACE_SCORING_STATUSES),
                       FaceDatasetImage.content_sig == content_sig,
                       FaceDatasetImage.content_sig_stat == content_sig_stat)
                .values(face_state=r.get('state'), face_score=r.get('sim'))
                .execution_options(synchronize_session=False))
            if write.rowcount != 1:
                # Another request won the row after inference.  It is newer than
                # this pass, so leave it exactly as it is.
                db.session.rollback()
                continue
            db.session.commit()
            state = r.get('state')
            counts[state] = counts.get(state, 0) + 1
        return counts, scoring_error
    finally:
        dataset_activity.end(token)
        score_lock.release()


def analyze_image_face(user_id, image_id):
    """Score one owned dataset image against its dataset reference on CPU only.

    The single-image action deliberately uses the same scorer contract as the
    batch pass. Operational scorer failures are returned with the untouched
    current fields, while invalid image/dataset input remains a validation
    error for the route to map.
    """
    img = _owned_image(user_id, image_id)
    if not img:
        return None
    ds = get_dataset(user_id, img.dataset_id)
    if not ds:
        return None

    def _result(scoring_error=None, stale=False, row=None):
        row = img if row is None else row
        result = {'image_id': row.id, 'face_state': row.face_state,
                  'face_score': row.face_score, 'scoring_error': scoring_error}
        if stale:
            result['stale'] = True
        return result

    def _stale_result():
        db.session.expire_all()
        fresh = _owned_image(user_id, image_id)
        if not fresh:
            return None
        return _result(stale=True, row=fresh)


    # Match the batch behaviour: explain the photographic-subject gate before
    # asking for a reference that could never make this kind of dataset scorable.
    blocked = face_scoring_block_reason(ds)
    if blocked:
        return _result({'kind': 'subject_not_photographic', 'detail': blocked})
    if not ds.ref_filename or not os.path.isfile(_ref_path(ds)):
        raise ValueError('reference photo missing')
    if img.status not in FACE_SCORING_STATUSES:
        raise ValueError('image is not eligible for face scoring')
    filename_snapshot = img.filename
    if not img.filename or not os.path.isfile(_img_path(img)):
        raise ValueError('image file missing')

    ref_path = _ref_path(ds)
    image_path = _img_path(img)
    score_lock = _face_scoring_lock(ds.id)
    if not score_lock.acquire(blocking=False):
        return _result(_face_scoring_busy_error())
    try:
        try:
            from . import face_similarity
        except ImportError:
            return _result({'kind': 'unavailable',
                            'detail': 'face scoring service not configured/available yet'})

        # Reserve the content identity before launching the subprocess.  A pixel
        # edit clears this cache pair, so the final write below can never promote
        # a score calculated for an earlier version of the same filename.
        revision = _face_score_content_revision(image_path)
        if revision is None:
            return _stale_result()
        content_sig, content_sig_stat = revision
        previous_sig = img.content_sig
        previous_stat = img.content_sig_stat
        from sqlalchemy import update
        reservation = db.session.execute(
            update(FaceDatasetImage)
            .where(FaceDatasetImage.id == img.id,
                   FaceDatasetImage.dataset_id == ds.id,
                   FaceDatasetImage.filename == filename_snapshot,
                   FaceDatasetImage.status.in_(FACE_SCORING_STATUSES),
                   _nullable_equals(FaceDatasetImage.content_sig, previous_sig),
                   _nullable_equals(FaceDatasetImage.content_sig_stat, previous_stat))
            .values(content_sig=content_sig, content_sig_stat=content_sig_stat)
            .execution_options(synchronize_session=False))
        if reservation.rowcount != 1:
            db.session.rollback()
            return _stale_result()
        db.session.commit()
        db.session.expire(img)

        # Recheck after the reservation: do not start the expensive process for
        # a file that changed while its identity was being recorded.
        from . import run_snapshot
        if run_snapshot._stat_key(image_path) != content_sig_stat:
            return _stale_result()

        try:
            results, scoring_error = face_similarity.score_dataset_faces(
                ref_path, [image_path], extra_ref_paths=_extra_ref_paths(ds))
        except Exception as e:
            logger.warning('single face scoring failed for image %s: %s', image_id, e)
            return _result({'kind': 'failed', 'detail': str(e) or 'face scoring failed'})
        if scoring_error:
            return _result(scoring_error)
        scored = results.get(image_path) if isinstance(results, dict) else None
        if not isinstance(scored, dict) or not scored.get('state'):
            return _result({'kind': 'failed',
                            'detail': 'face scorer returned no result for this image'})

        # A stat check is cheap but coarse on some filesystems; re-reading the
        # content signature here also catches a same-size edit in the same second.
        if _face_score_content_revision(image_path) != (content_sig, content_sig_stat):
            return _stale_result()

        write = db.session.execute(
            update(FaceDatasetImage)
            .where(FaceDatasetImage.id == img.id,
                   FaceDatasetImage.dataset_id == ds.id,
                   FaceDatasetImage.filename == filename_snapshot,
                   FaceDatasetImage.status.in_(FACE_SCORING_STATUSES),
                   FaceDatasetImage.content_sig == content_sig,
                   FaceDatasetImage.content_sig_stat == content_sig_stat)
            .values(face_state=scored['state'], face_score=scored.get('sim'))
            .execution_options(synchronize_session=False))
        if write.rowcount != 1:
            db.session.rollback()
            return _stale_result()
        db.session.commit()
        db.session.expire(img)
        return _result()
    finally:
        score_lock.release()

# --- Borrow: face_dataset_service.py primitives -----------------------------
# MUST stay at the bottom of this file, same reason as in the sibling split
# modules: whichever side loads first must find the other fully defined by the
# time the reach-back import resolves.
from .face_dataset_service import (
    get_dataset, dataset_payload, face_scoring_block_reason, _owned_image,
    _img_path, _ref_path, _nullable_equals, _face_scoring_lock,
    _face_scoring_busy_error, logger,
)
# Owned by a sibling split module -- imported from the owner, never through
# face_dataset_service's re-export.
from .reference_photos_service import _extra_ref_paths
