"""Promoting kept video clips into an IMAGE dataset of extracted frames.

The video bank already promotes to a VIDEO dataset (`video_bank_service.
start_promote`). This is the second target: the same kept clips, but every clip
contributes a handful of still frames chosen for sharpness, exposure and — when
the face pass is available — for actually containing a usable face.

WHY IT IS A PROMOTION TARGET AND NOT AN IMPORT SOURCE. Everything that makes
this usable already exists on the bank side: shot detection, the kept/rejected
decision, per-source caps, the metrics that say which clips are worth decoding
twice. An "import from a video file" entry point would have to rebuild all of it
before it could choose a single frame.

WHAT THIS MODULE DOES NOT DO. It does not write image files. Frame bytes go to
`dataset_import_service.import_images`, which owns naming, the encode policy,
the dimension limits and perceptual dedup against what the dataset already
holds. A second writer would be a second copy of those rules.

REFUSALS ARE SYNCHRONOUS, all of them, before a dataset row or a folder exists —
the same contract `start_promote` documents. A background job that fails on its
first item leaves the user a dataset to clean up.
"""

from __future__ import annotations

import logging
import os

from .. import config as cfg
from ..extensions import db
from ..models import VideoClip, VideoSource
from . import bank_jobs, video_frame_extract, video_frame_select
from . import video_bank_service as vbs

logger = logging.getLogger(__name__)

FRAMES_PER_CLIP_MAX = 20      # past this the clip is a video, not a source of stills


def job_key(bank_id):
    """The SAME key the video promotion uses: one bank runs one job at a time.

    A frame extraction and a clip encode on one bank would fight over the same
    decoder and the same disk for no gain, and the bank's own job UI has one
    slot."""
    return vbs.job_key(bank_id)


def resolve_refs(user_id, ref_dataset_id):
    """Every reference photo of one image dataset, as absolute paths.

    THE CLIENT NEVER SENDS PATHS. It names a dataset it owns and the paths are
    built here — a request body that could hand arbitrary absolute paths to a
    subprocess is a file-read primitive, and "it is only the face scorer" is not
    a boundary anything enforces.
    """
    if not ref_dataset_id:
        return []
    from .face_dataset_service import _ref_path, get_dataset
    from .reference_photos_service import _extra_ref_paths
    ds = get_dataset(user_id, int(ref_dataset_id))
    if ds is None:
        raise ValueError('that reference dataset does not exist')
    out = []
    if getattr(ds, 'ref_filename', None):
        primary = _ref_path(ds)
        if os.path.isfile(primary):
            out.append(primary)
    out.extend(_extra_ref_paths(ds))
    if not out:
        raise ValueError(f'“{ds.name}” has no reference photo to compare against '
                         '— set one on that dataset, or turn the face filter off.')
    return out


def _face_pass_or_raise(refs, require_face):
    """(python, script, models_root) for the face pass, or None when off.

    Refused LOUDLY rather than degraded when the caller asked for the face gate:
    silently producing an unfiltered dataset from a request that said "only
    frames with a usable face" is the kind of quiet mismatch nobody attributes
    to the right setting later.
    """
    if not require_face:
        return None
    python = (cfg.get('face_scoring.python') or '').strip()
    if not python:
        raise ValueError(
            'face-aware frame picking needs the face scoring interpreter — set '
            'face_scoring.python in Settings, or turn the face filter off.')
    if not refs:
        raise ValueError('face-aware frame picking needs at least one reference '
                         'photo of the person to keep.')
    script = str(cfg.BACKEND_DIR / 'infer' / 'face_score_infer.py')
    return python, script, (cfg.get('face_scoring.models_root') or '').strip() or None


def start_promote_to_images(app, user_id, bank_id, *, name, ids=None,
                            frames_per_clip=3, total_limit=None,
                            max_per_source=None,
                            min_gap_s=video_frame_select.MIN_GAP_S,
                            face_bbox_min=video_frame_select.FACE_BBOX_MIN,
                            require_face=True, ref_dataset_id=None,
                            trigger_word=None, kind=None):
    """Validate everything, create the image dataset, start the extraction job.

    Returns the dataset identity plus the same kind of `composition` report the
    clip promotion returns — how many sources are represented and how lopsided
    the selection is, counted BEFORE anything is decoded. 60 % of a dataset
    coming from one source is invisible on disk and is exactly the imbalance
    that quietly overfits.
    """
    bank = vbs._require_free_bank(user_id, bank_id)
    name = (name or '').strip()
    if not name:
        raise ValueError('name is required')
    try:
        frames_per_clip = int(frames_per_clip)
    except (TypeError, ValueError):
        raise ValueError('frames per clip must be a whole number')
    if not 1 <= frames_per_clip <= FRAMES_PER_CLIP_MAX:
        raise ValueError(f'frames per clip must be between 1 and '
                         f'{FRAMES_PER_CLIP_MAX}')
    refs = resolve_refs(user_id, ref_dataset_id) if require_face else []
    face_cfg = _face_pass_or_raise(refs, require_face)

    q = VideoClip.query.filter_by(bank_id=bank_id, status='keep')
    if ids:
        q = q.filter(VideoClip.id.in_([int(i) for i in ids]))
    rows = q.order_by(VideoClip.source_id.asc(), VideoClip.start_s.asc()).all()

    per_cap = vbs._resolve_max_per_source(max_per_source)
    if per_cap is not None:
        taken, kept = {}, []
        for clip in rows:
            n = taken.get(clip.source_id, 0)
            if n < per_cap:
                taken[clip.source_id] = n + 1
                kept.append(clip)
        rows = kept
    if not rows:
        raise ValueError('nothing to promote — keep some clips first')

    per_source = {}
    for clip in rows:
        per_source[clip.source_id] = per_source.get(clip.source_id, 0) + 1
    ceiling = len(rows) * frames_per_clip
    composition = {
        'clips': len(rows),
        'sources': len(per_source),
        'frames_ceiling': ceiling,
        'top_source_share': max(per_source.values()) / len(rows),
        # A ceiling, never a promise: a clip with nothing admissible contributes
        # nothing, and the job never pads to reach a number.
        'total_limit': int(total_limit) if total_limit else None,
        'face_filtered': face_cfg is not None,
    }

    from .face_dataset_service import create_dataset
    dataset = create_dataset(user_id, name, trigger_word or '', kind=kind)

    clip_ids = [c.id for c in rows]
    bank_jobs.start(app, job_key(bank_id), 'promote_images',
                    _extract_job(bank.id, dataset.id, user_id, clip_ids,
                                 frames_per_clip=frames_per_clip,
                                 total_limit=total_limit, min_gap_s=min_gap_s,
                                 face_bbox_min=face_bbox_min,
                                 face_cfg=face_cfg, refs=list(refs or [])),
                    total=len(clip_ids))
    return {'id': dataset.id, 'name': dataset.name,
            'clips': len(clip_ids), 'composition': composition}


def _extract_job(bank_id, dataset_id, user_id, clip_ids, *, frames_per_clip,
                 total_limit, min_gap_s, face_bbox_min, face_cfg, refs):
    """Decode clip by clip, import each clip's frames as it finishes.

    IMPORTED PER CLIP, not once at the end. A four-hour bank would otherwise
    hold every extracted frame in memory before writing anything, and a crash at
    clip 300 would throw away 299 clips of decoding. Per-clip import also means
    the dataset fills visibly while the job runs, which is what the progress bar
    is for.
    """
    def _score_faces(frames):
        """Write the candidates to a scratch dir, score them, throw it away.

        `face_score_infer` reads FILES; these frames exist only as bytes. The
        pictures are candidates — most of them are about to be rejected — so
        they never touch the dataset folder, and the directory is removed
        whether the scorer succeeded, refused or crashed.
        """
        if face_cfg is None:
            return None
        python, script, models_root = face_cfg
        import tempfile
        from pathlib import Path as _Path
        with tempfile.TemporaryDirectory(prefix='lds-frames-') as tmp:
            paths = []
            for i, frame in enumerate(frames):
                p = _Path(tmp) / f'cand_{i:04d}.png'
                p.write_bytes(frame['bytes'])
                paths.append(str(p))
            results = video_frame_extract.score_faces(
                python, script, refs, paths, models_root=models_root)
            if results is None:
                # The pass could not run. Returning None (not a list of "no
                # face") is what stops the selector from reading a broken
                # interpreter as a video with nobody in it.
                return None
            return [video_frame_extract.face_reading(results.get(p))
                    for p in paths]

    def run(job):
        from .dataset_import_service import import_images

        bank = db.session.get(vbs.VideoBank, bank_id)
        rows = {c.id: c for c in VideoClip.query.filter(
            VideoClip.id.in_(clip_ids)).all()}
        relpaths = dict(db.session.query(VideoSource.id, VideoSource.relpath)
                        .filter_by(bank_id=bank_id).all())

        written = 0
        skipped = 0
        bank_jobs.progress(job, done=0, total=len(clip_ids),
                           detail='extracting frames')

        for done, clip_id in enumerate(clip_ids, start=1):
            clip = rows.get(clip_id)
            if clip is None:
                continue
            if total_limit and written >= int(total_limit):
                break
            # Resolved through the bank's own containment check: a relpath is
            # database data a user can edit, and joining it blindly is how a
            # bank reads a file outside itself.
            rel = relpaths.get(clip.source_id)
            path = vbs._abs_source_path(bank, rel) if (bank and rel) else None
            if not path:
                skipped += 1
                continue

            budget = frames_per_clip
            if total_limit:
                budget = min(budget, int(total_limit) - written)
            if budget <= 0:
                break

            got = video_frame_extract.extract_from_clip(
                path=path, start_s=clip.start_s, end_s=clip.end_s,
                fps=None, limit=budget, min_gap_s=min_gap_s,
                face_bbox_min=face_bbox_min, face_scores=_score_faces,
                clip_id=clip.id, source_id=clip.source_id)
            if not got:
                skipped += 1
                bank_jobs.progress(job, done=done, total=len(clip_ids),
                                   detail='extracting frames')
                continue

            import_images(user_id, dataset_id,
                          [g['bytes'] for g in got],
                          source_metadata=[g['provenance'] for g in got])
            written += len(got)
            bank_jobs.progress(job, done=done, total=len(clip_ids),
                               detail=f'{written} frame(s) so far')

        return {'frames': written, 'clips': len(clip_ids),
                'clips_without_frames': skipped}

    return run
