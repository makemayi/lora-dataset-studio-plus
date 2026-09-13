"""Turning chosen timestamps into image bytes, in two passes over one clip.

WHY TWO PASSES AND NOT ONE. The face gate needs a real picture; the ranking
needs every frame. Holding every full-resolution frame of a clip to satisfy both
is the obvious design and the wrong one — a 30-second 4K shot is tens of
gigabytes of decoded RGB. So:

  PASS A  decodes the clip ONCE at analysis resolution and measures every frame
          (`video_metrics_scan._read_clip_frames`, the same readings the bank
          already stores — this adds no new decode to a scanned bank beyond the
          one it does here). Selection runs WITHOUT the face gate and with a
          widened budget, producing a shortlist of timestamps.
  PASS B  seeks back to those timestamps only, decodes them at full resolution,
          scores their faces in one batch, and runs the SAME selection again —
          this time with the face gate and the real budget.

The selector is called twice on purpose. It is a pure function, so "shortlist"
and "final pick" are the same rules with different inputs, not two policies that
can drift apart.

WHY SEEK BY TIMESTAMP. Scraped material is routinely variable-frame-rate, where
frame index n corresponds to no fixed instant. This is the same reason
`VideoClip.start_frame` is stored as informative and never cut from.

WHAT LEAVES THIS MODULE. Encoded image bytes plus a provenance dict per frame —
NOT files on disk. The dataset import path (`dataset_import_service.import_images`)
already owns naming, dedup, the encode policy and the dimension limits, and a
second writer would be a second set of those rules to keep in sync.
"""

from __future__ import annotations

import io
import json
import logging
import subprocess

from . import video_frame_select as vfs

logger = logging.getLogger(__name__)

# How much wider than the final budget the shortlist is. Pass B is the expensive
# one, so this is a direct cost multiplier — but a shortlist that is not wider
# than the budget makes the face gate unable to reject anything without leaving
# the caller short.
SHORTLIST_FACTOR = 4
SHORTLIST_MIN = 8


def _shortlist(readings, limit, min_gap_s, sharp_tolerance=None):
    """Pass A's pick: exposure + sharpness + spacing, no face gate.

    Returns the full select result (picked + rejected): the pass-A rejection
    counts are real frames this clip produced that the quality gates threw
    away, and the report must count them too — not just the final round over
    the shortlist.
    """
    wide = max(SHORTLIST_MIN, int(limit) * SHORTLIST_FACTOR)
    return vfs.select_frames(readings, limit=wide, min_gap_s=min_gap_s,
                             require_face=False,
                             sharp_tolerance=sharp_tolerance)


def _merge_rejected(first, second):
    """Sum two per-reason rejection maps (pass A over every measured frame,
    pass B over the shortlist — disjoint sets, no double counting)."""
    merged = dict(first or {})
    for reason, n in (second or {}).items():
        merged[reason] = merged.get(reason, 0) + n
    return merged


def decode_at(path, times, *, image_format='PNG'):
    """Decode the frames nearest ``times`` (seconds) at full resolution.

    Returns [{'t': actual_seconds, 'bytes': encoded_image}] in the order asked.
    A timestamp the container cannot reach is DROPPED, not approximated with a
    neighbour: a frame that is not the one the selector scored would carry that
    frame's sharpness and face numbers while being a different picture.
    """
    import av
    from PIL import Image

    out = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        tb = stream.time_base or 1
        for want in times:
            try:
                container.seek(int(want / tb), stream=stream)
            except Exception:                     # noqa: BLE001
                logger.debug('seek refused at %.3fs in %s', want, path)
            got = None
            for frame in container.decode(stream):
                t = float(frame.pts * tb) if frame.pts is not None else 0.0
                if t + 1e-6 < want:
                    continue
                got = (t, frame)
                break
            if got is None:
                continue
            t, frame = got
            buf = io.BytesIO()
            im = Image.fromarray(frame.to_ndarray(format='rgb24'))
            im.save(buf, format=image_format)
            # The size travels with the frame: the character pixel gate turns an
            # area FRACTION into a face's edge in pixels, and without these it
            # refuses rather than guesses.
            out.append({'t': t, 'bytes': buf.getvalue(),
                        'w': im.width, 'h': im.height})
    return out


def score_faces(python_exe, script_path, refs, image_paths, models_root=None,
                timeout=600):
    """Batch face readings for candidate frames, or None when unavailable.

    None means "the pass did not run", which the selector treats as absent
    evidence rather than as "no face here" — the distinction matters, because a
    missing interpreter would otherwise look exactly like a video with nobody
    in it. EMPTY ``refs`` is not "did not run": it is the refless
    detection-only pass (person_mode 'person').
    """
    # refs may legitimately be EMPTY: person_mode 'person' gates on presence
    # only (the _face_pass_or_raise contract) — the infer script runs a
    # detection-only pass for it. Only a missing interpreter or no candidates
    # make the pass unavailable.
    if not python_exe or not image_paths:
        return None
    payload = json.dumps({'refs': list(refs), 'images': list(image_paths),
                          'models_root': models_root or None})
    try:
        proc = subprocess.run([python_exe, script_path], input=payload,
                              capture_output=True, text=True, timeout=timeout)
        data = json.loads((proc.stdout or '').strip().splitlines()[-1])
    except Exception as exc:                       # noqa: BLE001
        logger.warning('face pass unavailable (%s) — frames will be chosen '
                       'without the face gate', exc)
        return None
    if not data.get('ref_ok'):
        logger.warning('face pass refused the references: %s',
                       data.get('error') or 'no reason given')
        return None
    return data.get('results') or {}


def face_reading(result):
    """Translate one `face_score_infer` result into the selector's shape."""
    if not result:
        return {'ok': False}
    state = result.get('state')
    return {'ok': state == 'scorable', 'det': result.get('det'),
            'bbox_frac': result.get('bbox_frac'), 'yaw': result.get('yaw'),
            'pitch': result.get('pitch'), 'n_faces': result.get('n_faces'),
            'nx0': result.get('nx0'), 'ny0': result.get('ny0'),
            'nx1': result.get('nx1'), 'ny1': result.get('ny1'),
            'sim': result.get('sim'), 'state': state,
            'face_sharp': result.get('face_sharp')}


def _crop_bytes(image_bytes, box, image_format='PNG'):
    """Crop one encoded frame to ``box`` pixels, re-encoded in place."""
    from PIL import Image
    im = Image.open(io.BytesIO(image_bytes))
    x0, y0, x1, y1 = box
    im = im.crop((max(0, int(x0)), max(0, int(y0)),
                  min(im.width, int(x1)), min(im.height, int(y1))))
    buf = io.BytesIO()
    im.save(buf, format=image_format)
    return buf.getvalue()


def extract_from_clip(*, path, start_s, end_s, fps, limit,
                      min_gap_s=vfs.MIN_GAP_S,
                      face_bbox_min=vfs.FACE_BBOX_MIN,
                      dedup_max_cosine=vfs.DEDUP_MAX_COSINE,
                      min_face_px=None, min_sim=None,
                      sharp_tolerance=None, face_tolerance=None,
                      read_frames=None, decode=None, face_scores=None,
                      clip_id=None, source_id=None,
                      framings=('full',), single_face=False):
    """The two passes, wired. Returns {'frames': [...], 'rejected': {...}}.

    ``read_frames``/``decode``/``face_scores`` are injectable so the ordering and
    the degradation can be tested without a video file. ``face_scores`` receives
    the decoded pass-B frames (``[{'t', 'bytes'}]``) and must return a list of
    readings parallel to them, or None when the pass could not run.
    """
    if read_frames is None:
        from .video_metrics_scan import _read_clip_frames
        read_frames = _read_clip_frames
    if decode is None:
        decode = decode_at

    try:
        readings = read_frames(path, start_s, end_s, fps)
    except Exception as exc:                       # noqa: BLE001
        # One unreadable clip must not take the batch with it — the same
        # contract `measure_one` already honours for metrics.
        logger.warning('clip %s unreadable (%s) — skipped', clip_id, exc)
        return []

    short_sel = _shortlist(readings, limit, min_gap_s, sharp_tolerance)
    shortlist = short_sel['picked']
    if not shortlist:
        return {'frames': [], 'rejected': dict(short_sel.get('rejected') or {})}

    frames = decode(path, [f['t'] for f in shortlist])
    if not frames:
        return []

    # The hook is handed the DECODED frames, bytes included, not just their
    # timestamps: the face scorer reads files, and these pictures do not exist
    # anywhere yet. A hook that only knew the timestamps would have to decode
    # them a third time to have something to write.
    faces = face_scores(frames) if face_scores else None
    by_time = {round(f['t'], 3): f for f in shortlist}
    candidates = []
    for i, frame in enumerate(frames):
        # Pass B decoded the frame NEAREST each request, so carry the measured
        # numbers of the frame that was actually chosen where we have them, and
        # fall back to the shortlist entry that asked for it.
        base = by_time.get(round(frame['t'], 3)) or shortlist[min(i, len(shortlist) - 1)]
        entry = dict(base)
        entry['t'] = frame['t']
        entry['bytes'] = frame['bytes']
        if frame.get('w'):
            entry['w'], entry['h'] = frame['w'], frame['h']
        if faces is not None:
            entry['face'] = faces[i] if i < len(faces) else {'ok': False}
        candidates.append(entry)

    final = vfs.select_frames(candidates, limit=limit, min_gap_s=min_gap_s,
                              face_bbox_min=face_bbox_min,
                              dedup_max_cosine=dedup_max_cosine,
                              min_face_px=min_face_px, min_sim=min_sim,
                              require_face=faces is not None,
                              sharp_tolerance=sharp_tolerance,
                              face_tolerance=face_tolerance,
                              single_face=single_face)

    rejected = _merge_rejected(short_sel.get('rejected') or {},
                               final['rejected'])

    def _emit(f, framing, image_bytes):
        return {
            'bytes': image_bytes,
            'provenance': {
                'source': 'video_frame',
                'source_id': source_id,
                'clip_id': clip_id,
                # Seconds into the SOURCE, so the frame can be found again and
                # re-extracted when a threshold changes. Never a frame index.
                'timestamp_s': round(float(f['t']), 3),
                'framing': framing,
                'sharpness': f.get('sharp'),
                'luma': f.get('luma'),
                'face': f.get('face'),
                'face_px': (round(vfs.face_pixels(f), 1)
                            if vfs.face_pixels(f) else None),
            },
        }

    out = [_emit(f, 'full', f['bytes']) for f in final['picked']]

    # THE CROP FRAMINGS take what the full frame did NOT: each runs its own
    # selection over the candidates still unclaimed after the full round, so a
    # single moment never lands as full + half + face triplets of one instant —
    # three look-alike pictures teach one thing, and the whole point of the
    # extra framings is MORE distinct usable moments. A crop needs a measured
    # face box to crop to, so candidates without one (or with the pass off)
    # are simply never offered to these rounds.
    wanted = set(framings or ('full',)) - {'full'}
    if wanted:
        taken = {round(f.get('t', 0.0), 3) for f in final['picked']}
        remaining = [c for c in candidates
                     if round(c.get('t', 0.0), 3) not in taken
                     and vfs.face_box_px(c.get('face') or {}, c.get('w'), c.get('h'))]
        for framing in ('half', 'face'):
            if framing not in wanted or not remaining:
                continue
            sel = vfs.select_frames(remaining, limit=limit, min_gap_s=min_gap_s,
                                    face_bbox_min=face_bbox_min,
                                    dedup_max_cosine=dedup_max_cosine,
                                    min_face_px=min_face_px, min_sim=min_sim,
                                    require_face=faces is not None,
                                    sharp_tolerance=sharp_tolerance,
                                    face_tolerance=face_tolerance,
                                    single_face=single_face)
            taken = {round(f.get('t', 0.0), 3) for f in sel['picked']}
            remaining = [c for c in remaining
                         if round(c.get('t', 0.0), 3) not in taken]
            rejected = _merge_rejected(rejected, sel['rejected'])
            for f in sel['picked']:
                box = vfs.face_box_px(f.get('face') or {}, f.get('w'), f.get('h'))
                window = (vfs.half_window(f['w'], f['h'], box) if framing == 'half'
                          else vfs.face_window(f['w'], f['h'], box))
                if window is None:
                    continue
                out.append(_emit(f, framing, _crop_bytes(f['bytes'], window)))
    return {'frames': out, 'rejected': rejected}
