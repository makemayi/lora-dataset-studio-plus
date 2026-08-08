"""Face masks for CONCEPT training — the INVERSE polarity of person_mask.py.

person_mask.py answers "learn the subject, not the room": person white, background
black, background weighted to 10% of the loss.

This module answers the opposite question, raised by shivdbz2010 (GitHub issue #15):
a concept LoRA also learns the FACES of its dataset, so combining it with a character
LoRA makes the two fight over the identity. Masking the faces during concept training
teaches the act without the identity.

It is a LOSS mask, not an image edit — ai-toolkit multiplies the per-pixel loss by
the mask (white 1.0, black -> mask_min_value). Nothing is painted into the training
image. Blurring or pixelating the faces instead would make the blur the regression
TARGET, and the LoRA would learn to render blurred faces.

Runs InsightFace in the DEDICATED ML interpreter (face_scoring.python), like
face_similarity.py — insightface is not in the Flask venv. Note it needs
`face_scoring`, NOT `masks`/rembg: an install with face scoring but no rembg can
still mask faces.
"""
from __future__ import annotations
import json
import logging
import os
import re
import sys

from .. import config as cfg
from .infer_stream import run_infer_script, stderr_tail

logger = logging.getLogger(__name__)

_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'face_mask_infer.py')

# The two shapes face_mask_infer prints on stderr. Same idiom as face_similarity's
# `[face] i/N`, with one addition: a NAMED PHASE. The counter alone lies about the
# wait — importing onnxruntime and preparing antelopev2 costs tens of seconds
# before image 1, and (first run only) the models are a ~350 MB download. A bar
# frozen at 0/N through all that is read as a hang, which is the whole reason this
# progress exists.
_PHASE_RE = re.compile(r'\[facemask\] phase=([a-z_]+)')
_COUNT_RE = re.compile(r'\[facemask\] (\d+)/(\d+)')

# Ordered, so the UI can name what is happening without knowing the script.
PHASES = ('starting', 'downloading', 'loading', 'detecting')


def parse_progress_line(line: str) -> dict | None:
    """One stderr line -> a progress record, or None if it says nothing about
    progress. PURE, so the grammar is testable without a subprocess."""
    m = _PHASE_RE.search(line or '')
    if m and m.group(1) in PHASES:
        return {'phase': m.group(1)}
    m = _COUNT_RE.search(line or '')
    if m:
        # Reaching image 1 means the model is loaded, whatever phase line was
        # last seen — so the counter carries the phase with it and a missed
        # `phase=detecting` cannot leave the UI stuck on "Loading…".
        return {'phase': 'detecting', 'done': int(m.group(1)), 'total': int(m.group(2))}
    return None

# Bounds for the two exposed knobs. Clamping happens SERVER-side so a hand-edited
# config.json (or a stale UI) degrades to a usable value instead of killing an
# export halfway through: a bad number must never be the reason a training run dies.
EXPAND_MIN, EXPAND_MAX = 1.0, 3.0
# The floor is NOT 0.0. A zero-weight region is not ignored, it is unpenalised (the
# model may render anything there for free), and ai-toolkit divides the mask by its
# own mean — an all-black mask would divide by zero and NaN the run. See config.py.
MIN_WEIGHT_MIN, MIN_WEIGHT_MAX = 0.05, 1.0


def _clamp(value, low, high, fallback):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    if v != v:  # NaN
        return fallback
    return max(low, min(high, v))


def expand_factor() -> float:
    """Configured face->head growth factor, clamped. 2.0 is the shipped default."""
    return _clamp(cfg.get('face_mask.expand'), EXPAND_MIN, EXPAND_MAX,
                  cfg.defaults()['face_mask']['expand'])


def min_weight() -> float:
    """Configured loss weight left inside the mask, clamped away from zero."""
    return _clamp(cfg.get('face_mask.min_weight'), MIN_WEIGHT_MIN, MIN_WEIGHT_MAX,
                  cfg.defaults()['face_mask']['min_weight'])


def is_available() -> bool:
    from ..capabilities import probe_face_scoring
    return probe_face_scoring()['ok']


def _face_python() -> str:
    return cfg.get('face_scoring.python') or sys.executable


def _asked(should_stop) -> bool:
    """Whether the stop really was requested. A predicate that raises is read as
    'no' — a broken callback must not turn a genuine crash into a silent 'you
    stopped it', which would hide the failure the preview exists to surface."""
    try:
        return bool(should_stop())
    except Exception:  # noqa: BLE001
        logger.debug('face_mask: should_stop raised', exc_info=True)
        return False


def _stop_plumbing(should_stop):
    """A (payload_field, on_stop, cleanup) triple for a stoppable pass, or
    (None, None, cleanup) when the caller asked for no stop.

    The sentinel lives in a private temp DIRECTORY, and the file is created only
    when the stop is actually asked — an existing file IS the request, so
    creating it up front would cancel the pass before it began. The directory is
    removed on the way out, whether the stop happened or not: a leftover sentinel
    would silently cancel the NEXT pass, which is a far nastier bug than the one
    being fixed."""
    import shutil
    import tempfile
    if not should_stop:
        return None, None, (lambda: None)
    tmp = tempfile.mkdtemp(prefix='lds-facemask-')
    path = os.path.join(tmp, 'stop')

    def _ask():
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write('stop')
        except OSError:
            logger.warning('face_mask: could not write the stop sentinel')

    return path, _ask, (lambda: shutil.rmtree(tmp, ignore_errors=True))


def _run_detail(images, out_dir, expand, timeout, on_progress=None,
                should_stop=None) -> dict:
    """The pass, with its failure REASON kept. Always returns a dict carrying
    `ok`; on failure it also carries a human `error`.

    _run() below throws that reason away on purpose (mask generation degrades to
    "train the historical way" and must never block an export), but the preview
    exists precisely to answer "is this going to work?" — there, a silent {} is
    the bug being fixed: an operation that fails must look failed, not like a wait
    that never ends."""
    images = [p for p in (images or []) if p and os.path.isfile(p)]
    if not images:
        return {'ok': True, 'written': 0, 'results': {}}
    if not is_available():
        return {'ok': False, 'error': 'face detection unavailable',
                'reason': 'face_scoring'}
    cancel_file, on_stop, cleanup = _stop_plumbing(should_stop)
    payload = json.dumps({'images': images, 'out_dir': out_dir, 'expand': expand,
                          'models_root': cfg.get('face_scoring.models_root') or None,
                          'cancel_file': cancel_file})

    def _on_line(line):
        rec = parse_progress_line(line)
        if rec and on_progress:
            on_progress(rec)

    try:
        stdout, stderr_lines, rc, timed_out = run_infer_script(
            _face_python(), _SCRIPT, payload, timeout, _on_line,
            should_stop=should_stop, on_stop=on_stop)
    except OSError as e:
        logger.warning('face_mask: subprocess échec : %s', e)
        return {'ok': False, 'error': f'could not start face detection: {e}'}
    finally:
        cleanup()
    if timed_out:
        return {'ok': False,
                'error': f'face detection timed out after {int(timeout)}s'}
    line = next((ln for ln in reversed((stdout or '').splitlines())
                 if ln.strip().startswith('{')), '')
    if not line:
        # A stop that had to be enforced with a kill lands here. It is NOT a
        # failure to report: the user asked for it, and the pass simply has
        # nothing to hand back (it never reached a polling point, so it never
        # reached image 1 either).
        if should_stop and _asked(should_stop):
            return {'ok': True, 'cancelled': True, 'results': {}}
        tail = stderr_tail(stderr_lines)
        logger.warning('face_mask: pas de JSON (rc=%s) stderr=%s', rc, tail)
        return {'ok': False,
                'error': f'face detection stopped unexpectedly (exit {rc})'
                         + (f': {tail}' if tail else '')}
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        logger.warning('face_mask: JSON illisible : %s', e)
        return {'ok': False, 'error': f'unreadable face-detection output: {e}'}
    if not data.get('ok'):
        logger.warning('face_mask: échec : %s', data.get('error'))
        return {'ok': False, 'error': str(data.get('error') or 'face detection failed')}
    return data


def _run(images, out_dir, expand, timeout) -> dict:
    # Unchanged contract for the GENERATION path: {} for "no masks", whatever the
    # reason. Nothing to do reads as nothing done, exactly as before.
    if not [p for p in (images or []) if p and os.path.isfile(p)]:
        return {}
    data = _run_detail(images, out_dir, expand, timeout)
    return data if data.get('ok') else {}


def generate_face_masks(image_paths, out_dir, expand=None, timeout: int = 1800) -> dict:
    """Write one mask PNG per image (face black, frame white) into `out_dir`.
    Returns {'ok', 'written', 'results'}; {} on any failure/unavailability —
    never blocking, exactly like generate_person_masks: a run without masks is
    still a valid run, it just trains the historical way."""
    return _run(image_paths, out_dir, expand if expand is not None else expand_factor(), timeout)


def detect_faces(image_paths, timeout: int = 900, on_progress=None,
                 should_stop=None) -> dict:
    """Detection ONLY — no file written. Feeds the preview, which grows the raw
    boxes itself so moving the expand slider costs nothing (the same arithmetic
    lives in infer/face_mask_infer.dilate_box and utils/faceMaskBox.js).

    `on_progress(record)` — optional — receives {'phase', 'done'?, 'total'?} from
    the stderr reader THREAD, so touch nothing but in-memory state in it.

    `should_stop()` — optional — is polled while the pass runs. When it turns
    true the child is asked to wind up and the reply carries `cancelled: True`
    together with the faces found SO FAR. Those partial results are the point:
    the caller can keep them and hand back only the remaining images next time,
    so stopping costs the detector load rather than the whole pass. `ok` stays
    true — a stop is not a failure.

    Unlike generate_face_masks this returns the failure reason (`ok`/`error`):
    the preview is the screen a user stares at, and "it failed" must be readable
    there rather than inferred from a spinner that never stops."""
    return _run_detail(image_paths, None, expand_factor(), timeout, on_progress,
                       should_stop)


def coverage_summary(results: dict) -> dict:
    """Fold a results map into the numbers the UI must show.

    `masked`/`total` is a SAFETY figure, not a statistic: a partially masked set is
    the worst outcome of all. The faces that stayed unmasked become the only ones
    carrying loss weight, so they end up over-represented in what the LoRA learns
    about faces — the "remaining guys have won" effect reported by the community
    workflow this feature follows. Surface it, do not bury it."""
    states = [(r or {}).get('state') for r in (results or {}).values()]
    total = len(states)
    return {
        'total': total,
        'masked': sum(1 for s in states if s == 'masked'),
        'no_face': sum(1 for s in states if s == 'no_face'),
        'too_large': sum(1 for s in states if s == 'too_large'),
        'failed': sum(1 for s in states if s in ('error', 'unreadable')),
    }
