"""Scoring de ressemblance faciale via InsightFace antelopev2, en SUBPROCESS dans un
interprete DEDIE (insightface absent du venv Flask). Meme pattern que
app/services/joycaption.py. CPU -> ne touche pas le GPU/ComfyUI."""
from __future__ import annotations
import json
import logging
import os
import re

from .. import config as cfg
from .infer_stream import run_infer_script, stderr_tail as _tail

logger = logging.getLogger(__name__)

# face_score_infer.py vit dans backend/infer/ (pas app/services/).
_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'face_score_infer.py')

# The scorer already announces every image it finishes on stderr — nobody read
# it, so the pass showed 0/N for its whole duration and then jumped to N/N. Same
# mechanism as the Bank's embedding pass (image_bank_service._PROGRESS_RE over
# face_embed_infer's "[embed] i/N"): one regex, one drain thread.
_PROGRESS_RE = re.compile(r'\[face\] (\d+)/(\d+)')


def _scoring_python() -> str:
    import sys
    return cfg.get('face_scoring.python') or sys.executable


def is_available() -> bool:
    from ..capabilities import probe_face_scoring
    return probe_face_scoring()['ok']


def _stderr_tail(lines) -> str:
    """Derniere ligne non vide de stderr — pour un crash Python c'est la ligne
    `SomeError: ...` du traceback, exactement ce qu'un humain veut lire."""
    return _tail(lines)


def _run_scorer(python, payload, timeout, on_progress):
    """Run face_score_infer, streaming its `[face] i/N` lines to ``on_progress``.

    Returns ``(stdout, stderr_lines, returncode, timed_out)``. The Popen/drain
    plumbing lives in infer_stream.run_infer_script, shared with the concept
    face-mask preview; only the line grammar is ours."""
    def _on_line(line):
        m = _PROGRESS_RE.search(line)
        if m and on_progress:
            on_progress(int(m.group(1)), int(m.group(2)))

    return run_infer_script(python, _SCRIPT, payload, timeout, _on_line)


# Budget temps par image, en secondes. antelopev2 sur CPU tourne autour de
# 0.3-1 s/image selon la taille ; 3 s laisse de la marge sur une machine lente
# sans jamais bloquer une session entiere. Le forfait couvre le chargement du
# modele (le plus gros cout fixe du subprocess).
_TIMEOUT_PER_IMAGE_S = 3
_TIMEOUT_FLOOR_S = 900


def default_timeout(n_images: int) -> int:
    """Budget d'un run de scoring, en secondes. Le timeout etait un forfait de
    900 s dimensionne pour le seul set GARDE ; depuis que la passe couvre aussi
    la pile de triage (les variations generees non encore ✓/✕), un gros dataset
    peut depasser ce forfait — et un timeout ne rend AUCUN resultat partiel, donc
    la passe entiere serait perdue. Le budget suit donc le nombre d'images."""
    return max(_TIMEOUT_FLOOR_S,
               120 + _TIMEOUT_PER_IMAGE_S * max(0, int(n_images or 0)))


def score_dataset_faces(ref_path, image_paths, timeout: int | None = None,
                        on_progress=None, extra_ref_paths=None):
    """Retourne ({path: {state, sim?, det, bbox_frac, yaw}}, error|None).

    Chaque image candidate est comparee a `ref_path` PLUS tout `extra_ref_paths`
    fourni (best-match-of-N : la meilleure ressemblance gagne, pas une moyenne,
    pas besoin que toutes les refs soient d'accord). `ref_path` reste
    obligatoire — la reference primaire ne devient jamais optionnelle, seuls les
    extras le sont ; `extra_ref_paths=None` (defaut) se comporte EXACTEMENT
    comme avant, refs = [ref_path] seul.

    `on_progress(done, total)` — optionnel — est appelé à chaque image finie par
    le scorer, depuis un thread de lecture (donc PAS dans un contexte Flask :
    n'y touchez qu'à de l'état en mémoire, comme dataset_activity). Sans lui la
    passe reste exactement ce qu'elle était.

    `error` est None quand le scorer a tourne, sinon {'kind', 'detail'} :
    'unavailable' (extras ML absents), 'failed' (subprocess/JSON casse — detail
    = derniere ligne du traceback), 'ref_unusable' (aucune des references n'a de
    visage exploitable). Les echecs restent NON-fatals ({} + error) mais
    doivent etre VISIBLES : les avaler en {} muet transformait un scorer casse
    en « Face scoring done — 0/14 » avec toast vert (user-reported)."""
    image_paths = [p for p in (image_paths or []) if p and os.path.isfile(p)]
    if not ref_path or not os.path.isfile(ref_path) or not image_paths:
        return {}, None
    if not is_available():
        return {}, {'kind': 'unavailable',
                    'detail': 'face scoring is not installed (Quality tools step in Setup)'}
    if timeout is None:
        timeout = default_timeout(len(image_paths))
    refs = [ref_path] + [p for p in (extra_ref_paths or []) if p and os.path.isfile(p)]
    payload = json.dumps({"refs": refs, "images": image_paths,
                          "models_root": cfg.get('face_scoring.models_root') or None})
    try:
        stdout, stderr_lines, returncode, timed_out = _run_scorer(
            _scoring_python(), payload, timeout, on_progress)
    except OSError as e:
        logger.warning('face_similarity: subprocess echec : %s', e)
        return {}, {'kind': 'failed', 'detail': str(e)}
    if timed_out:
        logger.warning('face_similarity: timeout apres %ss', timeout)
        return {}, {'kind': 'failed',
                    'detail': f'face scoring timed out after {timeout}s '
                              f'({len(image_paths)} image(s))'}
    line = next((ln for ln in reversed((stdout or '').splitlines())
                 if ln.strip().startswith('{')), '')
    if not line:
        tail = _stderr_tail(stderr_lines)
        logger.warning('face_similarity: pas de JSON (rc=%s) stderr=%s',
                       returncode, ' | '.join(stderr_lines))
        return {}, {'kind': 'failed',
                    'detail': tail or f'scorer produced no output (rc={returncode})'}
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        logger.warning('face_similarity: JSON illisible : %s', e)
        return {}, {'kind': 'failed', 'detail': f'unreadable scorer output: {e}'}
    if not data.get('ref_ok'):
        logger.warning('face_similarity: ref inutilisable : %s', data.get('error'))
        return {}, {'kind': 'ref_unusable',
                    'detail': data.get('error') or 'no usable face in the reference photo'}
    return data.get('results') or {}, None


# --- Reference-set self-check ----------------------------------------------
# Best-match-of-N scoring (`sim` = MAX over refs) has one hole, and it is not a
# small one: a single photo of the WRONG person is never outvoted. It becomes the
# ref that wins the max, so it RAISES every candidate's score instead of lowering
# it. A wrong reference is therefore invisible in the scores it corrupts — the
# only place it can be seen is against the OTHER references.
#
# Below this mean cosine a reference is flagged as "possibly not the same person".
#
# THIS NUMBER IS NOT PORTED FROM ANYWHERE, and in particular it is NOT Inline
# Studio's 25.0: that floor is calibrated on OpenCV SFace (128-d, scores x100),
# and a cosine threshold carries no meaning across two different encoders.
# antelopev2 is ArcFace-r100 on 512-d normed embeddings, so the scale here is a
# raw cosine in [-1, 1]. 0.20 sits deliberately LOW — impostor pairs cluster near
# 0.0 while genuine pairs run 0.4-0.7, but this scorer accepts refs up to
# YAW_MAX=70 deg, and a real 3/4 profile against a frontal is exactly the genuine
# pair that lands lowest. A warning nobody trusts is a warning nobody reads, so
# this is tuned to miss a marginal impostor rather than to accuse a real profile.
#
# It has NOT been calibrated on a labelled set on this machine. That is why the
# check WARNS and never blocks, and why the UI shows the raw agreement number
# next to the flag: the number is what lets you judge, and re-tune this if your
# own sets prove it wrong. Override without editing code:
# config.json -> "face_scoring": {"reference_agreement_floor": 0.25}
REFERENCE_AGREEMENT_FLOOR = 0.20


def reference_agreement_floor() -> float:
    """The configured floor, or the module default. Out-of-range values fall back
    rather than silently flagging every photo (a floor of 1.0) or none (-1.0)."""
    raw = cfg.get('face_scoring.reference_agreement_floor')
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return REFERENCE_AGREEMENT_FLOOR
    return value if -1.0 <= value <= 1.0 else REFERENCE_AGREEMENT_FLOOR


def check_reference_set(ref_path, extra_ref_paths=None, timeout: int | None = None):
    """Is every reference photo the SAME person? Returns ``(report, error|None)``.

    ``report`` = ``{'floor': float, 'compared': int, 'refs': {path: {...}}}`` where each
    ref carries the scorer's own verdict (``state``, ``det``, ``bbox_frac``, ``yaw``)
    plus ``agreement`` (mean cosine with the other USABLE refs, None when there is
    nothing to compare against) and ``flagged`` (agreement below the floor).

    Runs the scorer with NO candidate images, so the cost is the model load plus one
    detection per reference — a handful of seconds on CPU, and it never touches the
    GPU. ``error`` uses the same ``{'kind', 'detail'}`` vocabulary as
    ``score_dataset_faces``; 'ref_unusable' here means no reference had a usable face
    at all, which is a real answer and not a crash.

    Two usable refs give both the same single pairwise number: that says "these two
    do not look like the same person" WITHOUT saying which one is wrong. The caller
    is expected to show it that way rather than to pick a culprit.
    """
    refs = [ref_path] + [p for p in (extra_ref_paths or []) if p and os.path.isfile(p)]
    refs = [p for p in refs if p and os.path.isfile(p)]
    if not refs:
        return {'floor': reference_agreement_floor(), 'compared': 0, 'refs': {}}, None
    if not is_available():
        return {}, {'kind': 'unavailable',
                    'detail': 'face scoring is not installed (Quality tools step in Setup)'}
    if timeout is None:
        # NOT default_timeout(): that one is floored at 900s for the candidate pass,
        # and this call is synchronous behind an HTTP request. A reference set is a
        # handful of photos, so the budget is the model load plus a detection each.
        timeout = 180 + _TIMEOUT_PER_IMAGE_S * len(refs)
    payload = json.dumps({"refs": refs, "images": [],
                          "models_root": cfg.get('face_scoring.models_root') or None})
    try:
        stdout, stderr_lines, returncode, timed_out = _run_scorer(
            _scoring_python(), payload, timeout, None)
    except OSError as e:
        logger.warning('face_similarity: reference check subprocess failed: %s', e)
        return {}, {'kind': 'failed', 'detail': str(e)}
    if timed_out:
        return {}, {'kind': 'failed',
                    'detail': f'reference check timed out after {timeout}s '
                              f'({len(refs)} reference photo(s))'}
    line = next((ln for ln in reversed((stdout or '').splitlines())
                 if ln.strip().startswith('{')), '')
    if not line:
        tail = _stderr_tail(stderr_lines)
        return {}, {'kind': 'failed',
                    'detail': tail or f'scorer produced no output (rc={returncode})'}
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        return {}, {'kind': 'failed', 'detail': f'unreadable scorer output: {e}'}

    floor = reference_agreement_floor()
    raw = data.get('refs') or {}
    report = {}
    compared = 0
    for path in refs:
        row = dict(raw.get(path) or {})
        agreement = row.get('agreement')
        if agreement is None:
            row['agreement'] = None
            row['flagged'] = False
        else:
            compared += 1
            row['flagged'] = float(agreement) < floor
        report[path] = row
    # `compared` counts the refs that actually got a number -- which needs TWO usable
    # faces, so a lone usable ref scores 0 here. That is the point: the caller must be
    # able to say "nothing to compare" rather than "all clear", and those are not the
    # same answer.
    if not data.get('ref_ok'):
        return ({'floor': floor, 'compared': compared, 'refs': report},
                {'kind': 'ref_unusable',
                 'detail': data.get('error') or 'no usable face in the reference photos'})
    return {'floor': floor, 'compared': compared, 'refs': report}, None
