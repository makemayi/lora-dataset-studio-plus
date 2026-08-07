"""Face-dataset orchestration: CRUD, fan-out, import, classify, caption, export.

The vision passes (classify/caption) call describe_image_ollama; the CALLER (the
route) is responsible for wrapping them in the GPU-exclusive window. The ComfyUI
output dir is resolved via `cfg.comfyui_dir('output')` so tests can monkeypatch cfg.
"""
from __future__ import annotations
from decimal import Decimal
import io
import json
import logging
import math
import ntpath
import os
import posixpath
import random
import re
import shutil
import stat
import tempfile
import threading
import time
import uuid
import warnings
import zipfile
from functools import wraps
from types import SimpleNamespace
from typing import BinaryIO
from urllib.parse import urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError

from ..extensions import db
from ..models import (CanvasImageNode, CanvasNodePosition, FaceDataset,
                      FaceDatasetImage, ImageGenerationQueue, LoraTestImage,
                      RefPoseSlot)
from .. import config as cfg
from . import (bank_transfer_metadata, dataset_activity, image_encoding,
               reference_edit_jobs, trash)
from .dataset_storage import dataset_path, ensure_dataset_dir
from .image_provenance import provenance_metrics
from .image_quality import ANALYSIS_MAX_SIDE, quality_metrics
from .ollama_control import normalize_ollama_model_ref
# These used to be imported MID-FILE, inside what is now
# dataset_generation_service.py. The API-refusal handling left in this file
# still catches them, so the import moves to the head rather than vanishing.
from .engine_errors import EngineError
from .chatgpt_image import SubscriptionQuotaExceeded, SubscriptionUnavailable

# Garde le modèle vision chaud entre les images d'un même batch caption/classify
# (sinon Ollama le recharge - cold start ~10s - à CHAQUE image). Déchargé en fin
# de batch pour rendre la VRAM à ComfyUI. ComfyUI est déjà en pause pendant la passe.
_VISION_BATCH_KEEPALIVE = '5m'
from .face_variations import (CAPTION_PROMPT, CAPTION_PROMPT_BOORU,
                              DESCRIPTIVE_CAPTION_PROMPT,
                              CAPTION_REFINE_CONCEPT_PROMPT, CAPTION_LEAK_FIX_PROMPT,
                              EXPAND_CONCEPT_TERMS_PROMPT,
                              CLASSIFY_PROMPT, HEAD_BBOX_PROMPT, WATERMARK_BBOX_PROMPT,
                              JOYCAPTION_PROMPT, aspect_for_label, caption_prompt_for,
                              caption_prompt_for_style, caption_prompt_for_concept,
                              caption_has_identity_leak, caption_has_concept_leak,
                              identity_leak_terms, caption_concept_leaks,
                              compose_prompt_suffix, concept_lexical_field,
                              drop_identity_sentences, drop_identity_tags,
                              is_nsfw_label, prompt_by_label, wrap_variation,
                              wrap_variation_klein, wrap_variation_krea,
                              krea_pose_direction,
                              get_identity_prompt,
                              normalize_subject_type,
                              KLEIN_IMAGE_IMPROVE_PROMPT)

logger = logging.getLogger(__name__)


def _comfy_output_dir():
    d = cfg.comfyui_dir('output')
    return str(d) if d else None


# Garde-fou (PAS une limite produit) sur une caption STOCKÉE : la colonne est un TEXT
# sans contrainte DB, mais on borne quand même pour qu'une sortie vision emballée
# (boucle, collage pathologique) ne gonfle pas la base sans fin. Le vrai budget de
# longueur est l'encodeur de texte du trainer (T5 de FLUX/Klein, ~512 tokens ≈ bien
# au-delà d'une caption descriptive normale) et JoyCaption/Qwen bornent déjà leur propre
# sortie (max_new_tokens). Le plafond est donc volontairement TRÈS large et, quand il
# mord, _cap_caption coupe à une FIN DE PHRASE — jamais en plein mot. Historique : à 800
# il tranchait les captions descriptives en pleine phrase (« …a pale, neutral tone, and a »).
CAPTION_MAX_CHARS = 10000

# Exact Unicode whitespace set that Python's ``str.strip()`` recognizes.  SQLite's
# default ``trim`` only removes U+0020, so it would otherwise sample a caption made
# solely of (for example) U+2003 and let the final Python cleanup turn it into an
# apparent empty result.  Supplying this set to SQLite keeps the SQL eligibility
# predicate and the API's ``.strip()`` contract aligned without loading all rows.
_PYTHON_STRIP_CHARS = (
    '\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0\u1680'
    '\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a'
    '\u2028\u2029\u202f\u205f\u3000'
)


def _cap_caption(text):
    """Borne une caption à CAPTION_MAX_CHARS sans jamais couper en plein mot ni au
    milieu d'une phrase. Sous le plafond, le texte (strippé) est rendu tel quel ; au
    dessus, on garde les phrases entières jusqu'au plafond, sinon on retombe sur le
    dernier mot entier. Rend toujours une chaîne (l'entrée vide reste vide)."""
    text = (text or '').strip()
    if len(text) <= CAPTION_MAX_CHARS:
        return text
    head = text[:CAPTION_MAX_CHARS]
    last_end = 0
    for m in re.finditer(r'[.!?]["\'”’)\]]?(?=\s|$)', head):
        last_end = m.end()
    if last_end:
        return head[:last_end].strip()
    return head.rsplit(' ', 1)[0].strip() or head.strip()

# Padding du head-crop AUTO de la référence (côté du carré = grand côté de la bbox
# tête × pad). Volontairement plus large que l'ancien 1.7 (jugé « trop serré ») pour
# garder épaules + contexte par défaut ; le recadrage manuel depuis l'original permet
# d'ajuster ensuite dans les deux sens. Ne concerne QUE la référence (les imports
# gardent le défaut 1.7 de face_crop_to_square_webp).
REF_CROP_PAD = 2.0

# Un crop dont le côté source fait moins de size/1.5 se retrouve agrandi ≥50% par le
# LANCZOS du resize final — au-delà, la texture visible est majoritairement inventée
# par l'upscale plutôt que capturée du sujet. Seuil d'avertissement composition_upscaled
# (dataset_payload), pas un blocage : un unique gros plan upscalé n'est pas un problème,
# un dataset qui n'en a QUE des upscalés l'est (biais loss vers ce patch, cf. issue GitHub).
UPSCALE_WARN_THRESHOLD = 1.5


# Backward-compatible aliases for existing service consumers. New cross-module
# callers use the public names from dataset_storage so read paths cannot
# accidentally create directories.
_dataset_path = dataset_path
_dataset_dir = ensure_dataset_dir


def _restore_from_trash(trashed_path, original_path) -> None:
    """Best-effort filesystem compensation when a matching DB commit fails."""
    if not trashed_path or not original_path or not os.path.exists(trashed_path):
        return
    try:
        if os.path.exists(original_path):
            logger.error('cannot restore trashed path because destination exists: %s',
                         original_path)
            return
        os.makedirs(os.path.dirname(original_path), exist_ok=True)
        shutil.move(trashed_path, original_path)
    except OSError:
        # The bytes are still recoverable in Trash; never mask the DB exception.
        logger.exception('failed to restore %s from Trash after DB rollback',
                         original_path)


def _img_path(img) -> str:
    return os.path.join(_dataset_dir(img.dataset_id), img.filename)


def _invalidate_image_content_analysis(img):
    """Drop derived content and face analysis after a pixel-level mutation.

    The content cache doubles as the optimistic-concurrency token for a running
    one-image face score.  Clearing both fields is intentionally cheap: the
    next training snapshot simply re-hashes this one file.
    """
    img.content_sig = None
    img.content_sig_stat = None
    img.face_state = None
    img.face_score = None


def _ref_path(ds) -> str:
    return os.path.join(_dataset_dir(ds.id), ds.ref_filename)


# (path, mtime_ns, size) -> (w, h) | None. dataset_payload is POLLED, and it
# measured the reference on every single call: sub-millisecond, but a fresh disk
# open on a hot path, forever. Keyed on the file's identity rather than its name,
# so re-cropping the reference (which rewrites the same filename) invalidates the
# entry by itself — a stale shape here would silence the "your square reference
# will squeeze the body shots" warning, or raise a false one. Small and bounded:
# a handful of reference files per install, cleared wholesale when it grows.
_PIXEL_SIZE_CACHE: dict = {}
_PIXEL_SIZE_CACHE_MAX = 512


def image_pixel_size(path):
    """(w, h) of an image file, or None when it cannot be measured.

    PIL reads the header only — no decode — and the answer is cached per
    (path, mtime, size). The dataset payload exposes the reference dimensions
    for clients that need to describe or crop the source; Krea dataset cards now
    choose their own target aspect through the Fit v1.2 path. Degrades to None
    on ANY failure (missing file, exotic format, Pillow absent): an unmeasurable
    image must never turn a payload read into a 500. A file that cannot be
    stat'ed is measured without caching — never guessed."""
    try:
        st = os.stat(path)
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    if key in _PIXEL_SIZE_CACHE:
        return _PIXEL_SIZE_CACHE[key]
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        size = (int(w), int(h)) if w and h and w > 0 and h > 0 else None
    except Exception:
        size = None
    if len(_PIXEL_SIZE_CACHE) >= _PIXEL_SIZE_CACHE_MAX:
        _PIXEL_SIZE_CACHE.clear()
    _PIXEL_SIZE_CACHE[key] = size
    return size


_VALID_STATUS = ('pending', 'keep', 'reject', 'failed')
MAX_FANOUT = 60


def fanout_in_flight(dataset_id) -> int:
    """Generations already queued on this dataset (pending row, no file yet)."""
    return (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='pending')
            .filter(FaceDatasetImage.filename.is_(None)).count())


def check_fanout_budget(dataset_id, total):
    """Refuse a WHOLE multi-engine batch up front when it would blow MAX_FANOUT.

    generate_variations / generate_variations_nanobanana each enforce the cap on
    their own call, which is enough for a single engine but NOT for a run split
    across several: three 25-image calls each pass individually while the run
    totals 75, and the third one would be refused only after the first two had
    already created rows — a half-dispatched batch. The multi-engine route calls
    this with the aggregate BEFORE dispatching anything, so the run is all-or-
    nothing. The per-call checks stay as defense in depth."""
    total = int(total)
    if total > MAX_FANOUT:
        raise ValueError(f'fan-out too large ({total} > {MAX_FANOUT})')
    in_flight = fanout_in_flight(dataset_id)
    if in_flight + total > MAX_FANOUT:
        raise ValueError(f'too many generations in flight ({in_flight}), wait or cancel')
# Shown when a delete can't move a file to Trash because it's still open in
# another process (typically an antivirus scan of a just-cleaned image, or an
# open preview). Raised as a RuntimeError so the route maps it to a clean 409
# toast instead of a bare 500. The dataset is left fully intact (DB + disk).
_TRASH_LOCK_MESSAGE = (
    "Couldn't delete this because one of its files is still open in another "
    "program — most often an antivirus scan of a just-cleaned image, or an open "
    "preview. Close it or wait a few seconds, then try again.")
# Shown when a delete is refused because a training run (local or cloud) is still
# running on the dataset. Deleting under it would orphan the run's provenance row
# and — for a cloud run — leave a paid vast pod training against images we just
# trashed. RuntimeError -> 409 (routes._common._map_error); dataset untouched.
_ACTIVE_RUN_TEMPLATE = (
    'A training run is active on this dataset — stop it (or let it finish) '
    'before {action}.')
_ACTIVE_RUN_MESSAGE = _ACTIVE_RUN_TEMPLATE.format(action='deleting')
SMALL_IMAGE_SOURCE = 'small_image_source'
KLEIN_SMALL_IMAGE = 'klein_small_image'
KLEIN_IMAGE_IMPROVE = 'klein_image_improve'

# The three "Upscale & improve" knobs live in config (klein.improve_*). Read
# through clamps: a hand-edited config with a string, a negative or a wild value
# must degrade the pass to something sane, never raise inside the enqueue path.
_IMPROVE_MAX_STRENGTH = 2.0
_IMPROVE_MAX_STEPS = 50


# Config keys renamed after they shipped. improve_character_lora_strength was a
# MISNOMER: the value drives klein.consistency_strength (composition anchoring),
# never an identity LoRA. Renamed rather than left lying, but a value already saved
# under the old name must keep working — config keys live in users' config.json.
_IMPROVE_KEY_ALIASES = {
    'improve_consistency_strength': ('improve_character_lora_strength',),
}


def _improve_float(key, default, ceiling=_IMPROVE_MAX_STRENGTH) -> float:
    """Per-key ceiling: the consistency LoRA is itself clamped to 1.5 downstream, and
    the megapixel budget is a resolution, not a strength — one shared ceiling would
    either lie to the user or silently cap a value the UI had offered."""
    raw = cfg.get(f'klein.{key}')
    # cfg.get merges the shipped defaults, so the new key NEVER reads as absent —
    # "still at its default" is what actually means "the user has not set this one",
    # and only then may a value saved under the old name speak for it.
    if raw is None or raw == default:
        for legacy in _IMPROVE_KEY_ALIASES.get(key, ()):
            legacy_value = cfg.get(f'klein.{legacy}')
            if legacy_value is not None:
                raw = legacy_value
                break
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(ceiling, v))


def _improve_int(key, default) -> int:
    try:
        v = int(cfg.get(f'klein.{key}'))
    except (TypeError, ValueError):
        return default
    return max(1, min(_IMPROVE_MAX_STEPS, v))


def _generation_steps() -> int:
    """Sampler steps for a Klein GENERATION job (variations, regenerate, small-image
    rescue). The shipped workflow hardcodes 5 at node 77 and nothing ever passed
    `sampler_steps` on these paths, so the knob existed but was unreachable
    (reported by ashish.sinha, Discord). Default 5 = that exact behaviour; a bad
    config value degrades to it rather than crashing the enqueue."""
    return _improve_int('generation_steps', 5)


def _generation_base_lora_strength() -> float:
    """Enhancement-LoRA strength (node 139, klein/realistic.safetensors) for every
    Klein lane that is NOT "Upscale & improve": the reference edit, variations,
    regenerate, and the small-image rescue.

    The shipped workflow pins that node at 0.8 and none of those lanes ever passed
    a value, so the LoRA applied at full 0.8 with nothing to turn it down. That was
    invisible while the file existed on no install (enqueue_klein_edit bypasses a
    missing LoRA), and became real the day Setup started downloading it
    (klein_enhancement_lora, 031766f): a detail/style LoRA at 0.8 on top of every
    edit steers the render toward its own look instead of the instruction — the
    "Klein edits are not conformant" report.

    Default 0.0 = what every install rendered before that download existed, and the
    same default the improve pass already carries. Raising it is now a choice, on a
    setting, instead of a hardcoded workflow widget."""
    return _improve_float('edit_base_lora_strength', 0.0)
# KLEIN_IMAGE_IMPROVE_PROMPT is the shipped DEFAULT of the editable klein_improve
# prompt (imported from face_variations, which owns the identity/quality prompt
# registry). Re-exported here so `svc.KLEIN_IMAGE_IMPROVE_PROMPT` keeps resolving.
_SMALL_IMAGE_DERIVATIONS = (SMALL_IMAGE_SOURCE, KLEIN_SMALL_IMAGE)
# A striped in-process lock is sufficient for LDS's single local server process
# and makes the active-candidate check + row creation + enqueue one critical
# section.  In particular, a second simultaneous lightbox click waits until the
# first row has its job_id, then takes the idempotent return path below.
_IMAGE_IMPROVE_LOCKS = tuple(threading.Lock() for _ in range(64))
# An in-place pixel edit is a fold on the CURRENT file: two requests for the same
# image must run in order (two mirror clicks restore the original orientation,
# four rotate-right clicks come back round), not both read the same source pixels
# and race to promote a result computed from the same "before".  Mirror and
# rotation deliberately share ONE stripe set so they serialize against each other
# too.  Stripes avoid an unbounded lock map.
_IMAGE_PIXEL_EDIT_LOCKS = tuple(threading.Lock() for _ in range(64))
# Face scoring starts a heavyweight CPU subprocess.  Stripes keep one dataset's
# requests serial without retaining an unbounded lock map; a collision only
# makes an unrelated request retry, never permits concurrent scorers.
_FACE_SCORING_LOCKS = tuple(threading.Lock() for _ in range(64))
# LDS runs one threaded Flask process (backend/run.py). Striped
# locks therefore serialize dataset dedupe snapshots without an unbounded map.
# RLock permits a promotion to hold the stripe across all chunks while nested
# import_images calls retain the same protection.
_DATASET_INGEST_LOCKS = tuple(threading.RLock() for _ in range(64))
_FACE_SCORING_BUSY_DETAIL = 'face scoring is already running; try again shortly'


def _face_scoring_lock(dataset_id):
    return _FACE_SCORING_LOCKS[hash(str(dataset_id)) % len(_FACE_SCORING_LOCKS)]


def _dataset_ingest_lock(user_id, dataset_id):
    return _DATASET_INGEST_LOCKS[
        hash((str(user_id), str(dataset_id))) % len(_DATASET_INGEST_LOCKS)]


def _serialize_dataset_ingest(fn):
    @wraps(fn)
    def wrapped(user_id, dataset_id, *args, **kwargs):
        with _dataset_ingest_lock(user_id, dataset_id):
            return fn(user_id, dataset_id, *args, **kwargs)
    return wrapped


def _serialize_dataset_image_ingest(fn):
    @wraps(fn)
    def wrapped(user_id, image_id, *args, **kwargs):
        image = db.session.get(FaceDatasetImage, image_id)
        if image is None:
            return fn(user_id, image_id, *args, **kwargs)
        with _dataset_ingest_lock(user_id, image.dataset_id):
            return fn(user_id, image_id, *args, **kwargs)
    return wrapped


def _face_scoring_busy_error():
    return {'kind': 'busy', 'detail': _FACE_SCORING_BUSY_DETAIL}



class KleinNodesMissing(Exception):
    """Klein graph preflight failure carried from the service to the HTTP mapper."""

    def __init__(self, missing, missing_nodes):
        self.missing = list(missing or [])
        self.missing_nodes = list(missing_nodes or [])
        super().__init__('Klein custom nodes are missing')


# The modal exposes three request-scoped anchors. Enforce the same bound before
# route reads so a hand-written multipart request cannot create an unbounded
# in-memory snapshot.
MAX_EDIT_REFERENCE_UPLOADS = 3


def _read_external_reference(path, *, label: str) -> bytes:
    """Read at most one external-reference budget, then sanitize exact bytes."""
    try:
        with open(path, 'rb') as fh:
            raw = fh.read(EXTERNAL_REFERENCE_MAX_BYTES + 1)
    except (OSError, TypeError, MemoryError) as exc:
        raise ValueError(f'{label} is unavailable') from exc
    if len(raw) > EXTERNAL_REFERENCE_MAX_BYTES:
        raise ValueError(
            f'{label} is too large (max {EXTERNAL_REFERENCE_MAX_BYTES // (1024 * 1024)} MiB)')
    return sanitize_external_reference(raw, label=label)


_POSE_DIRECTION_TO_KEY = {'left': 'left45', 'right': 'right45', 'back': 'back'}


def _krea_pose_source_path(ds, prompt_text) -> str:
    """Which file this Krea shot should read as its reference. Front
    (`_ref_path`) is the default AND the entire zero-slot-data path: a dataset
    with no RefPoseSlot rows gets {} from enabled_pose_slot_paths, so every
    branch below falls through to the last line — byte-identical to before
    this feature existed."""
    direction = krea_pose_direction(prompt_text)
    enabled = enabled_pose_slot_paths(ds)
    if direction == 'ambiguous':
        # Only left45/right45 are side-facing candidates for this heuristic —
        # 'ambiguous' means "some side/three-quarter cue, no left/right word",
        # so a lone 'back' (or a future left90/right90) slot must NOT be picked
        # up here: it answers a different question than the prompt asked.
        side_enabled = {k: v for k, v in enabled.items() if k in POSE_SLOT_ACTIVE_KEYS}
        if len(side_enabled) == 1:
            return next(iter(side_enabled.values()))
        return _ref_path(ds)
    pose_key = _POSE_DIRECTION_TO_KEY.get(direction)
    if pose_key and pose_key in enabled:
        return enabled[pose_key]
    return _ref_path(ds)


# --- CRUD ------------------------------------------------------------------
# Natures de dataset. 'concept' inverse la logique personnage (cf import_images /
# caption_images). 'style' = esthétique globale : captions de CONTENU pur (le style
# n'est jamais décrit → il est absorbé par le LoRA), pas de trigger dans les captions
# ni dans la config. Tout le reste (dont NULL) = 'character' (défaut historique).
DATASET_KINDS = ('character', 'concept', 'style')


def normalize_kind(kind) -> str | None:
    """'concept'/'style' -> tels quels ; tout le reste -> None (character, stocké NULL)."""
    k = (kind or '').strip().lower()
    return k if k in ('concept', 'style') else None


def _safe_json(text):
    """None-safe json.loads for TEXT columns holding JSON (never raises)."""
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


_PEXELS_PAGE_HOSTS = frozenset({'pexels.com', 'www.pexels.com'})
_PEXELS_IMAGE_HOSTS = frozenset({'images.pexels.com'})
_SOURCE_URL_MAX_CHARS = 2048
_PHOTOGRAPHER_MAX_CHARS = 160


def _safe_source_https_url(value, allowed_hosts):
    """Return a stripped HTTPS URL on an exact allowlisted host, else None."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if (not value or len(value) > _SOURCE_URL_MAX_CHARS
            or any(ord(ch) < 32 for ch in value)):
        return None
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or '').lower()
        port = parsed.port
    except (TypeError, ValueError):
        return None
    if (parsed.scheme != 'https' or host not in allowed_hosts
            or parsed.username is not None or parsed.password is not None
            or port not in (None, 443)):
        return None
    return value


def normalize_source_metadata(value, *, image_url=None):
    """Validate the generic provenance object currently supported by LDS.

    Unknown platforms are deliberately dropped for backwards compatibility.
    Pexels provenance is accepted only when both attribution links are exact
    Pexels HTTPS hosts; at scrape-import time the downloaded image must also be
    hosted by the official Pexels image CDN. Extra keys never reach storage or
    the dataset payload.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, dict) or value.get('platform') != 'pexels':
        return None
    if image_url is not None and not _safe_source_https_url(
            image_url, _PEXELS_IMAGE_HOSTS):
        return None
    photographer = value.get('photographer')
    if not isinstance(photographer, str):
        return None
    photographer = photographer.strip()
    if not photographer or len(photographer) > _PHOTOGRAPHER_MAX_CHARS:
        return None
    photographer = ' '.join(photographer.split())
    source_url = _safe_source_https_url(value.get('source_url'), _PEXELS_PAGE_HOSTS)
    photographer_url = _safe_source_https_url(
        value.get('photographer_url'), _PEXELS_PAGE_HOSTS)
    if not source_url or not photographer_url:
        return None
    return {
        'platform': 'pexels',
        'source_url': source_url,
        'photographer': photographer,
        'photographer_url': photographer_url,
    }


def _source_metadata_storage(value, *, image_url=None):
    metadata = normalize_source_metadata(value, image_url=image_url)
    return (json.dumps(metadata, ensure_ascii=False, separators=(',', ':'))
            if metadata else None)


def _source_metadata_from_scrape_item(item):
    if not isinstance(item, dict) or item.get('platform') != 'pexels':
        return None
    return normalize_source_metadata(item, image_url=item.get('url'))


def _watermark_regions_payload(img) -> dict:
    """Return the nullable stored override and the editor's always-list value."""
    stored = _safe_json(img.watermark_regions)
    if not isinstance(stored, list):
        stored = None
    if stored is not None:
        effective = stored
    else:
        bbox = _safe_json(img.watermark_bbox)
        effective = ([bbox] if img.watermark_state == 'detected'
                     and isinstance(bbox, list) and len(bbox) == 4 else [])
    return {
        'watermark_regions': stored,
        'effective_watermark_regions': effective,
    }


def is_concept(ds) -> bool:
    return bool(ds) and (getattr(ds, 'kind', None) or '').lower() == 'concept'


def is_style(ds) -> bool:
    return bool(ds) and (getattr(ds, 'kind', None) or '').lower() == 'style'


def is_conceptual(ds) -> bool:
    """Concept OU style : les kinds où l'invariant du set n'est PAS une identité.
    Regroupe les comportements communs : heuristiques personnage (équilibre de
    composition, fuite d'identité) sans objet, masques personne interdits (ils
    effaceraient ce qu'on apprend), barème de steps sous-linéaire (√n)."""
    return is_concept(ds) or is_style(ds)


def face_masking_enabled(ds) -> bool:
    """True when a CONCEPT dataset opted into face masking (Advanced training
    options). Reported by shivdbz2010 (GitHub issue #15): a concept LoRA also
    learns the faces of its dataset and then fights a character LoRA over the
    identity; masking the faces teaches the act without the identity.

    OPT-IN, and deliberately stored in the train_settings JSON blob rather than
    on the request (like dual_captions): `masked` already threads through seven
    call sites in routes/training.py plus the cloud lane, and a parallel flag
    would double that. One read, at export time — which also means the local
    queue, the scheduler, a cloud run and a re-run of an OLD dataset all inherit
    it without a single extra line, and no existing dataset changes behaviour.

    Concept only. A Character wants its identity learned, and a Style must learn
    how it renders a face — masking there would amputate the thing being taught."""
    if not ds or not is_concept(ds):
        return False
    raw = getattr(ds, 'train_settings', None)
    if not raw:
        return False
    try:
        return bool(json.loads(raw).get('mask_faces'))
    except (ValueError, TypeError):
        return False


# Person masking (background down-weighted to 10 %) has always defaulted to ON.
# Keep the constant next to the reader: the default is what makes the migration
# to a stored setting a no-op for every dataset that never touched it.
PERSON_MASK_DEFAULT = True


def person_masking_enabled(ds) -> bool:
    """Whether this dataset trains with PERSON masks (subject isolated, background
    at 10 % loss weight). Default ON.

    Used to be a `masked` query parameter carried by the browser's localStorage,
    which the server only ever saw at launch. Three consequences, all real: the
    readiness badge could not warn that a dataset set to masked would train
    unmasked for want of rembg; opening the app from a phone silently reverted to
    the default; and no run snapshot recorded it, so two runs differing only by
    masking looked identical. Stored in the train_settings JSON blob now, exactly
    like `mask_faces` — one read, at export time, so the local queue, the
    scheduler, a cloud run and a re-run of an OLD dataset all inherit it.

    ABSENT key = the historical default (True): no existing dataset changes
    behaviour by upgrading. An explicit False is a VALUE, not a falsy no-op —
    the opposite of `mask_faces`, whose default is OFF.

    Concept/Style are forced OFF here, mirroring the export guard: a person mask
    erases a concept, and an always-on style must learn the whole frame."""
    if not ds:
        return PERSON_MASK_DEFAULT
    if is_conceptual(ds):
        return False
    raw = getattr(ds, 'train_settings', None)
    if not raw:
        return PERSON_MASK_DEFAULT
    try:
        stored = json.loads(raw).get('masked')
    except (ValueError, TypeError):
        return PERSON_MASK_DEFAULT
    return PERSON_MASK_DEFAULT if stored is None else bool(stored)


def person_masking_stored(ds):
    """The RAW stored opt-in — True / False / None when the dataset never answered.
    The panel needs the tri-state (not the resolved boolean) to know whether the
    one-time localStorage carry-over notice still has anything to disclose."""
    raw = getattr(ds, 'train_settings', None) if ds else None
    if not raw:
        return None
    try:
        stored = json.loads(raw).get('masked')
    except (ValueError, TypeError):
        return None
    return None if stored is None else bool(stored)


# Concept descriptions whose ACT lives on the face. Masking the head then erases
# the very thing being taught -- the community workflow this feature follows hit
# exactly this and had to subtract the mouth back out of its face masks. We WARN
# and let the user decide (they know their dataset); we never block.
_FACE_ANCHORED = frozenset({
    'face', 'faces', 'facial', 'head', 'mouth', 'lips', 'lip', 'tongue', 'teeth',
    'throat', 'chin', 'jaw', 'cheek', 'cheeks', 'eye', 'eyes', 'gaze', 'stare',
    'staring', 'expression', 'smile', 'smiling', 'grimace', 'ahegao', 'blowjob',
    'kiss', 'kissing', 'licking', 'lick', 'sucking', 'suck', 'oral', 'deepthroat',
    'facesitting', 'cum', 'cumshot', 'facial_expression', 'nose', 'ear', 'ears',
})


def concept_face_conflict(ds) -> bool:
    """True when this concept's own description names the face/mouth/gaze — i.e.
    when face masking would likely mask away the concept itself. Derived from the
    dataset's concept_desc, never a global list of 'risky' concepts."""
    if not ds or not is_concept(ds):
        return False
    toks = set(re.split(r'[^a-z]+', (getattr(ds, 'concept_desc', '') or '').lower()))
    return bool(toks & _FACE_ANCHORED)


def dual_captions_enabled(ds) -> bool:
    """True when the dataset opted into ai-toolkit dual long+short captioning (Advanced
    training options). The flag lives in the train_settings JSON blob (like the other
    expert levers); default OFF = the historical single-caption behaviour, byte-for-byte.
    Self-contained JSON read so lora_training can reuse it without a circular import."""
    if not ds:
        return False
    raw = getattr(ds, 'train_settings', None)
    if not raw:
        return False
    try:
        return bool(json.loads(raw).get('dual_captions'))
    except (ValueError, TypeError):
        return False


# --- Per-dataset caption method options --------------------------------------
# The Captions ⚙️ Options popover writes these to the caption_options JSON column.
# All three are OVERRIDES of the global captioning defaults: an empty value means
# "follow the global default" (captioning.backend / ollama.vision_model), so a
# dataset that never touched the popover behaves byte-for-byte as before.
_CAPTION_BACKENDS = ('auto', 'joycaption', 'ollama', 'none')
# Extra instructions are APPENDED to the built caption prompt (never replace it),
# so the kind rules (identity/concept/style omission) and the post-filter cleaners
# stay in force — a bounded length keeps a runaway paste out of every prompt.
_CAPTION_INSTRUCTIONS_MAX = 2000

# Vocabulary preset: a one-click register for how the model names nude/sexual content.
# '' = untouched (the model's own default — byte-identical to the pre-preset behaviour).
# This is the NSFW lane: paired with an abliterated Ollama model, 'explicit' makes the
# captioner spell the acts out. It rides in as an appended instruction just like the free
# text, so the identity/concept omission rules and the leak cleaners still post-filter —
# the preset changes wording, never what is allowed to bind to the trigger. NONE of the
# cleaners (identity/body/concept ban-lists) touch crude anatomical vocabulary, so an
# 'explicit' caption keeps its terms; only identity/concept leaks are ever scrubbed.
_CAPTION_VOCABULARIES = ('explicit', 'clinical', 'safe')
# Public so the image bank's caption lane validates against — and appends — the SAME
# vocabulary registers as the dataset pass, rather than duplicating the tuple or the
# text. Defined HERE, next to the private tuple it aliases, rather than in
# captioning_service.py: a module-level alias is evaluated while that module's body
# runs, which is before its bottom borrow-back has bound anything (2026-08 split).
CAPTION_VOCABULARIES = _CAPTION_VOCABULARIES
_VOCABULARY_INSTRUCTION = {
    'explicit': (
        'Describe any sexual, nude or intimate content directly and explicitly, using '
        'plain, crude anatomical terms. Do not censor, euphemise, hedge or soften — name '
        'the body parts and the acts as they are.'),
    'clinical': (
        'Describe any nude or sexual content in neutral, clinical, anatomical terms. '
        'Avoid crude slang and euphemism alike — be precise and matter-of-fact.'),
    'safe': (
        'Keep the description strictly non-explicit. Do not use sexual or crude terms; '
        'refer to any nudity only in general, non-graphic language.'),
}


def caption_options(ds) -> dict:
    """Normalized per-dataset caption overrides: {backend, ollama_model, instructions}.
    Empty strings = "use the global default". Never raises ({} defaults on a missing or
    corrupt blob) so every caption path can read it unconditionally."""
    out = {'backend': '', 'ollama_model': '', 'instructions': '', 'vocabulary': ''}
    raw = getattr(ds, 'caption_options', None) if ds else None
    if not raw:
        return out
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return out
    if not isinstance(data, dict):
        return out
    backend = str(data.get('backend') or '').strip().lower()
    if backend in _CAPTION_BACKENDS:
        out['backend'] = backend
    try:
        out['ollama_model'] = normalize_ollama_model_ref(
            data.get('ollama_model', ''), allow_empty=True)
    except ValueError:
        # Legacy/manual DB blobs are untrusted input too. Keep every other valid
        # option but fall back to the global model instead of propagating a bad ref.
        out['ollama_model'] = ''
    out['instructions'] = str(data.get('instructions') or '').strip()[:_CAPTION_INSTRUCTIONS_MAX]
    vocab = str(data.get('vocabulary') or '').strip().lower()
    if vocab in _CAPTION_VOCABULARIES:
        out['vocabulary'] = vocab
    return out


def set_caption_options(user_id, dataset_id, patch) -> dict:
    """Persist a caption-options patch (only the provided keys change). An invalid engine
    raises ValueError (mapped 400 by the route). Empty keys are dropped so a fully-default
    dataset stores NULL — identical to one that never opened the popover. Returns the
    resulting normalized options."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    cur = caption_options(ds)
    if 'backend' in patch:
        b = str(patch.get('backend') or '').strip().lower()
        if b and b not in _CAPTION_BACKENDS:
            raise ValueError(f'invalid captioning backend: {b}')
        cur['backend'] = b
    if 'ollama_model' in patch:
        cur['ollama_model'] = normalize_ollama_model_ref(
            patch.get('ollama_model'), allow_empty=True)
    if 'instructions' in patch:
        cur['instructions'] = str(patch.get('instructions') or '').strip()[:_CAPTION_INSTRUCTIONS_MAX]
    if 'vocabulary' in patch:
        v = str(patch.get('vocabulary') or '').strip().lower()
        if v and v not in _CAPTION_VOCABULARIES:
            raise ValueError(f'invalid caption vocabulary: {v}')
        cur['vocabulary'] = v
    stored = {k: v for k, v in cur.items() if v}
    ds.caption_options = json.dumps(stored) if stored else None
    db.session.commit()
    return cur


# --- Which Klein model this dataset runs on ----------------------------------
# Stored on the DATASET, not in localStorage: it describes what the dataset is
# made of, so it must survive a browser change and be the same from a phone. The
# generation picker had a per-browser value (editPage_flux2KleinModel_v1) that
# improve never even read — hence "no option anywhere to choose the model used
# for improve". NULL = auto (resolve_klein_unet decides), which is exactly what
# every improve did before this setting existed.
def dataset_klein_model(ds):
    """The bare Klein model file name this dataset chose, or None for auto."""
    name = (getattr(ds, 'klein_model', None) or '').strip() if ds else ''
    return name or None


def set_dataset_klein_model(user_id, dataset_id, name):
    """Persist the dataset's Klein model pick. '' / None clears it back to auto —
    un-choosing has to be a real gesture, not a value you can never take back.

    Only a BARE file name is accepted: the picker lists bare names (the loader
    prefix is resolve_klein_unet's job), so a value carrying a path separator is
    never something the UI produced. Existence is deliberately NOT checked here —
    a model can be moved away long after it was chosen, and the honest place to
    say so is the run (KleinModelGone names the file), not a settings write that
    would silently drop the user's answer."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    value = (name or '').strip()
    if value and (os.path.basename(value) != value or value in ('.', '..')):
        raise ValueError('a Klein model is named by its file name, without a folder')
    ds.klein_model = value or None
    db.session.commit()
    return dataset_klein_model(ds)


def _resolve_caption_backend(ds) -> str:
    """The engine a caption run uses: the dataset override when set, else the global
    captioning.backend (default 'auto')."""
    return (caption_options(ds).get('backend')
            or cfg.get('captioning.backend') or 'auto').lower()


def _with_caption_instructions(prompt: str, instructions: str) -> str:
    """Append the user's extra instructions to a built caption prompt. The base prompt
    (with its kind omission rules) stays first so the model still reads them; the extras
    ride at the end under a clear header. The output cleaners run regardless, so this can
    never reintroduce a banned identity/concept term."""
    extra = (instructions or '').strip()
    if not extra:
        return prompt
    return f'{prompt}\n\nAdditional instructions from the user:\n{extra}'


def _combined_caption_instructions(opts) -> str:
    """The text appended to a caption prompt for a run: the vocabulary preset (if any),
    then the user's free-text instructions. Empty when neither is set — so a dataset that
    never touched the popover produces byte-identical prompts. Both ride at the END of the
    prompt, after the kind omission rules, and the output cleaners still post-filter."""
    parts = []
    preset = _VOCABULARY_INSTRUCTION.get(opts.get('vocabulary'))
    if preset:
        parts.append(preset)
    extra = (opts.get('instructions') or '').strip()
    if extra:
        parts.append(extra)
    return '\n\n'.join(parts)


# Cibles de fidélité (datasets personnage). 'body' = le LoRA reproduit AUSSI la
# morphologie : captions bannissent en plus les marques corporelles permanentes
# (elles se lient au trigger), composition recommandée plus corps/buste, import
# plein cadre par défaut.
FIDELITIES = ('face', 'body')


def normalize_fidelity(f) -> str:
    f = (f or '').strip().lower()
    return f if f in FIDELITIES else 'face'


def is_body_fidelity(ds) -> bool:
    return bool(ds) and (getattr(ds, 'fidelity', None) or 'face').lower() == 'body'


def set_fidelity(user_id, dataset_id, fidelity) -> bool:
    """Switch face-only <-> full-body fidelity later. Affects FUTURE captions
    (re-caption to apply) + the composition target + the import crop default."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return False
    ds.fidelity = normalize_fidelity(fidelity)
    db.session.commit()
    return True


# Familles de modèle entraînables (= pipeline ai-toolkit). Source de vérité côté UI
# ET validation : choisie à la création, drive le format de caption (sdxl→booru, sinon
# prose) et le regroupement du menu. Reste modifiable ensuite (TrainingPanel).
# NB : 'flux2klein' (FLUX.2 Klein) — PAS 'klein' : ce namespace est déjà pris par
# le moteur de GÉNÉRATION (engines.klein, unet/klein/) ; un train_type 'klein'
# télescoperait les résolveurs de modèles et les chemins loras du Studio.
TRAIN_TYPES = ('zimage', 'sdxl', 'krea', 'flux', 'flux2klein', 'anima')


def normalize_train_type(t) -> str:
    """Famille valide en minuscules, défaut 'zimage' (toute valeur inconnue/None)."""
    t = (t or '').strip().lower()
    return t if t in TRAIN_TYPES else 'zimage'


# --- Prompt suffixes (creative direction, community feature request) ----------
# Free user text that rides on every generated variation: a GLOBAL suffix plus an
# optional per-framing map (same buckets as the composition). Persisted on the
# dataset row, applied at WRAP time only (never baked into variation_prompt — a
# regenerate would double-apply it). Composition: per-framing first, then global
# (see face_variations.compose_prompt_suffix).
SUFFIX_FRAMINGS = ('face', 'bust', 'body', 'back')
MAX_SUFFIX_LEN = 300


def _normalize_prompt_suffix(value):
    """Provided global-suffix string -> stripped/capped text or None (cleared)."""
    if not isinstance(value, str):
        raise ValueError('prompt_suffix must be a string')
    return value.strip()[:MAX_SUFFIX_LEN] or None


def _normalize_prompt_suffixes(value):
    """Provided per-framing map -> JSON text keeping only non-empty known keys,
    or None when nothing remains ({} therefore CLEARS the map). The whole map is
    replaced on each write — simple, predictable modal semantics."""
    if not isinstance(value, dict):
        raise ValueError('prompt_suffixes must be an object {face,bust,body,back}')
    out = {}
    for k in SUFFIX_FRAMINGS:
        v = value.get(k)
        if v is None:
            continue
        if not isinstance(v, str):
            raise ValueError(f'prompt_suffixes.{k} must be a string')
        v = v.strip()[:MAX_SUFFIX_LEN]
        if v:
            out[k] = v
    return json.dumps(out, ensure_ascii=False) if out else None


def prompt_suffixes_dict(ds) -> dict:
    """The stored per-framing suffix map as a clean dict (defensive JSON parse;
    unknown keys / non-string values dropped). {} when unset."""
    raw = getattr(ds, 'prompt_suffixes', None) if ds else None
    if not raw:
        return {}
    try:
        m = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(m, dict):
        return {}
    return {k: v.strip() for k, v in m.items()
            if k in SUFFIX_FRAMINGS and isinstance(v, str) and v.strip()}


def dataset_prompt_suffix(ds, framing=None) -> str:
    """The dataset's EFFECTIVE creative-direction suffix for one shot (per-framing
    then global). Every wrap call site funnels through here so the suffix is
    applied exactly once, at generation time — the stored variation_prompt stays
    raw and regeneration can never double-apply it."""
    if not ds:
        return ''
    return compose_prompt_suffix(getattr(ds, 'prompt_suffix', None),
                                 getattr(ds, 'prompt_suffixes', None), framing)


def subject_type_of(ds) -> str:
    """The dataset's subject type, normalised — NULL/legacy -> 'human'. The single
    reader every wrap call site uses so a legacy dataset (column NULL) generates
    exactly as before."""
    return normalize_subject_type(getattr(ds, 'subject_type', None) if ds else None)


# InsightFace/antelopev2 is a detector+embedder trained on PHOTOGRAPHED faces. On a
# drawn character it detects nothing most of the time, and the rare "detection" is a
# meaningless cosine — the pass used to fail OPEN: grey tiles or a plausible number,
# with nothing saying the tool simply cannot read this kind of image.
# The message states the way out on purpose: there is NO extra setting to force the
# pass. A knob whose only correct value is "off" is a knob nobody can set right; the
# subject type IS the switch, it is one click away, and it says what it means. A
# genuinely photographic dataset mislabelled anime is fixed where the mistake is.
FACE_SCORING_DRAWN_REASON = (
    'Face similarity needs a photographic face; it cannot read a drawn one. '
    'Set the subject type to Human if this dataset is photographic.')


def face_scoring_block_reason(ds):
    """Why InsightFace scoring must NOT run on this dataset, or None to go ahead.

    The SINGLE place the rule lives: the dataset pass, the Studio cell scoring and
    best-epoch selection all consult this one function, and the dataset payload
    republishes its result so the UI never re-derives the rule either. A gate
    posted at four sites would drift; this one cannot.

    Scoped to face SIMILARITY. Head-cropping (`face_crop_to_square_webp` ->
    `detect_head_bbox`) goes through Qwen3-VL, a general vision model that reads a
    drawn head perfectly well — it is deliberately NOT gated here."""
    if subject_type_of(ds) == 'anime':
        return FACE_SCORING_DRAWN_REASON
    return None


def create_dataset(user_id, name, trigger_word, kind=None, concept_desc=None, train_type=None,
                   fidelity=None, prompt_suffix=None, prompt_suffixes=None, subject_type=None,
                   *, commit=True):
    """Create a dataset and return its row.

    ``commit=False`` is reserved for callers that need to coordinate the row with
    another resource (for example a restored filesystem tree).  The row is still
    flushed so its id is available, but ownership of commit/rollback stays with
    the caller.  Ordinary callers keep the historical commit-on-return contract.
    """
    k = normalize_kind(kind)
    desc = (concept_desc or '').strip()
    if k == 'concept' and not desc:
        # The concept description is what the captioner OMITS; without it the
        # inverted-caption logic has nothing to bind the trigger to. Required.
        raise ValueError('concept_desc required for a concept dataset')
    ds = FaceDataset(user_id=str(user_id), name=(name or '').strip()[:100],
                     trigger_word=(trigger_word or '').strip()[:60] or 'zchar',
                     # concept_desc n'a de sens que pour un concept ; un STYLE n'a rien
                     # à omettre nommément (les captions décrivent le contenu, jamais le
                     # rendu — c'est le prompt de caption qui porte cette règle).
                     kind=k, concept_desc=(desc[:500] if k == 'concept' else None),
                     # subject_type steers the generation catalog + identity lock;
                     # None left as NULL (== 'human') so a plain create is unchanged.
                     subject_type=(normalize_subject_type(subject_type)
                                   if subject_type is not None else None),
                     train_type=normalize_train_type(train_type),
                     # fidelity ne concerne que les personnages (concept : l'acte est
                     # omis ; style : les sujets varient, aucune identité à protéger).
                     fidelity=(normalize_fidelity(fidelity) if k is None else None),
                     # Direction créative optionnelle (globale + par cadrage) appliquée
                     # au wrap de chaque variation générée — cf. dataset_prompt_suffix.
                     prompt_suffix=(_normalize_prompt_suffix(prompt_suffix)
                                    if prompt_suffix is not None else None),
                     prompt_suffixes=(_normalize_prompt_suffixes(prompt_suffixes)
                                      if prompt_suffixes is not None else None))
    db.session.add(ds)
    db.session.flush()
    if k == 'style' and not (trigger_word or '').strip():
        # Le token d'un style est un identifiant INTERNE, jamais un mot d'activation :
        # `_run_name`/`lora_{trigger}` nomment le run d'entraînement avec. Deux styles
        # créés sans trigger retomberaient tous deux sur 'zchar' → le garde anti-
        # collision bloquerait le 2e entraînement. On sale le défaut avec l'id.
        ds.trigger_word = f'zsty_{ds.id}'
    if commit:
        db.session.commit()
    return ds


def family_base_memory(ds) -> dict:
    """Parsed `train_family_bases` — {family: {'base': str, 'variant': str|None}}.

    Anything unparsable/foreign reads as {} (same discipline as _train_settings):
    a corrupted blob must degrade to "nothing remembered", never to a crash."""
    raw = getattr(ds, 'train_family_bases', None)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for fam, entry in data.items():
        if fam in TRAIN_TYPES and isinstance(entry, dict):
            out[fam] = {'base': entry.get('base') or '',
                        'variant': entry.get('variant') or None}
    return out


def remembered_family_base(ds, family):
    """(base, variant) this dataset last used on `family`, or (None, None) when
    that family has never been configured here. `None` is deliberately distinct
    from `''` (= "officially chose the official base")."""
    entry = family_base_memory(ds).get(normalize_train_type(family))
    if entry is None:
        return None, None
    return entry['base'], entry['variant']


def family_settings_memory(ds) -> dict:
    """Parsed `train_family_settings` — {family: {setting: value}}, restricted to
    the family-scoped keys. Same degrade-to-{} discipline as family_base_memory:
    a corrupted blob means "nothing remembered", never a crash."""
    from . import lora_training as _lt
    raw = getattr(ds, 'train_family_settings', None)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for fam, entry in data.items():
        if fam in TRAIN_TYPES and isinstance(entry, dict):
            out[fam] = {k: v for k, v in entry.items()
                        if k in _lt._FAMILY_SCOPED_SETTING_KEYS}
    return out


def remembered_family_settings(ds, family):
    """The family-scoped settings this dataset last used on `family`, or None
    when that family was never configured here. `{}` (configured, everything on
    Auto) is deliberately distinct from None (never configured)."""
    return family_settings_memory(ds).get(normalize_train_type(family))


def set_train_type(user_id, dataset_id, train_type, *, commit=True,
                   target_training_mode=None) -> bool:
    """Change the target model family later (kept in sync with the TrainingPanel
    selector so the menu re-groups). Normalizes; unknown -> zimage. False if absent.

    The base and the variant are FAMILY-SCOPED even though `train_base_model` /
    `train_variant` are single columns: a Z-Image merge is not a thing a Krea run
    can load, and 'turbo' means a different checkpoint on each family. So the
    outgoing family's pair is stashed in `train_family_bases` and the incoming
    family's remembered pair takes its place — a family never yet configured
    starts from the official base, and coming back to Z-Image finds the merge
    exactly where it was left. Nothing is destroyed and nothing is asked.

    The SAME treatment is given to the handful of `train_settings` keys whose
    meaning is bound to the family (lora_training._FAMILY_SCOPED_SETTING_KEYS —
    `timestep_type`, whose canonical value differs per family): stashed in
    `train_family_settings`, restored on the way back, and CLEARED (back to the
    incoming family's own default) when that family has nothing remembered. The
    other advanced settings stay global on purpose — see the comment on
    _FAMILY_SCOPED_SETTING_KEYS for why quantisation and resolution are not
    here. ``commit=False`` lets a caller join this family transition to a wider
    validated settings transaction without an intermediate database state.
    ``target_training_mode`` is reserved for that wider transaction: the legacy
    family-only endpoint must validate against the currently persisted mode."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return False
    new_fam = normalize_train_type(train_type)
    old_fam = normalize_train_type(getattr(ds, 'train_type', None))
    from . import lora_training as _lt
    intended_mode = (_lt.training_mode(ds) if target_training_mode is None
                     else _lt.normalize_training_mode(target_training_mode))
    if intended_mode == 'full_transformer' and new_fam != 'krea':
        raise ValueError(
            'full_transformer training requires the Krea 2 model family. '
            'Switch the training mode to LoRA in Training settings before '
            'changing the model family.')
    if new_fam == old_fam:
        if commit:
            db.session.commit()
        return True
    memory = family_base_memory(ds)
    # Never remember a base the OUTGOING family provably cannot load. Datasets
    # created before this column exist in exactly that state (a Z-Image merge
    # left attached to a Krea 2 dataset); stashing it under 'krea' would freeze
    # the bug into the memory and hand it back on the way home.
    outgoing = ds.train_base_model or ''
    if _lt.foreign_base_reason(old_fam, outgoing):
        outgoing = ''
    memory[old_fam] = {'base': outgoing,
                       'variant': ds.train_variant or None}
    remembered = memory.get(new_fam)
    ds.train_base_model = (remembered or {}).get('base') or None
    ds.train_variant = (remembered or {}).get('variant') or None
    ds.train_family_bases = json.dumps(memory)

    # --- family-scoped train_settings keys, same stash/restore contract --------
    scoped = _lt._FAMILY_SCOPED_SETTING_KEYS
    settings = _lt._train_settings(ds)
    # An applied preset is itself family-scoped. Invalidate that replacement
    # before stashing/restoring individual family values, otherwise its hidden
    # topology/optimizer/step fields could survive under the new family's UI.
    _lt.clear_active_preset_settings(settings)
    smemory = family_settings_memory(ds)
    smemory[old_fam] = {k: settings[k] for k in scoped if k in settings}
    incoming = smemory.get(new_fam)
    for k in scoped:
        if incoming is not None and k in incoming:
            settings[k] = incoming[k]
        else:
            # Never configured on the incoming family (or explicitly left on
            # Auto there) → drop the key so the family's own canonical default
            # applies. Dropping is what makes it byte-identical to a dataset
            # that never touched the setting, exactly like update_train_settings.
            settings.pop(k, None)
    ds.train_settings = json.dumps(settings) if settings else None
    ds.train_family_settings = json.dumps(smemory)

    ds.train_type = new_fam
    if commit:
        db.session.commit()
    return True


def _guard_kind_switch(dataset_id):
    """Raise RuntimeError (-> 409) when live work on the dataset still assumes the
    CURRENT kind: an active training run, a server-side batch (caption / re-caption
    / watermark / face / classify) or an in-flight generation. Switching the kind
    mid-flight would mix caption strategies, or land generated variations into a set
    that no longer generates. ``dataset_activity`` covers the batch AND generation
    cases (the Klein/API fan-out is tracked as a 'generate' activity)."""
    _guard_no_active_training(dataset_id)
    if dataset_activity.get(dataset_id) is not None:
        raise RuntimeError(
            'This dataset has work in progress (generation, captioning or a quality '
            'pass). Wait for it to finish before changing the kind.')


def update_dataset_settings(user_id, dataset_id, *, name=None, trigger_word=None,
                            concept_desc=None, kind=None, prompt_suffix=None,
                            prompt_suffixes=None, subject_type=None):
    """Edit a dataset's identity AFTER creation. Returns {'ok', 'concept_desc_changed'}
    (plus {'kind_changed', 'kind', 'previous_kind'} when the kind actually changed),
    or None if the dataset is absent; raises ValueError on invalid input and
    RuntimeError (-> 409) when a kind switch is asked while work is in progress.

    Changing the **trigger word** needs NO re-caption: captions are stored without it
    (it's prepended at export). It is, however, the ON-DISK naming key, so everything
    the dataset already produced is renamed to follow — see _propagate_trigger_rename,
    reported back as `trigger_rename`. Refused (409) while a run is live, because the
    run folder is what ai-toolkit auto-resumes from. Changing a concept dataset's **description**
    (what the captions must omit) invalidates the cached LLM avoid-list (concept_terms)
    so it regenerates — but images already captioned keep the OLD omission until
    re-captioned (same 'future captions' contract as set_fidelity).

    Changing the **kind** (character / concept / style) is the disruptive one: it flips
    the caption strategy and which workspace panels show. It is honest, not magic —
    NOTHING is deleted (images, captions, scores, watermark work and training history
    stay), but existing captions keep the OLD strategy until re-captioned (the route's
    caller nudges it). Invariants mirror create_dataset: fidelity is character-only
    (cleared for concept/style); the concept avoid-list cache is dropped so it rebuilds
    for the new kind; a concept target requires an omit-description (passed here or
    already stored); a style keeps its stored trigger token but never uses it as an
    activation word. Past run identifiers are unaffected — a run is named by the model
    family + trigger, never the kind (see lora_training._run_name).

    **prompt_suffix** (global text) / **prompt_suffixes** (map {face,bust,body,back}):
    None = untouched; '' / {} = cleared. Applied at generation time only, so editing
    them changes FUTURE generations/regenerations — existing images are untouched."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return None
    # The on-disk naming key, measured ONCE before any mutation and once after them
    # all. Two different edits can move it (the trigger word, or a style's name — see
    # below), and a dataset can be edited by both in a single save, so comparing the
    # start and end states is the only reading that can't disagree with itself.
    _lt = _lora_training()
    naming_before = _lt._safe_trigger(ds) if _lt else None
    prev_label = (ds.kind or '').lower() or 'character'
    kind_changed = False
    if kind is not None:
        new_kind = normalize_kind(kind)          # None | 'concept' | 'style'
        new_label = new_kind or 'character'
        if new_label != prev_label:
            _guard_kind_switch(dataset_id)
            if new_label == 'concept':
                # A concept needs the omit-description: take the one passed in this
                # same save, else any value already stored (a switch back to concept).
                desc_src = concept_desc if concept_desc is not None else ds.concept_desc
                if not (desc_src or '').strip():
                    raise ValueError('concept_desc required for a concept dataset')
            ds.kind = new_kind
            if new_label != 'character':
                # Fidelity is a character-only target (mirrors create_dataset). The
                # value is remembered by nothing else, so a switch back defaults to face.
                ds.fidelity = None
            # The cached concept avoid-list is concept-specific; drop it so the
            # detector/captioner rebuild it for the new kind. concept_desc itself is
            # left in place (harmless for other kinds, restored on a switch back).
            ds.concept_terms = None
            kind_changed = True
    if name is not None:
        n = (name or '').strip()
        if n:
            new_name = n[:100]
            # A STYLE has no visible trigger — it is always-on, so the field is hidden
            # and the token that names its files is retained internally, out of reach.
            # Its NAME is therefore the only identity it can edit, so for a style (and
            # only a style) the name drives the naming token too; without this, a style
            # dataset could never rename the LoRAs it had already produced. The token is
            # pure file naming for a style (never an activation word), so moving it
            # changes nothing about captions or generation.
            if is_style(ds) and new_name != (ds.name or '') and _lt:
                token = _lt._safe_trigger(SimpleNamespace(
                    trigger_word=new_name, id=ds.id))[:60]
                if token != (ds.trigger_word or ''):
                    _guard_no_active_training(dataset_id, action='renaming a style dataset')
                    ds.trigger_word = token
            ds.name = new_name
    trigger_rename = None        # (old_safe, new_safe) when the on-disk naming key moved
    # A STYLE has no trigger FIELD — the settings modal sends back the stored token
    # verbatim (`trigger_word: style ? d.trigger_word : ...`). Honouring that echo
    # here overwrote the token the name block had just derived from the new name, so
    # renaming a style changed its label and nothing else: the reported bug. For a
    # style the name is the only lever, so an incoming trigger is never an edit.
    if is_style(ds):
        trigger_word = None
    if trigger_word is not None:
        t = (trigger_word or '').strip()
        if t:
            if t[:60] != (ds.trigger_word or ''):
                # The trigger is the ON-DISK naming key (u{user}_{trigger} run folders,
                # lora_{trigger} deployed files), so changing it renames everything this
                # dataset already produced. Refuse mid-flight: the run folder IS what
                # ai-toolkit auto-resumes from, and moving it under a live job would
                # strand the run. The rename itself is decided from naming_before /
                # naming_after around the whole edit, not here.
                _guard_no_active_training(dataset_id, action='changing the trigger word')
            ds.trigger_word = t[:60]
        elif not is_style(ds):
            # A character/concept trigger is the summon token — it cannot be blank.
            # A style has no activation trigger, so an empty value just keeps the
            # retained internal token as-is.
            raise ValueError('trigger_word cannot be empty')
    concept_changed = False
    if concept_desc is not None and is_concept(ds):
        d = (concept_desc or '').strip()
        if not d:
            raise ValueError('concept_desc required for a concept dataset')
        if d[:500] != (ds.concept_desc or ''):
            ds.concept_desc = d[:500]
            ds.concept_terms = None   # invalidate the cached LLM avoid-list → regenerated next caption
            concept_changed = True
    if prompt_suffix is not None:
        ds.prompt_suffix = _normalize_prompt_suffix(prompt_suffix)
    if prompt_suffixes is not None:
        ds.prompt_suffixes = _normalize_prompt_suffixes(prompt_suffixes)
    # subject_type: None = untouched. Only steers FUTURE wraps (existing images keep
    # their stored variation_prompt), so no in-flight guard is needed.
    if subject_type is not None:
        ds.subject_type = normalize_subject_type(subject_type)
    naming_after = _lt._safe_trigger(ds) if _lt else None
    if naming_before and naming_after and naming_before != naming_after:
        trigger_rename = (naming_before, naming_after)
    db.session.commit()
    res = {'ok': True, 'concept_desc_changed': concept_changed}
    if kind_changed:
        res.update(kind_changed=True, kind=(ds.kind or 'character'),
                   previous_kind=prev_label)
    if trigger_rename:
        moved = _propagate_trigger_rename(ds, *trigger_rename)
        # Only reported when it actually did something: a dataset that never trained
        # has no artefacts to move, and a silent 0-file rename is indistinguishable
        # from no rename at all — so the response stays exactly as it was before.
        if moved['files'] or not moved['ok']:
            res['trigger_rename'] = moved
    return res


def _lora_training():
    """lora_training, or None in a phase-1 install where it isn't present yet.
    Lazy: face_dataset_service <-> lora_training is a circular import at module level."""
    try:
        from . import lora_training as lt
        return lt
    except ImportError:
        return None


def _propagate_trigger_rename(ds, old_safe, new_safe) -> dict:
    """Carry a trigger rename through to disk AND to the rows that point at the
    renamed files. Returns {'ok', 'files', 'rows', 'conflicts'} for the caller to
    report; never raises — a failed rename leaves a working dataset whose old
    artefacts simply keep the old name (exactly today's behaviour).

    The database rewrite is DERIVED FROM THE FILES ACTUALLY RENAMED rather than
    rebuilt from the trigger: stored checkpoint values carry a ComfyUI subfolder
    ('z image\\...') and a family/step suffix, so reconstructing them here would
    duplicate — and eventually contradict — the naming rules in lora_training.
    Matching on basename keeps this correct whatever those rules become."""
    lt = _lora_training()
    if lt is None:
        return {'ok': False, 'files': 0, 'rows': 0, 'conflicts': []}
    out = lt.rename_training_artifacts(ds.user_id, old_safe, new_safe)
    if not out['ok']:
        # A destination already existed (a dataset already using the new trigger).
        # Nothing was moved, so nothing in the DB may be rewritten either.
        return {'ok': False, 'files': 0, 'rows': 0, 'conflicts': out['conflicts']}

    renames = out['renamed']
    by_basename = {os.path.basename(src): os.path.basename(dest)
                   for src, dest in renames if src.endswith('.safetensors')}
    dir_moves = [(src, dest) for src, dest in renames if not os.path.splitext(src)[1]]
    rows = 0

    def _remap(value):
        """The new name for a stored LoRA reference, or None when it isn't one of
        the files we just moved. Compares on basename so a stored subfolder prefix
        ('z image\\lora_X.safetensors') survives untouched."""
        if not value:
            return None
        base = os.path.basename(str(value).replace('\\', '/'))
        new_base = by_basename.get(base)
        return str(value)[:-len(base)] + new_base if new_base else None

    if by_basename:
        for row in LoraTestImage.query.filter_by(dataset_id=ds.id).all():
            new_ck = _remap(row.checkpoint)
            if new_ck:
                row.checkpoint = new_ck
                rows += 1
        # The dataset's winning Test-Studio settings pin a LoRA filename too.
        settings = _safe_json(ds.best_settings)
        if isinstance(settings, dict):
            new_ck = _remap(settings.get('lora_filename'))
            if new_ck:
                settings['lora_filename'] = new_ck
                ds.best_settings = json.dumps(settings)
                rows += 1

    # Cloud runs store the local run identity (u{user}_{trigger}{tag}) and cache
    # absolute paths under the renamed run folders — both carry the old trigger.
    from ..models import CloudTrainingRun
    old_run, new_run = f'u{ds.user_id}_{old_safe}', f'u{ds.user_id}_{new_safe}'
    for run in CloudTrainingRun.query.filter_by(dataset_id=ds.id).all():
        if run.run_name and lt._trigger_boundary(run.run_name, old_run):
            run.run_name = new_run + run.run_name[len(old_run):]
            rows += 1
        for attr in ('staging_dir', 'checkpoint_local_path'):
            cur = getattr(run, attr, None)
            for src, dest in dir_moves:
                if cur and os.path.normcase(str(cur)).startswith(os.path.normcase(src)):
                    setattr(run, attr, dest + str(cur)[len(src):])
                    rows += 1
                    break
    db.session.commit()
    return {'ok': True, 'files': len(renames), 'rows': rows, 'conflicts': []}


def get_dataset(user_id, dataset_id):
    ds = db.session.get(FaceDataset, dataset_id)
    return ds if ds and str(ds.user_id) == str(user_id) else None


def random_kept_caption(user_id, dataset_id) -> str | None:
    """Return one cleaned caption from an owned dataset's kept images.

    The candidate count and random offset stay in SQL so a large dataset never
    has all of its captions materialized just to choose one.  ``None`` means
    the dataset exists but has no non-blank kept caption; an inaccessible
    dataset raises ``LookupError`` so the route can return a 404 without
    leaking ownership details.
    """
    if not get_dataset(user_id, dataset_id):
        raise LookupError('dataset not found')

    from sqlalchemy import func

    cleaned = func.trim(FaceDatasetImage.caption, _PYTHON_STRIP_CHARS)
    eligible = (db.session.query(FaceDatasetImage.caption)
                .join(FaceDataset, FaceDatasetImage.dataset_id == FaceDataset.id)
                .filter(FaceDatasetImage.dataset_id == dataset_id,
                        FaceDataset.user_id == str(user_id),
                        FaceDatasetImage.status == 'keep',
                        FaceDatasetImage.caption.isnot(None),
                        cleaned != ''))
    count = eligible.count()
    if not count:
        return None

    caption = (eligible.order_by(FaceDatasetImage.id.asc())
               .offset(random.randrange(count)).limit(1).scalar())
    # Keep the API contract robust if an unusual Unicode whitespace-only value
    # slipped through SQLite's trim character set.
    return (caption or '').strip() or None


def list_datasets(user_id):
    return (FaceDataset.query.filter_by(user_id=str(user_id))
            .order_by(FaceDataset.updated_at.desc()).all())


def dataset_list_stats(user_id):
    """Per-dataset aggregates for the library page — image counts and the
    families ever trained — in two grouped queries (never one per dataset).
    Returns {dataset_id: {'images_total', 'images_kept', 'images_captioned',
    'trained_families': [str]}}; datasets absent from a map just have zeros."""
    from sqlalchemy import case, func
    from ..models import TrainingRunRecord
    owned = (db.session.query(FaceDataset.id)
             .filter_by(user_id=str(user_id))).subquery()
    stats = {}
    img_rows = (db.session.query(
        FaceDatasetImage.dataset_id,
        func.count(FaceDatasetImage.id),
        func.sum(case((FaceDatasetImage.status == 'keep', 1), else_=0)),
        func.sum(case(((FaceDatasetImage.status == 'keep')
                       & (func.coalesce(FaceDatasetImage.caption, '') != ''), 1), else_=0)))
        .filter(FaceDatasetImage.dataset_id.in_(db.session.query(owned.c.id)))
        .group_by(FaceDatasetImage.dataset_id).all())
    for ds_id, total, kept, captioned in img_rows:
        stats[ds_id] = {'images_total': int(total or 0), 'images_kept': int(kept or 0),
                        'images_captioned': int(captioned or 0), 'trained_families': []}
    fam_rows = (db.session.query(TrainingRunRecord.dataset_id, TrainingRunRecord.family)
                .filter(TrainingRunRecord.dataset_id.in_(db.session.query(owned.c.id)))
                .distinct().all())
    for ds_id, fam in fam_rows:
        entry = stats.setdefault(ds_id, {'images_total': 0, 'images_kept': 0,
                                         'images_captioned': 0, 'trained_families': []})
        if fam and fam not in entry['trained_families']:
            entry['trained_families'].append(fam)
    for entry in stats.values():
        entry['trained_families'].sort()
    return stats


def _clear_watermark_metadata(img):
    img.watermark_state = None
    img.watermark_bbox = None
    img.watermark_regions = None


def _unkeep_parent_for_kept_improvement(img):
    """Make a kept Klein improvement the dataset's active choice.

    An improve result is a separate image row, so this deliberately changes only
    the original row's review state: no files, captions or lineage are removed.
    The parent lookup is scoped to the candidate's dataset because legacy
    ``parent_image_id`` has no foreign key and may be stale or point elsewhere.
    A queued result can be marked Keep before its bytes arrive; it cannot replace
    the source until a regular file has actually landed in the dataset folder.
    """
    filename = img.filename
    if (img.derivation_kind != KLEIN_IMAGE_IMPROVE
            or not img.parent_image_id
            or not isinstance(filename, str)
            or not filename
            or '/' in filename
            or '\\' in filename
            or os.path.basename(filename) != filename
            or ntpath.basename(filename) != filename
            or posixpath.basename(filename) != filename
            or img.parent_image_id == img.id):
        return False
    try:
        candidate_path = _img_path(img)
    except (TypeError, ValueError):
        return False
    if not os.path.isfile(candidate_path):
        return False
    # Keep can race with an unkeep/reject click while completion is linking its
    # file.  Flush our local completion/status work, then let one SQL statement
    # consult the CURRENT candidate row and update only a still-kept parent.
    # Reading ``img.status`` here would use a stale SQLAlchemy object and could
    # evict the parent after the user already changed their mind.
    from sqlalchemy import exists, update
    from sqlalchemy.orm import aliased

    db.session.flush()
    candidate = aliased(FaceDatasetImage)
    candidate_is_kept = exists().where(
        candidate.id == img.id,
        candidate.dataset_id == img.dataset_id,
        candidate.parent_image_id == img.parent_image_id,
        candidate.derivation_kind == KLEIN_IMAGE_IMPROVE,
        candidate.status == 'keep',
        candidate.filename == filename,
    )
    result = db.session.execute(
        update(FaceDatasetImage)
        .where(FaceDatasetImage.id == img.parent_image_id,
               FaceDatasetImage.dataset_id == img.dataset_id,
               FaceDatasetImage.status == 'keep',
               candidate_is_kept)
        .values(status='pending')
        .execution_options(synchronize_session=False))
    return bool(result.rowcount)


def _rekeep_pending_parent_for_reimprove(img):
    """CAS the source back to Keep while a currently kept result is re-run."""
    if (img.derivation_kind != KLEIN_IMAGE_IMPROVE
            or not img.parent_image_id
            or img.parent_image_id == img.id):
        return False
    from sqlalchemy import exists, update
    from sqlalchemy.orm import aliased

    candidate = aliased(FaceDatasetImage)
    candidate_is_kept = exists().where(
        candidate.id == img.id,
        candidate.dataset_id == img.dataset_id,
        candidate.parent_image_id == img.parent_image_id,
        candidate.derivation_kind == KLEIN_IMAGE_IMPROVE,
        candidate.status == 'keep',
    )
    result = db.session.execute(
        update(FaceDatasetImage)
        .where(FaceDatasetImage.id == img.parent_image_id,
               FaceDatasetImage.dataset_id == img.dataset_id,
               FaceDatasetImage.status == 'pending',
               candidate_is_kept)
        .values(status='keep')
        .execution_options(synchronize_session=False))
    return bool(result.rowcount)


def _nullable_equals(column, value):
    return column.is_(None) if value is None else column == value


def _matches_reimprove_state(row, img, state):
    """SQL predicates for the snapshot that the re-run is allowed to replace."""
    return (
        row.id == img.id,
        row.dataset_id == img.dataset_id,
        row.parent_image_id == img.parent_image_id,
        row.derivation_kind == KLEIN_IMAGE_IMPROVE,
        row.status == state['status'],
        _nullable_equals(row.filename, state['filename']),
        _nullable_equals(row.job_id, state['job_id']),
    )


def _transition_reimprove_candidate(img, old_state, parent, label, prompt, job_id,
                                    expected_transition_caption):
    """CAS one improvement into its in-flight replacement state.

    The job has already been queued, but a status click can land while enqueue is
    in progress.  Do not overwrite that newer decision; the caller cancels the
    unlinked job when this snapshot no longer matches.
    """
    from sqlalchemy import case, update

    values = {
        'filename': None,
        'status': 'pending',
        'job_id': job_id,
        'variation_label': label,
        'variation_prompt': prompt[:500],
        'framing': parent.framing,
        'fail_reason': None,
        'fail_kind': None,
        'watermark_state': None,
        'watermark_bbox': None,
        'watermark_regions': None,
    }
    if not old_state['caption']:
        # A blank caption inherits the parent on a normal re-run, but an editor
        # can save text while enqueue_klein_edit is waiting.  Fill only if it is
        # STILL blank in the database; otherwise preserve that newer work.
        still_blank = ((FaceDatasetImage.caption.is_(None))
                       | (FaceDatasetImage.caption == ''))
        values['caption'] = case(
            (still_blank, expected_transition_caption), else_=FaceDatasetImage.caption)
    result = db.session.execute(
        update(FaceDatasetImage)
        .where(*_matches_reimprove_state(FaceDatasetImage, img, old_state))
        .values(**values)
        .execution_options(synchronize_session=False))
    if result.rowcount:
        db.session.expire(img)
    return bool(result.rowcount)


def _restore_reimprove_candidate_after_trash_failure(
        img, old_state, job_id, expected_transition_caption):
    """Restore only the exact transient state written by this re-run."""
    from sqlalchemy import case, update

    transient = dict(old_state, status='pending', filename=None, job_id=job_id)
    # Restore the exact old caption only while it is still what this transition
    # would have written.  A caption changed during Trash I/O wins instead.
    restore_values = {field: value for field, value in old_state.items()
                      if field != 'caption'}
    restore_values['caption'] = case(
        (_nullable_equals(FaceDatasetImage.caption, expected_transition_caption),
         old_state['caption']),
        else_=FaceDatasetImage.caption)
    result = db.session.execute(
        update(FaceDatasetImage)
        .where(*_matches_reimprove_state(FaceDatasetImage, img, transient))
        .values(**restore_values)
        .execution_options(synchronize_session=False))
    if result.rowcount:
        db.session.expire(img)
    return bool(result.rowcount)


def _undo_rekeep_parent_after_reimprove_trash_failure(img, old_state):
    """Undo only a fallback whose candidate was successfully restored."""
    if (img.derivation_kind != KLEIN_IMAGE_IMPROVE
            or not img.parent_image_id
            or img.parent_image_id == img.id):
        return False
    from sqlalchemy import exists, update
    from sqlalchemy.orm import aliased

    candidate = aliased(FaceDatasetImage)
    candidate_is_restored = exists().where(
        *_matches_reimprove_state(candidate, img, old_state))
    result = db.session.execute(
        update(FaceDatasetImage)
        .where(FaceDatasetImage.id == img.parent_image_id,
               FaceDatasetImage.dataset_id == img.dataset_id,
               FaceDatasetImage.status == 'keep',
               candidate_is_restored)
        .values(status='pending')
        .execution_options(synchronize_session=False))
    return bool(result.rowcount)


@_serialize_dataset_image_ingest
def set_image_status(user_id, image_id, status):
    if status not in _VALID_STATUS:
        raise ValueError('invalid status')
    img = db.session.get(FaceDatasetImage, image_id)
    if not img:
        return False
    ds = db.session.get(FaceDataset, img.dataset_id)
    if not ds or str(ds.user_id) != str(user_id):
        return False
    if img.derivation_kind in _SMALL_IMAGE_DERIVATIONS:
        raise ValueError('resolve small-image rescue pairs with the dedicated review action')
    if status == 'reject':
        _clear_watermark_metadata(img)
    img.status = status
    if status == 'keep':
        _unkeep_parent_for_kept_improvement(img)
    db.session.commit()
    return True


def clear_unseen_flag(user_id, image_id):
    """Mark a tile as seen (opened) — the one-way transition off `unseen`,
    called when the tile's lightbox opens. A no-op (still returns True) when
    the flag is already clear, so opening an already-seen tile costs nothing."""
    img = db.session.get(FaceDatasetImage, image_id)
    if not img:
        return False
    ds = db.session.get(FaceDataset, img.dataset_id)
    if not ds or str(ds.user_id) != str(user_id):
        return False
    if img.unseen:
        img.unseen = False
        db.session.commit()
    return True


def set_image_locked(user_id, image_id, locked):
    """Toggle the delete-guard flag. Locking/unlocking never touches content —
    reject/regenerate/face-swap/crop/mirror stay available either way."""
    img = _owned_image(user_id, image_id)
    if not img:
        return False
    if bool(img.is_locked) != bool(locked):
        img.is_locked = bool(locked)
        db.session.commit()
    return True


def _owned_image(user_id, image_id):
    img = db.session.get(FaceDatasetImage, image_id)
    if not img:
        return None
    ds = db.session.get(FaceDataset, img.dataset_id)
    return img if ds and str(ds.user_id) == str(user_id) else None


def resolve_small_image_rescue(user_id, dataset_id, candidate_id, choice):
    """Resolve an original/Klein rescue pair in one DB commit.

    The pair is deliberately not mutable through the generic single/batch status
    paths: exactly one of these three decisions is the source of truth.
    Returns None when the owned dataset/candidate does not exist.
    """
    if choice not in ('original', 'klein', 'reject'):
        raise ValueError('choice must be original, klein, or reject')

    def _load_pair():
        ds = get_dataset(user_id, dataset_id)
        if not ds:
            return None, None
        candidate = (FaceDatasetImage.query
                     .filter_by(id=candidate_id, dataset_id=dataset_id).first())
        if not candidate:
            return None, None
        if candidate.derivation_kind != KLEIN_SMALL_IMAGE or not candidate.parent_image_id:
            raise ValueError('image is not a Klein small-image rescue candidate')
        source = (FaceDatasetImage.query
                  .filter_by(id=candidate.parent_image_id, dataset_id=dataset_id,
                             derivation_kind=SMALL_IMAGE_SOURCE).first())
        if not source:
            raise ValueError('small-image rescue source is missing or invalid')
        return source, candidate

    def _resolved_as(source, candidate):
        states = (source.status, candidate.status)
        return {('keep', 'reject'): 'original',
                ('reject', 'keep'): 'klein',
                ('reject', 'reject'): 'reject'}.get(states)

    def _payload(source, candidate):
        return {'choice': choice,
                'source': {'id': source.id, 'status': source.status},
                'candidate': {'id': candidate.id, 'status': candidate.status}}

    # Cancel before touching pair statuses: queue_manager uses the same scoped DB
    # session and commits its job row, so calling it after mutations would split
    # the supposedly atomic source/candidate decision.
    source, candidate = _load_pair()
    if source is None:
        return None
    already = _resolved_as(source, candidate)
    if already:
        result = _payload(source, candidate)
        db.session.rollback()
        if already != choice:
            raise RuntimeError(f'small-image rescue was already resolved as {already}')
        return result  # idempotent retry
    job_id = (candidate.job_id if choice != 'klein' and not candidate.filename else None)
    db.session.rollback()  # close the preflight read transaction before queue cancellation
    if job_id:
        try:
            from ..job_queue import queue_manager
            queue_manager.cancel_job(job_id, str(user_id), 'image')
        except Exception:
            logger.exception('small-image rescue: failed to cancel job %s', job_id)
    db.session.rollback()

    # SQLite's BEGIN IMMEDIATE serializes competing resolutions before either one
    # reads the transition state. The second caller therefore observes the first
    # committed choice and follows the idempotent/conflict branch.
    from sqlalchemy import text
    try:
        db.session.execute(text('BEGIN IMMEDIATE'))
        source, candidate = _load_pair()
        if source is None:
            db.session.rollback()
            return None
        already = _resolved_as(source, candidate)
        if already:
            if already != choice:
                raise RuntimeError(f'small-image rescue was already resolved as {already}')
            result = _payload(source, candidate)
            db.session.rollback()
            return result
        if source.status != 'pending' or candidate.status not in ('pending', 'failed'):
            raise RuntimeError('small-image rescue is not in a resolvable state')
        if choice == 'klein':
            if candidate.status == 'failed' or not candidate.filename:
                raise ValueError('Klein rescue result is not ready')
            source.status, candidate.status = 'reject', 'keep'
            _clear_watermark_metadata(source)
        elif choice == 'original':
            source.status, candidate.status = 'keep', 'reject'
            _clear_watermark_metadata(candidate)
        else:
            source.status = candidate.status = 'reject'
            _clear_watermark_metadata(source)
            _clear_watermark_metadata(candidate)
        db.session.commit()
        result = _payload(source, candidate)
    except Exception:
        db.session.rollback()
        raise
    _sync_generate_activity(dataset_id)
    return result


_UNSET = object()


def set_image_caption(user_id, image_id, caption, short=_UNSET):
    """Save one image's long caption; optionally its short variant. `short` defaults to a
    sentinel so a caller that only edits the long caption (the inline grid textarea) never
    wipes an existing short — only the expanded editor passes `short` to touch it."""
    img = _owned_image(user_id, image_id)
    if not img:
        return False
    img.caption = _cap_caption(caption) or None
    if short is not _UNSET:
        img.caption_short = _cap_caption(short) or None
    db.session.commit()
    return True


def _crop_resize_file(path, x, y, w, h, size=1024, dst=None):
    """Crop the file at `path` to (x,y,w,h) and normalise the crop's LONG side DOWN
    to at most `size`, PRESERVING the box's aspect ratio: a 2000x1500 box yields
    1024x768, a 2:3 box yields 683x1024 — no padding, no distortion (ai-toolkit
    buckets handle non-square training images). Writes to `dst` (default: overwrite
    `path`). Passing a distinct `dst` lets the reference crop read the untouched
    full-frame ORIGINAL and write the derived crop — so a re-crop can widen back
    out instead of only tightening the previous crop.

    A box SMALLER than `size` is left at its own size. The resize used to be
    unconditional, so a 240x180 crop was blown up to 1024x768 — and that upscale
    carried essentially nothing: shrinking the result back to 240 recovers the
    original at 48.96 dB (max channel error 10), for 2.3x the bytes. Since the
    encoder went lossless that is close to a megabyte of interpolated pixels per
    small crop, and it hands the trainer a tile whose apparent resolution is a
    fiction. Cropping in cannot create detail; it should not pretend to.

    Returns (ok, upscale_ratio), or (False, None) on failure. The ratio is
    unchanged in value and meaning — `size / long_side_of_box`, i.e. how far the
    box sits under the training resolution (>1 = under it) — because it is a
    STORED column (`FaceDatasetImage.upscale_ratio`) feeding the composition
    warning, and capping it along with the pixels would silently retire that
    warning. Only the pixels stopped pretending; the measurement did not move.

    ENCODING: the source format is preserved and written under
    `image_encoding.LOSSLESS`. This used to be an unconditional lossy WEBP q92, so
    cropping a PNG degraded it AND left PNG-named files holding WEBP bytes.

    Crop is the one operation for which lossless was a real trade rather than an
    obvious win — it RESAMPLES, so it destroys information whatever the encoder does,
    and lossless costs 4.59x the bytes. It was chosen on measurement, not principle:
    lossy WEBP has an error floor (chroma subsampled to 4:2:0 at every quality, so
    q100 still leaves max channel error 16 for 1.74x the size), and that error
    COMPOUNDS — five successive crops land at PSNR 45 dB whether they are q92 or
    q100, while lossless stays byte-identical to the first crop. See the measurement
    table in `image_encoding`'s module docstring.

    ⚠️ What this does NOT claim: only the ENCODING is lossless. A box longer than
    `size` is still resampled down, which destroys information whatever the encoder
    does. A box at or under `size` is now a pure cut, so it IS lossless end to end —
    as is the watermark crop (`_apply_watermark_crop`), which never resizes."""
    if not os.path.exists(path):
        return False, None
    with Image.open(path) as opened:
        # The DESTINATION name decides (it may differ from the source: the reference
        # editor reads the kept full frame and writes the derived crop), so the file
        # written always contains what its extension promises.
        fmt = image_encoding.format_for_path(dst or path, opened)
        opened.load()
        icc = _valid_icc_profile(opened.info.get('icc_profile'))
        # Browser crop coordinates describe the EXIF-oriented visual frame, not
        # the raw camera raster. Bake the orientation before interpreting x/y.
        oriented = ImageOps.exif_transpose(opened)
        # Narrow the mode BEFORE resampling: Pillow silently drops to nearest-neighbour
        # on paletted images, which would undo the point of removing the lossy encoder.
        src = oriented.convert(image_encoding.resample_mode(oriented))
    box = (max(0, int(x)), max(0, int(y)), min(src.width, int(x + w)), min(src.height, int(y + h)))
    if box[2] <= box[0] or box[3] <= box[1]:
        return False, None
    bw, bh = box[2] - box[0], box[3] - box[1]
    # Normalise DOWN only: `long` is what we actually render, `size` stays the
    # reference the reported ratio is measured against (see the docstring).
    long = min(size, max(bw, bh))
    if bw >= bh:
        out_w, out_h = long, max(1, round(long * bh / bw))
    else:
        out_w, out_h = max(1, round(long * bw / bh)), long
    scale = size / max(bw, bh)
    out = io.BytesIO()
    image_encoding.save_edit(src.crop(box).resize((out_w, out_h), Image.LANCZOS),
                             out, fmt, image_encoding.LOSSLESS, icc_profile=icc)
    with open(dst or path, 'wb') as fh:
        fh.write(out.getvalue())
    return True, scale


@_serialize_dataset_image_ingest
def crop_image(user_id, image_id, x, y, w, h):
    """Crop a dataset image to (x,y,w,h), long side capped at 1024, no pad (a box
    smaller than that keeps its own size). Returns bool."""
    img = _owned_image(user_id, image_id)
    if not img or not img.filename:
        return False
    ok, scale = _crop_resize_file(_img_path(img), x, y, w, h)
    if ok:
        _clear_watermark_metadata(img)
        img.upscale_ratio = scale
        _invalidate_image_content_analysis(img)
        db.session.commit()
    return ok


def _valid_icc_profile(raw):
    """Return an ICC payload only when LittleCMS can parse it.

    Pillow will otherwise copy arbitrary bytes into the rewritten image, and some
    encoders fail late on malformed profiles.  ICC is the one embedded metadata
    item worth retaining here (colour rendering); EXIF orientation is deliberately
    baked into the pixels by ``ImageOps.exif_transpose`` and must not be reattached.
    """
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        return None
    try:
        from PIL import ImageCms
        ImageCms.getOpenProfile(io.BytesIO(raw))
    except Exception:
        return None
    return bytes(raw)


def transformed_image_bytes(path, transform, *, max_source_bytes: int | None = None):
    """Apply ``transform`` (a PIL image -> PIL image callable) fully in memory,
    without touching ``path``, and return the re-encoded bytes. ``max_source_bytes``
    is an optional exact-read fence for a live external folder (Bank); ordinary
    Dataset edits retain their historical path-backed behaviour.

    THE shared encoder of every in-place pixel edit that REORDERS pixels without
    rebuilding any (mirror, rotation). Dataset rows may point at JPEG, PNG, WebP
    or BMP files (and restored legacy rows can still carry a misleading extension).
    Preserve the format Pillow actually detects and encode
    it under `image_encoding.LOSSLESS` — the policy this operation REQUIRES, passed
    explicitly so that tuning another operation's encoder can never silently
    degrade this one.

    ⚠️ Only JPEG loses anything here, and it loses it on EVERY edit — Pillow has
    no DCT-domain (jpegtran-style) path, so a 90° turn of a JPEG is a re-encode,
    not a lossless block transform. PNG, WebP and BMP preserve their decoded RGB
    pixels; BMP has no useful alpha path, so edits intentionally flatten it to RGB.

    The format is read from the CONTENT, not the file name: a mirror/rotation has
    no business converting a legacy extension mismatch it did not create. Crop,
    which rewrites the file wholesale and may write to a DIFFERENT destination,
    uses `image_encoding.format_for_path` instead.
    """
    source = path
    if max_source_bytes is not None:
        try:
            max_source_bytes = int(max_source_bytes)
        except (TypeError, ValueError) as exc:
            raise ValueError('invalid image source byte limit') from exc
        if max_source_bytes < 1:
            raise ValueError('invalid image source byte limit')
        try:
            with open(path, 'rb') as raw_source:
                raw = raw_source.read(max_source_bytes + 1)
        except (OSError, MemoryError) as exc:
            raise ValueError('invalid image file') from exc
        if len(raw) > max_source_bytes:
            raise ValueError(
                f'image source is too large (max {max_source_bytes // (1024 * 1024)} MiB)')
        # Decode the exact bounded bytes just read, not a path that a live Bank
        # folder could replace between a header check and the actual edit.
        source = io.BytesIO(raw)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(source) as src:
                image_encoding.validate_input_header_dimensions(src, label='image edit')
                fmt = (src.format or '').upper()
                if fmt not in image_encoding.EDITABLE_FORMATS:
                    raise ValueError(f'unsupported image format: {fmt or "unknown"}')
                if getattr(src, 'n_frames', 1) != 1:
                    raise ValueError('animated images are not supported')
                src.load()
                icc = _valid_icc_profile(src.info.get('icc_profile'))
                # EXIF orientation is baked into the pixels FIRST, so the edit the
                # user asked for is applied to the image they were shown — and the
                # tag is dropped (never reattached), so nothing rotates it twice.
                oriented = ImageOps.exif_transpose(src)
                edited = transform(oriented)

                out = io.BytesIO()
                edited, save_kwargs = image_encoding.save_params(
                    edited, fmt, image_encoding.LOSSLESS, icc_profile=icc)
                edited.save(out, fmt, **save_kwargs)
                payload = out.getvalue()
                # Read AFTER save_params: it may have converted the mode, and the
                # self-check below compares the decoded size against this.
                expected_size = edited.size
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, MemoryError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise ValueError('invalid image file') from e

    # Decode the exact encoded result before it is allowed near the live path.
    try:
        with Image.open(io.BytesIO(payload)) as check:
            check.load()
            if (check.format or '').upper() != fmt or check.size != expected_size:
                raise OSError('encoded edit validation failed')
    except (UnidentifiedImageError, OSError, SyntaxError, MemoryError) as e:
        raise ValueError('could not encode the edited image') from e
    return payload


def _mirrored_image_bytes(path):
    """Horizontal mirror — kept as a named wrapper for the mirror lane."""
    return transformed_image_bytes(path, ImageOps.mirror)


#: The only turns we offer, in degrees CLOCKWISE. Anything else is refused: a
#: free-angle rotation would need padding or cropping (it invents or drops
#: pixels), which is a different feature from "this photo is on its side".
ROTATION_DEGREES = (90, 180, 270)

#: Clockwise degrees -> Pillow transpose op. Pillow's ROTATE_* names are
#: COUNTER-clockwise, so 90 clockwise is ROTATE_270. These are exact pixel
#: permutations: no resampling, no interpolation, no pixel invented.
_ROTATE_OPS = {
    90: Image.Transpose.ROTATE_270,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,
}


def normalize_rotation(degrees):
    """Fold any int to 0/90/180/270 clockwise, or raise ValueError.

    Accepts negatives (-90 == 270) and multiples of 360 so callers can pass a
    delta without doing the modulo themselves.
    """
    try:
        value = int(degrees)
    except (TypeError, ValueError):
        raise ValueError('rotation must be 90, 180 or 270 degrees') from None
    value %= 360
    if value % 90:
        raise ValueError('rotation must be 90, 180 or 270 degrees')
    return value


def rotate_transform(degrees):
    """The PIL transform for a normalised clockwise angle (0 => identity)."""
    op = _ROTATE_OPS.get(normalize_rotation(degrees))
    if op is None:
        return lambda image: image
    return lambda image: image.transpose(op)


def _rotated_image_bytes(path, degrees):
    """Rotate ``path`` clockwise by ``degrees`` in memory, format preserved."""
    if normalize_rotation(degrees) == 0:
        raise ValueError('rotation must be 90, 180 or 270 degrees')
    return transformed_image_bytes(path, rotate_transform(degrees))


@_serialize_dataset_image_ingest
def _edit_image_in_place(user_id, image_id, make_payload, *, tag):
    """Promote a re-encoded copy of one owned dataset image over its own file.

    ``make_payload(path) -> bytes`` prepares the new bytes; this owns everything
    that makes the swap safe — the per-image lock, the "did something else touch
    the file while we worked" check, the atomic replace and the watermark
    metadata rollback. Mirror and rotation share it verbatim so a fix to one is
    a fix to both.
    """
    lock = _IMAGE_PIXEL_EDIT_LOCKS[
        hash((str(user_id), image_id)) % len(_IMAGE_PIXEL_EDIT_LOCKS)]
    with lock:
        img = _owned_image(user_id, image_id)
        if not img:
            return None
        if not img.filename:
            raise ValueError('image file required')
        path = _img_path(img)
        if not os.path.isfile(path):
            raise RuntimeError('image file missing')

        try:
            before = os.stat(path)
            payload = make_payload(path)
        except ValueError:
            raise
        except OSError as e:
            raise RuntimeError('could not read image file') from e

        tmp_path = None
        try:
            try:
                fd, tmp_path = tempfile.mkstemp(
                    prefix=f'.{os.path.basename(path)}.{tag}-', suffix='.tmp',
                    dir=os.path.dirname(path),
                )
                with os.fdopen(fd, 'wb') as fh:
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                # Validate the on-disk temp as well as the in-memory encoding.
                with Image.open(tmp_path) as check:
                    check.verify()
            except (UnidentifiedImageError, OSError, SyntaxError) as e:
                raise RuntimeError(f'could not prepare the {tag} result') from e

            # Do not overwrite a crop/clean that raced this preparation outside
            # the edit lock.  (Mirror and rotation share the SAME stripe, so two
            # pixel edits of one image can never read the same source twice.)
            try:
                current = os.stat(path)
            except OSError as e:
                raise RuntimeError('image file missing') from e
            if (current.st_mtime_ns, current.st_size) != (before.st_mtime_ns, before.st_size):
                raise RuntimeError('image changed while editing; retry')

            watermark_snapshot = (
                img.watermark_state, img.watermark_bbox, img.watermark_regions)
            watermark_changed = any(value is not None for value in watermark_snapshot)
            if watermark_changed:
                _clear_watermark_metadata(img)
                try:
                    db.session.commit()
                except Exception:
                    db.session.rollback()
                    raise

            try:
                # Same-directory replacement is atomic; the original remains live
                # until this single operation succeeds.
                os.replace(tmp_path, path)
                tmp_path = None
            except OSError as e:
                if watermark_changed:
                    (img.watermark_state, img.watermark_bbox,
                     img.watermark_regions) = watermark_snapshot
                    try:
                        db.session.commit()
                    except Exception:
                        db.session.rollback()
                        logger.exception(
                            'failed to restore watermark metadata after %s promotion failure', tag)
                raise RuntimeError('could not update image file') from e


            _invalidate_image_content_analysis(img)
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()
                raise
            return {
                'image_id': img.id,
                # A request token is intentionally independent of filename and
                # HTTP Last-Modified granularity; the frontend appends it to ?v=.
                'cache_bust': time.time_ns(),
            }
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    logger.warning('could not remove %s temp file %s', tag, tmp_path)


def mirror_image(user_id, image_id):
    """Permanently mirror one owned dataset image horizontally.

    Returns ``None`` for an unknown/foreign row, otherwise a cache-bust payload.
    The filename and all semantic/provenance metadata remain stable.  Only
    watermark metadata is cleared because its pixel coordinates are no longer
    valid after a horizontal flip.
    """
    return _edit_image_in_place(
        user_id, image_id, _mirrored_image_bytes, tag='mirror')


def rotate_image(user_id, image_id, degrees):
    """Permanently rotate one owned dataset image by 90/180/270° CLOCKWISE.

    Same contract as :func:`mirror_image` — ``None`` for an unknown/foreign row,
    otherwise a cache-bust payload; the filename and every semantic/provenance
    field stay put, and only the watermark metadata is cleared (its normalised
    bbox is expressed in the OLD frame and a quarter turn invalidates it).

    A quarter turn is an exact pixel permutation, so nothing is resampled; what
    it costs is the re-encode of the container (see ``transformed_image_bytes``),
    which is pixel-exact for PNG/WEBP and lossy for JPEG.
    """
    turn = normalize_rotation(degrees)
    if turn == 0:
        raise ValueError('rotation must be 90, 180 or 270 degrees')
    return _edit_image_in_place(
        user_id, image_id, lambda path: _rotated_image_bytes(path, turn),
        tag='rotate')


@_serialize_dataset_image_ingest
def delete_image(user_id, image_id):
    """Delete a dataset image row and move its file to the app trash.

    If the image is still a pending generation, its queue job is cancelled
    first. Returns bool.
    """
    img = _owned_image(user_id, image_id)
    if not img:
        return False
    if img.is_locked:
        raise RuntimeError('This image is locked — unlock it before deleting.')
    if img.derivation_kind in _SMALL_IMAGE_DERIVATIONS:
        raise ValueError('resolve the small-image rescue pair before cleanup')
    original_path = (os.path.join(_dataset_path(img.dataset_id), img.filename)
                     if img.filename else None)
    trashed_path = None
    try:
        if img.status == 'pending' and not img.filename and img.job_id:
            from ..job_queue import queue_manager
            if not queue_manager.cancel_job(
                    img.job_id, str(user_id), 'image', commit=False):
                raise RuntimeError(
                    'This generation still has unconfirmed ComfyUI work; cancel it safely before deleting.')
        if original_path and os.path.exists(original_path):
            trashed_path = trash.send_to_trash(
                original_path, context=f'dataset-{img.dataset_id}-image-{img.id}')
        db.session.delete(img)
        db.session.commit()
    except trash.TrashLockError as e:
        db.session.rollback()
        _restore_from_trash(trashed_path, original_path)
        raise RuntimeError(_TRASH_LOCK_MESSAGE) from e
    except Exception:
        db.session.rollback()
        _restore_from_trash(trashed_path, original_path)
        raise
    return True


def _guard_no_active_training(dataset_id, *, action='deleting'):
    """Raise RuntimeError (-> 409) when a LOCAL or CLOUD training run is mid-flight
    on this dataset, so delete_dataset refuses instead of silently orphaning the
    run. Lazy imports dodge the cloud_training/lora_training <-> face_dataset_service
    import cycle; a module absent in a phase-1 install just means 'no such run'.

    TERMINAL runs (done/stopped/error/error_pod_kept) don't block: their provenance
    rows stay behind with an orphaned dataset_id (the existing no-FK pattern), which
    preserves run history and importable-checkpoint records after the dataset is gone."""
    try:
        from . import cloud_training as ct
    except ImportError:
        ct = None
    if ct is not None and ct.active_runs_for(dataset_id):
        raise RuntimeError(_ACTIVE_RUN_TEMPLATE.format(action=action))
    try:
        from . import lora_training as lt
    except ImportError:
        lt = None
    if lt is not None and lt.is_local_run_active(dataset_id):
        raise RuntimeError(_ACTIVE_RUN_TEMPLATE.format(action=action))


@_serialize_dataset_ingest
def delete_dataset(user_id, dataset_id):
    """Delete an owned dataset and move its complete folder to app trash.

    Refuses (RuntimeError -> 409) while a local or cloud training run is active on
    the dataset — deleting under a running run orphans its record and abandons a
    paid vast pod. Child image and Studio rows are explicitly removed for legacy
    databases whose foreign key had neither enforcement nor ``ON DELETE CASCADE``;
    terminal training-run records are intentionally left behind (orphaned
    dataset_id) to keep run history. Cancels any in-flight generations first.
    Returns False if not owned.
    """
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return False
    _guard_no_active_training(dataset_id)
    # Capture le trigger AVANT de supprimer la ligne : sert à purger les artefacts
    # d'entraînement orphelins (LoRA déployés dans ComfyUI, run/export ai-toolkit,
    # job config) qui survivaient à la suppression du dataset et restaient
    # sélectionnables en génération. Import paresseux = pas d'import circulaire ;
    # lora_training n'existe pas encore en phase 1 -> purge silencieusement sautée.
    lt = None
    purge_user, purge_trigger = ds.user_id, None
    try:
        from . import lora_training as lt
        purge_trigger = lt._safe_trigger(ds)
    except ImportError:
        pass
    imgs = FaceDatasetImage.query.filter_by(dataset_id=dataset_id).all()
    locked_count = sum(1 for img in imgs if img.is_locked)
    if locked_count:
        raise RuntimeError(
            f'{locked_count} locked image(s) — unlock them or delete them '
            'individually before deleting the whole dataset.')
    studio_rows = LoraTestImage.query.filter_by(dataset_id=dataset_id).all()
    # ◉ LoRA Canvas card positions. The model declares a relationship() to
    # face_dataset so the unit of work orders the DELETEs, but a mapper-level
    # dependency only covers rows that are IN the session — so they are loaded
    # and deleted explicitly here like every other child, and flushed before the
    # parent below. A dataset must never fail to delete over a display
    # preference: that exact bug already answered HTTP 500 once in this project.
    canvas_rows = CanvasNodePosition.query.filter_by(dataset_id=dataset_id).all()
    # 🖼 Pinned-image nodes: same story, same trap. They reference
    # lora_test_image rows that are being deleted in this very transaction.
    canvas_imgs = CanvasImageNode.query.filter_by(dataset_id=dataset_id).all()
    dataset_path = _dataset_path(dataset_id)
    trashed_path = None
    try:
        # Keep Studio queue cancellation atomic with deleting its owning rows.
        # Exact job_id + owned dataset scope prevents cross-dataset cancellation.
        from ..job_queue import queue_manager
        for img in imgs:
            if img.status == 'pending' and not img.filename and img.job_id:
                if not queue_manager.cancel_job(
                        img.job_id, str(user_id), 'image', commit=False):
                    raise RuntimeError(
                        'A dataset generation still has unconfirmed ComfyUI work; cancel it safely first.')
        for cell in studio_rows:
            if (cell.job_id
                    and cell.status not in ('done', 'failed', 'cancelled')):
                if not queue_manager.cancel_job(
                        cell.job_id, str(user_id), 'image', commit=False):
                    raise RuntimeError(
                        'A Test Studio cell still has unconfirmed ComfyUI work; cancel it safely first.')
        if os.path.exists(dataset_path):
            trashed_path = trash.send_to_trash(
                dataset_path, context=f'dataset-{dataset_id}')
        for img in imgs:
            db.session.delete(img)
        # Explicit for old databases whose FK definition cannot be altered by
        # db.create_all(). New databases also have ON DELETE CASCADE as a guard.
        for cell in studio_rows:
            db.session.delete(cell)
        for pos in canvas_rows:
            db.session.delete(pos)
        for pin in canvas_imgs:
            db.session.delete(pin)
        # Force the child DELETEs to reach the DB BEFORE the parent's. The child
        # models declare only a table-level ForeignKey (no relationship()), so the
        # unit of work has no ordering dependency between them and would otherwise
        # emit `DELETE FROM face_dataset` first. On a legacy DB whose FK lacks
        # ON DELETE CASCADE that parent-first order raises IntegrityError (the
        # children still physically exist); on a cascade DB it works but leaves a
        # SAWarning. Flushing the children here makes the order deterministic on
        # every DB vintage — the belt no longer depends on the DB doing the cascade.
        db.session.flush()
        db.session.delete(ds)
        db.session.commit()
    except trash.TrashLockError as e:
        db.session.rollback()
        _restore_from_trash(trashed_path, dataset_path)
        raise RuntimeError(_TRASH_LOCK_MESSAGE) from e
    except Exception:
        db.session.rollback()
        _restore_from_trash(trashed_path, dataset_path)
        raise
    # Purge les artefacts d'entraînement (LoRA ComfyUI + ai-toolkit + config). Best
    # effort : un échec ici ne doit pas faire échouer la suppression du dataset.
    if lt is not None:
        try:
            removed = lt.purge_training_artifacts(purge_user, purge_trigger)
            if removed:
                logger.info('delete_dataset %s : %d artefact(s) LoRA purgé(s)', dataset_id, len(removed))
        except Exception as e:
            logger.warning('delete_dataset %s : purge artefacts LoRA échouée : %s', dataset_id, e)
    return True


def _finish_cancelled_generation_row(img):
    """Remove one safely terminal generation while preserving rescue originals."""
    if img.derivation_kind == KLEIN_SMALL_IMAGE:
        img.status = 'failed'
        img.fail_reason = 'Klein small-image rescue was cancelled.'
    else:
        db.session.delete(img)


def cancel_pending(user_id, dataset_id):
    """Cancel all in-flight (pending) generations of a dataset.

    A local queue row is removed only after ``cancel_job`` proves that its exact
    ComfyUI prompt is gone.  If that proof cannot be obtained yet, keep the image
    row and its ``job_id``: it is the only UI handle from which the user can press
    Stop again once ComfyUI answers.  Dropping that row used to leave the durable
    global recovery barrier orphaned, making every GPU action report ``GPU busy``
    with no recoverable card left.

    Returns explicit recovery counts. ``retry_pending`` means LDS can retry the
    exact known prompt; ``restart_required`` means ComfyUI must be restarted and
    that restart explicitly confirmed before LDS may clear an unknown submission.

    ⏹ Stop generation also stops the server-side ✨ improve BATCH: cancelling the
    rows alone used to be pointless, because whatever was feeding the queue simply
    queued the next wave. The flag is armed FIRST so the worker can't slip another
    image in between the arming and the row deletion."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return {'cancelled': 0, 'recovery_pending': 0,
                'retry_pending': 0, 'restart_required': 0,
                'recovery_error': 0}
    dataset_activity.request_cancel(dataset_id, dataset_activity.IMPROVE_KINDS)
    # Only in-flight generations (pending AND no result file yet) - leave
    # completed-but-uncurated images alone.
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='pending')
            .filter(FaceDatasetImage.filename.is_(None)).all())
    n = 0
    retry_pending = 0
    restart_required = 0
    recovery_error = 0
    for img in rows:
        if img.job_id:  # Klein rows only - API rows never carry a job_id
            try:
                from ..job_queue import queue_manager
                outcome = queue_manager.cancel_job_outcome(
                    img.job_id, str(user_id), 'image')
            except Exception:
                logging.getLogger(__name__).exception(
                    'could not safely cancel generation job %s', img.job_id)
                outcome = 'retry'
            if outcome == 'restart_required':
                restart_required += 1
                continue
            if outcome == 'barrier_corrupt':
                recovery_error += 1
                continue
            if outcome == 'retry':
                retry_pending += 1
                continue
            # cancelled / terminal / missing are all safe: cancel_job_outcome
            # proved that this exact job owns no durable recovery barrier.
        _finish_cancelled_generation_row(img)
        n += 1
    db.session.commit()
    # Stop deleted the in-flight rows: clear the Klein 'generate' indicator now
    # (its completion callbacks won't fire for cancelled jobs). An API batch's own
    # begin/end entry is untouched — its worker unwinds and end()s on its own.
    _sync_generate_activity(dataset_id)
    return {
        'cancelled': n,
        'retry_pending': retry_pending,
        'restart_required': restart_required,
        'recovery_error': recovery_error,
        'recovery_pending': retry_pending + restart_required + recovery_error,
    }


def confirm_unknown_generation_restart(user_id, dataset_id, *,
                                       restart_confirmed=False) -> int:
    """Clear this dataset's unknown-submit barrier after a human-confirmed restart.

    The reachability check belongs to the route. This service owns the atomic
    identity decision: only pending cards whose exact ``job_id`` matches the
    durable unknown-submit barrier are finalized.
    """
    if not restart_confirmed:
        raise ValueError('Confirm that ComfyUI was restarted before recovery.')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    from ..job_queue import queue_manager
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='pending')
            .filter(FaceDatasetImage.filename.is_(None))
            .filter(FaceDatasetImage.job_id.isnot(None)).all())
    n = 0
    for img in rows:
        if not queue_manager.confirm_unknown_comfyui_restart(
                img.job_id, str(user_id), restart_confirmed=True):
            continue
        _finish_cancelled_generation_row(img)
        n += 1
    db.session.commit()
    _sync_generate_activity(dataset_id)
    return n


def purge_unused(user_id, dataset_id):
    """Permanently delete all REJECTED and FAILED images of a dataset (rows +
    files). Returns the number purged."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return 0
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id)
            .filter(FaceDatasetImage.status.in_(('reject', 'failed')))
            .filter(FaceDatasetImage.derivation_kind.notin_(_SMALL_IMAGE_DERIVATIONS)
                    | FaceDatasetImage.derivation_kind.is_(None))
            .filter(FaceDatasetImage.is_locked.isnot(True)).all())
    n = 0
    for img in rows:
        if delete_image(user_id, img.id):
            n += 1
    return n


# --- Sauvegarde / restauration complète d'un dataset ---------------------------
# ZIP portable (≠ export d'entraînement) : manifest + réglages + TOUTES les images
# avec statuts/captions/scores — pour archiver ou déplacer un dataset entre machines.
BACKUP_FORMAT = 'lds-dataset-backup'
BACKUP_VERSION = 1
_BACKUP_MAX_FILES = 600
_BACKUP_MAX_ROWS = 600
_BACKUP_MAX_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB uncompressed (zip-bomb guard)
_BACKUP_MAX_METADATA_BYTES = 4 * 1024 * 1024
# The import raster budget permits at most 16 Mi pixels.  96 MiB leaves room for
# a valid 16 MP RGBA/BMP source plus container overhead, while making a single
# archive entry incapable of filling a disk or RAM by itself.
_BACKUP_MAX_IMAGE_BYTES = 96 * 1024 * 1024
_BACKUP_NAME_RE = re.compile(r'^[\w.-]+\.(webp|jpg|jpeg|png|bmp)$', re.IGNORECASE)
_BACKUP_EXTENSION_CANONICAL = {
    '.jpg': '.jpg', '.jpeg': '.jpg', '.png': '.png', '.webp': '.webp', '.bmp': '.bmp',
}

# Champs snapshotés tels quels par ligne image (job_id/klein_model exclus : liés
# à la machine source — un backup restauré ne peut pas « regénérer »).
_BACKUP_IMG_FIELDS = ('filename', 'source', 'framing', 'variation_label', 'status',
                      'caption', 'caption_short', 'variation_prompt', 'face_score', 'face_state',
                      'upscale_ratio', 'watermark_state', 'watermark_bbox',
                      'watermark_regions', 'parent_image_id', 'derivation_kind',
                      'fail_reason', 'fail_kind', 'source_metadata',
                      'bank_analysis_snapshot')


def _backup_basename(value):
    """Return a portable image basename, or None for paths/invalid values."""
    if not isinstance(value, str) or not value:
        return None
    if '/' in value or '\\' in value or not _BACKUP_NAME_RE.fullmatch(value):
        return None
    return value


def _read_validated_backup_image(z: zipfile.ZipFile, info: zipfile.ZipInfo,
                                 basename: str) -> bytes:
    """Return one bounded, fully-decoded static backup image.

    The outer archive's central-directory total protects the aggregate, but a
    single entry still needs its own raw cap before it is inflated.  Crucially,
    validation happens while the bytes are only in memory: a malformed image
    must never be copied into the restore staging directory and later promoted.
    """
    max_bytes = _BACKUP_MAX_IMAGE_BYTES
    if info.file_size > max_bytes:
        raise ValueError(
            f'backup image {basename} is too large '
            f'(max {max_bytes // (1024 * 1024)} MiB per image)')
    try:
        with z.open(info) as source:
            # Do not trust a crafted central-directory ``file_size`` alone.
            # Reading one extra byte limits actual decompression even if that
            # metadata is inconsistent with the entry payload.
            raw = source.read(max_bytes + 1)
    except (OSError, EOFError, RuntimeError, MemoryError, zipfile.BadZipFile) as exc:
        raise ValueError(f'backup image {basename} could not be read') from exc
    if len(raw) > max_bytes:
        raise ValueError(
            f'backup image {basename} is too large '
            f'(max {max_bytes // (1024 * 1024)} MiB per image)')
    try:
        content_ext = _preserved_import_extension(raw, label=f'backup image {basename}')
    except (ValueError, MemoryError) as exc:
        raise ValueError(f'backup image {basename} is invalid: {exc}') from exc
    named_ext = _BACKUP_EXTENSION_CANONICAL.get(os.path.splitext(basename)[1].lower())
    if named_ext != content_ext:
        raise ValueError(
            f'backup image {basename} extension does not match its decoded content')
    return raw


def _backup_extra_ref_names(raw, *, limit=_UNSET):
    """Parse the stored JSON list into unique portable basenames."""
    if limit is _UNSET:
        # MAX_EXTRA_REFS now lives in reference_photos_service.py and is only
        # borrowed back into this module's namespace by the bottom-of-file
        # re-export (see that block) — it isn't defined yet at THIS point in
        # module load, so the default must be resolved here, at call time,
        # not as a def-time default expression.
        limit = MAX_EXTRA_REFS
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return []
    if not isinstance(raw, list):
        return []
    out = []
    seen = set()
    for value in raw:
        name = _backup_basename(value)
        key = name.casefold() if name else None
        if not name or key in seen:
            continue
        seen.add(key)
        out.append(name)
        if limit is not None and len(out) >= limit:
            break
    return out


def _portable_train_base_model(value):
    """Keep model ids/relative paths, never machine-local absolute paths."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    drive, _ = ntpath.splitdrive(value)
    if (not value or drive or ntpath.isabs(value) or posixpath.isabs(value)):
        return None
    return value


def write_backup_zip(user_id: int, dataset_id: int, output: BinaryIO) -> None:
    """Self-contained backup of one dataset: manifest.json (settings) +
    images.json (rows) + ref/ + images/ files. Ordinary rows without a file are
    skipped, but small-image rescue metadata rows are retained so their pair can
    never become orphaned after restore."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    dsdir = _dataset_dir(dataset_id)
    from sqlalchemy import or_
    rows = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id)
            .filter(or_(FaceDatasetImage.filename.isnot(None),
                        FaceDatasetImage.derivation_kind.in_(_SMALL_IMAGE_DERIVATIONS)))
            .all())
    primary_ref_names = []
    ref_name_keys = set()
    for raw_name in (ds.ref_filename, ds.ref_original_filename):
        name = _backup_basename(raw_name)
        key = name.casefold() if name else None
        if (not name or key in ref_name_keys
                or not os.path.isfile(os.path.join(dsdir, name))):
            continue
        ref_name_keys.add(key)
        primary_ref_names.append(name)
    portable_extras = []
    for name in _backup_extra_ref_names(ds.ref_extra_filenames, limit=None):
        key = name.casefold()
        if (key in ref_name_keys
                or not os.path.isfile(os.path.join(dsdir, name))):
            continue
        ref_name_keys.add(key)
        portable_extras.append(name)
        if len(portable_extras) >= MAX_EXTRA_REFS:
            break
    # Extra-ref ORIGINALS travel too (as plain ref/ files, not in the manifest list):
    # restore keeps basenames, so the naming convention still ties them to their
    # extra — a restored dataset can widen a crop back out just like the source one.
    extra_originals = []
    for name in portable_extras:
        orig = extra_ref_original_name(name)
        key = orig.casefold() if orig else None
        if (not orig or key in ref_name_keys
                or not os.path.isfile(os.path.join(dsdir, orig))):
            continue
        ref_name_keys.add(key)
        extra_originals.append(orig)
    ref_names = primary_ref_names + portable_extras + extra_originals
    image_file_names = {
        name.casefold(): name for img in rows
        if (name := _backup_basename(img.filename))
        and os.path.isfile(os.path.join(dsdir, name))
    }
    collisions = ref_name_keys.intersection(image_file_names)
    if collisions:
        collision = image_file_names[next(iter(collisions))]
        raise ValueError(f'ref/image filename collision in dataset: {collision}')

    manifest = {
        'format': BACKUP_FORMAT, 'version': BACKUP_VERSION,
        'name': ds.name, 'trigger_word': ds.trigger_word,
        'kind': ds.kind, 'fidelity': ds.fidelity,
        'concept_desc': ds.concept_desc, 'concept_terms': ds.concept_terms,
        'train_type': ds.train_type,
        # Optional in backup v1: old archives omit it and restore as LoRA.
        'training_mode': (ds.training_mode
                          if ds.training_mode in ('lora', 'full_transformer')
                          else 'lora'),
        'train_base_model': _portable_train_base_model(ds.train_base_model),
        'train_variant': ds.train_variant, 'train_settings': ds.train_settings,
        'best_settings': ds.best_settings,
        'ref_filename': (_backup_basename(ds.ref_filename)
                         if _backup_basename(ds.ref_filename) in primary_ref_names else None),
        'ref_original_filename': (
            _backup_basename(ds.ref_original_filename)
            if _backup_basename(ds.ref_original_filename) in primary_ref_names else None),
        'ref_extra_filenames': json.dumps(portable_extras),
    }
    # backup_image_id is archive-local only. It lets restore remap parent_image_id
    # to the newly allocated row ids instead of retaining ids from the source DB.
    images_meta = []
    for img in rows:
        row = dict({'backup_image_id': img.id},
                   **{f: getattr(img, f) for f in _BACKUP_IMG_FIELDS})
        # Archive a structured, revalidated object rather than the raw TEXT
        # column. A malformed legacy/local row can never export arbitrary links.
        row['source_metadata'] = normalize_source_metadata(img.source_metadata)
        # A snapshot is durable only when it has the expected version, fingerprint
        # and bounded analysis shape.  Invalid legacy/local text is deliberately
        # omitted rather than becoming an opaque payload in a portable backup.
        row['bank_analysis_snapshot'] = bank_transfer_metadata.normalized_snapshot_storage(
            img.bank_analysis_snapshot)
        images_meta.append(row)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=1))
        z.writestr('images.json', json.dumps(images_meta, ensure_ascii=False, indent=1))
        for n in ref_names:
            p = os.path.join(dsdir, n)
            z.write(p, f'ref/{n}')
        for img in rows:
            name = _backup_basename(img.filename)
            if not name:
                continue   # metadata-only small-rescue candidate
            p = os.path.join(dsdir, name)
            if os.path.isfile(p):
                z.write(p, f'images/{name}')


def build_backup_zip(user_id: int, dataset_id: int) -> bytes:
    """Compatibility wrapper for callers that still need an in-memory archive."""
    output = io.BytesIO()
    write_backup_zip(user_id, dataset_id, output)
    return output.getvalue()


def _coerce_archive_stream(archive):
    """Return (seekable stream, owned stream or None) without copying file uploads."""
    if isinstance(archive, (bytes, bytearray, memoryview)):
        owned = io.BytesIO(bytes(archive))
        return owned, owned
    if not hasattr(archive, 'read') or not hasattr(archive, 'seek'):
        raise ValueError('not a zip file')
    try:
        archive.seek(0)
    except (OSError, ValueError) as exc:
        raise ValueError('zip archive is not seekable') from exc
    return archive, None


def import_backup_zip(user_id: int, archive: bytes | BinaryIO):
    """Restore a backup as a NEW dataset (never merges into an existing one).
    Hardened: manifest format/version check, per-entry filename whitelist (no
    separators/traversal), file-count and uncompressed-size caps. Returns the
    created FaceDataset."""
    stream, owned = _coerce_archive_stream(archive)
    try:
        try:
            z = zipfile.ZipFile(stream)
        except zipfile.BadZipFile as exc:
            raise ValueError('not a zip file') from exc
        try:
            return _import_backup_zipfile(user_id, z)
        finally:
            z.close()
    finally:
        if owned is not None:
            owned.close()


def _import_backup_zipfile(user_id: int, z: zipfile.ZipFile):
    # Validate the central directory BEFORE inflating JSON.  Previously a tiny
    # compressed manifest/images.json could bypass the image-only size total and
    # consume unbounded RAM during z.read/json.loads.
    all_infos = z.infolist()
    if len(all_infos) > _BACKUP_MAX_FILES + 2:
        raise ValueError(f'too many files in backup (max {_BACKUP_MAX_FILES})')
    if sum(info.file_size for info in all_infos) > _BACKUP_MAX_BYTES:
        raise ValueError('backup too large (max 2 GB uncompressed)')
    metadata = {}
    for info in all_infos:
        if info.filename not in ('manifest.json', 'images.json'):
            continue
        if info.filename in metadata:
            raise ValueError(f'duplicate {info.filename} in backup')
        if info.file_size > _BACKUP_MAX_METADATA_BYTES:
            raise ValueError(f'{info.filename} is too large')
        metadata[info.filename] = info
    if set(metadata) != {'manifest.json', 'images.json'}:
        raise ValueError('not a dataset backup (manifest.json/images.json missing or invalid)')
    try:
        manifest = json.loads(z.read(metadata['manifest.json']).decode('utf-8'))
        images_meta = json.loads(z.read(metadata['images.json']).decode('utf-8'))
    except (ValueError, UnicodeError, RecursionError, MemoryError, zipfile.BadZipFile):
        raise ValueError('not a dataset backup (manifest.json/images.json missing or invalid)')
    if not isinstance(manifest, dict):
        raise ValueError('invalid backup manifest')
    if manifest.get('format') != BACKUP_FORMAT:
        raise ValueError('not a dataset backup')
    version = manifest.get('version')
    if (isinstance(version, bool) or not isinstance(version, int)
            or version < 1):
        raise ValueError('invalid backup version')
    if version > BACKUP_VERSION:
        raise ValueError('backup made by a newer version of the app - update first')
    for field in ('name', 'trigger_word'):
        value = manifest.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f'invalid backup {field}')
    restored_training_mode = manifest.get('training_mode', 'lora')
    if restored_training_mode not in ('lora', 'full_transformer'):
        raise ValueError('invalid backup training_mode')
    if not isinstance(images_meta, list):
        raise ValueError('invalid backup image metadata')
    if len(images_meta) > _BACKUP_MAX_ROWS:
        raise ValueError(f'too many image rows in backup (max {_BACKUP_MAX_ROWS})')
    seen_backup_ids = set()
    rescue_sources = set()
    rescue_parent_counts = {}
    for meta in images_meta:
        if not isinstance(meta, dict):
            raise ValueError('invalid backup image metadata')
        filename = meta.get('filename')
        if filename is not None and not isinstance(filename, str):
            raise ValueError('invalid backup image filename')
        backup_id = meta.get('backup_image_id')
        if backup_id is not None:
            if isinstance(backup_id, bool) or not isinstance(backup_id, int) or backup_id <= 0:
                raise ValueError('invalid backup image id')
            if backup_id in seen_backup_ids:
                raise ValueError('duplicate backup image id')
            seen_backup_ids.add(backup_id)
        derivation = meta.get('derivation_kind')
        if derivation not in (None, SMALL_IMAGE_SOURCE, KLEIN_SMALL_IMAGE,
                              KLEIN_IMAGE_IMPROVE):
            raise ValueError('invalid image derivation in backup')
        if derivation == SMALL_IMAGE_SOURCE:
            if backup_id is None or meta.get('parent_image_id') is not None:
                raise ValueError('invalid small-image source provenance')
            rescue_sources.add(backup_id)
        elif derivation == KLEIN_SMALL_IMAGE:
            parent_id = meta.get('parent_image_id')
            if backup_id is None or isinstance(parent_id, bool) or not isinstance(parent_id, int):
                raise ValueError('invalid Klein rescue provenance')
            rescue_parent_counts[parent_id] = rescue_parent_counts.get(parent_id, 0) + 1
            if rescue_parent_counts[parent_id] > 1:
                raise ValueError('multiple Klein rescue candidates for one source')
    if any(parent_id not in rescue_sources for parent_id in rescue_parent_counts):
        raise ValueError('Klein rescue candidate has no valid source')
    infos = [i for i in all_infos
             if not i.is_dir() and i.filename.startswith(('ref/', 'images/'))]
    if len(infos) > _BACKUP_MAX_FILES:
        raise ValueError(f'too many files in backup (max {_BACKUP_MAX_FILES})')
    archive_names = {'ref': {}, 'images': {}}
    for info in infos:
        prefix, candidate = info.filename.split('/', 1)
        name = _backup_basename(candidate)
        if name:
            key = name.casefold()
            if key in archive_names[prefix]:
                raise ValueError(
                    f'backup has duplicate {prefix} filename: {name}')
            archive_names[prefix][key] = name
    collisions = set(archive_names['ref']).intersection(archive_names['images'])
    if collisions:
        collision = archive_names['images'][next(iter(collisions))]
        raise ValueError(f'backup has colliding ref/image filename: {collision}')
    name = (manifest.get('name') or 'Restored dataset')[:100]
    trigger = (manifest.get('trigger_word') or 'restored')[:60]
    # Extract first into a sibling directory: it is on the same volume as the final
    # dataset folder, so promotion is a single rename.  The database transaction is
    # only opened after extraction succeeds; no empty dataset can become visible.
    root = str(cfg.dataset_images_root())
    staging_dir = os.path.join(root, f'.restore-{uuid.uuid4().hex}.tmp')
    os.mkdir(staging_dir)
    final_dir = None
    promoted = False
    db_started = False
    try:
        extracted_images = set()
        extracted_refs = {}
        for info in infos:
            prefix, candidate = info.filename.split('/', 1)
            base = _backup_basename(candidate)
            if not base:
                continue   # nested path or weird name -> skip, never traverse
            # Decode/verify all archive image bytes before they ever reach the
            # restore staging directory. That keeps the eventual rename/promotion
            # atomic even for compact pixel bombs or content/extension lies.
            raw = _read_validated_backup_image(z, info, base)
            with open(os.path.join(staging_dir, base), 'wb') as dst:
                # Keep this copy seam (rather than ``dst.write(raw)``): it is
                # deliberately fault-injectable by the atomic restore regression.
                shutil.copyfileobj(io.BytesIO(raw), dst, 1024 * 1024)
            if prefix == 'ref':
                extracted_refs.setdefault(base.casefold(), base)
            else:
                extracted_images.add(base)

        db_started = True
        ds = create_dataset(user_id, name, trigger, kind=manifest.get('kind'),
                            concept_desc=manifest.get('concept_desc'),
                            train_type=manifest.get('train_type'), commit=False)
        for field in ('concept_terms', 'train_variant', 'train_settings',
                      'best_settings', 'fidelity'):
            setattr(ds, field, manifest.get(field))
        ds.training_mode = restored_training_mode
        ds.train_base_model = _portable_train_base_model(manifest.get('train_base_model'))
        ds.ref_filename = _backup_basename(manifest.get('ref_filename'))
        ds.ref_original_filename = _backup_basename(
            manifest.get('ref_original_filename'))
        final_dir = os.path.join(root, str(ds.id))
        if os.path.exists(final_dir):
            # Never merge with or delete a pre-existing orphan directory.
            raise RuntimeError(f'dataset folder already exists for id {ds.id}')

        n_rows = 0
        restored_rows = []
        valid_source_ids = {
            meta.get('backup_image_id') for meta in images_meta
            if isinstance(meta, dict)
            and meta.get('derivation_kind') == SMALL_IMAGE_SOURCE
            and meta.get('filename') in extracted_images
        }
        for meta in images_meta:
            if not isinstance(meta, dict):
                continue
            fn = meta.get('filename')
            derivation = meta.get('derivation_kind')
            is_candidate = derivation == KLEIN_SMALL_IMAGE
            if fn and fn not in extracted_images:
                continue
            if not fn and not is_candidate:
                continue   # only rescue candidates have meaningful metadata-only rows
            if is_candidate and meta.get('parent_image_id') not in valid_source_ids:
                continue   # never restore an orphaned candidate
            values = {f: meta.get(f) for f in _BACKUP_IMG_FIELDS
                      if f not in ('filename', 'parent_image_id')}
            # Backup input is untrusted. Unknown/invalid provenance is dropped,
            # while valid Pexels metadata is canonicalized back to JSON TEXT.
            values['source_metadata'] = _source_metadata_storage(
                values.get('source_metadata'))
            values['bank_analysis_snapshot'] = (
                bank_transfer_metadata.normalized_snapshot_storage(
                    values.get('bank_analysis_snapshot')))
            if is_candidate and not fn and values.get('status') in ('pending', 'keep'):
                values['status'] = 'failed'
                values['fail_reason'] = (
                    'Klein rescue was in flight when this backup was created; '
                    'the original image is preserved, but the job must be started again.'
                )
            img = FaceDatasetImage(dataset_id=ds.id,
                                   **values,
                                   filename=fn)
            db.session.add(img)
            restored_rows.append((img, meta))
            n_rows += 1
        # Allocate new ids first, then restore the graph strictly within this backup.
        # A missing/skipped parent clears the relationship rather than pointing at an
        # unrelated row that happens to reuse the old numeric id.
        db.session.flush()
        id_map = {meta.get('backup_image_id'): img.id for img, meta in restored_rows
                  if meta.get('backup_image_id') is not None}
        for img, meta in restored_rows:
            img.parent_image_id = id_map.get(meta.get('parent_image_id'))
        # Reference fields are rebuilt exclusively from actual ref/ archive files.
        # Never retain paths, missing names, image-only files, or case variants.
        ds.ref_filename = (extracted_refs.get(ds.ref_filename.casefold())
                           if ds.ref_filename else None)
        ds.ref_original_filename = (
            extracted_refs.get(ds.ref_original_filename.casefold())
            if ds.ref_original_filename else None)
        used_ref_keys = {
            ref.casefold() for ref in (ds.ref_filename, ds.ref_original_filename) if ref
        }
        restored_extras = []
        for requested in _backup_extra_ref_names(
                manifest.get('ref_extra_filenames'), limit=None):
            key = requested.casefold()
            actual = extracted_refs.get(key)
            if not actual or key in used_ref_keys:
                continue
            used_ref_keys.add(key)
            restored_extras.append(actual)
            if len(restored_extras) >= MAX_EXTRA_REFS:
                break
        ds.ref_extra_filenames = json.dumps(restored_extras)

        os.replace(staging_dir, final_dir)
        promoted = True
        db.session.commit()
    except Exception:
        try:
            if db_started:
                db.session.rollback()
        finally:
            if promoted and final_dir:
                shutil.rmtree(final_dir, ignore_errors=True)
        raise
    finally:
        # Exists on extraction/build/promotion failure; after promotion the old path
        # is already gone.  Never leave hidden partial restores behind.
        shutil.rmtree(staging_dir, ignore_errors=True)
    logger.info(f"dataset backup restored: '{name}' -> #{ds.id} ({n_rows} image rows)")
    return ds


def replace_in_captions(user_id, dataset_id, find, replace, mode='text'):
    """Bulk-edit the captions of KEPT images (the ones that train). Two modes:

    - 'text': whole-word replace, CASE-INSENSITIVE — the same match rule as the
      grid filter ("smile" hits "a warm smile" but not "smiling") and the most-
      frequent-words counter, both case-insensitive. So clicking a "bulldog ×41"
      chip and stripping it removes all 41 whatever their casing (the captions
      hold "Bulldog"); a case-sensitive substring replace matched 0 and looked
      broken. Whole-word so "red" never eats the "red" inside "colored". When
      `replace` is empty the gaps a stripped word leaves in prose are tidied.
    - 'tag':  the caption is treated as a comma-separated tag list (booru); `find`
      must match a WHOLE tag (trimmed, case-insensitive) and is replaced by
      `replace` — or dropped when `replace` is empty. Avoids the ', ,' artifacts a
      substring removal would leave in tag captions. Result is deduped
      case-insensitively (keeping first occurrence / original casing).

    Returns the number of captions actually changed."""
    if mode not in ('text', 'tag'):
        raise ValueError('invalid mode')
    find = (find or '').strip() if mode == 'tag' else (find or '')
    if not find:
        raise ValueError('find is required')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return 0
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='keep')
            .filter(FaceDatasetImage.caption.isnot(None)).all())
    changed = 0
    for img in rows:
        old = img.caption or ''
        if mode == 'text':
            pattern = re.compile(rf'\b{re.escape(find)}\b', re.IGNORECASE)
            new = pattern.sub(replace or '', old)
            if not (replace or '').strip():          # stripping: tidy prose gaps
                new = re.sub(r'\s+([,.;:])', r'\1', new)   # space before punctuation
                new = re.sub(r'(,\s*){2,}', ', ', new)     # collapsed repeated commas
                new = re.sub(r'\s{2,}', ' ', new)          # collapsed double spaces
                new = new.strip(' ,;')
        else:
            tags = [t.strip() for t in old.split(',')]
            out, seen = [], set()
            for t in tags:
                if not t:
                    continue
                nt = (replace or '').strip() if t.lower() == find.lower() else t
                if not nt or nt.lower() in seen:
                    continue
                seen.add(nt.lower())
                out.append(nt)
            new = ', '.join(out)
        new = _cap_caption(new) or None
        if new != img.caption:
            img.caption = new
            changed += 1
    if changed:
        db.session.commit()
    return changed


# Batch curation (multi-select in the grid). 'pending' = reset the triage state.
BATCH_ACTIONS = ('keep', 'reject', 'pending', 'delete', 'clear_caption')


@_serialize_dataset_ingest
def batch_image_action(user_id, dataset_id, image_ids, action):
    """Apply one whitelisted action to a set of this dataset's images in one call
    (the grid's multi-select). Ownership is checked once on the dataset; ids that
    don't belong to it (or don't exist) are silently skipped, so a stale selection
    after a poll refresh can't touch another dataset's rows. Returns
    (affected, skipped_locked) — the latter is always 0 for actions other than
    'delete' (locked rows are otherwise untouched, never refused)."""
    if action not in BATCH_ACTIONS:
        raise ValueError('invalid action')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return 0, 0
    ids = [int(i) for i in (image_ids or []) if isinstance(i, (int, float, str)) and str(i).lstrip('-').isdigit()]
    if not ids:
        return 0, 0
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id)
            .filter(FaceDatasetImage.id.in_(ids)).all())
    n = 0
    skipped_locked = 0
    if action != 'clear_caption' and any(
            img.derivation_kind in _SMALL_IMAGE_DERIVATIONS for img in rows):
        raise ValueError('resolve small-image rescue pairs with the dedicated review action')
    if action == 'delete':
        # Per-image path: reuses delete_image (file removal + pending-job cancel).
        # A locked row is a partial success, not a refusal — skip it and keep
        # going, same as an id that doesn't belong to this dataset.
        for img in rows:
            if img.is_locked:
                skipped_locked += 1
                continue
            if delete_image(user_id, img.id):
                n += 1
        return n, skipped_locked
    for img in rows:
        if action == 'clear_caption':
            img.caption = None
        else:
            # Never resurrect a failed generation into keep/reject — the tile has
            # no file; regenerate is the only way out of 'failed'.
            if img.status == 'failed':
                continue
            if action == 'reject':
                _clear_watermark_metadata(img)
            img.status = action
        n += 1
    if action == 'keep':
        # This is deliberately a second phase.  If both a source and its
        # improvement are selected, every explicit choice is applied first,
        # then the kept candidate wins regardless of database/query order.
        for img in rows:
            if img.status == 'keep':
                _unkeep_parent_for_kept_improvement(img)
    db.session.commit()
    return n, skipped_locked


def _watermark_route_payload(img):
    """The routes Clean WOULD take for a 'detected' image, as a dict spread into the
    image payload:
      - 'watermark_route'        : the DEFAULT route ('crop' | 'lama' | 'review'), used
                                   by the 🚩 tooltip and the batch/lightbox planned line;
      - 'watermark_route_nocrop' : the SAME routing with auto-crop disabled ('lama' |
                                   'review') -- only ever differs when the default is
                                   'crop'. It lets the review lightbox offer a per-image
                                   crop-vs-inpaint choice (and name the inpaint fallback)
                                   without duplicating _route_watermark in JS.
    Both are None for a non-'detected' row. It needs the pixel dims (the grid doesn't
    carry them), so it opens the file ONCE -- but only for 'detected' rows (a bounded
    subset), so the single-dataset payload never reads every image header. Defensive: any
    read/parse error yields None routes and the UI falls back to the generic hint."""
    none = {'watermark_route': None, 'watermark_route_nocrop': None}
    if img.watermark_state != 'detected':
        return none
    bbox = _safe_json(img.watermark_bbox)
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return none
    try:
        with Image.open(_img_path(img)) as im:
            # The bbox comes from the browser/VLM visual frame, so route against
            # the same EXIF-oriented dimensions the user can see. This is a
            # payload/polling read: use the header-only helper, never decode the
            # whole master merely to draw a route badge.
            W, H = image_encoding.visual_size_from_header(im)
    except (OSError, ValueError):
        return none
    box = tuple(bbox)
    route, _ = _route_watermark(box, W, H)
    # Only recompute the crop-disabled route when crop is what the default picked --
    # otherwise the two are identical, so skip the redundant pure-function call.
    route_nc = route if route != 'crop' else _route_watermark(box, W, H, allow_crop=False)[0]
    return {'watermark_route': route, 'watermark_route_nocrop': route_nc}


def _image_engine(img):
    """Which engine produced this image — 'klein' | 'nanobanana' | 'chatgpt' — or
    None when it CANNOT be told.

    `klein_model` carries two different kinds of value: an engine id for the API
    rows (set by generate_variations_nanobanana) and a local .safetensors file
    name for the Klein rows. That is enough to answer honestly for both, but not
    for every legacy row: images generated before the column was populated, and
    imported photos, hold nothing. Those get None → the UI shows NO badge, which
    is the right answer. Guessing 'klein' for an empty value would label old
    Nano Banana images as local, and a wrong badge is worse than none."""
    value = (img.klein_model or '').strip()
    if not value:
        return None
    if value in API_ENGINES:
        return value
    # Krea 2 Edit rows store the engine id here, like the API ones: the engine
    # resolves its base model deterministically at enqueue AND at regenerate
    # (krea_edit_helper.resolve_krea_unet), so there is no per-row model to keep.
    if value == KREA_ENGINE:
        return KREA_ENGINE
    return 'klein'   # a local model file name — the row was rendered on the GPU


def dataset_payload(user_id, dataset_id):
    from . import lora_test_studio as studio
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return None
    imgs = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id)
            .order_by(FaceDatasetImage.id.desc()).all())
    ref_size = image_pixel_size(_ref_path(ds)) if ds.ref_filename else None
    comp = {'face': 0, 'bust': 0, 'body': 0, 'back': 0}
    # Combien, PAR bucket, viennent d'une box bien plus petite que la résolution
    # d'entraînement (upscale_ratio >= UPSCALE_WARN_THRESHOLD) plutôt que d'une prise
    # native : le compte `comp` seul traite un gros plan natif et un gros plan
    # recadré x3 comme équivalents vis-à-vis de la cible — ce sous-compte permet à
    # l'UI de signaler un dataset qui « remplit » sa cible face/bust surtout en
    # recadrant (texture agrandie à l'import, ou tuile sous-résolue en manuel).
    comp_upscaled = {'face': 0, 'bust': 0, 'body': 0, 'back': 0}
    for i in imgs:
        # Composition counts only usable images: rejected and failed ones don't
        # contribute to the training-target tally the UI tracks deficits against.
        if i.framing in comp and i.status not in ('reject', 'failed'):
            comp[i.framing] += 1
            if (i.upscale_ratio or 0) >= UPSCALE_WARN_THRESHOLD:
                comp_upscaled[i.framing] += 1
    # concept OU style : le champ `fidelity`/`concept_desc` du payload est gouverné par
    # is_conceptual (character-only). La DÉTECTION de fuite, elle, est spécifique au KIND :
    #   - character : fuite d'IDENTITÉ (hair/skin/eyes)  → caption_has_identity_leak
    #   - concept   : fuite de CONCEPT (le set nomme le concept au lieu du trigger) →
    #                 caption_has_concept_leak — on ne force PLUS 0 (le badge « 0 leak »
    #                 faussement rassurant de l'incident leg_behind)
    #   - style     : rien (la description des sujets EST le contenu contrôlable) → 0 honnête
    concept = is_conceptual(ds)
    kind_concept = is_concept(ds)
    kind_style = is_style(ds)
    body = is_body_fidelity(ds)
    # Cached concept ban-list (JSON on the row) → the concept-leak detector unions it with
    # concept_desc + the derived body/pose field, so the badge and the caption-time
    # enforcement agree on what "leaking" means. Ignored for non-concept kinds.
    _concept_terms = ds.concept_terms if kind_concept else None
    _pose_rows = pose_slot_rows(ds)

    def _img_leaks(i):
        if i.status != 'keep' or not i.caption:
            return False
        if kind_concept:
            return caption_has_concept_leak(i.caption, ds.concept_desc, _concept_terms)
        if kind_style:
            return False
        return caption_has_identity_leak(i.caption, body=body)

    def _img_leak_terms(i):
        """The exact leaking words for i, so the caption-leak review can highlight
        them in place instead of just counting them. Same kind branching as
        _img_leaks — empty whenever that returns False."""
        if i.status != 'keep' or not i.caption:
            return []
        if kind_concept:
            return caption_concept_leaks(i.caption, ds.concept_desc, _concept_terms)
        if kind_style:
            return []
        return identity_leak_terms(i.caption, body=body)

    # Which pending rows are ACTUALLY being worked on right now, vs merely
    # queued behind them — a 50-shot batch left every not-yet-done tile on the
    # same amber "pending" border, with no way to tell which one the worker is
    # actually cooking. One batched lookup (never one query per row) against
    # the SEPARATE job-queue table (FaceDatasetImage.status is the curation
    # verdict; ImageGenerationQueue.status is the job's own lifecycle:
    # pending -> processing -> sent_to_comfy). Only 'processing'/'sent_to_comfy'
    # count as "running now" — a job still 'pending' in the queue hasn't been
    # claimed by a worker yet, same as the tile itself.
    _job_ids = [i.job_id for i in imgs if i.status == 'pending' and i.job_id]
    _running_jobs = set()
    if _job_ids:
        _running_jobs = {row.job_id for row in
                         ImageGenerationQueue.query
                         .filter(ImageGenerationQueue.job_id.in_(_job_ids),
                                 ImageGenerationQueue.status.in_(('processing', 'sent_to_comfy')))
                         .all()}

    return {
        'id': ds.id, 'name': ds.name, 'trigger_word': ds.trigger_word,
        'train_type': (ds.train_type or 'zimage'),
        'kind': (ds.kind or 'character'),
        # WHAT the subject is (NULL/legacy -> 'human'); drives the generation
        # catalog + identity lock. Orthogonal to `kind`.
        'subject_type': subject_type_of(ds),
        # Why face-similarity scoring is refused for this dataset (string), or null
        # to go ahead. Published so the UI disables the button and states the reason
        # from the SAME rule the server enforces, instead of re-implementing
        # "subject_type === 'anime'" in JSX and drifting from it later.
        'face_scoring_blocked': face_scoring_block_reason(ds),
        # How much work 🎭 Analyze faces actually has: {total, unscored} over the
        # kept set PLUS the undecided triage pile (FACE_SCORING_STATUSES). Lets the
        # button name its scope instead of running a mystery pass.
        'face_scoring_scope': face_scoring_counts(imgs),
        # Dual long+short captioning toggle (Advanced options) → the caption editor shows
        # the short field only when this is on.
        'dual_captions': dual_captions_enabled(ds),
        # Concept face masking (Advanced options) + whether this concept's own
        # description names the face/mouth/gaze. The second one drives a WARNING,
        # not a block: only the user knows whether the face carries their concept.
        'mask_faces': face_masking_enabled(ds),
        'concept_face_conflict': concept_face_conflict(ds),
        'fidelity': (ds.fidelity or 'face') if not concept else 'face',
        'concept_desc': (ds.concept_desc or '') if concept else '',
        # Creative-direction suffixes (global + per-framing) → settings modal
        # prefill. Applied at wrap time; never part of the stored per-image prompt.
        'prompt_suffix': ds.prompt_suffix or '',
        'prompt_suffixes': prompt_suffixes_dict(ds),
        # Where this dataset's images actually live. It was displayed NOWHERE,
        # which is how people ended up hunting for it in the file manager and
        # pasting it into "create a bank" — a bank over a dataset's live files,
        # whose 🗑 Delete rejected then deleted images out of the dataset. Showing
        # the path (with the sentence that it belongs to the dataset) removes the
        # reason to go looking; `services.path_guard` refuses the paste anyway.
        'storage_path': _dataset_path(ds.id),
        'ref_filename': ds.ref_filename,
        # Pixel size of the ACTIVE reference (the cropped one — that is the file
        # every engine is handed). Kept for crop-aware clients; Krea dataset cards
        # now use the selected card's target frame. None when unmeasurable.
        'ref_width': (ref_size or (None, None))[0],
        'ref_height': (ref_size or (None, None))[1],
        'ref_original_filename': ds.ref_original_filename or '',
        'ref_extra_filenames': extra_ref_filenames(ds),
        # Per extra ref, the file its ✂ editor must open (full-frame original when
        # kept, else the extra itself) — aligned index-by-index with the list above.
        'ref_extra_crop_sources': [extra_ref_crop_source(ds, fn)
                                   for fn in extra_ref_filenames(ds)],
        # ALL 5 POSE_SLOT_KEYS, always — deliberately the opposite shape from
        # pose_slot_rows() (which omits keys with no upload). This lets the
        # frontend render every slot's card the same way, with no
        # "never uploaded" special-casing on a missing dict key.
        'pose_slots': {
            key: ({'filename': r.filename, 'original_filename': r.original_filename or '',
                  'enabled': bool(r.enabled)} if (r := _pose_rows.get(key)) else
                  {'filename': None, 'original_filename': '', 'enabled': False})
            for key in POSE_SLOT_KEYS
        },
        'composition': comp,
        'composition_upscaled': comp_upscaled,
        # Réglages gagnants du Studio (JSON → objet). Manquait du payload : le badge
        # ★ du workspace ne s'affichait jamais, et le garde-fou « suppression d'un
        # checkpoint référencé » en a besoin.
        'best_settings': _safe_json(ds.best_settings),
        # The pinned LoRA filenames, FLATTENED out of the per-family map above.
        # The delete guard-rail used to read `best_settings.lora_filename`, a key
        # that only exists in the legacy flat shape — so on any dataset pinned
        # since best settings went per-family the ⚠ warning was silently dead.
        'best_settings_loras': studio.best_settings_lora_filenames(ds),
        'face_thresholds': {'green': cfg.get('face_scoring.green'), 'orange': cfg.get('face_scoring.orange')},
        'images': [{'id': i.id, 'filename': i.filename, 'source': i.source,
                    # Which engine made this image, when it can be told HONESTLY
                    # (see _image_engine) — the tile badge that makes a
                    # multi-engine run comparable. None = no badge.
                    'engine': _image_engine(i),
                    'framing': i.framing, 'variation_label': i.variation_label,
                    'status': i.status,
                    # True for the (usually few) pending tiles a worker has
                    # actually claimed right now, vs merely queued behind them —
                    # see the batched lookup above.
                    'is_generating': i.job_id in _running_jobs,
                    # True once a generation OR regenerate finishes until the
                    # tile is opened — the "which ones are new" marker in a
                    # big grid.
                    'unseen': bool(i.unseen),
                    'is_locked': bool(i.is_locked),
                    'caption': i.caption,
                    'caption_short': i.caption_short,
                    'fail_reason': i.fail_reason,
                    # 'refused' | 'empty' | 'error' | None — de quelle NATURE est
                    # l'échec, pour que l'UI puisse compter les refus fournisseur
                    # sans relire la phrase (cf. models.FaceDatasetImage).
                    'fail_kind': i.fail_kind,
                    'parent_image_id': i.parent_image_id,
                    'derivation_kind': i.derivation_kind,
                    'source_metadata': normalize_source_metadata(i.source_metadata),
                    'upscale_ratio': i.upscale_ratio,
                    # Core creative prompt (generated tiles) → seeds the ✏️ edit
                    # bubble so the user edits the real prompt, not a blank box.
                    'variation_prompt': i.variation_prompt,
                    # Per-image leak flag (identity for character, concept for concept,
                    # never for style): lets the UI LIST the offending captions for quick
                    # manual treatment (the aggregate badge alone forced a grid hunt).
                    'leak': _img_leaks(i),
                    # The exact leaking words, so the review panel can highlight them
                    # inside the caption instead of leaving the user to hunt by eye.
                    'leak_terms': _img_leak_terms(i),
                    'face_score': i.face_score, 'face_state': i.face_state,
                    # Watermark V1: state drives the tile badge (🚩 detected / ⊘ dismissed
                    # / ✨ cleaned / ⚠ failed) and the "Clean (N)" count; bbox lets the UI
                    # draw the detected box (review lightbox); watermark_route(_nocrop)
                    # name the planned action ('crop'|'lama'|'review') with auto-crop on
                    # and off, so the lightbox can offer a per-image crop-vs-inpaint choice.
                    'watermark_state': i.watermark_state,
                    'watermark_bbox': _safe_json(i.watermark_bbox),
                    **_watermark_regions_payload(i),
                    **_watermark_route_payload(i)} for i in imgs],
        # Kind-specific leak count (see _img_leaks): character = identity, concept = the
        # caption naming the concept (NEVER forced 0 any more), style = 0 (not applicable).
        # `captioned` bounds the badge ("N leaking / M checked") so a 0 reads as a real
        # result on M captions, not a check that never ran.
        'caption_leak': {
            'leaking': sum(1 for i in imgs if _img_leaks(i)),
            'captioned': sum(1 for i in imgs if i.status == 'keep' and i.caption),
        },
        # Live server-side batch on this dataset (watermark detect/clean, caption/
        # re-caption, face analysis, framing classify) as {kind, done, total,
        # started_at} — or None. The front-end RESTORES the in-progress button state
        # from this on reload and polls the payload until it clears (the indicator was
        # React-local before, so a refresh mid-batch dropped it). In-memory registry:
        # empty after a server restart, so a batch killed with the process leaves no
        # phantom indicator.
        'activity': dataset_activity.get(dataset_id),
        # Pending reference EDIT (server background job) as {status, engine, prompt,
        # candidate_filename, error, started_at} — or None. The modal RESTORES its
        # Before/After from this after a tab sleep or reload; the 'edit_reference'
        # activity above keeps this polled while it runs. get() lazily purges an
        # abandoned candidate past its TTL, so this can't strand a stale file.
        'reference_edit': reference_edit_jobs.get(dataset_id),
    }


# --- Image normalization ---------------------------------------------------
def write_image_atomic(path, data: bytes) -> None:
    """Publish an image file in one step: it is either absent or COMPLETE.

    `open(path, 'wb')` truncates immediately, and the bytes usually arrive a
    second or two later (a WEBP re-encode of a 1024px generation is not free).
    Under its FINAL name that leaves an empty file on disk for the whole
    encode, and the grid polls the dataset while a batch runs: the browser
    asks for it, the server answers 200 with zero bytes, and the tile renders
    black. Reported after an OpenRouter generation, but nothing about it was
    engine-specific — every generated image had the same window.

    Writing beside the target and renaming closes it: os.replace is atomic on
    the same filesystem, so a reader sees the old state or the new one, never
    a half-written one. A missing file is already handled everywhere (the tile
    shows its pending state), which is the honest answer while it is encoding.
    """
    tmp = f'{path}.part'
    try:
        with open(tmp, 'wb') as fh:
            fh.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)          # never leave a .part behind on failure
        except OSError:
            pass
        raise


def normalize_to_webp(image_bytes: bytes, size: int = 1024,
                      quality: int = 92, lossless: bool = False) -> bytes:
    """Resize so the longest side ≤ `size`, KEEP the aspect ratio (no square pad),
    return WEBP. Pour les variations Nano Banana : un plan corps reste en portrait
    (pas de bandes noires que le LoRA apprendrait). ai-toolkit gère le bucketing.

    `size=0` means "do not resample at all" — the ceiling below still applies
    because it is a FORMAT limit, not a taste: Pillow refuses to write a WebP
    past 16383 px ("Image size exceeds WebP limit of 16383 pixels"), so an
    uncapped call would turn a big panorama into a failed import.

    DERIVATIVE ON PURPOSE — this is INGEST/TRANSPORT (the ≤2048 px copy handed to a
    generation API, and the normalisation of freshly generated bytes), not an edit of
    an image the user already curated. It must NOT be routed through `image_encoding`:
    inflating an upload 4x to protect pixels the remote engine will re-encode anyway
    buys nothing. See the module docstring of `image_encoding` for the split."""
    # A normalised WebP has no reason to retain a camera orientation tag. The
    # shared loader validates header geometry before any full decode, then bakes
    # the visible orientation into its temporary pixels.
    im = _load_import_derivative_image(image_bytes).convert('RGB')
    limit = min(size, IMPORT_MAX_SIDE_CEILING) if size else IMPORT_MAX_SIDE_CEILING
    im.thumbnail((limit, limit), Image.LANCZOS)   # only ever shrinks
    out = io.BytesIO()
    if lossless:
        im.save(out, 'WEBP', lossless=True)
    else:
        im.save(out, 'WEBP', quality=quality)
    return out.getvalue()


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


# --- Completion linking (called from the job queue) -------------------------
def link_completed_dataset_image(job_id, filename, failed=False, reason=None):
    """Attach a finished fan-out job to its FaceDatasetImage row.

    Called from the job-queue completion/failure/cancel paths, which may run in
    a long-lived monitor thread whose SQLAlchemy session holds a STALE read
    snapshot (rows committed by other threads are invisible). If the first
    lookup misses, end the transaction (rollback) and retry on a fresh snapshot
    before concluding the row really doesn't exist.
    `reason` (the job row's error_message, e.g. a ComfyUI execution error) shows
    on the failed tile so the user sees WHY, not a generic 'see the log'."""
    img = FaceDatasetImage.query.filter_by(job_id=job_id).first()
    if img is None:
        db.session.rollback()  # drop the stale read snapshot, then re-read
        img = FaceDatasetImage.query.filter_by(job_id=job_id).first()
    if img is None:
        logger.warning(f"dataset link: no FaceDatasetImage row for job {job_id}")
        return
    if img.derivation_kind == KLEIN_SMALL_IMAGE and img.status in ('keep', 'reject'):
        # The user already resolved the pair while this job/callback was racing.
        # The terminal review decision wins: do not attach
        # a late file and do not turn reject into failed. This is a temporary,
        # unlinked Comfy output (never user data), so direct removal is intentional.
        output_dir = _comfy_output_dir()
        late_output = os.path.join(output_dir, filename) if output_dir and filename else None
        if late_output and os.path.isfile(late_output):
            try:
                os.remove(late_output)
            except OSError:
                pass
        try:
            _sync_generate_activity(img.dataset_id)
        except Exception:
            logger.exception(
                'dataset link: terminal rescue activity sync failed for job %s', job_id)
        return
    if failed:
        # A cancel racing with the worker dispatches a failure callback. Never let
        # that callback overwrite an already-resolved rescue choice (keep/reject).
        if not (img.derivation_kind == KLEIN_SMALL_IMAGE
                and img.status in ('keep', 'reject')):
            img.status = 'failed'
            img.fail_reason = (img.fail_reason or reason
                               or 'Klein generation failed (see 🪵 Server log in Settings for the ComfyUI error)')
    else:
        output_dir = _comfy_output_dir()
        src = os.path.join(output_dir, filename) if output_dir else None
        dst = os.path.join(_dataset_dir(img.dataset_id), filename)
        if src and os.path.exists(src) and os.path.exists(dst):
            # Collision guard: NEVER overwrite another tile's file. ComfyUI's
            # SaveImage counter re-issued the same name when earlier results
            # were moved out of its output folder — every tile then displayed
            # the same (last) image. The prefix is unique per job now, but a
            # residual collision must degrade to a rename, not a silent loss.
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{uuid.uuid4().hex[:6]}{ext}"
            dst = os.path.join(_dataset_dir(img.dataset_id), filename)
            logger.warning(f"dataset link: name collision, storing as {filename}")
        img.filename = filename
        if src and os.path.exists(src):
            shutil.move(src, dst)          # file where we expected it on disk
        elif os.path.exists(dst):
            pass                           # already brought in (retry / dup completion)
        else:
            # The file isn't on disk where we look — ComfyUI was pointed at a
            # custom output path, or none is configured. Fetch it over the /view
            # API instead (path-independent, like other ComfyUI front-ends). #2
            from ..utils.comfyui import fetch_output_image_bytes
            data = fetch_output_image_bytes(filename)
            if data:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, 'wb') as f:
                    f.write(data)
            else:
                img.status = 'failed'
                img.fail_reason = ('The finished image could not be retrieved from ComfyUI '
                                   '(not on disk, and the /view API fetch failed).')
                logger.warning(f"dataset link: file not on disk and /view API fetch failed (job {job_id})")
        # Only a genuine success (not the late-turned-'failed' /view fetch miss
        # right above) counts as fresh, unopened content.
        if img.status != 'failed':
            img.unseen = True
        # A user may have marked this in-flight improvement Keep while waiting.
        # Only the freshly linked, on-disk result may now replace its parent;
        # the helper also preserves a later explicit return to Pending.
        _unkeep_parent_for_kept_improvement(img)
    db.session.commit()
    # This job just left the in-flight set: reconcile the Klein 'generate'
    # indicator (clears it when this was the last job of the batch). Guarded — a
    # bookkeeping hiccup must never break completion linking; the TTL is the net.
    try:
        _sync_generate_activity(img.dataset_id)
    except Exception:
        logger.exception(f"dataset link: generate-activity sync failed for job {job_id}")


# --- Migration helper (run once manually after deploy) ---------------------
def migrate_existing_images_to_per_dataset():
    """Migration helper - run once manually after deploy. Not called automatically."""
    counts = {'moved': 0, 'skipped': 0, 'missing': 0}
    output_dir = _comfy_output_dir()
    if output_dir is None:
        return counts
    datasets = FaceDataset.query.all()
    for ds in datasets:
        if ds.ref_filename:
            src = os.path.join(output_dir, ds.ref_filename)
            dst = os.path.join(_dataset_dir(ds.id), ds.ref_filename)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)
                counts['moved'] += 1
            elif os.path.exists(dst):
                counts['skipped'] += 1
            else:
                counts['missing'] += 1
        for img in FaceDatasetImage.query.filter_by(dataset_id=ds.id).all():
            if not img.filename:  # pending/failed rows without a file
                continue
            src = os.path.join(output_dir, img.filename)
            dst = os.path.join(_dataset_dir(img.dataset_id), img.filename)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)
                counts['moved'] += 1
            elif os.path.exists(dst):
                counts['skipped'] += 1
            else:
                counts['missing'] += 1
    return counts


# --- Export ----------------------------------------------------------------
_TRAIN_FAMILY_LABELS = {
    'zimage': 'Z-Image',
    'krea': 'Krea 2',
    'flux2klein': 'FLUX.2 Klein',
    'flux': 'FLUX.1',
    'sdxl': 'SDXL',
    'anima': 'Anima',
}


def _dataset_info(ds, n, composition) -> str:
    """Factual, family/kind-aware README without stale tuning advice."""
    family = (getattr(ds, 'train_type', None) or 'zimage').lower()
    kind = (getattr(ds, 'kind', None) or 'character').lower()
    lines = [
        '# LoRA Dataset Studio export',
        '',
        f'Dataset kind: {kind}',
        f'Training family: {_TRAIN_FAMILY_LABELS.get(family, family)}',
        f'Images: {n}',
        f'Composition: {composition}',
        '',
    ]
    if kind == 'style':
        lines.extend([
            'Activation: always-on Style (no trigger token).',
            'Captions describe visible content only; the aesthetic is omitted.',
        ])
    else:
        lines.extend([
            f'Activation token: {ds.trigger_word}',
            'Caption sidecars already include this token.',
        ])
    return '\n'.join(lines) + '\n'


def style_content_caption(ds, caption) -> str:
    """Return a Style caption without a legacy leading internal identifier.

    New captions are already content-only. This final seam also repairs sidecars
    generated by older LDS releases (``trigger, content``) without deleting an
    ordinary content word that merely happens to equal the id: only an exact id or
    an id followed by explicit caption punctuation is stripped.
    """
    cap = (caption or '').strip()
    if not is_style(ds):
        return cap
    trigger = (getattr(ds, 'trigger_word', None) or '').strip()
    if not trigger:
        return cap
    if cap.strip(' .!?:;,').strip().casefold() == trigger.casefold():
        return ''
    return re.sub(
        rf'^{re.escape(trigger)}\s*[,;:.!?]\s*', '', cap,
        count=1, flags=re.IGNORECASE).strip()


def _export_caption(ds, caption) -> str:
    """The exact text a trainer reads for one image: the dataset trigger prepended
    to the stored caption for character/concept datasets. A style LoRA is always-on:
    its sidecars contain CONTENT ONLY, with no hidden activation token. Single source
    of truth shared by the ZIP export and write_caption_files, so on-disk .txt
    sidecars always match what the ZIP would contain."""
    cap = style_content_caption(ds, caption)
    if is_style(ds):
        return cap
    return f"{ds.trigger_word}, {cap}" if cap else ds.trigger_word


def write_export_zip(user_id: int, dataset_id: int, output: BinaryIO) -> None:
    """Training-ready ZIP in the PUBLIC-TOOL layout, not an app-internal format:
    one `10_<trigger>/` folder of `image.png` + same-stem `image.txt` caption
    pairs (captions carry the resolved trigger inline, except always-on Style
    datasets whose sidecars are content-only). That single shape feeds
    every mainstream trainer as-is: ai-toolkit (point the dataset at the folder;
    the folder name is ignored), kohya_ss / sd-scripts (drop under img/ — the
    `10_` prefix IS kohya's repeats convention), OneTrainer & friends (image+txt
    pairs). The info file is .md so no caption-scanner ever picks it up."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    kept = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep')
            .order_by(FaceDatasetImage.id.asc()).all())
    if not kept:
        raise ValueError('no kept images to export')
    safe = ''.join(c for c in ds.name if c.isalnum() or c in ('-', '_')) or 'dataset'
    safe_trigger = ''.join(c for c in ds.trigger_word if c.isalnum() or c in ('-', '_')) or 'lora'
    folder = f"10_{safe_trigger}"
    comp = {'face': 0, 'bust': 0, 'body': 0, 'back': 0}
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Garder la PHOTO RÉELLE de référence dans le set : les datasets 100 %
        # synthétiques dérivent de la distribution réelle (deep-research 2026-06-14).
        # On l'inclut comme ancre réelle (_000), caption = trigger seul.
        ref_path = _ref_path(ds) if ds.ref_filename else ''
        # The reference row has no content caption. Exporting it for a Style set
        # would force either a blank sidecar or the internal run identifier into
        # training, both of which violate the always-on/content-only contract.
        if ref_path and os.path.exists(ref_path) and not is_style(ds):
            try:
                rpng = io.BytesIO()
                with Image.open(ref_path) as source:
                    ImageOps.exif_transpose(source).convert('RGB').save(rpng, 'PNG')
                zf.writestr(f"{folder}/{safe}_000_ref.png", rpng.getvalue())
                zf.writestr(f"{folder}/{safe}_000_ref.txt", ds.trigger_word)
            except OSError:
                pass
        for n, img in enumerate(kept, 1):
            path = _img_path(img) if img.filename else ''
            if not img.filename or not os.path.exists(path):
                continue
            png = io.BytesIO()
            with Image.open(path) as source:
                ImageOps.exif_transpose(source).convert('RGB').save(png, 'PNG')
            base = f"{folder}/{safe}_{n:03d}"
            zf.writestr(f"{base}.png", png.getvalue())
            zf.writestr(f"{base}.txt", _export_caption(ds, img.caption))
            if img.framing in comp:
                comp[img.framing] += 1
        zf.writestr(f"{folder}/_dataset_info.md",
                    _dataset_info(ds, len(kept), comp))


def build_export_zip(user_id: int, dataset_id: int) -> bytes:
    """Compatibility wrapper for callers that still need an in-memory archive."""
    output = io.BytesIO()
    write_export_zip(user_id, dataset_id, output)
    return output.getvalue()


def write_caption_files(user_id, dataset_id) -> dict:
    """Write a kohya/ai-toolkit-style `<image>.txt` sidecar NEXT TO each kept
    captioned image in the dataset folder (data/datasets/<id>/) — same caption
    text as the ZIP export (trigger prepended except for content-only Style), for
    tools that read the folder directly instead of downloading the ZIP. Overwrites
    existing .txt files (it's a resync after re-captioning/edits); kept images
    without a caption are counted, not written — they'd get only a bare trigger
    (character/concept) or an empty Style sidecar, so caption them first. Returns
    {'ok', 'written', 'skipped_uncaptioned'}."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    kept = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep')
            .order_by(FaceDatasetImage.id.asc()).all())
    written = skipped_uncaptioned = removed_stale = 0
    for img in kept:
        if not img.filename or not os.path.exists(_img_path(img)):
            continue                       # nothing on disk to sit next to
        stem = os.path.splitext(os.path.basename(img.filename))[0]
        sidecar = os.path.join(_dataset_dir(dataset_id), f'{stem}.txt')
        if not (img.caption or '').strip():
            if os.path.isfile(sidecar):
                os.remove(sidecar)
                removed_stale += 1
            skipped_uncaptioned += 1
            continue
        body = _export_caption(ds, img.caption)
        if not body:                      # legacy Style caption = internal id only
            if os.path.isfile(sidecar):
                os.remove(sidecar)
                removed_stale += 1
            skipped_uncaptioned += 1
            continue
        with open(sidecar, 'w', encoding='utf-8') as fh:
            fh.write(body)
        written += 1
    return {'ok': True, 'written': written,
            'skipped_uncaptioned': skipped_uncaptioned,
            'removed_stale': removed_stale}


# --- Re-exports: reference_edit_service.py (Phase 6 file split, 2026-08) ----
from .reference_edit_service import (
    LOCAL_EDIT_REF_SUPPORT, crop_reference, recrop_reference_auto,
    normalize_edit_engines, start_reference_edit, link_completed_reference_edit,
    keep_reference_edit, discard_reference_edit, reference_mutation,
    invalidate_reference_edit, commit_edited_reference,
    _ref_crop_source_path, _edit_engine_call, _preflight_local_reference_edit,
    _enqueue_local_reference_edit, _cancel_local_edit_job,
    _finish_reference_edit_activity, _validated_comfy_output_name,
    _resolve_comfy_output, _comfy_output_path, _is_reparse_stat, _safe_lstat,
    _read_comfy_output, _drop_comfy_output, _run_reference_edit,
    _clear_reference_edit, _commit_edited_reference_locked,
)

# --- Re-exports: dataset_generation_service.py (Phase 5 file split, 2026-08) -
# dataset_import_service needs `_sync_generate_activity` from here and imports
# it from that module directly, so these blocks stay order-independent.
from .dataset_generation_service import (
    IMPROVE_ENGINES, API_ENGINES, LOCAL_ENGINES, KNOWN_ENGINES, KREA_ENGINE,
    LOCAL_ENGINE_LABELS, API_ENGINE_LABELS, _ENGINE_FILE_TAG,
    REIMPROVE_PARENT_GONE, REIMPROVE_SOURCE_FILE_GONE, REIMPROVE_IN_FLIGHT,
    REIMPROVE_STATE_CHANGED, IMPROVE_SLOT_POLL_SECONDS,
    IMPROVE_SLOT_TIMEOUT_SECONDS, _IMPROVE_ID_CHUNK,
    _EMPTY_MSG, _QUOTA_MSG, _LOST_MSG,
    generate_variations, generate_variations_krea, generate_variations_nanobanana,
    regenerate_image, face_swap_image, resolve_improve_engine,
    improve_existing_image, reimprove_image, bulk_improve_eligible_ids,
    start_bulk_improve, engine_labels, editable_engines, edit_engine_choice_message,
    _sync_generate_activity, _improve_prompt, _improve_candidate_label,
    _improve_extra_metadata, _improve_preflight, _enqueue_improve,
    _improve_existing_image_locked, _reimprove_image_locked, _improve_in_flight,
    _drain_improve_queue, _api_generate_fn, _run_nanobanana_batch,
)

# --- Re-exports: dataset_import_service.py (Phase 4 file split, 2026-08) ----
# Order-independent, like every block here: a split module that needs a name
# from ANOTHER split module imports it from the owner directly, never through
# this re-export. Routing those borrows via the parent is what would make the
# order of these blocks load-bearing.
from .dataset_import_service import (
    IMPORT_MAX_SIDE_CEILING, PRESERVED_IMPORT_MAX_SIDE, PRESERVED_IMPORT_MAX_PIXELS,
    DATASET_ZIP_MAX_FILES, DATASET_ZIP_MAX_BYTES, DATASET_ZIP_MAX_IMAGE_BYTES,
    SCRAPE_IMPORT_MAX, SCRAPE_IMPORT_MIN_SIDE, SCRAPE_IMPORT_MAX_RATIO,
    SCRAPE_DHASH_MAX_DISTANCE, BANK_ANALYSIS_MAX_SIDE, BANK_ANALYSIS_MAX_PIXELS,
    _IMPORT_ENCODINGS, _PRESERVED_IMPORT_EXTENSIONS, _WATERMARK_BBOX_MARGIN,
    _DATASET_ZIP_IMG_EXTS, _SCRAPE_DL_TYPES, _SCRAPE_DL_MAX_BYTES, _SCRAPE_DL_WORKERS,
    import_encode_policy, import_store_image, import_encode, import_images,
    import_dataset_zip, import_dataset_folder, scrape_import_urls,
    detect_head_bbox, detect_watermark_bbox, classify_images,
    face_crop_to_square_webp, bank_deterministic_analysis,
    _validate_import_header_dimensions, _import_header_dimensions,
    _preserved_import_header_extension, _preserved_import_extension,
    _load_import_derivative_image, _parse_watermark_bbox, _merge_training_images,
    _dhash, _hamming, _existing_dhash_rows, _existing_dhashes,
    _bank_analysis_dimensions_allowed, _loaded_bank_deterministic_analysis,
    _bank_deterministic_values, _attach_bank_provenance, _accept_scrape_bytes,
    _scrape_resolution_key, _save_small_scrape_pair, _download_scrape_item,
    _parse_classify,
)

# --- Re-exports: reference_photos_service.py (Phase 1 file split, 2026-08) ---
# MUST stay at the bottom of this file. reference_photos_service.py imports
# primitives (get_dataset, write_image_atomic, face_crop_to_square_webp, ...)
# FROM this module at its own top; if this import ran any earlier, some of
# those primitives wouldn't be defined yet and the import would fail. See
# "Why the import goes at the bottom" in
# docs/superpowers/plans/2026-08-03-split-face-dataset-service-phase1.md.
# Every name here is either called by code still in THIS file, or accessed
# externally as `svc.<name>` by routes/tests (grep found both usages) — both
# need it present on this module, not just on reference_photos_service.
from .reference_photos_service import (
    MAX_EXTRA_REFS, EXTERNAL_REFERENCE_MAX_BYTES, EXTERNAL_REFERENCE_MAX_SIDE,
    extra_ref_filenames, sanitize_external_reference, _all_ref_bytes, _extra_ref_paths,
    extra_ref_original_name, extra_ref_crop_source,
    add_extra_ref, crop_extra_ref, remove_extra_ref,
    POSE_SLOT_KEYS, POSE_SLOT_ACTIVE_KEYS, pose_slot_rows, enabled_pose_slot_paths,
    set_pose_slot, crop_pose_slot, mirror_pose_slot, set_pose_slot_enabled, remove_pose_slot,
)

# --- Re-exports: watermark_service.py (Phase 2 file split, 2026-08) ---------
# MUST stay at the bottom, same reason as the Phase 1 block above. Every name
# here is either called by code still in THIS file or reached as
# `svc.<name>` from routes/tests/image_bank_service -- including the private
# ones, which four test modules and image_bank_service address directly.
from .watermark_service import (
    WATERMARK_BORDER_BAND, WATERMARK_MAX_INPAINT_AREA, WATERMARK_MIN_SIDE,
    WATERMARK_REGION_LIMIT, WATERMARK_REGION_MIN_SIDE,
    normalize_watermark_regions, set_watermark_regions, _route_watermark,
    _preserve_original, _stage_oriented_watermark_edit,
    _promote_staged_watermark_edit, _discard_staged_watermark_edit,
    _apply_watermark_crop, detect_watermarks, dismiss_watermarks,
    _clean_inpaint_engine, clean_watermarks, restore_watermark_original,
)

# --- Re-exports: captioning_service.py (Phase 3 file split, 2026-08) --------
# MUST stay at the bottom, same reason as the Phase 1/2 blocks above. Private
# names included on purpose: test_concept_dataset and test_concept_caption_omission
# address five of them directly.
from .captioning_service import (
    vocabulary_instruction, caption_images, caption_paths,
    preview_caption, derive_short_captions,
    _refine_output_ok, _usable_caption, _fallback_concept_terms, _concept_terms_re,
    _scrub_concept_clauses, _parse_terms_json, _get_concept_terms,
    _enforce_concept_omission, _caption_concept, _compose_preview_instructions,
    _shorten_prompt, _scrub_short_like_long,
    _REFINE_REASONING_RE, _MIN_CONCEPT_CAPTION_CHARS, _TERMS_STOP,
    _CAPTURE_TRIGGERS, _CAPTURE_LEXICON, _SHORTEN_BASE,
)
