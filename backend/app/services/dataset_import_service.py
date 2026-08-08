"""Everything that brings images INTO a dataset: the import encode policy
(resolution/format resolved from Settings), the single-image and batch import
paths, the ZIP / folder import of an existing training set, direct scrape
import, and the Qwen3-VL passes that classify a freshly imported image or find
its head / watermark bounding box.

Split out of face_dataset_service.py (2026-08, Phase 4 of a multi-phase file
split) -- pure move, no behavior change.
"""
import io
import json
import lzma
import math
import os
from pathlib import Path
import re
import stat
import time
import uuid
import warnings
import zipfile
import zlib
from functools import wraps
from typing import BinaryIO

from PIL import Image, ImageOps, UnidentifiedImageError

from ..extensions import db
from ..models import FaceDatasetImage
from .. import config as cfg
from . import bank_transfer_metadata, caption_origin, dataset_activity, image_encoding
from .image_provenance import provenance_metrics
from .image_quality import ANALYSIS_MAX_SIDE, quality_metrics
# Prompts straight from face_variations: it has no dependency on this module.
from .face_variations import CLASSIFY_PROMPT, HEAD_BBOX_PROMPT, WATERMARK_BBOX_PROMPT


def _serialize_dataset_ingest(fn):
    """Late-bound twin of face_dataset_service._serialize_dataset_ingest.

    A DECORATOR runs while this module's body executes, before the borrow-back
    import at the bottom has bound anything, so the real decorator cannot be a
    borrowed name. This resolves it at CALL time and delegates rather than
    restating the locking, so there is one implementation. Safe because that
    decorator is stateless: it takes the per-(user, dataset) lock when the
    wrapped function runs and holds nothing between calls.
    """
    @wraps(fn)
    def wrapped(*args, **kwargs):
        from . import face_dataset_service
        return face_dataset_service._serialize_dataset_ingest(fn)(*args, **kwargs)
    return wrapped


# --- Import resolution & encoding (Settings ▸ Captioning & quality) ----------
# Hard ceiling on a *normalised WebP* dataset image, in px. MEASURED, not chosen:
# Pillow's WebP encoder raises "Image size exceeds WebP limit of 16383 pixels" past
# that side. It applies only to the opt-in normalisation modes; preserving a source
# file never re-encodes it just to meet a WebP implementation limit.
# NOT the input budget: this one bounds what a WebP normalisation mode WRITES,
# and it stays put when the user raises the ingress budget. Conflating the two
# would let "accept a 24 000 px panorama" silently become "write a 24 000 px
# WebP derivative", which the encoder cannot do anyway.
IMPORT_MAX_SIDE_CEILING = 8192
_IMPORT_ENCODINGS = {                       # label -> storage policy
    'preserve': {'preserve': True, 'quality': None, 'lossless': False},
    'standard': {'preserve': False, 'quality': 92, 'lossless': False},
    'high': {'preserve': False, 'quality': 100, 'lossless': False},
    'lossless': {'preserve': False, 'quality': 100, 'lossless': True},
}

# Only static formats both the dataset tools and the trainer's disposable PNG
# staging pass can read. The extension comes from decoded CONTENT, never from the
# upload name: an `image.jpg` carrying PNG bytes must not leave a lying filename.
_PRESERVED_IMPORT_EXTENSIONS = {
    'JPEG': '.jpg',
    'PNG': '.png',
    'WEBP': '.webp',
    'BMP': '.bmp',
}

# A raw master is intentionally NOT resized on import, but importing it must not
# turn the process into an unbounded decompressor. These limits apply uniformly to
# every image ingress path (preserve, crop, explicit normalisation, ZIP and scrape)
# and are checked from Pillow's header before ``load()``. They are a SETTING now
# (`image_input.*`, default 64 Mi-pixels / 16384 px per side, 0 = no limit), so
# they are read through a function: a module-level snapshot taken at import would
# freeze the first value the process ever saw. The budget is a memory guard, not
# an encoder limit; an accepted preserved image remains byte-for-byte untouched.


def preserved_import_limits() -> tuple[int, int]:
    """The effective (max_side, max_pixels) ingress budget; 0 = no limit."""
    return image_encoding.input_budget()


def import_encode_policy() -> dict:
    """What an imported image will ACTUALLY be stored as, resolved once so the
    UI, the toast and the encoder all quote the same policy.

    Total by construction: an unusable configured value logs and degrades to the
    shipped default rather than breaking every import. `capped` is True when a
    WebP-normalisation mode asked for more than that format allows. In `preserve`
    mode the value is retained for a future explicit normalisation choice, but has
    no effect on the stored source bytes."""
    defaults = cfg.DEFAULTS['dataset_import']
    raw_side = cfg.get('dataset_import.max_side', defaults['max_side'])
    try:
        max_side = int(raw_side)
        if max_side < 0:
            raise ValueError(raw_side)
    except (TypeError, ValueError):
        logger.warning('ignoring unusable dataset_import.max_side %r', raw_side)
        max_side = int(defaults['max_side'])
    capped = max_side > IMPORT_MAX_SIDE_CEILING
    if capped:
        max_side = IMPORT_MAX_SIDE_CEILING
    encoding = str(cfg.get('dataset_import.encoding', defaults['encoding']) or '')
    if encoding not in _IMPORT_ENCODINGS:
        if encoding:
            logger.warning('ignoring unusable dataset_import.encoding %r', encoding)
        encoding = defaults['encoding']
    policy = _IMPORT_ENCODINGS[encoding]
    input_max_side, input_max_pixels = preserved_import_limits()
    # A preserved image is never sent through a WebP encoder, so a WebP ceiling
    # cannot cap it. Keep the resolved max_side in the payload so switching
    # back to a normalising mode remains predictable.
    return {'max_side': max_side, 'encoding': encoding,
            'capped': capped and not policy['preserve'],
            'ceiling': IMPORT_MAX_SIDE_CEILING,
            # Explicit names for the ingress safety budget. The older
            # `preserve_*` aliases stay below for clients released while this
            # policy only described raw-preserve imports.
            'input_max_side': input_max_side,
            'input_max_pixels': input_max_pixels,
            'preserve_max_side': input_max_side,
            'preserve_max_pixels': input_max_pixels,
            **policy}


def _validate_import_header_dimensions(im: Image.Image, *, label: str) -> None:
    """Reject an unsafe raster header before any caller asks Pillow to decode it."""
    image_encoding.validate_input_header_dimensions(im, label=label)


def _import_header_dimensions(image_bytes: bytes, *, label: str = 'import') -> tuple[int, int]:
    """Read bounded image dimensions without ever asking Pillow for pixel data.

    This is the common ingress check for the small-image warning, scrape sorting,
    crop and explicit normalisation paths.  It is deliberately separate from the
    preserve validator: those paths additionally require a static, supported
    container, while this helper is only about stopping an unsafe raster header
    before any caller decides to call ``load()``.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as im:
                _validate_import_header_dimensions(im, label=label)
                return im.size
    except ValueError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError(f'{label} rejected an unsafe image header') from exc
    except (OSError, UnidentifiedImageError, MemoryError) as exc:
        raise ValueError(f'{label} received an unreadable image') from exc


def _preserved_import_header_extension(im: Image.Image, *, label: str = 'preserve mode') -> str:
    """Validate a raw static source header and return its content extension.

    This is deliberately called before ``im.load()``.  A file can advertise a
    huge raster in a very small compressed payload, so dimensions must be
    rejected before the decoder allocates its full pixel buffer.
    """
    fmt = (getattr(im, 'format', None) or '').upper()
    ext = _PRESERVED_IMPORT_EXTENSIONS.get(fmt)
    if ext is None:
        raise ValueError(
            f'{label} supports only static JPEG, PNG, WebP, or BMP images '
            f'(got {fmt or "unknown"})')
    if getattr(im, 'n_frames', 1) != 1:
        raise ValueError(
            f'{label} supports only static JPEG, PNG, WebP, or BMP images '
            '(animated images are not supported)')
    _validate_import_header_dimensions(im, label=label)
    return ext


def _preserved_import_extension(image_bytes: bytes, *, label: str = 'preserve mode') -> str:
    """Validate a raw static source and return its canonical content extension.

    Preserving bytes must not mean accepting arbitrary browser/media formats the
    rest of the dataset pipeline cannot safely edit, serve or train from. GIF,
    TIFF, AVIF and animated WebP are deliberately refused here instead of being
    silently flattened or saved under a made-up extension.  Header dimensions
    are checked before ``load()`` under a local Pillow bomb-warning policy; no
    process-global warning filter is ever changed.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as im:
                ext = _preserved_import_header_extension(im, label=label)
                # Fully decode only after the explicit header budget passed. PIL can
                # identify a truncated file from the header alone; `load` enforces the
                # same readability guarantee the old normalisation path provided.
                im.load()
    except ValueError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError(f'{label} rejected an unsafe image header') from exc
    except (OSError, UnidentifiedImageError, MemoryError) as exc:
        raise ValueError(f'{label} received an unreadable image') from exc
    return ext


def _load_import_derivative_image(image_bytes: bytes) -> Image.Image:
    """Decode a bounded, visually upright temporary image for derived imports.

    Preserve mode calls its stricter static-format validator above.  Crop and
    explicit WebP-normalisation use this shared geometry guard so they cannot
    decode a compressed bomb merely because they are allowed to derive pixels.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as opened:
                _validate_import_header_dimensions(opened, label='import')
                opened.load()
                return ImageOps.exif_transpose(opened).copy()
    except ValueError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('import rejected an unsafe image header') from exc
    except (OSError, UnidentifiedImageError, MemoryError) as exc:
        raise ValueError('import received an unreadable image') from exc


def import_store_image(image_bytes: bytes) -> tuple[bytes, str]:
    """Return exactly what a non-cropped import should store and its extension.

    `preserve` keeps approved source bytes byte-for-byte. The three legacy
    encoding modes intentionally still create WebP derivatives, preserving their
    historical resizing and quality controls for users who explicitly choose them.
    """
    p = import_encode_policy()
    if p['preserve']:
        return image_bytes, _preserved_import_extension(image_bytes)
    return (normalize_to_webp(image_bytes, size=p['max_side'],
                              quality=p['quality'], lossless=p['lossless']),
            '.webp')


def import_encode(image_bytes: bytes) -> bytes:
    """Backward-compatible bytes-only view of :func:`import_store_image`.

    New ingest lanes need the true extension as well and use
    :func:`import_store_image` directly. Generated images and API transport
    copies deliberately keep their own fixed sizes: this policy is about what the
    user hands in, not about what the app produces.
    """
    return import_store_image(image_bytes)[0]


def detect_head_bbox(image_bytes):
    """Return normalized (x1, y1, x2, y2) of the main head via Qwen3-VL, or None.

    None also covers Ollama being unreachable/misconfigured (describe_image_ollama
    never raises) -- the caller (face_crop_to_square_webp) already treats "no
    detection" as a normal case and falls back to a centered crop, so uploads
    keep working (degraded but functional)."""
    try:
        from .vision_ollama import describe_image_ollama
    except ImportError:
        return None
    # fmt='json' forces Ollama's grammar mode: the model must emit a JSON object from
    # the first token, so reasoning-prone (abliterated) checkpoints can't ramble a
    # <think> trace past num_predict and never reach the coords (a silent-None cause).
    #
    # keep_alive is decided by CONTENTION, not by this call site (see
    # services/vision_keepalive.py). This is the burst case the policy exists for:
    # cropping five references in a row used to pay the 12.8 s cold load five times
    # because each upload is its own isolated call. When the card is contended — or
    # when the signal can't be read — the policy returns 0 and nothing changes.
    from .vision_keepalive import keep_alive_for_isolated_call
    raw = describe_image_ollama(image_bytes, HEAD_BBOX_PROMPT, num_predict=400,
                                prefer_json=True, fmt='json',
                                keep_alive=keep_alive_for_isolated_call())
    try:
        s = raw.index('{')
        obj = json.loads(raw[s:raw.index('}', s) + 1])
        y1, x1, y2, x2 = (float(obj[k]) for k in ('y1', 'x1', 'y2', 'x2'))
    except (ValueError, KeyError, AttributeError, TypeError):
        return None
    # Qwen3-VL frequently SWAPS corners (returns y1>y2 or x1>x2). Normalize to
    # min/max instead of rejecting — rejecting was a silent-None cause that fell back
    # to a body-centered crop even when the head was correctly located.
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    return (x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0)


# Marge d'elargissement de la bbox watermark (fraction du cote). Les bbox VLM sont
# GROSSIERES et souvent trop serrees : sans marge, le crop/inpaint laisse un lisere du
# watermark. 2.5% de chaque cote = filet de securite sans engloutir le sujet.
_WATERMARK_BBOX_MARGIN = 0.025


def _parse_watermark_bbox(raw):
    """PURE parser for a WATERMARK_BBOX_PROMPT answer. Returns a MARGIN-EXPANDED
    normalized (x1,y1,x2,y2) in [0,1], or None (no watermark / unparseable). Split out
    from the vision call so the batch can tell an EMPTY vision output (Ollama down ->
    leave the state untouched) apart from a clean 'present:false' answer (-> 'none').

    Same bbox handling as detect_head_bbox: 0-1000 grid, swapped corners normalized to
    min/max. A `present:false` (or a missing/invalid box) -> None. VLM boxes run tight,
    so we pad by _WATERMARK_BBOX_MARGIN and clamp -- the router needs the whole mark."""
    try:
        s = raw.index('{')
        obj = json.loads(raw[s:raw.index('}', s) + 1])
    except (ValueError, AttributeError, TypeError):
        return None
    if 'present' in obj and not obj.get('present'):
        return None
    try:
        y1, x1, y2, x2 = (float(obj[k]) for k in ('y1', 'x1', 'y2', 'x2'))
    except (KeyError, TypeError, ValueError):
        return None
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    m = _WATERMARK_BBOX_MARGIN
    return (max(0.0, x1 / 1000.0 - m), max(0.0, y1 / 1000.0 - m),
            min(1.0, x2 / 1000.0 + m), min(1.0, y2 / 1000.0 + m))


def detect_watermark_bbox(image_bytes, *, keep_alive=0):
    """Return normalized (x1, y1, x2, y2) of an OVERLAID watermark via Qwen3-VL, or
    None (no overlaid watermark, or the model is unreachable / the JSON won't parse).
    fmt='json' forces Ollama's grammar mode, same as detect_head_bbox.

    The prompt targets watermark/logo/URL/username text ADDED ON TOP of the photo, NOT
    scene text (signs, clothing prints) -- see WATERMARK_BBOX_PROMPT. Box is margin-
    expanded (see _parse_watermark_bbox). `keep_alive` mirrors describe_image_ollama:
    0 unloads after this call; a batch passes a duration and unloads at the end."""
    try:
        from .vision_ollama import describe_image_ollama
    except ImportError:
        return None
    raw = describe_image_ollama(image_bytes, WATERMARK_BBOX_PROMPT, num_predict=400,
                                prefer_json=True, fmt='json', keep_alive=keep_alive)
    return _parse_watermark_bbox(raw)


def face_crop_to_square_webp(image_bytes: bytes, size: int = 1024, pad: float = 1.7,
                             *, return_detected: bool = False, use_vision: bool = True,
                             return_scale: bool = False):
    """Head-crop (Qwen3-VL bbox, generous padding for hair + shoulders) into a
    SQUARE that FILLS `size` - no black padding, no distortion (the square is
    shrunk to fit inside the image so it never needs letterboxing). Falls back to
    a centered-square crop if no head is detected. CALLER holds the GPU window.

    `return_detected=True` -> (webp_bytes, head_detected) so the caller can WARN the
    user when it silently fell back to a centered crop (e.g. vision model not pulled)
    instead of leaving them puzzled by a body-centered reference.

    `return_scale=True` -> also returns the upscale ratio applied to reach `size`
    (>1 means the detected/fallback box was smaller than `size` and got LANCZOS-
    enlarged — see UPSCALE_WARN_THRESHOLD). Additive and independent from
    `return_detected` so existing 2-tuple callers (the /ref route) are unaffected.

    `use_vision=False` -> skip the bbox detection entirely (fast pure-PIL centered
    square, no GPU window needed) — the manual-first reference flow.

    INGEST, not an edit: this runs once on the bytes being IMPORTED, and its name is
    part of its contract (callers write the result to a `.webp`). Re-cropping that
    reference afterwards goes through `_crop_resize_file`, which does preserve the
    format losslessly."""
    # The VLM sees an upright transport derivative. Work in that same visual
    # coordinate space so its normalized head box lands on the visible subject.
    im = _load_import_derivative_image(image_bytes).convert('RGB')
    W, H = im.size
    norm = detect_head_bbox(image_bytes) if use_vision else None
    half = 0
    if norm:
        x1, y1, x2, y2 = norm[0] * W, norm[1] * H, norm[2] * W, norm[3] * H
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2 - (y2 - y1) * 0.10  # shift up to keep the hair
        half = max(x2 - x1, y2 - y1) * pad / 2
        half = min(half, cx, W - cx, cy, H - cy)  # keep the square inside the image
    head_detected = half >= 8
    if head_detected:
        box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    else:  # no/failed detection → centered largest square
        side = min(W, H)
        left, top = (W - side) // 2, (H - side) // 2
        box = (left, top, left + side, top + side)
    box_side = max(1, box[2] - box[0])
    scale = size / box_side
    out = io.BytesIO()
    im.crop(box).resize((size, size), Image.LANCZOS).save(out, 'WEBP', quality=92)
    webp = out.getvalue()
    if return_detected and return_scale:
        return webp, head_detected, scale
    if return_detected:
        return webp, head_detected
    if return_scale:
        return webp, scale
    return webp


# --- Import + classify (Qwen3-VL) ------------------------------------------
@_serialize_dataset_ingest
def import_images(user_id, dataset_id, files_bytes, crop=False, dedupe=False, stats=None,
                  source_metadata=None, captions=None, caption_origins=None,
                  bank_image_ids=None,
                  framings=None, bank_analysis_snapshots=None,
                  watermark_states=None, watermark_bboxes=None,
                  watermark_regions=None, watermark_sources=None,
                  watermark_scores=None, statuses=None,
                  transfer_metadatas=None, dedupe_seen=None,
                  preserve_exact_bytes=False, created_ids_sink=None,
                  provenance_changes_sink=None):
    """Store original static bytes (or head-crop) + create import rows (status=keep).
    When crop=True, each image is auto head-cropped via Qwen3-VL - the CALLER
    must then hold the GPU-exclusive window - and is by construction a face,
    so framing='face' is set directly (no classify pass needed).

    dedupe=True (the /import route) drops perceptual duplicates by dHash — both
    within the batch and vs the dataset's existing files. The hash is computed on
    the final stored image, so a re-import of the same photo matches its earlier
    crop instead of comparing a full frame to a head crop. Skips are counted in
    stats['duplicates'] when a stats dict is passed.
    Default stays False: service-level callers (scrape flow dedupes upstream on
    the ORIGINALS, before paying the crop) keep the historical behavior.

    ``source_metadata`` is an optional list parallel to ``files_bytes``. Only
    validated Pexels or web-search provenance is stored; existing callers can omit it.

    ``captions`` is an optional list parallel to ``files_bytes`` — a pre-existing
    caption to carry onto the new row (the image-bank promotion path passes the bank
    captions here, so a promoted selection starts already captioned). Empty/None entries
    leave the row uncaptioned. A skipped duplicate simply drops its caption with it.

    ``framings`` is an optional list parallel to ``files_bytes`` — a framing
    ALREADY known for the blob (the image-bank promotion path passes the framing
    its own classify pass wrote, so a promoted selection lands counted in the
    composition instead of sitting at 0 until something re-classifies it). Only
    the catalog buckets are accepted; anything else lands as None so the dataset
    classifier can still fill it. Ignored when crop=True (a head crop IS a face).

    ``bank_image_ids`` is an optional list parallel to ``files_bytes`` — the
    bank_image each blob came from, recorded on the new row. A blob dropped as a
    perceptual DUPLICATE hands its bank id to the row it matched (when that row
    carries none yet): the dataset does hold that bank image, just under another
    row, and the bank's "already promoted here" answer must say so. That link is
    what lets the bank re-offer an image once the user deletes it here. Bank ids
    that could NOT be linked (the matched row already belongs to another bank —
    a scalar column can only credit one) are listed in ``stats['bank_unlinked']``.

    ``bank_analysis_snapshots`` is an internal Bank-promotion marker parallel to
    ``files_bytes``.  When present, this importer recalculates deterministic
    quality/provenance from the final Dataset bytes and seals a v3 snapshot with
    their SHA-256.  A byte-identical Bank capture also retains its complete row
    analysis plus path-free Score/Face embeddings in a bounded sidecar; a
    transformed image gets deterministic analysis only.  The regular current
    Dataset fields stay separate and remain user-owned.

    ``dedupe_seen`` is an optional internal mutable cache of ``(dhash, row_id)``
    pairs for chunked imports. When omitted, the importer loads the dataset's
    existing hashes itself, preserving the standalone-call behavior.

    Returns (ids, failed_count)."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return [], 0
    # Sans head-crop, on préserve le ratio ET les octets source autorisés : l'ancien
    # chemin « carré padé » ajoutait des bandes noires que le LoRA apprendrait, et
    # forçait tous les imports personnage en carré — un plan buste/corps importé
    # doit rester tel quel (ai-toolkit gère le bucketing multi-ratios).
    seen = (dedupe_seen if dedupe_seen is not None
            else _existing_dhash_rows(dataset_id)) if dedupe else None
    metadata_by_index = list(source_metadata) if source_metadata is not None else []
    captions_by_index = list(captions) if captions is not None else []
    # Parallel to ``captions`` and travelling WITH it. Without this list a bank
    # caption a human wrote or corrected arrives in the dataset stamped as
    # nothing, i.e. re-writable — the protection would survive exactly one hop.
    caption_origins_by_index = (list(caption_origins)
                                if caption_origins is not None else [])
    bank_ids_by_index = list(bank_image_ids) if bank_image_ids is not None else []
    framings_by_index = list(framings) if framings is not None else []
    snapshots_by_index = (list(bank_analysis_snapshots)
                          if bank_analysis_snapshots is not None else [])
    watermark_states_by_index = list(watermark_states) if watermark_states is not None else []
    watermark_bboxes_by_index = list(watermark_bboxes) if watermark_bboxes is not None else []
    watermark_regions_by_index = list(watermark_regions) if watermark_regions is not None else []
    watermark_sources_by_index = list(watermark_sources) if watermark_sources is not None else []
    watermark_scores_by_index = list(watermark_scores) if watermark_scores is not None else []
    statuses_by_index = list(statuses) if statuses is not None else []
    transfer_metadata_by_index = (list(transfer_metadatas)
                                  if transfer_metadatas is not None else [])

    def bank_id_at(i):
        return bank_ids_by_index[i] if i < len(bank_ids_by_index) else None

    def caption_origin_at(i, cap):
        """The stamp that rides with this caption — validated, never trusted raw.

        An unknown token would be stored and then compared against 'asserted'
        forever without ever matching, which is a protection that silently is
        not one. A caption with no stamp stays NULL: "never recorded".
        """
        if not (cap or '').strip():
            return None
        value = (caption_origins_by_index[i]
                 if i < len(caption_origins_by_index) else None)
        return value if value in caption_origin.VALUES else None

    def framing_at(i):
        # A head crop IS a face by construction; otherwise take the caller's value
        # when it is one of the composition buckets (an 'unknown'/None verdict must
        # stay NULL so the dataset classifier can still pick the row up).
        if crop:
            return 'face'
        fr = framings_by_index[i] if i < len(framings_by_index) else None
        return fr if fr in ('face', 'bust', 'body', 'back') else None

    def snapshot_at(i):
        return snapshots_by_index[i] if i < len(snapshots_by_index) else None

    def watermark_state_at(i):
        state = (watermark_states_by_index[i]
                 if i < len(watermark_states_by_index) else None)
        return state if state in ('none', 'detected', 'dismissed', 'cleaned', 'failed', 'error') else None

    def watermark_bbox_at(i):
        value = (watermark_bboxes_by_index[i]
                 if i < len(watermark_bboxes_by_index) else None)
        return value if isinstance(value, str) else None

    def watermark_regions_at(i):
        value = (watermark_regions_by_index[i]
                 if i < len(watermark_regions_by_index) else None)
        return value if isinstance(value, str) else None

    def watermark_source_at(i):
        value = (watermark_sources_by_index[i]
                 if i < len(watermark_sources_by_index) else None)
        return value if value in ('detector', 'vision') else None

    def watermark_score_at(i):
        value = (watermark_scores_by_index[i]
                 if i < len(watermark_scores_by_index) else None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None

    def status_at(i):
        value = statuses_by_index[i] if i < len(statuses_by_index) else None
        return value if value in ('pending', 'keep', 'reject') else 'keep'

    def transfer_metadata_at(i):
        value = (transfer_metadata_by_index[i]
                 if i < len(transfer_metadata_by_index) else None)
        if value is None:
            return None
        normalized = (
            bank_transfer_metadata.normalized_transfer_metadata_storage(value))
        if normalized is None:
            raise RuntimeError('invalid Bank/Dataset transfer metadata')
        return normalized

    ids = []
    failed = 0
    for index, raw in enumerate(files_bytes):
        # Garde-fou qualité : ai-toolkit ne fait que RÉDUIRE — une image sous
        # 768 px de petit côté reste floue à l'entraînement. Comptée (toast),
        # jamais bloquée : c'est parfois la seule photo disponible.
        if stats is not None:
            try:
                if min(_import_header_dimensions(raw)) < SCRAPE_IMPORT_MIN_SIDE:
                    stats['small'] = stats.get('small', 0) + 1
            except Exception:
                pass
        try:
            if preserve_exact_bytes:
                if crop:
                    raise ValueError('exact-byte Bank promotion cannot crop')
                # This path is a preservation operation, independent of the
                # user's normal Dataset import encoding preference.  Validate
                # the static supported format, then store the identical bytes.
                extension = _preserved_import_extension(
                    raw, label='bank dataset promotion')
                stored, scale = bytes(raw), None
            elif crop:
                stored, scale = face_crop_to_square_webp(raw, return_scale=True)
                extension = '.webp'
            else:
                stored, extension = import_store_image(raw)
                scale = None
        except Exception as e:
            if preserve_exact_bytes:
                raise RuntimeError(
                    f'could not preserve Bank image bytes: {e}') from e
            failed += 1
            logger.warning(f"dataset import: image skipped (dataset {dataset_id}): {e}")
            continue
        final_analysis = None
        captured_analysis = None
        captured_cache_bundle = None
        if snapshot_at(index) is not None:
            final_analysis = bank_deterministic_analysis(stored)
            if final_analysis is None:
                raise RuntimeError('could not seal Bank analysis for Dataset image')
            captured_analysis = (
                snapshot_at(index) if isinstance(snapshot_at(index), dict) else None)
            captured_matches = (
                captured_analysis is not None
                and captured_analysis.get('fingerprint')
                == bank_transfer_metadata.content_fingerprint_bytes(stored))
            if captured_matches and captured_analysis.get('caches'):
                captured_cache_bundle = captured_analysis['caches']

        def seal_analysis_snapshot():
            """Persist the sidecar only when this candidate is about to commit."""
            if final_analysis is None:
                return None, None
            cache_ref = None
            try:
                if captured_cache_bundle:
                    cache_ref = bank_transfer_metadata.write_cache_sidecar(
                        _bank_analysis_cache_dir(dataset_id), captured_cache_bundle)
                    if cache_ref is None:
                        raise RuntimeError(
                            'could not preserve Bank Score/Face cache in Dataset')
                snapshot = bank_transfer_metadata.snapshot_storage(
                    final_analysis, stored, captured=captured_analysis,
                    cache_ref=cache_ref)
                if snapshot is None:
                    raise RuntimeError(
                        'could not seal Bank analysis for Dataset image')
                return snapshot, cache_ref
            except Exception:
                _remove_unreferenced_bank_analysis_cache(dataset_id, cache_ref)
                raise
        fp = None
        if dedupe:
            try:
                with Image.open(io.BytesIO(stored)) as im:
                    fp = _dhash(im)
            except (OSError, ValueError):
                fp = None   # unreadable output would have failed above; belt & braces
            if fp is not None:
                match = None
                stale_ids = set()
                for cached_hash, mid in tuple(seen):
                    if _hamming(fp, cached_hash) > SCRAPE_DHASH_MAX_DISTANCE:
                        continue
                    live = (FaceDatasetImage.query
                            .filter(
                                FaceDatasetImage.id == mid,
                                FaceDatasetImage.dataset_id == dataset_id,
                                FaceDatasetImage.status.in_(('keep', 'pending')))
                            .first())
                    if live is None or not live.filename:
                        stale_ids.add(mid)
                        continue
                    try:
                        live_path = os.path.join(
                            _dataset_dir(dataset_id), live.filename)
                        if (preserve_exact_bytes
                                and Path(live_path).read_bytes() != stored):
                            # Perceptually similar is not byte-identical and
                            # cannot carry this image's exact analysis vault.
                            continue
                        with Image.open(live_path) as im:
                            live_hash = _dhash(im)
                    except (OSError, ValueError):
                        stale_ids.add(mid)
                        continue
                    if live_hash != cached_hash:
                        for cache_index, (_old_hash, cached_id) in enumerate(seen):
                            if cached_id == mid:
                                seen[cache_index] = (live_hash, mid)
                                break
                    if _hamming(fp, live_hash) <= SCRAPE_DHASH_MAX_DISTANCE:
                        match = mid
                        break
                if stale_ids:
                    seen[:] = [
                        (h, mid) for h, mid in seen if mid not in stale_ids
                    ]
                if match is not None:
                    if stats is not None:
                        stats['duplicates'] = stats.get('duplicates', 0) + 1
                    # The dataset already holds this image — hand the provenance to
                    # the row that holds it, so the source can tell it landed. When
                    # that row is already claimed (another bank supplied the same
                    # photo first), report the id back: the caller has no verifiable
                    # trace here and needs to fall back on its own bookkeeping.
                    bid = bank_id_at(index)
                    analysis_snapshot = None
                    analysis_cache_ref = None
                    try:
                        if bid:
                            analysis_snapshot, analysis_cache_ref = (
                                seal_analysis_snapshot())
                        linked_before = db.session.get(FaceDatasetImage, match)
                        previous = ((linked_before.bank_image_id,
                                     linked_before.bank_analysis_snapshot)
                                    if linked_before is not None else None)
                        linked = (bool(bid) and _attach_bank_provenance(
                            match, bid, bank_analysis_snapshot=analysis_snapshot,
                            bank_analysis_cache_dir=_bank_analysis_cache_dir(
                                dataset_id)))
                        linked_after = db.session.get(FaceDatasetImage, match)
                        if (provenance_changes_sink is not None
                                and previous is not None
                                and linked_after is not None
                                and previous != (linked_after.bank_image_id,
                                                linked_after.bank_analysis_snapshot)):
                            provenance_changes_sink.append({
                                'image_id': match,
                                'old_bank_image_id': previous[0],
                                'old_snapshot': previous[1],
                                'new_bank_image_id': linked_after.bank_image_id,
                                'new_snapshot': linked_after.bank_analysis_snapshot,
                            })
                        if bid and not linked and stats is not None:
                            stats.setdefault('bank_unlinked', []).append(bid)
                    except Exception:
                        # `_attach_bank_provenance` commits on success.  A fault
                        # before that commit leaves the session unusable until
                        # rollback; a fault after it is resolved by the durable
                        # ownership proof in `finally` below.
                        db.session.rollback()
                        raise
                    finally:
                        _remove_unreferenced_bank_analysis_cache(
                            dataset_id, analysis_cache_ref)
                    logger.info(f"dataset import: perceptual duplicate skipped (dataset {dataset_id})")
                    continue
        analysis_snapshot, analysis_cache_ref = seal_analysis_snapshot()
        transfer_metadata = transfer_metadata_at(index)
        restored = bank_transfer_metadata.dataset_restore_values(
            transfer_metadata,
            bank_transfer_metadata.content_fingerprint_bytes(stored))
        fn = f"{user_id}_dataset_{uuid.uuid4().hex[:8]}{extension}"
        stored_path = os.path.join(_dataset_dir(dataset_id), fn)
        try:
            write_image_atomic(stored_path, stored)
            cap = (captions_by_index[index] if index < len(captions_by_index) else None)
            cap = _cap_caption(cap) if (cap or '').strip() else None
            restored_short = restored.get('caption_short')
            restored_short = (_cap_caption(restored_short)
                              if isinstance(restored_short, str)
                              and restored_short.strip() else None)
            restored_short_origin = (restored.get('caption_short_origin')
                                     if restored_short else None)
            img = FaceDatasetImage(
                                   dataset_id=dataset_id,
                                   source=restored.get('source') or 'import',
                                   status=status_at(index),
                                   filename=fn, framing=framing_at(index),
                                   variation_label=restored.get('variation_label'),
                                   variation_prompt=restored.get('variation_prompt'),
                                   klein_model=restored.get('klein_model'),
                                   face_score=restored.get('face_score'),
                                   face_state=restored.get('face_state'),
                                   fail_reason=restored.get('fail_reason'),
                                   fail_kind=restored.get('fail_kind'),
                                   upscale_ratio=(restored.get('upscale_ratio')
                                                  if restored.get('upscale_ratio')
                                                  is not None else scale),
                                   caption=cap, caption_short=restored_short,
                                   caption_origin=caption_origin_at(index, cap),
                                   caption_short_origin=restored_short_origin,
                                   bank_image_id=bank_id_at(index),
                                   bank_analysis_snapshot=analysis_snapshot,
                                   transfer_metadata=transfer_metadata,
                                   watermark_state=watermark_state_at(index),
                                   watermark_bbox=watermark_bbox_at(index),
                                   watermark_regions=watermark_regions_at(index),
                                   watermark_source=watermark_source_at(index),
                                   watermark_score=watermark_score_at(index),
                                   source_metadata=_source_metadata_storage(
                                       metadata_by_index[index]
                                       if index < len(metadata_by_index) else None))
            db.session.add(img)
            db.session.commit()
        except Exception:
            db.session.rollback()
            try:
                os.unlink(stored_path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning('dataset import: could not remove uncommitted image %s',
                               stored_path, exc_info=True)
            _remove_unreferenced_bank_analysis_cache(
                dataset_id, analysis_cache_ref)
            raise
        if dedupe and fp is not None:
            seen.append((fp, img.id))
        ids.append(img.id)
        if created_ids_sink is not None:
            created_ids_sink.append(img.id)
    return ids, failed


# --- Import d'un dataset d'entraînement existant (ZIP kohya-style / dossier) --
# Des images + sidecars .txt de même nom (la convention kohya/ai-toolkit), soit
# dans un ZIP uploadé, soit dans un dossier du disque du serveur (app locale
# mono-user : le chemin est SON disque). Les images gardent leur ratio
# (source préservée par défaut, sans crop), les captions atterrissent sur les rows,
# dédup perceptuelle vs le lot ET le dataset. Les fichiers sont réécrits sous
# des noms générés (jamais celui de la source → aucune traversée possible),
# profondeur de dossiers libre (le ZIP accepte toute arborescence ; le dossier
# est parcouru récursivement pour rester aligné).
DATASET_ZIP_MAX_FILES = 400
DATASET_ZIP_MAX_BYTES = 2 * 1024 * 1024 * 1024
DATASET_ZIP_MAX_IMAGE_BYTES = 128 * 1024 * 1024
_DATASET_ZIP_MAX_CENTRAL_DIRECTORY_BYTES = 8 * 1024 * 1024
_DATASET_ZIP_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


@_serialize_dataset_ingest
def _merge_training_images(user_id, dataset_id, entries, captions, stats=None):
    """Cœur commun ZIP/dossier : `entries` = liste de (stem, display_name, getter)
    où `getter()` rend les bytes de l'image, `captions` = {stem: texte}. Chaque
    image lisible devient une row 'import' (status=keep, ratio préservé), la
    caption de même stem est attachée (tronquée à CAPTION_MAX_CHARS), les
    doublons perceptuels (dHash) vs le lot ET le dataset sont sautés — mais leur
    caption, elle, atterrit sur la row déjà présente si celle-ci n'en a pas
    (aller-retour « je légende ailleurs »). Returns (ids, failed)."""
    seen = _existing_dhash_rows(dataset_id)   # [(dhash, image_id)]
    ids, failed = [], 0
    for stem, display, getter in entries:
        try:
            raw = getter()
        except (OSError, ValueError, MemoryError, zipfile.BadZipFile,
                zipfile.LargeZipFile, zlib.error, lzma.LZMAError,
                NotImplementedError, RuntimeError):
            failed += 1
            continue
        if stats is not None:   # même garde qualité que l'import de photos
            try:
                if min(_import_header_dimensions(raw)) < SCRAPE_IMPORT_MIN_SIDE:
                    stats['small'] = stats.get('small', 0) + 1
            except Exception:
                pass
        try:
            stored, extension = import_store_image(raw)
        except Exception as e:
            failed += 1
            logger.warning(f"dataset import: image skipped ({display}): {e}")
            continue
        try:
            with Image.open(io.BytesIO(stored)) as im:
                fp = _dhash(im)
        except (OSError, ValueError):
            fp = None
        incoming = (captions.get(stem) or '').strip() or None
        if fp is not None:
            match = next((mid for h, mid in seen
                          if _hamming(fp, h) <= SCRAPE_DHASH_MAX_DISTANCE), None)
            if match is not None:
                # THE round trip: export the images, caption them in another
                # tool, bring the .txt files back. Those images are duplicates
                # BY DESIGN — dropping the row silently dropped the caption with
                # it ("0 imported · N duplicates skipped"), which made the whole
                # trip a dead end (reported by Qeeyana on Reddit). The pixels are
                # already here; what is new is the text, so the text lands on the
                # row that holds them. A caption written HERE is never
                # overwritten — an import cannot silently rewrite curated work.
                if stats is not None:
                    stats['duplicates'] = stats.get('duplicates', 0) + 1
                row = FaceDatasetImage.query.get(match) if incoming else None
                if row is not None:
                    if (row.caption or '').strip():
                        if stats is not None:
                            stats['captions_kept'] = stats.get('captions_kept', 0) + 1
                    else:
                        # A .txt sidecar is work done by a human in another tool —
                        # the whole point of the round-trip. It lands 'asserted',
                        # which is the same rule the branch above already applies by
                        # hand ("a caption written HERE is never overwritten"),
                        # generalised so a LATER forced pass honours it too.
                        caption_origin.stamp(row, _cap_caption(incoming),
                                             caption_origin.ASSERTED)
                        db.session.commit()
                        if stats is not None:
                            stats['captions_applied'] = \
                                stats.get('captions_applied', 0) + 1
                continue
        fn = f"{user_id}_dsimport_{uuid.uuid4().hex[:8]}{extension}"
        write_image_atomic(os.path.join(_dataset_dir(dataset_id), fn), stored)
        cap = _cap_caption(incoming) if incoming else None
        if cap and stats is not None:
            stats['captions'] = stats.get('captions', 0) + 1
        img = FaceDatasetImage(
            dataset_id=dataset_id, source='import', status='keep', filename=fn,
            caption=cap,
            caption_origin=caption_origin.ASSERTED if cap else None)
        db.session.add(img)
        db.session.commit()
        if fp is not None:
            seen.append((fp, img.id))     # so the rest of the batch dedupes too
        ids.append(img.id)
    return ids, failed


def import_dataset_zip(user_id: int, dataset_id: int,
                       archive: bytes | BinaryIO, stats=None):
    """Import an existing training dataset into THIS dataset (merge, not create):
    every image in the zip becomes an 'import' row (status=keep), a same-stem
    .txt sidecar becomes its caption (truncated to CAPTION_MAX_CHARS). Returns
    (ids, failed). ValueError on a non-zip / oversized archive."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    stream, owned = _coerce_archive_stream(archive)
    try:
        _preflight_zip_central_directory(
            stream, max_entries=DATASET_ZIP_MAX_FILES,
            max_central_bytes=_DATASET_ZIP_MAX_CENTRAL_DIRECTORY_BYTES,
            label='zip')
        try:
            z = zipfile.ZipFile(stream)
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile,
                zlib.error, lzma.LZMAError, NotImplementedError,
                RuntimeError) as exc:
            raise ValueError('not a zip file') from exc
        try:
            infos = [i for i in z.infolist() if not i.is_dir()]
            if len(infos) > DATASET_ZIP_MAX_FILES:
                raise ValueError(
                    f'too many files in the zip (max {DATASET_ZIP_MAX_FILES})')
            if sum(i.file_size for i in infos) > DATASET_ZIP_MAX_BYTES:
                raise ValueError('zip too large (max 2 GB uncompressed)')
            oversized = next((
                i for i in infos
                if i.filename.lower().endswith(_DATASET_ZIP_IMG_EXTS)
                and i.file_size > DATASET_ZIP_MAX_IMAGE_BYTES
            ), None)
            if oversized is not None:
                raise ValueError('image too large in zip (max 128 MiB per image)')
            captions = {}
            for i in infos:
                if i.filename.lower().endswith('.txt') and i.file_size <= 64 * 1024:
                    try:
                        captions[os.path.splitext(i.filename)[0]] = \
                            z.read(i).decode('utf-8', 'replace').strip()
                    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile,
                            zlib.error, lzma.LZMAError, NotImplementedError,
                            RuntimeError):
                        pass
            entries = [
                (os.path.splitext(i.filename)[0], i.filename,
                 lambda i=i: z.read(i))
                for i in infos if i.filename.lower().endswith(_DATASET_ZIP_IMG_EXTS)
            ]
            return _merge_training_images(
                user_id, dataset_id, entries, captions, stats=stats)
        finally:
            z.close()
    finally:
        if owned is not None:
            owned.close()


def import_dataset_folder(user_id, dataset_id, folder, stats=None):
    """Same merge as import_dataset_zip but straight from a folder on the
    server's disk — no need to zip an existing kohya dataset first. Recursive
    (the zip accepts any folder depth, the folder walk mirrors that); non-image
    files are ignored, same-stem .txt sidecars become captions. Returns
    (ids, failed). ValueError on a missing folder / oversized content."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    # Windows «Copier en tant que chemin» colle le chemin entre guillemets —
    # on les retire pour que le coller-direct marche du premier coup.
    folder = (folder or '').strip().strip('"\'')
    if not folder or not os.path.isdir(folder):
        raise ValueError(f'folder not found or not readable: {folder or "(empty)"}')
    paths = []
    for root, _dirs, files in os.walk(folder):
        paths.extend(os.path.join(root, f) for f in files)
    if len(paths) > DATASET_ZIP_MAX_FILES:
        raise ValueError(f'too many files in the folder (max {DATASET_ZIP_MAX_FILES})')
    sizes, regular_paths = {}, set()
    for p in paths:
        try:
            # Do not follow a symlink into an arbitrary file/pipe outside the
            # folder the user selected. A named pipe must never reach ``open``:
            # it can block the request forever before there are image bytes to
            # validate.
            source_stat = os.lstat(p)
        except OSError:
            sizes[p] = 0
            continue
        if stat.S_ISREG(source_stat.st_mode):
            regular_paths.add(p)
            sizes[p] = source_stat.st_size
        else:
            sizes[p] = 0
    if sum(sizes.values()) > DATASET_ZIP_MAX_BYTES:
        raise ValueError('folder too large (max 2 GB)')
    oversized = next((
        p for p in paths
        if p.lower().endswith(_DATASET_ZIP_IMG_EXTS)
        and p in regular_paths
        and sizes.get(p, 0) > DATASET_ZIP_MAX_IMAGE_BYTES
    ), None)
    if oversized is not None:
        # Match ZIP import's per-image rule before a regular/sparse file is ever
        # opened. The bounded reader below repeats it to cover a live-folder race.
        raise ValueError('image too large in folder (max 128 MiB per image)')
    captions = {}
    for p in paths:
        if (p in regular_paths and p.lower().endswith('.txt')
                and sizes.get(p, 0) <= 64 * 1024):
            try:
                with open(p, 'rb') as fh:
                    captions[os.path.splitext(p)[0]] = \
                        fh.read().decode('utf-8', 'replace').strip()
            except OSError:
                pass

    def _read(p):
        source_stat = os.lstat(p)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError('folder image is not a regular file')
        if source_stat.st_size > DATASET_ZIP_MAX_IMAGE_BYTES:
            raise ValueError('image too large in folder (max 128 MiB per image)')
        with open(p, 'rb') as fh:
            opened_stat = os.fstat(fh.fileno())
            if not stat.S_ISREG(opened_stat.st_mode):
                raise ValueError('folder image is not a regular file')
            raw = fh.read(DATASET_ZIP_MAX_IMAGE_BYTES + 1)
        if len(raw) > DATASET_ZIP_MAX_IMAGE_BYTES:
            raise ValueError('image too large in folder (max 128 MiB per image)')
        return raw

    def _non_regular_image():
        raise ValueError('folder image is not a regular file')

    entries = [
        (os.path.splitext(p)[0], p,
         (lambda p=p: _read(p)) if p in regular_paths else _non_regular_image)
        for p in paths if p.lower().endswith(_DATASET_ZIP_IMG_EXTS)
    ]
    return _merge_training_images(user_id, dataset_id, entries, captions, stats=stats)


# --- Scrape direct → dataset concept ----------------------------------------
# Construction de dataset AUTONOME : on scanne une URL de galerie (routes scrape
# READ-ONLY, /api/scrape/scan + /thumb) et on télécharge les images choisies
# DIRECTEMENT dans le dataset — le pool scrape partagé de l'app source n'est PAS
# porté (cette app ne scrape que pour construire des datasets concept). Filtres :
# dedup perceptuel + résolution + ratio = les 3 filtres « toujours rentables » ;
# flou/watermark restent une décision HUMAINE (la sélection dans la grille de scan).
SCRAPE_IMPORT_MAX = 60             # cap par import (download synchrone parallélisé)
SCRAPE_IMPORT_MIN_SIDE = 768       # ai-toolkit ne fait que downscaler : 768 reste exploitable
SCRAPE_IMPORT_MAX_RATIO = 3.0      # au-delà de 3:1, aucun bucket trainer ne gère proprement
SCRAPE_DHASH_MAX_DISTANCE = 8      # Hamming ≤ 8 sur 64 bits = doublon perceptuel
_SCRAPE_DL_TYPES = ('image/jpeg', 'image/jpg', 'image/png', 'image/webp',
                    'image/bmp')  # pas de gif/svg
_SCRAPE_DL_MAX_BYTES = 25 * 1024 * 1024
_SCRAPE_DL_WORKERS = 6


def _dhash(im: Image.Image) -> int:
    """dHash 64 bits (gradient horizontal sur grayscale 9×8) — PIL pur, insensible
    au resize/re-encodage, donc stable entre un scrape original et sa version
    normalisée webp déjà importée."""
    g = im.convert('L').resize((9, 8), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


# Bank analysis needs to decode before its pure-Pillow metrics can downscale, so
# it keeps a header guard at this call.  A Bank may still copy a larger file; it
# simply starts unanalysed and can be reviewed separately.
# It follows the SHARED input budget rather than a private copy of it:
# an image the user was allowed to import must not come back unanalysable because
# a second, stricter number lives here. Kept as module attributes for the tests
# and callers that read them, but resolved live by the check below.
BANK_ANALYSIS_MAX_SIDE = image_encoding.DEFAULT_INPUT_MAX_SIDE
BANK_ANALYSIS_MAX_PIXELS = image_encoding.DEFAULT_INPUT_MAX_PIXELS


def _bank_analysis_dimensions_allowed(im: Image.Image) -> bool:
    """Reject headers whose full decode would exceed the local analysis budget."""
    max_side, max_pixels = preserved_import_limits()
    try:
        width, height = im.size
        return (isinstance(width, int) and isinstance(height, int)
                and width > 0 and height > 0
                and (not max_side or (width <= max_side and height <= max_side))
                and (not max_pixels or width * height <= max_pixels))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False


def _loaded_bank_deterministic_analysis(im: Image.Image) -> dict | None:
    """Apply the header guard, then decode only an image safe for this analysis."""
    if not _bank_analysis_dimensions_allowed(im):
        logger.warning('bank analysis skipped image beyond the %s input budget',
                       image_encoding.input_budget_sentence())
        return None
    # Match the Bank scan's JPEG fast path. It bounds decode work before the
    # quality metric performs its own <=1024px analysis copy; other formats
    # keep their native decoder behavior after the local header guard.
    im.draft(None, (ANALYSIS_MAX_SIDE * 2, ANALYSIS_MAX_SIDE * 2))
    im.load()
    return _bank_deterministic_values(im)


def bank_deterministic_analysis(image_source) -> dict | None:
    """Measure the deterministic Bank fields from one final image.

    Bank -> Dataset always invokes this on its emitted final file, and every Bank
    -> Bank copy invokes it on the destination file. Keeping it here next to the
    Dataset dHash makes both transfer directions use exactly the same pure-Pillow
    formulas as the Bank quality scan, without carrying stale source ML outputs.
    """
    try:
        # Pillow may warn at open time, but the explicit header guard below runs
        # before ``load()``. If an installation promotes that warning to an
        # exception, the dedicated catch remains safe without changing global
        # warning filters shared by concurrent Bank scans.
        if isinstance(image_source, (bytes, bytearray)):
            handle = io.BytesIO(image_source)
            with Image.open(handle) as im:
                return _loaded_bank_deterministic_analysis(im)
        with Image.open(image_source) as im:
            return _loaded_bank_deterministic_analysis(im)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        logger.warning('bank analysis skipped Pillow decompression bomb')
        return None
    except (OSError, TypeError, ValueError, SyntaxError, UnidentifiedImageError):
        return None


def _bank_deterministic_values(im: Image.Image) -> dict:
    """The strict v2 snapshot schema, computed from an already decoded image."""
    metrics = quality_metrics(im)
    provenance = provenance_metrics(im)
    return {
        'quality_state': 'ok',
        'blur_score': metrics['blur_score'],
        'noise_score': metrics['noise_score'],
        'uniformity_score': metrics['uniformity_score'],
        'dhash': f'{_dhash(im):016x}',
        'detail_ratio': provenance['detail_ratio'],
        'bars_ratio': provenance['bars_ratio'],
        'jpeg_quality': provenance['jpeg_quality'],
        'origin': provenance['origin'],
        'origin_evidence': provenance['origin_evidence'],
    }


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


def _existing_dhash_rows(dataset_id) -> list:
    """[(dHash, image_id)] des images déjà dans le dataset (keep/pending),
    recalculés à la volée : resize 9×8 ≈ qq ms/image et un dataset plafonne à
    ~200 images — pas de colonne/migration pour si peu. L'id accompagne le hash
    pour que l'appelant sache QUELLE image un doublon a rencontrée (l'import
    depuis une bank y raccroche sa provenance)."""
    out = []
    rows = FaceDatasetImage.query.filter(
        FaceDatasetImage.dataset_id == dataset_id,
        FaceDatasetImage.status.in_(('keep', 'pending'))).all()
    for r in rows:
        if not r.filename:
            continue
        try:
            with Image.open(os.path.join(_dataset_dir(dataset_id), r.filename)) as im:
                out.append((_dhash(im), r.id))
        except (OSError, ValueError):
            continue
    return out


def _existing_dhashes(dataset_id) -> list:
    """Les seuls dHashes (sans les ids) — voir _existing_dhash_rows."""
    return [h for h, _id in _existing_dhash_rows(dataset_id)]


def _attach_bank_provenance(image_id, bank_image_id, *, bank_analysis_snapshot=None,
                            bank_analysis_cache_dir=None) -> bool:
    """Raccroche une image de dataset DÉJÀ présente à la bank_image dont elle est
    le doublon perceptuel, et dit si le lien a été pris. N'écrase jamais une
    provenance existante : la première bank qui a fourni l'image la garde (sinon
    deux banks se voleraient le lien à chaque promotion croisée) — l'appelant
    apprend alors que CETTE bank n'a pas de trace vérifiable ici."""
    if not image_id:
        return False
    row = db.session.get(FaceDatasetImage, image_id)
    if row is None:
        return False
    # A dHash duplicate can be only visually similar, not byte-identical.  Its
    # source Bank scores are useful only when the Dataset file proves it is the
    # exact normalized transfer output; never attach a stale-looking snapshot.
    changed = False
    # First Bank wins for the snapshot exactly as it does for bank_image_id.  A
    # later Bank may hold a perceptual duplicate with different scores, but it
    # must never rewrite the analysis attributed to the original provenance.
    # The sole exception fills a legacy/mid-upgrade row that is already linked to
    # this SAME Bank but has no snapshot yet.
    owns_snapshot = (row.bank_image_id is None
                     or (row.bank_image_id == bank_image_id
                         and row.bank_analysis_snapshot is None))
    if owns_snapshot and bank_analysis_snapshot and row.filename:
        path = os.path.join(_dataset_dir(row.dataset_id), row.filename)
        snapshot = bank_transfer_metadata.compatible_snapshot(
            bank_analysis_snapshot, path)
        cache_ok = (snapshot is not None and (
            not snapshot.get('cache_ref')
            or (bank_analysis_cache_dir is not None
                and bank_transfer_metadata.read_cache_sidecar(
                    bank_analysis_cache_dir, snapshot['cache_ref']) is not None)))
        if cache_ok:
            row.bank_analysis_snapshot = bank_analysis_snapshot
            changed = True
    linked = bool(bank_image_id and row.bank_image_id == bank_image_id)
    if bank_image_id and row.bank_image_id is None:
        row.bank_image_id = bank_image_id
        linked = True
        changed = True
    if changed:
        db.session.commit()
    return linked


def _accept_scrape_bytes(raw, seen_hashes, skipped, rescue_small=False):
    """Filtre une image téléchargée : résolution / ratio / dedup perceptuel.
    Retourne les bytes si acceptée (et enregistre son dHash dans seen_hashes),
    sinon None en incrémentant le compteur skipped adéquat. Quand rescue_small
    est vrai, une petite image continue vers ratio+dedup au lieu d'être rejetée;
    elle ne sera jamais importée directement dans l'entraînement."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                # Scrape quality/dHash must not become the first full decode of a
                # crafted response: run the same header budget as every import lane.
                _preserved_import_header_extension(im)
                im.load()
                w, h = im.size
                if min(w, h) < SCRAPE_IMPORT_MIN_SIDE and not rescue_small:
                    skipped['low_res'] += 1
                    return None
                if max(w, h) > SCRAPE_IMPORT_MAX_RATIO * min(w, h):
                    skipped['extreme_ratio'] += 1
                    return None
                fp = _dhash(im)
    except (OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning):
        skipped['errors'] += 1
        return None
    if any(_hamming(fp, s) <= SCRAPE_DHASH_MAX_DISTANCE for s in seen_hashes):
        skipped['duplicates'] += 1
        return None
    seen_hashes.append(fp)
    return raw


def _scrape_resolution_key(downloaded):
    """Sort key for rescue batches: the best-resolution duplicate must win."""
    reason, raw = downloaded
    if reason != 'ok' or not raw:
        return (0, 0)
    try:
        width, height = _import_header_dimensions(raw, label='scrape import')
        return (min(width, height), width * height)
    except ValueError:
        return (0, 0)


def _save_small_scrape_pair(user_id, dataset_id, raw, prompt, source_metadata=None):
    """Persist the untouched scrape source and enqueue one Klein candidate.

    Returns True when queued, False when enqueue failed. The original and result
    rows are committed before enqueue so a failed queue operation never loses the
    source file or leaves an untracked job.
    """
    from .klein_edit_helper import enqueue_klein_edit

    # This helper is usually called only after `_accept_scrape_bytes`, but it is
    # also a service seam. Validate again rather than letting a direct caller make
    # this its first unbounded decoder.
    ext = _preserved_import_extension(raw)
    filename = f"{user_id}_scrape_small_{uuid.uuid4().hex[:8]}{ext}"
    source_path = os.path.join(_dataset_dir(dataset_id), filename)
    with open(source_path, 'wb') as fh:
        fh.write(raw)

    stored_metadata = _source_metadata_storage(source_metadata)
    source = FaceDatasetImage(
        dataset_id=dataset_id, source='import', status='pending', filename=filename,
        derivation_kind=SMALL_IMAGE_SOURCE,
        variation_label='Small scraped image · original',
        source_metadata=stored_metadata,
    )
    db.session.add(source)
    db.session.flush()
    label = 'Klein rescue · small scraped image'
    candidate = FaceDatasetImage(
        dataset_id=dataset_id, source='generated', status='pending',
        parent_image_id=source.id, derivation_kind=KLEIN_SMALL_IMAGE,
        variation_label=label, variation_prompt=prompt,
        source_metadata=stored_metadata,
    )
    db.session.add(candidate)
    db.session.commit()

    try:
        job_id = enqueue_klein_edit(
            user_id=str(user_id), source_filename=filename, source_path=source_path,
            # Same model as everything else this dataset makes — a rescued 512 px
            # scrape ends up in the SAME training set as the improved images, so
            # running it on another model is exactly the drift the setting exists
            # to stop. None (never chose) = the historical auto pick.
            klein_model=dataset_klein_model(get_dataset(user_id, dataset_id)),
            edit_prompt=prompt, sampler_steps=_generation_steps(),
            base_lora_strength=_generation_base_lora_strength(),
            extra_metadata={'is_dataset': True, 'dataset_id': dataset_id,
                            'variation_label': label,
                            'derivation_kind': KLEIN_SMALL_IMAGE,
                            'parent_image_id': source.id},
        )
    except Exception as exc:
        candidate.status = 'failed'
        candidate.fail_reason = f'Klein small-image rescue could not be queued: {exc}'
        db.session.commit()
        logger.exception('small-image rescue enqueue failed for dataset %s source %s',
                         dataset_id, source.id)
        return False
    candidate.job_id = job_id
    db.session.commit()
    return True


def _download_scrape_item(item):
    """Télécharge UNE image d'un item de scan ({url,title}) en mémoire, durci
    anti-SSRF (mêmes garanties que /thumb). Retourne (reason, data|None) où
    reason ∈ {'ok','not_image','errors'}. Sûr hors app-context (thread pool)."""
    from ..scrape.netfetch import fetch_hardened_bytes, _validate_public_http_url
    url = (item or {}).get('url')
    if not url:
        return ('errors', None)
    ok_url, _err = _validate_public_http_url(url)
    if not ok_url:
        return ('errors', None)
    ok, data, _ctype, reason = fetch_hardened_bytes(
        url, allowed_types=_SCRAPE_DL_TYPES, max_bytes=_SCRAPE_DL_MAX_BYTES,
        require_image_magic=True)
    if not ok:
        # 'type'/'noimage' = pas une vraie image raster ; le reste = erreur réseau.
        return ('not_image' if reason in ('type', 'noimage') else 'errors', None)
    return ('ok', data)


def scrape_import_urls(user_id, dataset_id, items, rescue_small=False):
    """Télécharge les images scannées SÉLECTIONNÉES directement dans le dataset
    concept — flux AUTONOME. `items` = [{'url','title'}]. Download parallélisé
    (borné), puis filtre + dedup séquentiels (état partagé), puis import brut
    aspect-kept via import_images(crop=False). Renvoie
    {'imported': n, 'rescue_queued': n, 'rescue_failed': n,
     'skipped': {duplicates, low_res, extreme_ratio, not_image, errors}}."""
    from concurrent.futures import ThreadPoolExecutor
    skipped = {'duplicates': 0, 'low_res': 0, 'extreme_ratio': 0,
               'not_image': 0, 'errors': 0}
    items = [it for it in (items or []) if isinstance(it, dict) and it.get('url')]
    if not items:
        return {'imported': 0, 'rescue_queued': 0, 'rescue_failed': 0,
                'skipped': skipped}
    with ThreadPoolExecutor(max_workers=_SCRAPE_DL_WORKERS) as pool:
        # Keep each response tied to its scan item. Rescue sorting changes order,
        # so a separate byte list would otherwise attach the wrong photographer.
        downloaded = list(zip(items, pool.map(_download_scrape_item, items)))

    # In rescue mode a low-resolution duplicate must never claim the dHash first
    # and make the usable HD source look like the duplicate. The legacy path keeps
    # request order exactly as before.
    if rescue_small:
        downloaded.sort(key=lambda pair: _scrape_resolution_key(pair[1]), reverse=True)

    seen_hashes = _existing_dhashes(dataset_id)
    accepted, rescue_candidates = [], []
    for item, (reason, data) in downloaded:
        if reason != 'ok':
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        ok_bytes = _accept_scrape_bytes(data, seen_hashes, skipped,
                                        rescue_small=rescue_small)
        if ok_bytes is not None:
            if rescue_small:
                try:
                    is_small = (min(_import_header_dimensions(
                        ok_bytes, label='scrape import')) < SCRAPE_IMPORT_MIN_SIDE)
                except ValueError:
                    skipped['errors'] += 1
                    continue
                target = rescue_candidates if is_small else accepted
                target.append((ok_bytes, _source_metadata_from_scrape_item(item)))
            else:
                accepted.append((ok_bytes, _source_metadata_from_scrape_item(item)))

    # Capacity and model preflight happen once, after every quality/dedup filter,
    # but before creating a source/result pair. No small candidate => no Klein scan.
    if rescue_candidates:
        in_flight = (FaceDatasetImage.query
                     .filter_by(dataset_id=dataset_id, status='pending')
                     .filter(FaceDatasetImage.filename.is_(None)).count())
        if in_flight + len(rescue_candidates) > MAX_FANOUT:
            raise ValueError(f'too many generations in flight ({in_flight}), wait or cancel')
        from .klein_edit_helper import (KLEIN_REQUIRED, KleinModelsMissing,
                                        klein_missing_assets)
        missing = klein_missing_assets()
        if any(asset in missing for asset in KLEIN_REQUIRED):
            raise KleinModelsMissing(missing)

    ids, failed = import_images(
        user_id, dataset_id, [raw for raw, _metadata in accepted], crop=False,
        source_metadata=[metadata for _raw, metadata in accepted])
    skipped['errors'] += failed
    raw_prompt = cfg.get('klein.small_image_prompt', '')
    prompt = '' if raw_prompt is None else str(raw_prompt)
    rescue_queued = rescue_failed = 0
    for raw, source_metadata in rescue_candidates:
        try:
            queued = _save_small_scrape_pair(
                user_id, dataset_id, raw, prompt, source_metadata=source_metadata)
        except Exception:
            rescue_failed += 1
            logger.exception('small-image rescue save failed for dataset %s', dataset_id)
            continue
        if queued:
            rescue_queued += 1
        else:
            rescue_failed += 1
    if rescue_candidates:
        _sync_generate_activity(dataset_id)
    return {'imported': len(ids), 'rescue_queued': rescue_queued,
            'rescue_failed': rescue_failed, 'skipped': skipped}


def _parse_classify(raw):
    try:
        start = raw.index('{')
        obj = json.loads(raw[start:raw.index('}', start) + 1])
    except (ValueError, AttributeError):
        return 'unknown', None
    fr = obj.get('framing')
    fr = fr if fr in ('face', 'bust', 'body', 'back') else 'unknown'
    label = ', '.join(str(obj.get(k)) for k in ('angle', 'expression') if obj.get(k))
    return fr, (label or None)


def classify_images(user_id, dataset_id):
    """Classify imported images lacking a framing via Qwen3-VL. Returns count."""
    _guard_not_bank_export(dataset_id)
    try:
        from .vision_ollama import describe_image_ollama, unload_vision_model
    except ImportError:
        raise RuntimeError('vision (Ollama) service not configured/available yet')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return 0
    rows = FaceDatasetImage.query.filter_by(
        dataset_id=dataset_id, source='import', framing=None).all()
    # Ids, not ORM objects: see _live_image_row. The commit at the bottom of this
    # loop expires every row it has not reached, and a tile deleted from the grid
    # meanwhile used to kill the whole classification.
    row_ids = [img.id for img in rows]
    n = 0
    vanished = 0
    # Persistent progress indicator (survives a page reload): try/finally guarantees
    # end() runs even if the batch raises → no phantom "Classifying…" spinner.
    token = dataset_activity.begin(dataset_id, 'classify', total=len(row_ids))
    try:
        for i, image_id in enumerate(row_ids):
            dataset_activity.progress(token, done=i + 1)
            img = _live_image_row(image_id)
            if img is None:      # deleted while the pass ran
                vanished += 1
                continue
            path = _img_path(img) if img.filename else ''
            if not os.path.exists(path):
                continue
            with open(path, 'rb') as fh:
                raw = describe_image_ollama(fh.read(), CLASSIFY_PROMPT, num_predict=1200,
                                            prefer_json=True, keep_alive=_VISION_BATCH_KEEPALIVE)
            if not (raw or '').strip():
                # Échec vision (Ollama indisponible) ≠ « framing indéterminé » :
                # on laisse framing=None (retry possible) au lieu d'écrire 'unknown'
                # définitivement, qui bloquerait toute reclassification.
                continue
            framing, label = _parse_classify(raw)
            img.framing = framing
            img.variation_label = label
            db.session.commit()
            n += 1
    finally:
        unload_vision_model()  # libère la VRAM pour ComfyUI en fin de batch
        dataset_activity.end(token)
    if vanished:
        logger.info('classify: %s image(s) were deleted while the pass ran, skipped',
                    vanished)
    return n


# --- Borrow: face_dataset_service.py primitives -----------------------------
# MUST stay at the bottom of this file, same reason as in the sibling split
# modules: this module and face_dataset_service.py import names from each
# other, and whichever side loads first must find the other fully defined by
# the time the reach-back import resolves.
# --- Bank promotion rollback (ported from upstream, 2026-08-08) --------------
# The multi-chunk Bank promotion needs a way to undo a half-written batch. This
# function is upstream's, moved here because THIS module owns dataset ingest
# since the Phase 4 split. It borrows `_restore_from_trash` and
# `_bank_analysis_cache_dir` from face_dataset_service at call time, like the
# rest of this module does.

def rollback_imported_images(user_id, dataset_id, image_ids,
                             provenance_changes=None) -> bool:
    """Remove only rows created by a failed multi-chunk Bank promotion.

    The caller holds the Dataset ingest stripe for the whole promotion.  Files
    are moved to the recoverable app trash before one DB commit; any failure
    restores every moved file and leaves the rows intact.
    """
    wanted = {int(value) for value in image_ids}
    provenance_changes = list(provenance_changes or ())
    if not wanted and not provenance_changes:
        return True
    ds = get_dataset(user_id, dataset_id)
    if ds is None:
        return False
    rows = (FaceDatasetImage.query
            .filter(FaceDatasetImage.dataset_id == dataset_id,
                    FaceDatasetImage.id.in_(wanted)).all())
    if {row.id for row in rows} != wanted:
        return False
    moved = []
    cache_refs = set()
    try:
        for change in reversed(provenance_changes):
            row = (FaceDatasetImage.query
                   .filter_by(id=change.get('image_id'), dataset_id=dataset_id)
                   .one_or_none())
            if (row is None
                    or row.bank_image_id != change.get('new_bank_image_id')
                    or row.bank_analysis_snapshot != change.get('new_snapshot')):
                raise RuntimeError('dedupe provenance changed during rollback')
            new_snapshot = bank_transfer_metadata.parse_snapshot(
                row.bank_analysis_snapshot)
            if new_snapshot and new_snapshot.get('cache_ref'):
                cache_refs.add(new_snapshot['cache_ref'])
            row.bank_image_id = change.get('old_bank_image_id')
            row.bank_analysis_snapshot = change.get('old_snapshot')
        for row in rows:
            snapshot = bank_transfer_metadata.parse_snapshot(
                row.bank_analysis_snapshot)
            if snapshot and snapshot.get('cache_ref'):
                cache_refs.add(snapshot['cache_ref'])
            if row.filename:
                original = os.path.join(_dataset_path(dataset_id), row.filename)
                if os.path.exists(original):
                    trashed = trash.send_to_trash(
                        original,
                        context=f'dataset-{dataset_id}-failed-bank-promotion-{row.id}')
                    moved.append((trashed, original))
            db.session.delete(row)
        db.session.commit()
    except Exception:
        db.session.rollback()
        for trashed, original in reversed(moved):
            _restore_from_trash(trashed, original)
        return False
    cache_dir = _bank_analysis_cache_dir(dataset_id)
    for cache_ref in cache_refs:
        shared = False
        for other in (FaceDatasetImage.query
                      .filter(FaceDatasetImage.dataset_id == dataset_id,
                              FaceDatasetImage.bank_analysis_snapshot.isnot(None))):
            snapshot = bank_transfer_metadata.parse_snapshot(
                other.bank_analysis_snapshot)
            if snapshot and snapshot.get('cache_ref') == cache_ref:
                shared = True
                break
        if not shared:
            bank_transfer_metadata.remove_cache_sidecar(cache_dir, cache_ref)
    return True


from .face_dataset_service import (
    _live_image_row,
    _guard_not_bank_export,
    get_dataset, dataset_klein_model, normalize_to_webp, write_image_atomic,
    _img_path, _dataset_dir, _crop_resize_file, _cap_caption,
    _source_metadata_from_scrape_item, _source_metadata_storage,
    _generation_steps, _generation_base_lora_strength,
    _restore_from_trash, _bank_analysis_cache_dir, _dataset_path,
    _remove_unreferenced_bank_analysis_cache,
    CAPTION_MAX_CHARS, KLEIN_SMALL_IMAGE, SMALL_IMAGE_SOURCE, MAX_FANOUT,
    UPSCALE_WARN_THRESHOLD, _VISION_BATCH_KEEPALIVE, logger,
)
# Straight from the module that OWNS it, not via face_dataset_service's
# re-export: routing a cross-module borrow through the parent would make the
# ORDER of the parent's re-export blocks load-bearing.
from .dataset_generation_service import _sync_generate_activity
from .dataset_backup_service import (
    _coerce_archive_stream, _preflight_zip_central_directory,
)