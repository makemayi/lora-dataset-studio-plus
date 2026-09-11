"""MiniMax-H3 RefMod generation for a dataset (the 🧩 button).

Turns a dataset's kept images into one `minimaxh3_<name>_v1_refmod.safetensors`
reference file for the ComfyUI-MiniMaxH3Mod node pack. The heavy lifting runs in
the COMFYUI interpreter (it owns torch) through refmod_extract_worker.py — this
module only resolves paths, picks the images, and drives the subprocess.

Every path derives from the already-configured ComfyUI base_dir, so the feature
has no settings block of its own: python from the portable bundle layout,
the node pack and the VAE from the ComfyUI tree. A missing piece is a ValueError
with the remedy in the text — the toast shows it verbatim.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .. import config as cfg
from ..models import FaceDatasetImage
from ..services import face_mask as face_mask_service
from ..services.dataset_storage import dataset_path

# Per-framing picks, identity-first: the face signal lives in close-ups and
# half/full frames are what carry CLOTHING into the latent (the #1 complaint).
# Ladder face → half → full tops up only when a dataset lacks close-ups.
_MAX_FACE, _MAX_HALF = 12, 4
_MIN_TOTAL = 6
_TOKEN_BUDGET = 16384   # keeps every picked angle un-resampled (12×1024 + slack)
_BACKGROUND_RETENTION = 0.0   # outside the face mask collapses to a blurred copy
_MASKS_DIR_NAME = 'refmod'
_SUBPROCESS_TIMEOUT_S = 1800


def _comfy_root() -> Path:
    raw = (cfg.get('comfyui.base_dir') or '').strip()
    if not raw:
        raise ValueError('ComfyUI base_dir is not configured — set it in Settings ▸ ComfyUI.')
    root = Path(raw)
    if not root.is_dir():
        raise ValueError(f'ComfyUI base_dir does not exist: {root}')
    return root


def _python_exe(root: Path) -> Path:
    """The interpreter that owns torch: the portable bundle's python first
    (aki builds ship `<bundle>/python`, stock portables `python_embeded`),
    then a venv checkout."""
    candidates = [root.parent / 'python' / 'python.exe',
                  root.parent / 'python_embeded' / 'python.exe',
                  root / 'venv' / 'Scripts' / 'python.exe']
    for cand in candidates:
        if cand.is_file():
            return cand
    raise ValueError('No ComfyUI python found (looked for python/python.exe, '
                     'python_embeded/python.exe, venv/Scripts/python.exe next to '
                     'the ComfyUI base dir).')


def _node_dir(root: Path) -> Path:
    node = root / 'custom_nodes' / 'ComfyUI-MiniMaxH3Mod'
    if not node.is_dir():
        raise ValueError('ComfyUI-MiniMaxH3Mod is not installed — '
                         'clone it into custom_nodes/ and restart ComfyUI.')
    return node


def _vae_path(root: Path) -> Path:
    """The H3 VIDEO vae. Named loosely on purpose (fp16/fp32/int8 variants come
    and go); anything with 'h3' that is not the audio codec qualifies, and a
    name carrying 'video' wins the tie."""
    vae_dir = root / 'models' / 'vae'
    if not vae_dir.is_dir():
        raise ValueError('No models/vae directory under the ComfyUI base dir.')
    candidates = [f for f in vae_dir.glob('*.safetensors')
                  if 'h3' in f.name.lower() and 'audio' not in f.name.lower()]
    if not candidates:
        raise ValueError('No H3 video VAE found in models/vae — '
                         'download minimax_h3_video_vae first.')
    return sorted(candidates, key=lambda f: ('video' not in f.name.lower(), f.name))[0]


def pick_images(images) -> list:
    """Identity-first pick, ≤16 rows: up to 12 face close-ups, then half shots,
    full frames only as top-up (they are the clothing carrier). Framings
    outside the training trio count as full."""
    by: dict[str, list] = {'face': [], 'half': [], 'full': []}
    for row in images:
        if getattr(row, 'status', None) != 'keep' or not getattr(row, 'filename', None):
            continue
        framing = (getattr(row, 'framing', None) or 'full').lower()
        by.setdefault(framing, by['full']).append(row)
    picked = by['face'][:_MAX_FACE] + by['half'][:_MAX_HALF]
    if len(picked) < _MIN_TOTAL:
        picked += by['full'][:_MIN_TOTAL - len(picked)]
    if not (by['face'] or by['half']):
        # A dataset with only full frames has no closer option — clothing is
        # unavoidable there, so at least keep 12 angles for coverage.
        picked = by['full'][:12]
    return picked[:16]


def _kept_rows(ds):
    """The dataset's kept rows with files. FaceDataset has no images
    relationship — the store is queried by dataset_id, like everywhere else
    (measured: an ``ds.images`` guess 500s the whole route)."""
    return (FaceDatasetImage.query
            .filter_by(dataset_id=ds.id, status='keep')
            .filter(FaceDatasetImage.filename.isnot(None))
            .all())


def _draw_identity_mask(image_path, boxes, out_dir, expand=2.0):
    """Write a face mask for a 'too_large' frame: same dilate_box + ellipse +
    feather math as face_mask_infer (LDS convention, face=BLACK), but WITHOUT
    the 0.5-coverage refusal — a face crop where the face fills the frame is
    exactly the identity-dense image this feature must mask. Returns the PNG
    path or None."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    with Image.open(image_path) as img:
        w, h = img.size
    shift_up, feather_frac = 0.10, 0.03
    mask = Image.new('L', (w, h), 255)
    draw = ImageDraw.Draw(mask)
    for b in boxes or []:
        x1, y1, x2, y2 = b
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0 - (y2 - y1) * shift_up
        hw, hh = (x2 - x1) * expand / 2.0, (y2 - y1) * expand / 2.0
        draw.ellipse([(cx - hw) * w, (cy - hh) * h, (cx + hw) * w, (cy + hh) * h], fill=0)
    r = max(1, int(min(w, h) * feather_frac))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=r))
    name = os.path.splitext(os.path.basename(image_path))[0] + '.png'
    out = os.path.join(out_dir, 'identity_' + name)
    mask.save(out, 'PNG')
    return out


def _face_masks_for(image_paths, ds_id):
    """Face-mask PNG per picked image (None where no confident face), via the
    same InsightFace pass masked training uses. LDS mask convention is face=BLACK
    / background=WHITE; the worker inverts to the RefMod keep=1 convention.
    Unavailable detection degrades to no masks — an unmasked RefMod is still a
    valid RefMod, it just carries what the frame carries."""
    if not image_paths:
        return [None] * len(image_paths)
    out_dir = os.path.join(os.path.dirname(dataset_path(ds_id)), os.pardir,
                           'masks', _MASKS_DIR_NAME, str(ds_id))
    out_dir = os.path.abspath(out_dir)
    try:
        result = face_mask_service.generate_face_masks(image_paths, out_dir)
    except Exception:  # noqa: BLE001 - masks are an enhancement, never a gate
        return [None] * len(image_paths)
    results = (result or {}).get('results') or {}
    masks = []
    for p in image_paths:
        entry = results.get(p) or {}
        state = entry.get('state')
        stem = os.path.splitext(os.path.basename(p))[0]
        mask_path = os.path.join(out_dir, stem + '.png')
        if state == 'masked' and os.path.isfile(mask_path):
            masks.append(mask_path)
        elif state == 'too_large' and entry.get('boxes'):
            # The training-oriented 0.5-coverage cap refuses exactly the
            # face-filling close-ups identity needs; redraw without the cap.
            try:
                masks.append(_draw_identity_mask(p, entry['boxes'], out_dir))
            except OSError:
                masks.append(None)
        else:
            masks.append(None)
    return masks


def generate_for_dataset(ds, masked=True) -> dict:
    """Encode ds's kept images into one RefMod. Synchronous (~1-3 min: the VAE
    load dominates); the caller holds the GPU vision window.

    ``masked=True`` applies face-mask suppression and names the output with a
    ``_mask`` suffix; ``masked=False`` skips masks entirely so the plain-named
    unmasked baseline stays available for A/B comparison."""
    root = _comfy_root()
    python_exe = _python_exe(root)
    node_dir = _node_dir(root)
    vae_path = _vae_path(root)

    picked = pick_images(_kept_rows(ds))
    if not picked:
        raise ValueError('No kept images to encode.')
    storage = Path(dataset_path(ds.id))
    image_paths = [str(storage / row.filename) for row in picked]
    masks = _face_masks_for(image_paths, ds.id) if masked else [None] * len(image_paths)
    masked_n = sum(1 for m in masks if m)
    suffix = '_mask' if masked else ''

    worker = Path(__file__).with_name('refmod_extract_worker.py')
    manifest = {
        'node_dir': str(node_dir),
        'images': image_paths,
        'vae': str(vae_path),
        'output_dir': str(root / 'models' / 'refmods'),
        'description': f'identity baseline from LDS dataset {ds.id} ({ds.name})',
        'max_tokens': _TOKEN_BUDGET,
        'masks': masks,
        'background_retention': _BACKGROUND_RETENTION,
    }
    manifest['name'] = f"minimaxh3_{_safe_name(ds.name, ds.id)}_v1{suffix}_refmod"
    os.makedirs(manifest['output_dir'], exist_ok=True)

    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [str(python_exe), str(worker)],
        input=json.dumps(manifest).encode('utf-8'),
        capture_output=True, timeout=_SUBPROCESS_TIMEOUT_S,
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'})
    lines = [ln for ln in proc.stdout.decode('utf-8', 'replace').splitlines()
             if ln.strip().startswith('{')]
    try:
        result = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        result = {}
    if proc.returncode != 0 or not result.get('ok'):
        detail = result.get('error') or proc.stderr.decode('utf-8', 'replace')[-400:]
        raise RuntimeError(f'RefMod extraction failed: {detail}')
    return {'name': manifest['name'], 'tokens': result.get('tokens'),
            'frames': result.get('frames'), 'path': result.get('path'),
            'mb': result.get('mb'), 'masked': result.get('masked', 0)}


def _safe_name(name: str, ds_id: int) -> str:
    out = ''.join(c if (c.isalnum() or c == '_') else '_' for c in name.strip())
    return out.strip('_').lower() or f'dataset{ds_id}'
