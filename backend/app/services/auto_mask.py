"""Auto-masking: a mask for ANY region you can name, produced by the APP.

    auto_mask.mask_for('C:/…/tile.png', 'head')   -> path to an 8-bit PNG

WHY THIS IS A SERVICE AND NOT PART OF ONE FEATURE
--------------------------------------------------
Masks are needed all over this app and every place grew its own: a person matte
from rembg for masked training, a face ellipse from InsightFace boxes for
concept training, a box from the watermark detector, and — until this module —
a head mask computed INSIDE a ComfyUI graph by a custom node, where the app
could neither see it, cache it, test it nor reuse it. Image repair ("clean
this, keep that") would have been the fifth.

So a mask is one call: an image plus a short noun phrase in, a PNG out, white
where the region is.

SAM 3, IN THE APP'S OWN INTERPRETER, ON COMFYUI'S WEIGHTS
----------------------------------------------------------
SAM 3 is open-vocabulary, so ONE mechanism covers head, hair, hands, clothing, a
watermark, a logo — which is the whole reason to build this instead of a head
detector. It runs in an app-managed venv (`auto_mask.python`, built by Setup),
the same isolation every other model here gets, so nothing depends on ComfyUI
being up, on a graph, or on a custom-node pack that a user cannot install.

The WEIGHTS are borrowed rather than downloaded: `sam3.pt` is Meta's own
checkpoint and a ComfyUI install that does any segmentation already has it. It
is resolved through `comfy_model_paths`, so an extra_model_paths.yaml root works
exactly like the default tree. There is no auto-download: the honest answer for
a 3.4 GB file is a named gap and one link.

THE ENVIRONMENT, AS MEASURED (Windows, 2026-08-11)
---------------------------------------------------
Built at data/envs/automask from Python 3.13. `pip install sam3` pulls almost
none of what it needs on Windows, and each gap fails as a bare ImportError deep
inside the first call — so the working set is written down here rather than
rediscovered:

    torch==2.7.0+cu126, torchvision==0.22.0+cu126   (SAM 3's floor is 2.7)
    git+https://github.com/facebookresearch/sam3
    setuptools<81      model_builder imports pkg_resources, gone in 81
    einops             sam3.sam.rope
    triton-windows     sam3.model.edt imports triton unconditionally; upstream
                       triton has no Windows wheels, this fork does
    pycocotools        dragged in through the TRAINING data loaders, which the
                       image path imports whether or not it trains
    psutil, decord     sam3.model.sam3_video_predictor
    numpy<2            sam3 pins it; do NOT let opencv-python pull numpy 2 in

Measured on an RTX 3090: 23 s for the first mask (the 3.4 GB checkpoint load is
almost all of it), 0.001 s for a cached one. That ratio is why `mask_images`
takes a LIST — a batch pays the load once.

CACHED ON CONTENT, NOT ON PATH
------------------------------
The same mask gets asked for repeatedly — a swap wants the head, then a repair
pass wants the same head, then the user re-runs after changing something else.
The key is the sha256 of the image BYTES plus the phrase and the threshold, so a
renamed or copied file hits the entry and an edited one misses it. That is also
what makes a mask preview cheap enough to show before running anything.
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading

from .. import config as cfg
from . import comfy_model_paths
from .infer_stream import run_infer_script, stderr_tail

logger = logging.getLogger(__name__)

_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'sam3_mask_infer.py')

# ONE masking child at a time, process-wide.
#
# Every caller here is a request thread, so a batch of N head swaps calls
# `mask_for` N times CONCURRENTLY — and each call spawns a child that loads its
# own 3.4 GB SAM 3 onto the same card. Measured on a 24 GB card on 2026-08-14:
# five children four seconds apart, 21 GB resident, the GPU at 96% utilisation
# while drawing 125 W — the shape of memory contention, not of work. The render
# those masks were for then had ~400 MB to start in.
#
# Serialising costs nothing real: the children were competing for one GPU, so
# they were never parallel in the way that matters. It does mean a long
# `mask_images` batch makes a single-image caller wait, which is the correct
# order of harm — waiting finishes, an OOM does not.
#
# This is a per-process lock, so it does NOT protect against a second backend
# (or ComfyUI itself) reaching for the card at the same time. Nothing here can:
# the card has no admission control. It removes the one source of pile-up this
# app creates on its own.
_INFER_LOCK = threading.Lock()

# Meta's own checkpoint, as fetched by any ComfyUI segmentation pack from
# facebook/sam3. NOT the Comfy-Org repackaged `sam3.1_multiplex_fp16.safetensors`:
# that one is remapped for ComfyUI's own loader and will not load here.
SAM3_CANONICAL = 'sam3.pt'
SAM3_TOKENS = ('sam3', 'sam_3')
_CHECKPOINT_SUFFIXES = ('.pt', '.pth', '.bin')
# ComfyUI folder types worth scanning, in the order a user is likely to have
# used. 'sam3' is where the packs put it; the rest are where a person filing it
# by hand would.
_SEARCH_TYPES = ('sam3', 'sams', 'checkpoints', 'diffusion_models')

DEFAULT_THRESHOLD = 0.5
DEFAULT_TIMEOUT_S = 900
# Past this share of the frame a mask has stopped being a mask: "everything
# matched" is what a too-generic phrase returns, and compositing it back is not
# an edit, it is a re-render of the whole picture.
MAX_COVERAGE = 0.92

_PHASE_RE = re.compile(r'\[sam3\] phase=([a-z_]+)')
_COUNT_RE = re.compile(r'\[sam3\] (\d+)/(\d+)')
PHASES = ('starting', 'loading', 'masking')


class AutoMaskUnavailable(Exception):
    """No mask can be produced: no interpreter, no checkpoint, or the run
    failed. Carries the `{files, actions}` vocabulary the Setup banners already
    render — `actions` naming the installer step that fixes it, when one does."""

    def __init__(self, message, files=None, actions=None):
        self.files = list(files or [])
        self.actions = list(actions or [])
        super().__init__(message)


class NoMatch(AutoMaskUnavailable):
    """The phrase matched nothing, or matched the whole frame. Its own type
    because it is the one failure that is about the REQUEST rather than the
    install: the caller may want to retry with another phrase, and the UI must
    not offer an "install" button for it."""


# --- What is installed -------------------------------------------------------

def python_path() -> str:
    return (cfg.get('auto_mask.python') or '').strip()


def resolve_checkpoint(selected=None):
    """Absolute path of Meta's `sam3.pt`, or None.

    `selected` (a Settings value) wins when it is still on disk; a stale pin
    degrades to the scan rather than failing — same contract as every engine
    resolver here. An absolute path outside every ComfyUI root is honoured too:
    this model is not loaded BY ComfyUI, so it does not have to live under one."""
    selected = (selected or '').strip()
    if selected and os.path.isfile(selected):
        return selected
    if selected:
        logger.info('auto_mask: pinned checkpoint %r is gone, scanning', selected)
    roots = []
    for comfy_type in _SEARCH_TYPES:
        try:
            roots.extend(comfy_model_paths.search_roots(comfy_type))
        except Exception:                               # noqa: BLE001
            logger.debug('auto_mask: no %s root', comfy_type, exc_info=True)
    seen, canonical, fallback = set(), None, None
    for root in roots:
        if not os.path.isdir(root) or root in seen:
            continue
        seen.add(root)
        for dirpath, _sub, filenames in os.walk(root, followlinks=True):
            for fn in sorted(filenames):
                low = fn.lower()
                if not low.endswith(_CHECKPOINT_SUFFIXES):
                    continue
                full = os.path.join(dirpath, fn)
                if low == SAM3_CANONICAL:
                    canonical = canonical or full
                elif any(tok in low for tok in SAM3_TOKENS) and fallback is None:
                    fallback = full
        if canonical:
            return canonical
    return canonical or fallback


def availability():
    """{'ok', 'python', 'checkpoint', 'files', 'actions'} — what a Settings card
    or a preflight banner needs, without raising."""
    python = python_path()
    has_python = bool(python) and os.path.isfile(python)
    checkpoint = resolve_checkpoint(cfg.get('auto_mask.checkpoint'))
    files = []
    if not checkpoint:
        files.append({
            'kind': 'SAM 3 segmentation checkpoint',
            'path': 'ComfyUI/models/sam3/sam3.pt',
            'source': 'https://huggingface.co/facebook/sam3',
        })
    return {
        'ok': has_python and bool(checkpoint),
        'python': python if has_python else '',
        'checkpoint': checkpoint or '',
        'files': files,
        'actions': [] if has_python else ['automask'],
    }


def is_available() -> bool:
    return availability()['ok']


def preflight():
    """(python, checkpoint), or raise AutoMaskUnavailable naming every gap at
    once. No auto-download and no auto-install: both are explicit Setup actions,
    and a mask click is the wrong place to start a 3 GB one."""
    state = availability()
    if state['ok']:
        return state['python'], state['checkpoint']
    parts = ["Automatic masking can't run yet."]
    if not state['python']:
        parts.append('Build the automatic-masking environment in Setup ▸ Local '
                     'tools (it installs SAM 3 into its own Python, never the '
                     "app's).")
    for f in state['files']:
        parts.append(f"{f['kind']} → place it at {f['path']} ({f['source']}); "
                     'any ComfyUI segmentation pack downloads the same file.')
    raise AutoMaskUnavailable(' '.join(parts), files=state['files'],
                              actions=state['actions'])


# --- Progress ----------------------------------------------------------------

def parse_progress_line(line: str):
    """One stderr line -> a progress record, or None. PURE, so the grammar is
    testable without a subprocess. Reaching image 1 means the model is loaded
    whatever phase line was last seen, so the counter carries the phase with
    it — a missed `phase=masking` cannot leave the UI stuck on "Loading…"."""
    m = _PHASE_RE.search(line or '')
    if m and m.group(1) in PHASES:
        return {'phase': m.group(1)}
    m = _COUNT_RE.search(line or '')
    if m:
        return {'phase': 'masking', 'done': int(m.group(1)), 'total': int(m.group(2))}
    return None


# --- Cache -------------------------------------------------------------------

def cache_dir():
    d = cfg._data_dir() / 'masks'
    d.mkdir(parents=True, exist_ok=True)
    return d


def normalize_prompts(prompt):
    """A phrase, a comma-separated list, or a list -> an ordered, de-duplicated,
    lower-cased list of phrases.

    Several phrases exist because a REGION is usually several OBJECTS. To an
    open-vocabulary model 'head' is the head and the glasses on it are a
    different thing, so a head mask asked for in one word comes back with holes
    exactly where the accessories are — and a swap then repaints a head around
    the old pair of glasses. The phrases' masks are unioned."""
    if isinstance(prompt, (list, tuple, set)):
        raw = list(prompt)
    else:
        raw = str(prompt or '').split(',')
    out, seen = [], set()
    for item in raw:
        text = str(item or '').strip().lower()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def cache_key(image_bytes, prompt, *, threshold):
    """Content-addressed: the same picture under another name is the same mask,
    and an edited picture is a different one. The phrases and the threshold are
    part of the key because they change the answer — normalized and sorted
    first, so 'Head, glasses' and 'glasses,head' are one entry, not two."""
    h = hashlib.sha256()
    h.update(hashlib.sha256(image_bytes).digest())
    h.update(b'\x00')
    h.update('\x1f'.join(sorted(normalize_prompts(prompt))).encode('utf-8'))
    h.update(b'\x00')
    h.update(f'{float(threshold):.3f}'.encode('ascii'))
    return h.hexdigest()[:32]


def cached_path(key):
    path = cache_dir() / f'{key}.png'
    return str(path) if path.is_file() else None


# --- Running it --------------------------------------------------------------

def mask_images(image_paths, prompt, *, threshold=DEFAULT_THRESHOLD,
                out_dir=None, timeout=DEFAULT_TIMEOUT_S, on_progress=None,
                should_stop=None, device=None):
    """Mask every image in one child process. Returns the child's JSON dict
    ({'ok', 'written', 'cancelled', 'results'}), with 'out_dir' added.

    BATCHED because the cost here is the load, not the image: importing torch
    and reading a 3.4 GB checkpoint is tens of seconds, and one child per image
    would pay it every time. Every caller that has more than one image should
    come through here rather than looping over `mask_for`."""
    paths = [p for p in (image_paths or []) if p and os.path.isfile(p)]
    if not paths:
        return {'ok': True, 'written': 0, 'cancelled': False, 'results': {},
                'out_dir': out_dir}
    prompts = normalize_prompts(prompt)
    if not prompts:
        raise ValueError('a phrase naming the region is required')
    python, checkpoint = preflight()
    out_dir = out_dir or tempfile.mkdtemp(prefix='lds-masks-')
    os.makedirs(out_dir, exist_ok=True)

    cancel_file = os.path.join(out_dir, '.stop')
    payload = json.dumps({
        'images': paths, 'prompts': prompts, 'out_dir': out_dir,
        'checkpoint': checkpoint, 'threshold': float(threshold),
        'device': device or (cfg.get('auto_mask.device') or None),
        'cancel_file': cancel_file,
    })

    def _line(line):
        if on_progress:
            record = parse_progress_line(line)
            if record:
                on_progress(record)

    # Held across the whole child, not just its launch: two SAM 3 processes
    # overlapping for one second is already two checkpoints on the card.
    if not _INFER_LOCK.acquire(blocking=False):
        logger.info('auto_mask: waiting for the running masking child (%d image%s queued)',
                    len(paths), '' if len(paths) == 1 else 's')
        _INFER_LOCK.acquire()
    try:
        stdout, stderr_lines, rc, timed_out = run_infer_script(
            python, _SCRIPT, payload, timeout, on_line=_line,
            should_stop=should_stop,
            on_stop=lambda: open(cancel_file, 'w').close())
    finally:
        _INFER_LOCK.release()
    if timed_out:
        raise AutoMaskUnavailable(
            f'the masking run did not finish within {timeout}s')
    line = next((ln for ln in reversed((stdout or '').splitlines())
                 if ln.strip().startswith('{')), '')
    if not line:
        raise AutoMaskUnavailable(
            f'the masking run produced no result (rc={rc}): '
            f'{stderr_tail(stderr_lines)}')
    try:
        result = json.loads(line)
    except ValueError as e:
        raise AutoMaskUnavailable(f'unreadable masking result: {e}') from e
    if not result.get('ok'):
        raise AutoMaskUnavailable(result.get('error') or 'the masking run failed')
    result['out_dir'] = out_dir
    return result


def mask_for(image_path, prompt, *, threshold=DEFAULT_THRESHOLD, use_cache=True,
             on_progress=None):
    """The mask for ONE image: a path to an 8-bit PNG, white where `prompt`
    matched. Cached on the image's content.

    Raises NoMatch when the phrase matched nothing or matched the whole frame —
    both are answers that would silently do nothing downstream — ValueError on a
    missing image or an empty phrase, and AutoMaskUnavailable when the engine
    itself cannot run."""
    prompts = normalize_prompts(prompt)
    if not prompts:
        raise ValueError('a phrase naming the region is required')
    label = ' + '.join(prompts)
    if not image_path or not os.path.isfile(image_path):
        raise ValueError(f'image not found: {image_path}')
    with open(image_path, 'rb') as fh:
        raw = fh.read()
    key = cache_key(raw, prompt, threshold=threshold)
    if use_cache:
        hit = cached_path(key)
        if hit:
            return hit

    work_dir = tempfile.mkdtemp(prefix='lds-mask-')
    try:
        result = mask_images([image_path], prompt, threshold=threshold,
                             out_dir=work_dir, on_progress=on_progress)
        info = (result.get('results') or {}).get(image_path) or {}
        state = info.get('state')
        if state == 'no_match':
            raise NoMatch(f'nothing in this image matched “{label}” — try '
                          'another phrase, or a lower threshold')
        if state != 'ok':
            raise AutoMaskUnavailable(state or 'the mask could not be produced')
        coverage = float(info.get('coverage') or 0.0)
        if coverage >= MAX_COVERAGE:
            raise NoMatch(
                f'“{label}” matched {coverage:.0%} of the picture, which is not '
                'a region — name something narrower')
        produced = os.path.join(
            work_dir, os.path.splitext(os.path.basename(image_path))[0] + '.png')
        if not os.path.isfile(produced):
            raise AutoMaskUnavailable('the mask file was not written')
        target = cache_dir() / f'{key}.png'
        tmp = target.with_suffix('.png.tmp')
        shutil.copyfile(produced, tmp)
        os.replace(tmp, target)
        return str(target)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def env_python_default() -> str:
    """Where Setup builds the automask interpreter. Kept here so the service and
    the installer cannot disagree about it."""
    env = cfg.data_dir() / 'envs' / 'automask'
    return str(env / ('Scripts' if sys.platform == 'win32' else 'bin')
               / ('python.exe' if sys.platform == 'win32' else 'python'))
