"""ChatGPT images through ComfyUI's OpenAI API node — the THIRD auth lane.

WHAT THIS IS
------------
`engines.chatgpt_auth` had two lanes: an OpenAI API key, or a ChatGPT
subscription (Codex OAuth). This adds `comfyui`, which sends the same shot to
`OpenAIGPTImage1` inside the ComfyUI you already run, and bills comfy.org
credits instead of your OpenAI account.

The engine id does NOT change: rows keep `klein_model='chatgpt'`, the tile keeps
its badge, and a retry stays on the lane the config names at that moment. This
is an auth choice, not a new engine.

WHY IT NEEDS A KEY AND NOT A LOGIN
----------------------------------
ComfyUI's API nodes declare their credentials as HIDDEN inputs
(`IO.Hidden.api_key_comfy_org` / `auth_token_comfy_org`) and its `execution.py`
fills them from ONE place: the `extra_data` of the submitted prompt. The web UI
injects its session token there for you; a headless caller inherits nothing, so
LDS carries a comfy.org **API key** (Settings ▸ Engines) with each prompt. That
key is a secret like every other: `.env`, never `config.json`.

WHAT THIS LANE IS NOT
---------------------
It is not "local". The picture is still generated on a third-party server, so
`chatgpt` stays in API_ENGINES and the NSFW fail-closed rule still applies —
routing an NSFW shot here because the request left through 127.0.0.1 would be
the same disclosure with an extra hop.

It DOES occupy your ComfyUI queue: ComfyUI runs one prompt at a time, so a
GPT image waiting on OpenAI holds the slot a Klein/Krea render would use. That
is the cost of the lane, and it is why the two other lanes stay available.
"""
from __future__ import annotations
import logging
import os
import threading
import time
import uuid

from .. import config as cfg
from ..utils import comfy_fs
from ..utils.comfyui import (api_address, fetch_object_info_classes,
                             fetch_output_image_bytes, get_comfyui_history,
                             queue_prompt_to_comfyui)

logger = logging.getLogger(__name__)

#: The node this lane prefers, and the one it falls back to.
#:
#: V2's `model` input is a COMFY_DYNAMICCOMBO_V3, which this file first read as
#: "a hand-built graph cannot fill it". WRONG, and a workflow the maintainer had
#: on disk proved it: the sub-inputs are flattened with a dotted prefix
#: (`model.quality`, `model.images.image_1`), so an API-format graph fills them
#: like any other input. Verified against a live /object_info, 2026-08-09.
#:
#: V2 is worth preferring for one reason that is not cosmetic: its `images` is a
#: COMFY_AUTOGROW_V3 with 16 named slots, so the dataset's references arrive as
#: SEPARATE images — the shape gpt-image's edit endpoint actually takes, and the
#: same 16 the direct API lane allows. V1 has a single IMAGE input, so several
#: references have to be flattened into one batched picture first.
NODE_CLASS_V2 = 'OpenAIGPTImageNodeV2'
NODE_CLASS = 'OpenAIGPTImage1'
BATCH_NODE_CLASS = 'ImageBatch'

#: V2's own ceiling ("Up to 16 images" in its tooltip).
MAX_REFS_V2 = 16

#: Only gpt-image-2 declares the Custom-size pair, and V2 requires the pair to
#: be PRESENT whatever `size` says (see build_workflow). The floor is that
#: input's own `min`, and it is never read while size is a named enum value.
MODELS_WITH_CUSTOM_SIZE = ('gpt-image-2',)
CUSTOM_SIZE_FLOOR = 1024

#: comfy.org API key. A SECRET_KEYS entry, so it lives in .env like every other
#: credential and never reaches config.json or a diagnostics paste.
API_KEY_ENV = 'COMFY_ORG_API_KEY'

#: What the node's `model` combo accepts (asked of a live install, 2026-08-09).
#: The engine's own `engines.chatgpt_image_model` is free text — it has to be,
#: because the direct API lane must survive OpenAI renaming a model without a
#: release — so an unusable value degrades to the node's default rather than
#: failing the shot.
MODELS = ('gpt-image-1', 'gpt-image-1.5', 'gpt-image-2')
DEFAULT_MODEL = 'gpt-image-2'

#: What the node's `quality` combo accepts. THE cost dial of this lane: it moves
#: both the credits spent and the wall-clock wait, and nothing else here does.
#: Measured on the maintainer's install, three shots at `high`: 1'55", 2'11",
#: 2'12" — the time is OpenAI's, not ours, so this is the only lever that
#: touches it.
QUALITIES = ('low', 'medium', 'high')
DEFAULT_QUALITY = 'low'

#: The node's `size` enum, mapped from the catalog card's own ratio. Anything
#: else asks the node for 'auto' rather than inventing a size it would refuse.
_SIZE_BY_RATIO = {
    '1:1': '1024x1024',
    '2:3': '1024x1536',
    '3:4': '1024x1536',
    '9:16': '1024x1536',
    '3:2': '1536x1024',
    '4:3': '1536x1024',
    '16:9': '1536x1024',
}

#: An OpenAI render is a network wait, not a GPU one: the poll is slow on
#: purpose, and the ceiling is generous because the queue slot is already paid
#: for by the time we are waiting.
#:
#: 15 minutes, not the 7 this shipped with. MEASURED on the maintainer's install:
#: one reference at `low` came back in 47 s, and a four-reference shot with a
#: 1 166-character prompt was still running at 4 minutes — every reference is
#: uploaded and processed by the provider, so the shots that need the most
#: references are exactly the ones the old ceiling cut off. A ceiling that fires
#: while the picture is still coming costs the credits AND the picture.
_POLL_SECONDS = 2.0
_TIMEOUT_SECONDS = 900.0

#: One shot at a time on this lane. The API fan-out runs rows in parallel, which
#: is right for a provider that answers per request — but here every row lands in
#: the SAME single-prompt ComfyUI queue, so parallelism buys nothing and costs
#: something: four threads probing "is ComfyUI up?" at once against a busy
#: instance had three of them time out and fail rows that had nothing wrong with
#: them. Measured on the maintainer's install, 2026-08-09.
_LANE_LOCK = threading.Lock()

#: How long to keep asking before calling a busy ComfyUI "down". Its readiness
#: probe is a 3 s HTTP GET; one slow answer must not fail a row.
_READY_TIMEOUT_SECONDS = 45.0
_READY_POLL_SECONDS = 1.5


class ComfyGptUnavailable(Exception):
    """This lane cannot run, with a reason the user can act on. Raised BEFORE
    anything is staged or queued so a batch fails on its first row instead of
    once per tile."""


def api_key() -> str:
    return (os.environ.get(API_KEY_ENV) or '').strip()


def resolve_node() -> str | None:
    """Which OpenAI image node THIS ComfyUI exposes: V2 for preference, V1 as the
    fallback, None when neither is there (an install older than the API nodes, or
    one running with them disabled).

    A probe that cannot answer must not decide: an unreachable /object_info
    returns the preferred node and lets the submit fail with ComfyUI's own words,
    rather than reporting the engine as unavailable over a network blip."""
    try:
        classes = fetch_object_info_classes()
    except Exception:                              # a broken probe never decides
        return NODE_CLASS_V2
    if not classes:
        return NODE_CLASS_V2
    if NODE_CLASS_V2 in classes:
        return NODE_CLASS_V2
    return NODE_CLASS if NODE_CLASS in classes else None


def node_available() -> bool:
    return resolve_node() is not None


def status() -> dict:
    """Readiness for the capabilities payload / Settings, without calling OpenAI."""
    node = resolve_node()
    return {'key': bool(api_key()), 'node': node is not None,
            'node_class': node or NODE_CLASS_V2, 'url': api_address()}


def preflight() -> None:
    """Raise ComfyGptUnavailable unless this lane can actually run."""
    if not api_key():
        raise ComfyGptUnavailable(
            'ChatGPT through ComfyUI needs a comfy.org API key — add one in '
            'Settings ▸ Engines (create it at platform.comfy.org).')
    if resolve_node() is None:
        raise ComfyGptUnavailable(
            f'This ComfyUI exposes neither {NODE_CLASS_V2} nor {NODE_CLASS}. '
            'Update ComfyUI, or switch the ChatGPT lane back to an API key / '
            'your subscription.')


def wait_until_comfyui_answers() -> None:
    """Block until ComfyUI answers its readiness probe, or raise.

    `queue_prompt_to_comfyui` runs that probe itself and refuses on ONE failed
    answer. That is right for a local render — a GPU job on an unreachable
    ComfyUI is pointless — but this lane arrives with several rows at once and a
    3 s HTTP timeout against an instance already working is a coin flip. Waiting
    is free here: the wait is what the row would be doing anyway."""
    from .comfyui_service import ensure_comfyui_before_generation
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    detail = 'ComfyUI did not answer'
    while True:
        try:
            ok, message = ensure_comfyui_before_generation()
        except Exception as exc:                   # a broken probe never decides
            logger.debug('chatgpt/comfyui: readiness probe failed (%s)', exc)
            return
        if ok:
            return
        detail = message or detail
        if time.monotonic() >= deadline:
            raise ComfyGptUnavailable(f'ChatGPT via ComfyUI: {detail}')
        time.sleep(_READY_POLL_SECONDS)


def size_for_aspect(aspect_ratio) -> str:
    return _SIZE_BY_RATIO.get((aspect_ratio or '').strip(), 'auto')


def model_for(model) -> str:
    name = (model or '').strip()
    if name in MODELS:
        return name
    if name:
        logger.info('chatgpt/comfyui: %r is not a model this node offers — '
                    'using %s', name, DEFAULT_MODEL)
    return DEFAULT_MODEL


def quality_for(quality) -> str:
    """The configured quality, or the default — never a value the node refuses.
    An unusable setting must cost the user a cheaper picture, not the shot."""
    name = str(quality or '').strip().lower()
    if name in QUALITIES:
        return name
    if name:
        logger.info('chatgpt/comfyui: %r is not a quality this node offers — '
                    'using %s', quality, DEFAULT_QUALITY)
    return DEFAULT_QUALITY


def build_workflow(image_names, prompt, *, size='auto', model=DEFAULT_MODEL,
                   quality=DEFAULT_QUALITY, filename_prefix='LDS_ChatGPT',
                   node_class=None) -> dict:
    """LoadImage(s) -> the OpenAI image node -> SaveImage.

    V2 takes the references as SEPARATE slots (`model.images.image_1…`, up to
    16) — the shape gpt-image's edit endpoint actually has. V1 has one IMAGE
    input, so the same references are chained through ImageBatch into a single
    picture first. With no reference at all the input is simply absent on either
    node, and it generates instead of editing.

    `seed` is documented by the node as "not implemented yet in backend"; it is
    sent anyway so a build that starts honouring it sees a fresh value per shot
    rather than a constant."""
    node_class = node_class or NODE_CLASS_V2
    graph = {}
    loaded = []
    limit = MAX_REFS_V2 if node_class == NODE_CLASS_V2 else None
    for index, name in enumerate(image_names or ()):
        if limit is not None and index >= limit:
            logger.info('chatgpt/comfyui: %s takes %d references, dropping the rest',
                        node_class, limit)
            break
        node_id = str(10 + index)
        graph[node_id] = {'class_type': 'LoadImage',
                          'inputs': {'image': name},
                          '_meta': {'title': f'Reference {index + 1}'}}
        loaded.append(node_id)

    seed = uuid.uuid4().int % 2_147_483_647
    if node_class == NODE_CLASS_V2:
        # The dynamic combo, flattened: every sub-input of the chosen model is
        # sent under a `model.` prefix, and the autogrow list under
        # `model.images.<slot name>`.
        inputs = {'prompt': prompt, 'model': model, 'n': 1, 'seed': seed,
                  'model.size': size, 'model.quality': quality,
                  'model.background': 'auto'}
        if model in MODELS_WITH_CUSTOM_SIZE:
            # REQUIRED even when `size` is not 'Custom'. Their tooltip says
            # "used only when size is 'Custom'", and that is about the VALUE —
            # ComfyUI still validates the sub-input's presence and answers a
            # 400 `required_input_missing` without them. Measured: an 18-row
            # batch refused in two seconds (which at least cost nothing, since
            # validation happens before the provider is ever called). The
            # reference workflow carries the same pair at the same 1024.
            inputs['model.custom_width'] = CUSTOM_SIZE_FLOOR
            inputs['model.custom_height'] = CUSTOM_SIZE_FLOOR
        for index, node_id in enumerate(loaded, start=1):
            inputs[f'model.images.image_{index}'] = [node_id, 0]
    else:
        image_link = None
        if loaded:
            image_link = [loaded[0], 0]
            for index, node_id in enumerate(loaded[1:], start=1):
                batch_id = str(50 + index)
                graph[batch_id] = {'class_type': BATCH_NODE_CLASS,
                                   'inputs': {'image1': image_link,
                                              'image2': [node_id, 0]},
                                   '_meta': {'title': 'Chain the references'}}
                image_link = [batch_id, 0]
        inputs = {'prompt': prompt, 'size': size, 'quality': quality,
                  'model': model, 'n': 1, 'seed': seed}
        if image_link is not None:
            inputs['image'] = image_link

    graph['100'] = {'class_type': node_class, 'inputs': inputs,
                    '_meta': {'title': 'ChatGPT image (comfy.org credits)'}}
    graph['101'] = {'class_type': 'SaveImage',
                    'inputs': {'images': ['100', 0],
                               'filename_prefix': filename_prefix},
                    '_meta': {'title': 'Save'}}
    return graph


#: ComfyUI turns comfy.org's HTTP status into a sentence written for someone
#: sitting in ITS web UI, where "log in" is a button. From here that reads as
#: advice the user cannot follow — they did set a key — so the two statuses that
#: are really about the credential say what to do in THIS app instead. Matched on
#: the sentence rather than the status because the status never reaches us.
_EXPLAINED = (
    ('please login first',
     'comfy.org rejected the credential (401). The key in Settings ▸ Engines ▸ '
     'comfy.org API key is missing, mistyped or revoked — create a fresh one at '
     'platform.comfy.org. Signing into the ComfyUI app does not help: a request '
     'from LDS carries no browser session, only that key.'),
    ('payment required',
     'comfy.org accepted the key and refused the charge (402): the account is '
     'out of credits. Top it up at platform.comfy.org, or switch the ChatGPT '
     'lane back to an API key / your subscription in Settings ▸ Engines.'),
)


def _explain(message: str) -> str:
    lowered = (message or '').lower()
    for needle, replacement in _EXPLAINED:
        if needle in lowered:
            return replacement
    return message


def _give_up_message(prompt_id, client_id) -> str:
    """Stop waiting — and take the prompt with us when that is still possible.

    Giving up on the WAIT is not the same as giving up on the WORK: ComfyUI
    keeps running what it accepted, and this lane's prompt holds the single
    execution slot every local render needs. So the exact prompt is deleted when
    it is still queued (an exact-id delete, never the global /interrupt, which
    would kill whatever else is running).

    A prompt already RUNNING cannot be stopped that way, and the message says so
    rather than implying the credits were saved: OpenAI is mid-answer, and it
    will finish or fail on its own."""
    from ..utils.comfyui import ComfyPromptState, cancel_comfyui_prompt_state
    minutes = int(_TIMEOUT_SECONDS // 60)
    try:
        state = cancel_comfyui_prompt_state(prompt_id, client_id)
    except Exception:                          # never fail the failure path
        state = None
    if state is ComfyPromptState.DELETED:
        return (f'ChatGPT via ComfyUI gave up after {minutes} min — the shot was '
                'still queued behind other ComfyUI work, so it was removed from '
                'the queue. Nothing was generated and nothing was charged.')
    if state is ComfyPromptState.RUNNING:
        return (f'ChatGPT via ComfyUI gave up waiting after {minutes} min. The '
                'shot is STILL RUNNING in ComfyUI — it holds the queue slot '
                'until OpenAI answers, and it may still consume credits. A shot '
                'with several reference photos is the slow case; lowering the '
                'image quality or using fewer references makes it cheaper.')
    return (f'ChatGPT via ComfyUI did not finish within {minutes} min — '
            'check the ComfyUI queue.')


def _saved_image(history_entry) -> tuple:
    """(filename, subfolder) of the SaveImage output, or (None, None)."""
    outputs = (history_entry or {}).get('outputs') or {}
    for node_output in outputs.values():
        for image in (node_output or {}).get('images') or ():
            if isinstance(image, dict) and image.get('filename'):
                return image['filename'], image.get('subfolder') or ''
    return None, None


def _failure_message(history_entry) -> str | None:
    """ComfyUI's own words for a failed prompt — an API node reports a refused
    key, an exhausted credit balance and a provider refusal all through here, so
    quoting it beats inventing a generic 'generation failed'."""
    status_block = (history_entry or {}).get('status') or {}
    if status_block.get('status_str') != 'error':
        return None
    for message in reversed(status_block.get('messages') or ()):
        if not isinstance(message, (list, tuple)) or len(message) < 2:
            continue
        kind, payload = message[0], message[1]
        if kind in ('execution_error', 'execution_interrupted') and isinstance(payload, dict):
            text = payload.get('exception_message') or payload.get('exception_type')
            if text:
                return str(text)[:400]
    return 'ComfyUI reported an error for this prompt'


def generate_variation(ref_bytes, prompt: str, model: str | None = None,
                       aspect_ratio: str = '1:1') -> bytes | None:
    """One shot through the ComfyUI lane. Returns image bytes, or None.

    Same signature shape as the other two lanes, and the same synchronous
    contract: the API worker thread that called this owns the wait. That is what
    keeps ⏹ Stop, the activity indicator, per-row failures and retries working
    exactly as they do for the direct lanes — none of them can tell where the
    bytes came from."""
    preflight()
    with _LANE_LOCK:
        wait_until_comfyui_answers()
        return _generate_one(ref_bytes, prompt, model, aspect_ratio)


def _generate_one(ref_bytes, prompt: str, model: str | None,
                  aspect_ratio: str) -> bytes | None:
    refs = [r for r in (ref_bytes if isinstance(ref_bytes, (list, tuple)) else [ref_bytes]) if r]

    input_dir = comfy_fs.ensure_input_usable(cfg.comfyui_dir('input'))
    tag = uuid.uuid4().hex[:8]
    staged = []
    try:
        for index, raw in enumerate(refs):
            # Shape required by comfy_fs's sweep: `<lane>_<8 hex uid>_<name>`.
            # The `finally` below is what normally removes these; matching the
            # convention is what lets the 48 h safety net reach one left behind
            # by a process that died mid-shot.
            name = f'gptref{index}_{tag}_reference.png'
            comfy_fs.stage_input_write(
                name, lambda path, raw=raw: _write_png(path, raw), input_dir)
            staged.append(name)

        workflow = build_workflow(
            staged, prompt, size=size_for_aspect(aspect_ratio),
            model=model_for(model if model is not None
                            else cfg.get('engines.chatgpt_image_model')),
            quality=quality_for(cfg.get('engines.chatgpt_comfy_quality')),
            filename_prefix=f'LDS_ChatGPT_{tag}',
            node_class=resolve_node())
        client_id = f'lds-chatgpt-{tag}'
        response, error = queue_prompt_to_comfyui(
            workflow, client_id,
            extra_data={'api_key_comfy_org': api_key()})
        if error or not isinstance(response, dict) or not response.get('prompt_id'):
            raise ComfyGptUnavailable(error or 'ComfyUI refused the prompt')

        prompt_id = response['prompt_id']
        deadline = time.monotonic() + _TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            history = get_comfyui_history(prompt_id)
            entry = (history or {}).get(prompt_id)
            if entry:
                failure = _failure_message(entry)
                if failure:
                    raise ComfyGptUnavailable(
                        f'ChatGPT via ComfyUI: {_explain(failure)}')
                filename, subfolder = _saved_image(entry)
                if filename:
                    return fetch_output_image_bytes(filename, subfolder)
            time.sleep(_POLL_SECONDS)
        raise ComfyGptUnavailable(_give_up_message(prompt_id, client_id))
    finally:
        # The staged copies are full-resolution duplicates of the user's
        # reference; drop them whatever happened (see comfy_fs's own note on the
        # 3 896 orphans that convention exists to prevent).
        if staged:
            comfy_fs.drop_staged_inputs(staged, input_dir)


def _write_png(path, raw: bytes) -> None:
    """Bounded, EXIF-free PNG — the same disclosure boundary every other staged
    input crosses (comfy_fs.stage_input_image does this for files on disk; this
    lane starts from bytes already in memory)."""
    from PIL import Image
    import io
    from . import image_encoding
    with Image.open(io.BytesIO(raw)) as im:
        image_encoding.validate_input_header_dimensions(im, label='ChatGPT reference')
        im.load()
        im.convert('RGB').save(path, 'PNG')
