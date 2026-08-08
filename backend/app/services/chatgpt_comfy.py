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
import time
import uuid

from .. import config as cfg
from ..utils import comfy_fs
from ..utils.comfyui import (api_address, fetch_object_info_classes,
                             fetch_output_image_bytes, get_comfyui_history,
                             queue_prompt_to_comfyui)

logger = logging.getLogger(__name__)

#: The node the lane drives. `OpenAIGPTImageNodeV2` exists too, but its `model`
#: input is a COMFY_DYNAMICCOMBO_V3 whose sub-inputs change with the chosen
#: model — a shape a hand-built graph cannot fill without mirroring ComfyUI's
#: own combo resolution. `OpenAIGPTImage1` takes plain inputs and reaches the
#: same models, `gpt-image-2` included (it is that node's own default).
NODE_CLASS = 'OpenAIGPTImage1'
BATCH_NODE_CLASS = 'ImageBatch'

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
_POLL_SECONDS = 2.0
_TIMEOUT_SECONDS = 420.0


class ComfyGptUnavailable(Exception):
    """This lane cannot run, with a reason the user can act on. Raised BEFORE
    anything is staged or queued so a batch fails on its first row instead of
    once per tile."""


def api_key() -> str:
    return (os.environ.get(API_KEY_ENV) or '').strip()


def node_available() -> bool:
    """Does the target ComfyUI expose the API node at all? An install older than
    the API nodes, or one running with them disabled, answers no."""
    try:
        classes = fetch_object_info_classes()
    except Exception:                              # a broken probe never decides
        return True
    return not classes or NODE_CLASS in classes


def status() -> dict:
    """Readiness for the capabilities payload / Settings, without calling OpenAI."""
    return {'key': bool(api_key()), 'node': node_available(),
            'node_class': NODE_CLASS, 'url': api_address()}


def preflight() -> None:
    """Raise ComfyGptUnavailable unless this lane can actually run."""
    if not api_key():
        raise ComfyGptUnavailable(
            'ChatGPT through ComfyUI needs a comfy.org API key — add one in '
            'Settings ▸ Engines (create it at platform.comfy.org).')
    if not node_available():
        raise ComfyGptUnavailable(
            f'This ComfyUI does not expose {NODE_CLASS}. Update ComfyUI, or '
            'switch the ChatGPT lane back to an API key / your subscription.')


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


def build_workflow(image_names, prompt, *, size='auto', model=DEFAULT_MODEL,
                   quality='high', filename_prefix='LDS_ChatGPT') -> dict:
    """LoadImage(s) -> [ImageBatch chain] -> OpenAIGPTImage1 -> SaveImage.

    Several references become ONE batched IMAGE, which is what the node's single
    `image` input takes — the same "every reference rides along" contract the
    direct API lane has. With no reference at all the input is simply absent,
    and the node generates instead of editing."""
    graph = {}
    loaded = []
    for index, name in enumerate(image_names or ()):
        node_id = str(10 + index)
        graph[node_id] = {'class_type': 'LoadImage',
                          'inputs': {'image': name},
                          '_meta': {'title': f'Reference {index + 1}'}}
        loaded.append(node_id)

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
              'model': model, 'n': 1,
              # The node documents `seed` as "not implemented yet in backend";
              # it is sent anyway so a future build that honours it sees a fresh
              # value per shot rather than a constant.
              'seed': uuid.uuid4().int % 2_147_483_647}
    if image_link is not None:
        inputs['image'] = image_link
    graph['100'] = {'class_type': NODE_CLASS, 'inputs': inputs,
                    '_meta': {'title': 'ChatGPT image (comfy.org credits)'}}
    graph['101'] = {'class_type': 'SaveImage',
                    'inputs': {'images': ['100', 0],
                               'filename_prefix': filename_prefix},
                    '_meta': {'title': 'Save'}}
    return graph


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
            quality=str(cfg.get('engines.chatgpt_comfy_quality') or 'high'),
            filename_prefix=f'LDS_ChatGPT_{tag}')
        response, error = queue_prompt_to_comfyui(
            workflow, f'lds-chatgpt-{tag}',
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
                    raise ComfyGptUnavailable(f'ChatGPT via ComfyUI: {failure}')
                filename, subfolder = _saved_image(entry)
                if filename:
                    return fetch_output_image_bytes(filename, subfolder)
            time.sleep(_POLL_SECONDS)
        raise ComfyGptUnavailable(
            f'ChatGPT via ComfyUI did not finish within '
            f'{int(_TIMEOUT_SECONDS)}s — check the ComfyUI queue')
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
