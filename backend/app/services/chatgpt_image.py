"""ChatGPT image (OpenAI gpt-image-2) variation generator for the face Dataset Maker.

Same contract as `nanobanana.generate_variation` so the dataset fan-out treats
both API engines uniformly: reference photo(s) + variation prompt -> generated
image bytes (or None). Uses the multipart /images/edits endpoint (image[0] = the
base to edit, the rest = extra identity references, 16 max). No GPU, no ComfyUI —
SFW only by provider policy (moderation 400 -> None, the row just fails).

CHOOSING THE MODEL
------------------
`engines.chatgpt_image_model` is free text (Settings ▸ Image engines): OpenAI
ships image models faster than this app ships releases. Resolution order, read at
CALL time so a change in Settings applies without a restart:

    engines.chatgpt_image_model  >  CHATGPT_IMAGE_MODEL (env)  >  DEFAULT_IMAGE_MODEL

The environment variable is still honoured, and above the built-in default: it
existed before the setting did. Only an explicit slug typed in Settings outranks
it — which is why the config default is BLANK rather than a copy of
DEFAULT_IMAGE_MODEL (see config.DEFAULTS['engines']).

SCOPE: this is the image model of the API-KEY lane. The subscription lane renders
through OpenAI's own `image_generation` tool, which serves that plan's image model
(gpt-image-2 today) and takes no slug from us — so the setting does not apply
there, and the field says so. `engines.chatgpt_subscription_model` is a third
thing again: the Codex ROUTER model of that lane, which decides nothing about the
pixels. None of the three are ever merged.

FAILING LOUDLY
--------------
A rejected key, an unknown model, or a model gated behind OpenAI organization
verification raises a NAMED, FATAL error (see engine_errors) that stops the
batch instead of returning None — which the fan-out would have worded as "empty
response (often a content-policy refusal)", the wrong sentence entirely.

BOTH LANES ANSWER THE SAME EVENT THE SAME WAY
---------------------------------------------
They did not. The API-key lane raised for a 5xx and for an unreachable host; the
subscription lane returned None for EVERY non-200 except 429/401, and for every
transport error — a dropped connection, a read timeout, a 500, a lane OpenAI had
closed, all came back as the same blank shrug, and the fan-out then wrote the
sentence reserved for "the provider answered and produced nothing". One file,
one event, two opposite answers, and the user in front of the muter of the two.

Both lanes now sort the SAME event into the same four buckets, and no bucket
borrows another's words:

  refusal    the provider answered and declined THIS request -> EngineRefused
             (not fatal: the next row may well pass)
  breakdown  network, timeout, 5xx -> EngineError, this row only
  quota      429 -> SubscriptionQuotaExceeded / EngineError (rate limit)
  door shut  the token is rejected after a refresh, or the experimental
             subscription endpoint answers 403/404/410 -> the run stops, and it
             says WHICH of the two happened

Nothing anywhere offers a remedy that is not one. No message says retry, try
again or rephrase: whether the same call would pass a second time is exactly
what we do not know, and the previous wording guessed it wrong in both
directions at once (see nanobanana.py and test_chatgpt_refusal.py).

WHAT STAYS HONESTLY UNKNOWN
---------------------------
A 200 with no image and no readable reason is still `None`. So is a 400 whose
body names no cause we can recognise: the fan-out words those as "a
content-policy refusal and a transient API error look identical here", which is
the truth, and inventing a cause to fill the silence would be the worse bug.
"""
from __future__ import annotations
import base64
import json
import logging
import os
import uuid

import requests

from .. import config as cfg
from .engine_errors import EngineError, EngineFatal, EngineRefused

logger = logging.getLogger(__name__)

# gpt-image-2 = the only current model usable without OpenAI organization
# verification (gpt-image-1.5 / chatgpt-image-latest 403 without it). That trap
# is surfaced in the Settings field's own help, not just here: someone who types
# a newer slug and gets a 403 must not conclude their key is broken.
DEFAULT_IMAGE_MODEL = 'gpt-image-2'
_ENV_VAR = 'CHATGPT_IMAGE_MODEL'
# Dataset images are final training material -> default to 'high' (≈ Nano
# Banana's price point). Override with CHATGPT_IMAGE_QUALITY=medium to iterate.
CHATGPT_IMAGE_QUALITY = os.environ.get('CHATGPT_IMAGE_QUALITY', 'high')
_API = "https://api.openai.com/v1/images/edits"

_NO_KEY = ('no OpenAI API key saved — add OPENAI_API_KEY in '
           'Settings > Image engines, or connect a ChatGPT subscription')

# Fragments OpenAI uses when a 400 blames the MODEL rather than this prompt: an
# unknown slug, or one that /images/edits does not serve (a text-to-image-only
# model can never work here — the dataset generator always sends reference
# images). Matching the provider's own words keeps a genuine moderation 400 —
# which IS per-prompt — from cancelling the batch.
_MODEL_FAULT_HINTS = (
    'model_not_found', 'does not exist', 'do not have access',
    'supported values are', 'not supported', 'does not support', 'unsupported',
    'must be verified', 'organization must be verified',
)

# Fragments OpenAI uses when a 400 blames the CONTENT — its own moderation stack
# speaking, not a malfunction. Narrow on purpose: everything not matched here
# stays "cause unknown" rather than being talked into a refusal it may not be.
_MODERATION_HINTS = (
    'moderation_blocked', 'content_policy_violation', 'content policy',
    'safety system', 'safety_violation', 'rejected by our safety',
)

# Fragments OpenAI uses when a 400 blames the REFERENCE PHOTOS we uploaded. This
# one is worth telling apart from a moderation 400 because the remedy is the
# opposite end of the app: a broken/oversized reference is a file to replace, not
# a prompt to reconsider — and the whole batch shares the same references, so it
# would fail identically on every row.
_IMAGE_FAULT_CODES = ('invalid_image', 'image_parse_error', 'invalid_image_format',
                      'unsupported_image', 'image_too_large')
_IMAGE_FAULT_HINTS = ('invalid image', 'could not be decoded', 'unsupported image',
                      'image is too large', 'invalid file format')

# The one sentence this project uses when it genuinely cannot tell the two
# apart. Reused verbatim by the fan-out for a 200-with-no-image.
_AMBIGUOUS = ('a content-policy refusal and a transient API error look '
              'identical here')

# --- Subscription lane (Codex OAuth) -----------------------------------------
# EXPERIMENTAL: renders gpt-image-2 on the user's ChatGPT subscription quota via
# the Codex Responses backend. Undocumented lane — may break if OpenAI closes it.
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
# The Codex lane accepts far fewer input images than /images/edits (16).
SUBSCRIPTION_MAX_REFS = 5
SUBSCRIPTION_ROUTER_MODEL = 'gpt-5.4-mini'   # routing model only; images are gpt-image-2


class ChatGPTImageError(EngineError):
    """A named ChatGPT-image failure. User-facing text, never carries the key."""


class ChatGPTImageFatal(ChatGPTImageError, EngineFatal):
    """A failure that would repeat on every remaining row (missing/rejected key,
    unknown model, model gated behind organization verification)."""


class ChatGPTImageRefused(ChatGPTImageError, EngineRefused):
    """OpenAI answered and declined THIS request (its moderation stack, or the
    model replying in prose instead of pixels). Not fatal: the batch keeps going,
    and the fan-out counts these apart from breakdowns."""


class SubscriptionQuotaExceeded(RuntimeError):
    """429 on the subscription lane: the plan's image quota is exhausted, so
    every later call in the batch would fail too. Callers stop the batch —
    None (row-level failure) and this (batch-level stop) are different channels."""


class SubscriptionUnavailable(RuntimeError):
    """The subscription lane was selected but the ChatGPT connection is gone
    (token expired and refresh failed mid-batch). Stop the batch — never fall
    back to the paid API key. Distinct from SubscriptionQuotaExceeded so the
    batch can show the right message."""


def _api_key():
    return cfg.secret('OPENAI_API_KEY')


def get_image_model() -> str:
    """The image model this engine will ask for: setting > env var > built-in
    default. Read fresh on every call, so a slug typed in Settings applies to the
    very next generation with no restart."""
    return ((cfg.get('engines.chatgpt_image_model') or '').strip()
            or (os.environ.get(_ENV_VAR) or '').strip()
            or DEFAULT_IMAGE_MODEL)


def _error_body(resp):
    """The parsed JSON of an error response, or None. `resp.text` on a mocked or
    streamed response can be anything, so every read is defensive."""
    try:
        body = resp.json()
    except Exception:                                  # noqa: BLE001 — non-JSON edge error
        body = None
    if isinstance(body, dict):
        return body
    try:
        return json.loads(resp.text or '')
    except Exception:                                  # noqa: BLE001
        return None


def _error_message(resp) -> str:
    """The provider's own explanation, trimmed. The API lane's documented
    envelope is {"error": {"message", "type", "param", "code"}}; the Codex
    subscription backend uses {"detail": "..."} instead, and an edge/proxy
    failure answers neither — so fall back to the raw text. Reading only the
    documented shape is how the subscription lane's reasons were being thrown
    away before they ever reached a tile."""
    body = _error_body(resp)
    if isinstance(body, dict):
        err = body.get('error')
        if isinstance(err, dict):
            msg = str(err.get('message') or '').strip()
            if msg:
                return msg[:300]
        elif isinstance(err, str) and err.strip():
            return err.strip()[:300]
        for key in ('detail', 'message'):              # Codex / generic gateways
            val = body.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()[:300]
    try:
        return (resp.text or '').strip()[:300]
    except Exception:                                  # noqa: BLE001
        return ''


def _err_field(resp, name: str) -> str:
    body = _error_body(resp) or {}
    err = body.get('error')
    return str((err or {}).get(name) or '') if isinstance(err, dict) else ''


def _blames_the_content(resp, detail: str) -> bool:
    """Is this 400 OpenAI's moderation stack, in its own words? Matched on the
    documented code first, then on wording — and on NOTHING else, so an
    unrecognised 400 stays unexplained instead of being called a refusal."""
    if _err_field(resp, 'code') in ('moderation_blocked', 'content_policy_violation'):
        return True
    low = detail.lower()
    return any(h in low for h in _MODERATION_HINTS)


def _blames_our_images(resp, detail: str) -> bool:
    """Is this 400 about the reference photos we uploaded rather than the prompt?
    A malformed or oversized reference reads exactly like a content refusal in
    the old wording, and sends the user to rewrite a prompt that was never the
    problem."""
    if _err_field(resp, 'code') in _IMAGE_FAULT_CODES:
        return True
    if _err_field(resp, 'param').startswith('image'):
        return True
    low = detail.lower()
    return any(h in low for h in _IMAGE_FAULT_HINTS)


def _blames_the_model(resp, detail: str) -> bool:
    """Does this 400 blame the model rather than the prompt? OpenAI marks it in
    the envelope (`param: "model"`, `code: "model_not_found"`) when it can; the
    wording check catches the rest."""
    try:
        err = (resp.json() or {}).get('error') or {}
    except Exception:                                  # noqa: BLE001
        err = {}
    if isinstance(err, dict):
        if str(err.get('param') or '') == 'model':
            return True
        if str(err.get('code') or '') in ('model_not_found', 'invalid_model'):
            return True
    low = detail.lower()
    return any(h in low for h in _MODEL_FAULT_HINTS)


def _raise_for_api_status(resp, *, model: str) -> None:
    """Turn a non-200 from the API lane into the most specific outcome we can
    justify — and no more specific than that.

    401/403 (the key, or an unverified organization) and 404 / a model-blaming
    400 would refuse every other row identically, so they are FATAL. 429 and 5xx
    are transient and stay per-row. A 400 is sorted by what OpenAI says it is:
    the model, the reference photos, its moderation stack, or — the case that
    still returns instead of raising — none of those, where the caller words it
    as the genuine ambiguity it is."""
    status = resp.status_code
    if status == 200:
        return
    detail = _error_message(resp)
    suffix = f': {detail}' if detail else ''
    if status == 401:
        raise ChatGPTImageFatal(f'OpenAI rejected the API key (HTTP 401){suffix}')
    if status == 403:
        # The organization-verification wall lands here: `gpt-image-2` is the one
        # model that does not need it, so name that instead of leaving the user to
        # suspect their key.
        raise ChatGPTImageFatal(
            f'OpenAI refused the model "{model}" (HTTP 403){suffix} — models newer '
            f'than {DEFAULT_IMAGE_MODEL} need OpenAI organization verification; '
            f'set the model back to {DEFAULT_IMAGE_MODEL} in Settings > Image '
            'engines, or verify your organization with OpenAI')
    if status == 404:
        raise ChatGPTImageFatal(
            f'OpenAI does not serve the model "{model}" (HTTP 404){suffix} — '
            'check the model in Settings > Image engines')
    if status == 429:
        raise ChatGPTImageError(f'OpenAI rate-limited the request (HTTP 429){suffix}')
    if status == 400:
        if _blames_the_model(resp, detail):
            raise ChatGPTImageFatal(
                f'OpenAI refused the request for model "{model}" (HTTP 400){suffix} — '
                'this engine always sends your reference photos to the image-editing '
                'endpoint, so the model must accept image input; check the model in '
                'Settings > Image engines')
        if _blames_our_images(resp, detail):
            # Every row of a batch is sent the SAME reference photos, so this
            # repeats identically to the last one: stop, and point at the file.
            raise ChatGPTImageFatal(
                f'OpenAI could not read the reference photos (HTTP 400){suffix} — '
                'replace the reference image for this dataset; the prompt is not '
                'what was refused')
        if _blames_the_content(resp, detail):
            raise ChatGPTImageRefused(
                f"OpenAI's safety system refused this request (HTTP 400){suffix} — "
                'that filter is not configurable and LDS cannot turn it off')
        return                                         # cause unnamed -> stays ambiguous
    if 500 <= status <= 599:
        raise ChatGPTImageError(
            f'OpenAI is having trouble (HTTP {status}){suffix} — nothing was '
            'generated for this image; your prompt was not refused')
    raise ChatGPTImageError(f'OpenAI returned HTTP {status}{suffix}')


def size_for_aspect(aspect_ratio: str) -> str:
    """Map the dataset aspect strings ('1:1', '3:4', '16:9'…) onto the three
    sizes gpt-image supports. Portrait-ish -> 1024x1536, landscape-ish ->
    1536x1024, anything else (or unknown) -> square."""
    try:
        w, h = (int(x) for x in str(aspect_ratio).split(':', 1))
        if w > 0 and h > 0:
            if h > w:
                return '1024x1536'
            if w > h:
                return '1536x1024'
    except (ValueError, TypeError):
        pass
    return '1024x1024'


def parse_image_response(data) -> bytes | None:
    """Extract the first b64 image from an /images responses payload."""
    try:
        b64 = (data.get('data') or [{}])[0].get('b64_json')
        return base64.b64decode(b64) if b64 else None
    except (TypeError, ValueError, KeyError, IndexError):
        return None


def _generate_via_api(ref_bytes: bytes | list[bytes], prompt: str, model: str | None = None,
                      aspect_ratio: str = '1:1') -> bytes | None:
    """Reference photo(s) + variation prompt -> generated image bytes, or None.
    API-key lane: the multipart /images/edits endpoint.

    `ref_bytes`: one image (bytes) or a LIST (primary first — it becomes the
    edit base; extras ride along as identity references, capped at 16 total).
    NB: gpt-image-2 does NOT accept `input_fidelity` (400) — never send it."""
    key = _api_key()
    if not key:
        # An exception, not None: a missing key must never read to the user as
        # "the provider refused your prompt".
        raise ChatGPTImageFatal(_NO_KEY)
    mdl = (model or '').strip() or get_image_model()
    refs = list(ref_bytes) if isinstance(ref_bytes, (list, tuple)) else [ref_bytes]
    refs = refs[:16]
    files = [('image[]', (f'ref{i}.webp', rb, 'image/webp')) for i, rb in enumerate(refs)]
    data = {
        'model': mdl,
        'prompt': prompt,
        'size': size_for_aspect(aspect_ratio),
        'quality': CHATGPT_IMAGE_QUALITY,
    }
    try:
        # 'high' renders take 1-3 min -> generous read timeout (connect stays short).
        r = requests.post(_API, headers={"Authorization": f"Bearer {key}"},
                          data=data, files=files, timeout=(10, 420))
    except requests.RequestException as e:
        raise ChatGPTImageError(f'could not reach OpenAI: {e}')
    if r.status_code != 200:
        # Raises for everything the user can act on; returns for a moderation 400,
        # where the row simply fails and the shot can be switched to Klein (local).
        logger.warning(f"chatgpt_image: HTTP {r.status_code}: {r.text[:300]}")
        _raise_for_api_status(r, model=mdl)
        return None
    img = parse_image_response(r.json())
    if img is None:
        logger.warning("chatgpt_image: no image in response")
    return img


def _use_subscription() -> bool:
    mode = cfg.get('engines.chatgpt_auth') or 'auto'
    if mode == 'api':
        return False
    if mode == 'subscription':
        return True
    from . import chatgpt_oauth
    return chatgpt_oauth.status()['connected']


def generate_variation(ref_bytes: bytes | list[bytes], prompt: str, model: str | None = None,
                       aspect_ratio: str = '1:1', force_lane: str | None = None) -> bytes | None:
    """Reference photo(s) + variation prompt -> generated image bytes, or None.
    Routes on engines.chatgpt_auth: API key (default lane) or ChatGPT
    subscription (Codex OAuth). Raises SubscriptionQuotaExceeded on a
    subscription-quota 429, and SubscriptionUnavailable if the subscription
    lane loses its token mid-call, so batch callers can stop instead of
    burning rows / silently falling back to the paid API key.

    `force_lane`: None -> decide from engines.chatgpt_auth (single-call
    callers); 'subscription' | 'api' -> pinned lane (batch callers pin once so
    a mid-batch disconnect can't reroute later rows onto the paid API key)."""
    use_sub = (force_lane == 'subscription') or (force_lane is None and _use_subscription())
    if use_sub:
        refs = list(ref_bytes) if isinstance(ref_bytes, (list, tuple)) else [ref_bytes]
        return _generate_via_subscription(refs, prompt, aspect_ratio)
    return _generate_via_api(ref_bytes, prompt, model, aspect_ratio)


def _image_from_output(output) -> bytes | None:
    if not isinstance(output, (list, tuple)):
        return None
    for item in output:
        # The subscription backend is undocumented: a shape we did not expect
        # must read as "no image", never as an AttributeError the user meets as
        # an app crash on a tile.
        if not isinstance(item, dict):
            continue
        if item.get('type') == 'image_generation_call' and item.get('result'):
            try:
                return base64.b64decode(item['result'])
            except (ValueError, TypeError):
                return None
    return None


def _parse_sse_for_image(text: str) -> bytes | None:
    """Minimal SSE walk for the terminal image event. With stream:true (Codex
    requires it) the image arrives in a `response.output_item.done` event whose
    item is an `image_generation_call` carrying the base64 `result`; the final
    `response.completed` event ships an empty `output` (store:false), so that
    per-item event is the only place the image appears."""
    for line in text.splitlines():
        if not line.startswith('data:'):
            continue
        try:
            evt = json.loads(line[5:].strip())
        except ValueError:
            continue
        if not isinstance(evt, dict):                  # `data: null`, `data: 42`
            continue
        if evt.get('type') == 'response.completed':
            resp = evt.get('response')
            return _image_from_output(
                resp.get('output') if isinstance(resp, dict) else None)
        if evt.get('type') == 'response.output_item.done':
            img = _image_from_output([evt.get('item') or {}])
            if img:
                return img
    return None


def _text_from_output(output) -> str:
    """The words the model wrote when it answered in prose instead of pixels, and
    the error an `image_generation_call` carries when the tool itself declined.
    Relayed verbatim: paraphrasing OpenAI's own refusal would be a second guess
    on top of the one this whole change exists to remove."""
    for item in (output if isinstance(output, (list, tuple)) else []):
        if not isinstance(item, dict):
            continue
        if item.get('type') == 'image_generation_call':
            err = item.get('error')
            if isinstance(err, dict) and str(err.get('message') or '').strip():
                return str(err['message']).strip()[:300]
            if isinstance(err, str) and err.strip():
                return err.strip()[:300]
        for part in (item.get('content') or []):
            if not isinstance(part, dict):
                continue
            txt = str(part.get('text') or '').strip()
            if txt:
                return txt[:300]
    return ''


def _refusal_text_from_body(text: str) -> str:
    """Walk the streamed (or plain) Codex reply for a refusal in words. Returns
    '' when there is none — and then the caller keeps saying it does not know."""
    items = []
    for line in (text or '').splitlines():
        if not line.startswith('data:'):
            continue
        try:
            evt = json.loads(line[5:].strip())
        except ValueError:
            continue
        if not isinstance(evt, dict):
            continue
        if evt.get('type') == 'response.output_item.done':
            items.append(evt.get('item') or {})
        elif evt.get('type') == 'response.completed':
            resp = evt.get('response')
            out = resp.get('output') if isinstance(resp, dict) else None
            if isinstance(out, (list, tuple)):
                items.extend(out)
    if not items:
        try:
            parsed = json.loads(text or '')
        except ValueError:
            parsed = None
        out = parsed.get('output') if isinstance(parsed, dict) else None
        items = list(out) if isinstance(out, (list, tuple)) else []
    return _text_from_output(items)


def _raise_for_subscription_status(resp) -> None:
    """The subscription twin of `_raise_for_api_status`, and the reason this file
    stopped answering the same event two ways.

    The one status handled by the caller instead of here is 401 on the FIRST
    attempt: that is a stale token to refresh, not a failure yet."""
    status = resp.status_code
    if status == 200:
        return
    detail = _error_message(resp)
    suffix = f': {detail}' if detail else ''
    if status == 401:
        # Second attempt, i.e. still refused with a freshly refreshed token.
        raise SubscriptionUnavailable(
            f'ChatGPT refused the connection even after refreshing the token '
            f'(HTTP 401){suffix} — reconnect your ChatGPT account in '
            'Settings > Image engines')
    if status in (403, 404, 410):
        # The single most useful thing this lane can report. It rides on an
        # undocumented endpoint OpenAI never promised us; the day they close it,
        # "unknown error" would send everyone hunting through their own settings
        # for a fault that is not there.
        raise ChatGPTImageFatal(
            f'OpenAI is no longer serving image generation on this ChatGPT '
            f'subscription (HTTP {status}){suffix} — this lane is experimental '
            'and OpenAI can withdraw it at any time; switch ChatGPT access to '
            'API key in Settings > Image engines to keep generating')
    if 500 <= status <= 599:
        raise ChatGPTImageError(
            f'ChatGPT is having trouble (HTTP {status}){suffix} — nothing was '
            'generated for this image; your prompt was not refused')
    if status == 400 and _blames_the_content(resp, detail):
        raise ChatGPTImageRefused(
            f"OpenAI's safety system refused this request (HTTP 400){suffix} — "
            'that filter is not configurable and LDS cannot turn it off')
    # Everything left is a status this lane has never been documented to send.
    # Name the code and hand over OpenAI's own words; claim nothing about which
    # of the two causes it was.
    raise ChatGPTImageError(
        f'ChatGPT returned HTTP {status} on the subscription lane{suffix} — '
        f'{_AMBIGUOUS}')


def _generate_via_subscription(refs: list, prompt: str, aspect_ratio: str) -> bytes | None:
    from . import chatgpt_oauth
    refs = refs[:SUBSCRIPTION_MAX_REFS]              # primary first, extras ride along
    content = [{'type': 'input_image',
                'image_url': 'data:image/webp;base64,' + base64.b64encode(rb).decode('ascii')}
               for rb in refs]
    content.append({'type': 'input_text', 'text': prompt})
    body = {
        'model': cfg.get('engines.chatgpt_subscription_model') or SUBSCRIPTION_ROUTER_MODEL,
        'input': [{'role': 'user', 'content': content}],
        'tools': [{'type': 'image_generation', 'size': size_for_aspect(aspect_ratio),
                   'quality': CHATGPT_IMAGE_QUALITY, 'moderation': 'auto'}],
        'tool_choice': 'required',
        # The Codex responses backend has two hard requirements or it 400s:
        # store:false ({"detail":"Store must be set to false"}) and stream:true
        # ({"detail":"Stream must be set to true"}). The reply is then an SSE
        # event stream, parsed by _parse_sse_for_image below.
        'store': False,
        'stream': True,
    }
    for attempt in (0, 1):                           # attempt 1 = after a forced refresh
        token = chatgpt_oauth.access_token(force_refresh=bool(attempt))
        if not token:
            raise SubscriptionUnavailable(
                'ChatGPT connection lost — reconnect in Settings')
        headers = {'Authorization': f'Bearer {token}',
                   'chatgpt-account-id': chatgpt_oauth.account_id() or '',
                   'OpenAI-Beta': 'responses=experimental',
                   'originator': 'codex_cli_rs',
                   'session_id': str(uuid.uuid4())}
        try:
            # Same generous read timeout as the API lane: 'high' renders take minutes.
            r = requests.post(CODEX_RESPONSES_URL, headers=headers, json=body,
                              timeout=(10, 420))
        except requests.RequestException as e:
            # Was `return None`, i.e. a dropped connection reported to the user
            # as "the provider produced no image" — the API lane has raised here
            # since it shipped, and this is the same event.
            logger.warning(f"chatgpt_image: subscription request error: {e}")
            raise ChatGPTImageError(f'could not reach ChatGPT: {e}')
        if r.status_code == 401 and attempt == 0:
            continue                                   # stale token: refresh and retry
        if r.status_code == 429:
            raise SubscriptionQuotaExceeded(
                'ChatGPT subscription image quota reached — rerun in API-key mode '
                'or wait for your plan quota to reset')
        if r.status_code != 200:
            logger.warning(f"chatgpt_image: subscription HTTP {r.status_code}: {r.text[:300]}")
            _raise_for_subscription_status(r)
        # The Codex backend streams the reply (stream:true is mandatory) but
        # frequently sends NO content-type header, so we cannot trust it: sniff
        # the body head instead. An SSE stream opens with `data:`/`event:` lines;
        # anything else we treat as a plain JSON response.
        body = r.text
        head = body[:64].lstrip()
        if 'text/event-stream' in (r.headers.get('content-type') or '') \
                or head.startswith(('data:', 'event:')):
            img = _parse_sse_for_image(body)
        else:
            try:
                img = _image_from_output((r.json() or {}).get('output'))
            except ValueError:
                img = None
        if img is None:
            logger.warning("chatgpt_image: no image in subscription response")
            said = _refusal_text_from_body(body)
            if said:
                # The model answered in words. Those words ARE the reason, so
                # they are relayed as one — with no promise attached to them.
                raise ChatGPTImageRefused(
                    f'ChatGPT answered with text instead of an image: "{said}"')
            # No image, no words: this is the case we genuinely cannot read.
            # None keeps the fan-out's honest sentence instead of inventing one.
        return img
    return None
