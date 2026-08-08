"""Is a run's Hugging Face repository STILL THERE — asked now, not remembered.

WHY THIS EXISTS
---------------
``artifact_status = 'available'`` is a minute of the past. It is written once,
by ``_verify_full_transformer_artifact`` at delivery, and nothing ever writes it
again. Both panels that draw a dense run rendered that record in the PRESENT
tense: a repository whose owner deleted it overnight still read "Full model
available — Dense checkpoint and compliance metadata verified", under a link
that answers 404. The Checkpoints panel managed to print, in one sentence,
"· missing — the model is there, not on this computer".

So this module answers the question those screens were claiming to answer, with
a request that costs nothing: the Hub's model-metadata endpoint. No file is
listed, no byte of the ~26 GB checkpoint is fetched.

THREE ANSWERS, AND WHY THE THIRD IS NOT THE SECOND
--------------------------------------------------
* ``present`` — the repository answered. It is there, at this second.
* ``gone``    — the Hub answered 404 to a token it otherwise recognises.
* ``unknown`` — no token, no network, a 5xx, a refused token. NOT an absence.

Folding ``unknown`` into ``gone`` would be a worse bug than the one this fixes:
telling someone eight hours of paid GPU are lost because their Wi-Fi dropped.
Callers therefore get a state they must render, never a boolean they can misread.

WHY RAW HTTP AND NOT ``HfApi.repo_info``
----------------------------------------
``huggingface_hub`` raises ``RepositoryNotFoundError`` for 401 AND 404 alike —
it deliberately cannot tell "deleted" from "you may not look", which is exactly
the distinction this module exists to make. ``urllib`` returns the status code.

A 404 is still ambiguous on its own: the Hub hides a private repository from a
token that may not see it behind the same 404 it uses for a deleted one. So a
404 is never a verdict by itself. It is retried with every configured token, and
then confirmed against ``whoami`` — a token the Hub does not recognise can only
ever produce ``unknown``. What remains (a live token narrowed out of the
namespace it once pushed to) is named inside the sentence the user reads rather
than hidden behind it.

CACHING
-------
The Checkpoints panel re-polls itself; the Hub must not be re-asked at that
rate. A definite answer is held for five minutes and an ``unknown`` for one: an
outage should heal within a poll or two, whereas a deletion is not going to
un-happen.
"""
from __future__ import annotations

import logging
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from .. import config as cfg

logger = logging.getLogger(__name__)

PRESENT = 'present'
GONE = 'gone'
UNKNOWN = 'unknown'

_MODEL_API = 'https://huggingface.co/api/models/'
_WHOAMI_API = 'https://huggingface.co/api/whoami-v2'
_TIMEOUT = 6

# A verdict is stable; "could not check" should be retried soon.
_TTL = {PRESENT: 300, GONE: 300, UNKNOWN: 60}

# One panel open must never turn into an unbounded burst of Hub requests.
_MAX_PER_CALL = 12
# Nor may a long-lived process accumulate a row per repository it has ever seen.
_MAX_CACHED = 200

_lock = threading.Lock()
_cache: dict[str, tuple[float, dict]] = {}

_NO_TOKEN = ('No Hugging Face token is configured, so the repository could not '
             'be checked — add HF_CLOUD_TOKEN in Settings ▸ Local tools.')
_REFUSED = ('Hugging Face refused the configured token, so the repository could '
            'not be checked — check HF_CLOUD_TOKEN in Settings ▸ Local tools.')
_UNVALIDATED = ('Hugging Face answered "not found", but the configured token '
                'could not be validated — that answer cannot be trusted as a '
                'deletion. Check HF_CLOUD_TOKEN in Settings ▸ Local tools.')
_OFFLINE = 'Hugging Face could not be reached, so the repository was not checked.'
_DELETED = ('Hugging Face has no such repository: it was deleted or renamed, or '
            'the configured token is no longer allowed to see it.')
_MALFORMED = ('The repository name recorded for this run is not a Hugging Face '
              'repository id, so nothing could be checked.')

# `namespace/name`, the only shape that may be pasted into a URL here. These ids
# come from our own params, but a value that is not one must fail as "could not
# check" rather than travel into a request path.
_REPO_ID_RE = re.compile(r'^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$')


def _http_get(url, token, timeout=_TIMEOUT):
    """``(status, body)``; ``status`` is None when nothing answered at all.

    THE injection seam — tests replace this attribute and no test ever reaches
    the network. Nothing here is logged: a request diagnostic can echo an
    authorization header, and the states below are what is actionable anyway.
    """
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    req = urllib.request.Request(url, headers=headers, method='GET')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(getattr(resp, 'status', 200) or 200), resp.read(2048)
    except urllib.error.HTTPError as e:
        return int(e.code), b''
    except Exception:                       # noqa: BLE001 — offline is a state, not a crash
        return None, b''


def _tokens():
    """Every token that could legitimately see a delivery repo, best first.

    HF_CLOUD_TOKEN is the one the pod pushed with, so it is the one that must be
    able to read it back. HF_TOKEN is tried second and only ever to AVOID a
    false "gone": two 404s from two different tokens is a much stronger signal
    than one, and the cost is paid on the rare path only.
    """
    out = []
    for key in ('HF_CLOUD_TOKEN', 'HF_TOKEN'):
        token = cfg.secret(key)
        if token and token not in out:
            out.append(token)
    return out


def _token_is_live(token) -> bool:
    """Does the Hub still recognise this token? Only asked to qualify a 404."""
    status, _body = _http_get(_WHOAMI_API, token)
    return status == 200


def _probe(repo_id, tokens) -> tuple[str, str]:
    """``(state, detail)`` for ONE repository. Never raises, never asserts more
    than it measured."""
    if not _REPO_ID_RE.match(str(repo_id)):
        return UNKNOWN, _MALFORMED
    if not tokens:
        return UNKNOWN, _NO_TOKEN
    url = _MODEL_API + str(repo_id)
    not_found_with = None
    for token in tokens:
        status, _body = _http_get(url, token)
        if status == 200:
            return PRESENT, 'Hugging Face answered for this repository.'
        if status == 404:
            # Not a verdict yet: another token may simply be the one allowed to
            # see it, and a dead token 404s on everything.
            not_found_with = token
            continue
        if status in (401, 403):
            continue                        # refused ≠ absent
        return UNKNOWN, (
            f'Hugging Face answered {status}, so the repository was not checked.'
            if status else _OFFLINE)
    if not_found_with is None:
        return UNKNOWN, _REFUSED
    if not _token_is_live(not_found_with):
        return UNKNOWN, _UNVALIDATED
    return GONE, _DELETED


def _now_iso():
    # Naive UTC on purpose: every other stamp this app persists or renders
    # (delivery_last_checked_at, artifact_verified_at) has that shape, and a
    # lone '+00:00' would sort and compare differently from all of them.
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec='seconds')


def _cached(repo_id):
    with _lock:
        hit = _cache.get(repo_id)
    if not hit:
        return None
    stamped, payload = hit
    if time.monotonic() - stamped > _TTL.get(payload['state'], 60):
        return None
    return {**payload, 'cached': True}


def _store(repo_id, payload):
    with _lock:
        _cache[repo_id] = (time.monotonic(), payload)
        # A process that never restarts must not accumulate one entry per
        # repository it has ever seen. Expired rows are dead weight by
        # definition; dropping them when the map gets big is enough here — this
        # holds a handful of runs, not a workload.
        if len(_cache) > _MAX_CACHED:
            now = time.monotonic()
            for key, (stamped, value) in list(_cache.items()):
                if now - stamped > _TTL.get(value['state'], 60):
                    _cache.pop(key, None)


def clear_cache():
    """For tests and for a token change — the next question is asked for real."""
    with _lock:
        _cache.clear()


def check(repo_id, *, force=False) -> dict:
    """Whether ``repo_id`` is on Hugging Face right now.

    ``{'repo_id', 'state', 'detail', 'checked_at', 'cached'}``. Never raises:
    every failure is an ``unknown`` with a sentence saying so.
    """
    repo_id = str(repo_id or '').strip()
    if not repo_id:
        return {'repo_id': '', 'state': UNKNOWN, 'cached': False,
                'checked_at': _now_iso(),
                'detail': 'This run recorded no Hugging Face repository.'}
    if not force:
        hit = _cached(repo_id)
        if hit:
            return hit
    try:
        state, detail = _probe(repo_id, _tokens())
    except Exception:                       # noqa: BLE001 — a bug here is not an absence
        logger.warning('Hugging Face presence check failed for a dense run',
                       exc_info=True)
        state, detail = UNKNOWN, _OFFLINE
    payload = {'repo_id': repo_id, 'state': state, 'detail': detail,
               'checked_at': _now_iso(), 'cached': False}
    _store(repo_id, payload)
    return payload


def check_many(repo_ids, *, force=False) -> dict:
    """``{repo_id: check(...)}`` for a panel's worth of runs, de-duplicated.

    Stops asking the network the moment one probe says nothing answered: when
    the Hub is unreachable the next eleven requests will time out identically,
    and the panel would wait a minute to be told the same thing twelve times.
    """
    out, offline = {}, False
    for repo_id in list(dict.fromkeys(str(r or '').strip() for r in repo_ids
                                      if str(r or '').strip()))[:_MAX_PER_CALL]:
        if offline:
            out[repo_id] = {'repo_id': repo_id, 'state': UNKNOWN,
                            'detail': _OFFLINE, 'checked_at': _now_iso(),
                            'cached': False}
            continue
        result = check(repo_id, force=force)
        out[repo_id] = result
        if result['state'] == UNKNOWN and result['detail'] == _OFFLINE \
                and not result.get('cached'):
            offline = True
    return out
