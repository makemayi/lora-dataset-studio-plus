"""Opt-in check against a REAL local Ollama. Skipped by default, on purpose.

The unit suite is now hermetic: it never reads this machine's Ollama, because
letting it do so made ~24 tests pass or fail on whatever model happened to be
resident. That hermeticity is worth keeping, but it also means nothing in the
suite proves the fence still speaks Ollama's ACTUAL protocol — that /api/ps
answers the shape `_probe` parses, and that a `keep_alive: 0` generate really
frees the runner. A stub can drift from the daemon without either changing.

So the live proof stays available and is never automatic:

    LDS_TEST_LIVE_OLLAMA=1 python -m pytest tests/test_ollama_fence_live.py

It is read-only unless it loaded the model itself, and it refuses to run at all
when a model it did not load is resident — evicting someone else's work is the
exact thing the fence exists to prevent, and a test is not an exception.
"""
import os

import pytest

pytestmark = [pytest.mark.ollama_fence, pytest.mark.live_ollama,
              pytest.mark.skipif(os.environ.get('LDS_TEST_LIVE_OLLAMA') != '1',
                                 reason='set LDS_TEST_LIVE_OLLAMA=1 to run against a real Ollama')]

ENDPOINT = 'http://127.0.0.1:11434'


def _ps():
    import requests
    return requests.get(f'{ENDPOINT}/api/ps', timeout=(3, 5))


def test_probe_parses_what_a_real_ollama_actually_answers():
    """The shape contract, against the daemon rather than a fixture."""
    from app.services import ollama_gpu_fence as fence
    try:
        response = _ps()
    except Exception as exc:                                   # pragma: no cover
        pytest.skip(f'no local Ollama answering: {exc}')
    assert response.status_code == 200

    state, names, expiry = fence._probe(ENDPOINT)
    assert state in ('empty', 'models')
    live = {m.get('name') or m.get('model') for m in response.json().get('models', [])}
    assert names == {n for n in live if n}
    # Every residency Ollama reports must be readable as a deadline, or the
    # restart-adoption guard silently loses the fact it is built on.
    for name in names:
        assert expiry.get(name) is not None, f'unparseable expires_at for {name}'


def test_a_model_this_test_did_not_load_is_left_strictly_alone():
    from app.services import ollama_gpu_fence as fence
    try:
        state, names, _ = fence._probe(ENDPOINT)
    except Exception as exc:                                   # pragma: no cover
        pytest.skip(f'no local Ollama answering: {exc}')
    if state != 'models':
        pytest.skip('nothing resident: load a model to exercise the refusal')
    verdict = fence.mark_before_generate(ENDPOINT, 'lds-not-this-one', keep_alive=0)
    assert verdict == 'blocked'
    # And it is still there: the refusal touched nothing.
    assert fence._probe(ENDPOINT)[1] == names
