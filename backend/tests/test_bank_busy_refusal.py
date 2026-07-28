"""A bank refusing a click because it is BUSY must say WHICH pass is in the way.

The report behind this file: 🎚 Filter thresholds ▸ "↻ Re-group duplicates"
returned a red banner reading "a scan job is already running on this bank" and
nothing else — no progress, no remedy. The front-end can only rephrase that into
"✨ Score pass is running — 137/412, press Stop above" if it is told the KIND;
parsing our own English sentence would be a rename away from breaking.

The refusal frequently lands BEFORE the first 2 s progress poll, so the response
body is at that instant the only thing on the client that knows what is running.
Hence: every occupied-bank 409 carries `busy_kind`.
"""
import os
import time

import pytest

from tests.test_image_bank import checkerboard  # noqa: F401 — image factory


def _mkbank(client, tmp_path, names):
    src = tmp_path / 'src'
    os.makedirs(src, exist_ok=True)
    for n in names:
        checkerboard().save(str(src / n), 'JPEG', quality=92)
    r = client.post('/api/bank/create', json={'name': 'busy', 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


@pytest.fixture()
def occupied(client, tmp_path):
    """A bank with a live `score` job pinned in the registry — the exact state
    the report was filed from (an ✨ Analyze all was walking the bank)."""
    from app.services import bank_jobs
    bank_id, _src = _mkbank(client, tmp_path, ['a.jpg'])
    bank_jobs._jobs[bank_id] = {
        'kind': 'score', 'done': 137, 'total': 412, 'error': None,
        'cancelled': False, 'finished': False, 'detail': None,
        'started_at': time.time(), '_touched': time.time(),
        '_cancel_hook': None, 'pipeline': None,
    }
    yield bank_id
    bank_jobs.reset()


# Bank POSTs whose occupancy refusal is reachable on a bare bank. Every start_*
# checks its OWN prerequisite before asking bank_jobs for the slot, so on a bank
# with nothing scanned yet the other passes answer their prerequisite (400/503)
# rather than the occupancy — a deliberate order, and the reason this list is not
# "every route". They all come back through the same `_busy` helper regardless.
BUSY_ROUTES = [
    ('scan', {}),          # ↻ Re-group duplicates — the button in the report
    ('watermark', {}),
    ('framing', {}),
    ('caption', {}),
]


@pytest.mark.parametrize('path,body', BUSY_ROUTES)
def test_every_busy_409_names_the_pass_holding_the_bank(client, occupied, path, body):
    r = client.post(f'/api/bank/{occupied}/{path}', json=body)
    assert r.status_code == 409, (path, r.status_code, r.get_json())
    data = r.get_json()
    assert data['busy_kind'] == 'score', (path, data)


def test_the_other_threshold_button_refuses_the_same_way(client, occupied, monkeypatch):
    """↻ Re-find the same shots (semantic-dedup) — the panel's second button.
    Its "run ✨ Score first" prerequisite is checked before the slot, so the
    occupancy refusal only appears once embeddings exist."""
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_load_score_embeddings', lambda _bank: {1: [0.0]})
    r = client.post(f'/api/bank/{occupied}/semantic-dedup', json={})
    assert r.status_code == 409, r.get_json()
    assert r.get_json()['busy_kind'] == 'score'


def test_delete_rejected_refuses_with_the_same_shape(client, occupied):
    """The one occupied-bank refusal raised as a RuntimeError rather than
    BankJobBusy. It must still answer in the shape the UI rephrases."""
    r = client.post(f'/api/bank/{occupied}/delete-rejected', json={})
    assert r.status_code == 409
    assert r.get_json()['busy_kind'] == 'score'


def test_the_english_sentence_is_kept_for_non_ui_callers(client, occupied):
    """`error` stays what it always was: removing it would break anything that
    only knows how to print a message (scripts, the API docs' examples)."""
    data = client.post(f'/api/bank/{occupied}/scan', json={}).get_json()
    assert 'already running on this bank' in data['error']


def test_a_free_bank_still_starts_its_pass(client, tmp_path):
    """The guard must not have turned into a blanket refusal."""
    from app.services import bank_jobs
    bank_jobs.reset()
    bank_id, _src = _mkbank(client, tmp_path, ['a.jpg'])
    r = client.post(f'/api/bank/{bank_id}/scan', json={})
    assert r.status_code == 202, r.get_json()
    assert 'busy_kind' not in (r.get_json() or {})
