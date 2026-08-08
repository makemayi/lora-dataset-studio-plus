"""🗃️ Bank passes — an exact number is not an honest one on its own.

Three symptoms, one family: a figure the code could prove, printed without the
context that makes it readable.

  * ✨ Score went MUTE for minutes after its per-image counter reached N/N — the
    cache write and the n² style grouping publish nothing — so a full bar and a
    stale count sat there while the pass worked. The Stop button stays offered
    throughout, and what a Stop destroys is not the same at every step.
  * the style grouping reported "1 style group(s) of 2+" both when almost
    nothing grouped and when EVERYTHING did (measured: one group holding 24 928
    of 24 931 images). Two opposite failures, one sentence.
  * 🎨 Medium reported "0 classified, 2 skipped (not scored yet)" on a bank of
    50 397 images, saying nothing about the 25 464 the scope dropped before the
    pool was built.

Every assertion below is on the SENTENCE a user reads, and each negative claim
("no note when there is nothing to say") is paired with the mutation that makes
it appear — a probe that cannot fail is not a probe.
"""
import json
import os
import re
import threading
import time
from contextlib import nullcontext

from app.extensions import db
from app.models import BankImage
from app.services import image_bank_service as banks
from test_image_bank import _mkbank, flat


# --- the two-opposite-failures sentence (pure) --------------------------------

def test_one_group_swallowing_the_bank_says_so_instead_of_counting_one():
    """The measured case: 24 931 grouped images, 1 group of 24 928 + 3 of one."""
    s = banks.group_summary([24928, 1, 1, 1], 'style group', 0.6,
                            '🎚 Filter thresholds ▸ style_threshold')
    assert '24928 of 24931' in s
    assert 'too permissive' in s
    assert '0.6' in s and 'style_threshold' in s
    # …and it must NOT be readable as "almost nothing grouped".
    assert '1 style group(s) of 2+' not in s


def test_nothing_grouping_at_all_is_the_opposite_sentence():
    s = banks.group_summary([1] * 40, 'style group', 0.99)
    assert 'no two images grouped' in s and 'too strict' in s
    assert 'too permissive' not in s


def test_a_healthy_grouping_keeps_the_count_and_gains_its_shape():
    s = banks.group_summary([12, 9, 5, 1, 1], 'style group', 0.8)
    assert s.startswith('3 style group(s) of 2+')
    assert '12 of 28' in s          # the shape travels with the count
    assert 'too permissive' not in s and 'too strict' not in s


def test_no_verdict_on_a_bank_too_small_to_have_one():
    """A handful of images in one group is an ordinary answer, not a symptom —
    and a warning there would be noise on every small bank."""
    assert 'too permissive' not in banks.group_summary([5], 'style group', 0.6)
    # The floor is what suppresses it: the same shape above it does warn.
    assert 'too permissive' in banks.group_summary(
        [banks._STYLE_DIAGNOSIS_MIN], 'style group', 0.6)


def test_an_empty_grouping_names_the_reason():
    s = banks.group_summary([], 'person cluster')
    assert s == 'no person cluster — no image carried a usable embedding'


# --- what the scope left out (pure) -------------------------------------------

def test_scope_left_out_names_the_pile_the_default_scope_drops():
    """The measured bank: 25 464 rejected images with no medium, 2 undecided."""
    n, words = banks.scope_left_out(
        {'keep': 0, 'pending': 2, 'reject': 25464}, None)
    assert (n, words) == (25464, 'rejected')


def test_scope_left_out_is_silent_when_the_scope_reaches_everything():
    assert banks.scope_left_out({'keep': 3, 'pending': 2, 'reject': 0}, None) == (0, '')
    assert banks.scope_left_out({'keep': 3, 'pending': 2, 'reject': 9},
                                ['keep', 'pending', 'reject']) == (0, '')


def test_scope_left_out_lists_every_pile_it_drops():
    n, words = banks.scope_left_out({'keep': 4, 'pending': 7, 'reject': 9}, ['keep'])
    assert n == 16
    assert 'undecided' in words and 'rejected' in words


# --- the 🎨 Medium end-of-pass line (integration) -----------------------------

def _fake_medium_env(monkeypatch, bank, scored_rows):
    """The two things 🎨 Medium needs and this test refuses to load: the CLIP
    text prototypes and the cached image embeddings."""
    import numpy as np
    names = list(banks.MEDIUM_PROTOTYPES)
    base = np.zeros((len(names), 8), dtype='float32')
    for i, n in enumerate(names):
        base[i, 0] = 0.30 if n == 'photo' else 0.10
        base[i, 1 + i] = 0.90
        base[i] /= np.linalg.norm(base[i])
    monkeypatch.setattr(banks, '_medium_prototype_matrix', lambda: (names, base))
    from app.services import clip_text_encoder
    monkeypatch.setattr(clip_text_encoder, 'unavailable_reason', lambda: None)
    e = np.zeros(8, dtype='float32')
    e[0] = 1.0
    root = os.path.realpath(bank.source_path)
    by_path = {os.path.normpath(os.path.join(root, r.relpath)): e
               for r in scored_rows}
    fingerprints = {
        path: banks.bank_transfer_metadata.content_fingerprint_path(path)
        for path in by_path}
    for row in scored_rows:
        path = os.path.normpath(os.path.join(root, row.relpath))
        row.analysis_fingerprint = fingerprints[path]
    db.session.commit()
    monkeypatch.setattr(banks, '_load_score_embeddings', lambda _b: by_path)
    monkeypatch.setattr(
        banks, '_score_embedding_fingerprint', fingerprints.get)


def _bank_with_a_full_bin(app, client, tmp_path, monkeypatch):
    """1 undecided unscored image + 3 rejected ones — the measured shape of the
    bank the symptom came from, scaled down."""
    bank_id, _src = _mkbank(client, tmp_path,
                            {f'{i}.png': flat() for i in range(4)})
    with app.app_context():
        bank = db.session.get(banks.ImageBank, bank_id)
        rows = BankImage.query.filter_by(bank_id=bank_id).order_by(BankImage.id).all()
        for r in rows[1:]:
            r.status = 'reject'
        db.session.commit()
        # ✨ Score reached the rejected ones (before they were rejected) but not
        # the undecided one — so the default scope's pool is real and unusable.
        _fake_medium_env(monkeypatch, bank, rows[1:])
        for r in rows[1:]:
            r.aesthetic_score = 5.0
        db.session.commit()
    return bank_id


def test_medium_names_what_the_scope_dropped_and_what_it_could_not_score(
        app, client, tmp_path, monkeypatch):
    bank_id = _bank_with_a_full_bin(app, client, tmp_path, monkeypatch)
    with app.app_context():
        r = client.post(f'/api/bank/{bank_id}/medium', json={})
        assert r.status_code in (200, 202), r.get_json()
        detail = (banks.bank_jobs.get(bank_id) or {}).get('detail') or ''
    assert '0 classified' in detail
    assert '1 skipped (not scored yet)' in detail
    # THE MISSING FIGURE: the three rejected rows the pass never even considered.
    assert '3 image(s) left out by the scope (rejected)' in detail


def test_the_same_run_aimed_at_the_bin_has_nothing_to_report(
        app, client, tmp_path, monkeypatch):
    """The discriminating half: the note is not decoration, it appears only when
    the scope really drops work. Widen the scope and the sentence must go."""
    bank_id = _bank_with_a_full_bin(app, client, tmp_path, monkeypatch)
    with app.app_context():
        r = client.post(f'/api/bank/{bank_id}/medium',
                        json={'statuses': ['keep', 'pending', 'reject']})
        assert r.status_code in (200, 202), r.get_json()
        detail = (banks.bank_jobs.get(bank_id) or {}).get('detail') or ''
    assert '3 classified' in detail          # the bin was scored, so it answers
    assert 'left out by the scope' not in detail


def test_the_window_counts_the_rows_the_pass_cannot_answer_for(
        app, client, tmp_path, monkeypatch):
    """`blocked` is what stops the launch button promising "Classify 1 image"
    over a row with no embedding. Measured per pile, from the same clause the
    pass's own "not scored yet" comes from."""
    bank_id = _bank_with_a_full_bin(app, client, tmp_path, monkeypatch)
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    med = payload['pass_scopes']['medium']
    assert med['todo'] == {'keep': 0, 'pending': 1, 'reject': 3}
    # Only the undecided row lacks a score…
    assert med['blocked'] == {'keep': 0, 'pending': 1, 'reject': 0}
    # …so the default scope (kept + undecided) is ENTIRELY blocked, and the bin
    # scope is not. Two different answers off one payload — which is the whole
    # point of publishing it per pile.
    assert med['blocked']['pending'] == med['todo']['pending']
    assert med['blocked']['reject'] < med['todo']['reject']


# --- the mute tail of ✨ Score -------------------------------------------------

_FAKE_PHASED = '''\
import sys, json
json.loads(sys.stdin.read())
sys.stderr.write("[score] 7/7 ok\\n")
sys.stderr.write("[phase] grouping styles over 7 image(s) — Stop now keeps every "
                 "score but discards the grouping\\n")
sys.stderr.flush()
print(json.dumps({"ok": True, "results": {}, "clusters": {}}))
'''


def test_a_step_with_no_counter_replaces_the_full_bar_with_a_sentence(app, tmp_path):
    """Reproduces the symptom: the child's last counted line said 7/7 and the
    screen kept showing it (bar at 100 %) through minutes of real work. A phase
    line now takes over AND clears the count — "7 / 7 · grouping styles" would
    still read as a finished pass that hung."""
    script = tmp_path / 'fake_phased.py'
    script.write_text(_FAKE_PHASED, encoding='utf-8')
    cache_path = str(tmp_path / 'score_cache.npz')
    job = {'kind': 'score', 'done': 0, 'total': 0, 'error': None,
           'cancelled': False, 'finished': False, 'detail': None,
           'started_at': time.time(), '_touched': time.time(),
           '_cancel_hook': None, 'pipeline': None}
    out = {}

    def worker():
        out['res'] = banks._drive_infer_subprocess(
            job, __import__('sys').executable, str(script),
            json.dumps({'images': [], 'cache': cache_path,
                        'cancel_file': cache_path + '.cancel'}),
            cache_path, re.compile(r'\[score\] (\d+)/(\d+)'), nullcontext())

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive()
    data, _tail, rc = out['res']
    assert rc == 0 and data['ok']
    assert 'grouping styles' in (job['detail'] or '')
    # The Stop button is still offered here, so the sentence has to say what a
    # Stop costs AT THIS STEP — not what it costs during the inference.
    assert 'Stop' in job['detail']
    assert (job['done'], job['total']) == (0, 0), \
        'a phase with no counter must not leave the previous count on screen'


def test_the_score_child_announces_every_step_that_has_no_counter():
    """The child is the only one that knows when it leaves the counted loop. If
    these lines stop being emitted the parent has nothing to forward, so the
    contract is pinned on the source that produces them."""
    src = (banks.cfg.BACKEND_DIR / 'infer' / 'bank_score_infer.py').read_text(
        encoding='utf-8')
    assert src.count('_phase(') >= 4          # 3 call sites + the definition
    assert 'grouping styles' in src
    assert 'saving the score cache' in src
