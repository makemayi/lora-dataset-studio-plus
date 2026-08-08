"""🎨 Medium runs by itself at the end of ✨ Score.

Why it may: Medium performs NO image inference. It reads the embeddings Score
just cached and multiplies them by a handful of text vectors — no GPU, and
0.16 s for 23 000 images. Leaving it behind its own button meant a bank sat
there holding the data for the answer but not the answer.

What is pinned here is what makes that safe to do automatically:
  * the chain never turns a finished Score red — not when the text encoder is
    missing, not when the classification itself blows up;
  * whatever happens is SAID on the pass's own line, including "skipped" with
    its reason, because a pass that silently did less than you think is worse
    than one that did nothing;
  * a stopped Score does not chain;
  * the manual button still works, and still re-classifies on rescan, which the
    chain deliberately never does.

No model is loaded: the prototype matrix is monkeypatched exactly as
test_bank_medium_angle.py does.
"""
import os
from collections import deque

import numpy as np

from app.extensions import db
from app.models import BankImage
from app.services import image_bank_service as banks
from test_image_bank import _mkbank, flat


def _fresh_job(kind):
    return {'kind': kind, 'done': 0, 'total': 0, 'error': None, 'cancelled': False,
            'finished': False, 'detail': None, '_touched': 0, '_cancel_hook': None,
            'pipeline': None}


def _fake_prototypes(monkeypatch, winner='photo'):
    names = ['photo', 'anime', 'render3d', 'illustration', '_text', '_screen']
    base = np.zeros((len(names), 8), dtype='float32')
    for i, n in enumerate(names):
        base[i, 0] = 0.30 if n == winner else 0.10
        base[i, 1 + i] = 0.90
        base[i] /= np.linalg.norm(base[i])
    monkeypatch.setattr(banks, '_medium_prototype_matrix', lambda: (names, base))
    from app.services import clip_text_encoder
    monkeypatch.setattr(clip_text_encoder, 'unavailable_reason', lambda: None)


def _fake_embeddings(monkeypatch, bank, rows):
    e = np.zeros(8, dtype='float32')
    e[0] = 1.0
    by_path = {os.path.normpath(os.path.join(os.path.realpath(bank.source_path),
                                             r.relpath)): e for r in rows}
    fingerprints = {
        path: banks.bank_transfer_metadata.content_fingerprint_path(path)
        for path in by_path}
    for row in rows:
        path = os.path.normpath(os.path.join(
            os.path.realpath(bank.source_path), row.relpath))
        row.analysis_fingerprint = fingerprints[path]
    db.session.commit()
    monkeypatch.setattr(banks, '_load_score_embeddings', lambda _b: by_path)
    monkeypatch.setattr(
        banks, '_score_embedding_fingerprint', fingerprints.get)


def _score_driver(state='ok'):
    """A ✨ Score child that returns an aesthetic + NSFW score per image."""
    def fake_driver(job, python, script, payload, cache_path, rx, window):
        import json
        imgs = json.loads(payload)['images']
        results = {
            p: {
                'state': state, 'aesthetic': 6.0, 'nsfw': 0.1,
                'fingerprint': (
                    banks.bank_transfer_metadata.content_fingerprint_path(p)),
            }
            for p in imgs}
        return {'ok': True, 'results': results, 'clusters': {}}, deque(), 0
    return fake_driver


def _run_score(app, bank_id, monkeypatch):
    monkeypatch.setattr(banks, '_drive_infer_subprocess', _score_driver())
    job = _fresh_job('score')
    with app.app_context():
        banks._score_job(bank_id)(job)
    return job


def test_score_classifies_the_medium_without_being_asked(app, client, tmp_path,
                                                         monkeypatch):
    bank_id, _src = _mkbank(client, tmp_path, {'a.png': flat(), 'b.png': flat()})
    with app.app_context():
        bank = db.session.get(banks.ImageBank, bank_id)
        rows = BankImage.query.filter_by(bank_id=bank_id).order_by(BankImage.id).all()
        _fake_prototypes(monkeypatch)
        _fake_embeddings(monkeypatch, bank, rows)
    job = _run_score(app, bank_id, monkeypatch)

    # The bank now HAS its mediums, from one click instead of two.
    with app.app_context():
        got = {r.relpath: r.medium for r in
               BankImage.query.filter_by(bank_id=bank_id).all()}
    assert got == {'a.png': 'photo', 'b.png': 'photo'}
    # And the pass says so — a pass that quietly did more is as bad as one that
    # quietly did less.
    assert '🎨 Medium: 2 classified' in job['detail']
    assert 'scored' in job['detail']          # its own report is still there


def test_a_missing_text_encoder_is_said_not_swallowed(app, client, tmp_path,
                                                      monkeypatch):
    """The common case on a fresh machine. It is a Setup step, not a failure of
    this bank — so Score still succeeds and the line names the reason."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.png': flat()})
    from app.services import clip_text_encoder
    monkeypatch.setattr(clip_text_encoder, 'unavailable_reason',
                        lambda: 'the CLIP text encoder is not installed')
    job = _run_score(app, bank_id, monkeypatch)
    assert job['error'] is None
    assert '🎨 Medium skipped' in job['detail']
    assert 'text encoder' in job['detail']
    with app.app_context():
        assert all(r.medium is None for r in
                   BankImage.query.filter_by(bank_id=bank_id).all())


def test_a_blowing_up_classification_never_reddens_a_finished_score(
        app, client, tmp_path, monkeypatch):
    bank_id, _src = _mkbank(client, tmp_path, {'a.png': flat()})
    with app.app_context():
        _fake_prototypes(monkeypatch)

    def boom(*_a, **_k):
        raise RuntimeError('prototype matrix exploded')

    monkeypatch.setattr(banks, '_medium_prototype_matrix', boom)
    job = _run_score(app, bank_id, monkeypatch)
    assert job['error'] is None                  # Score SUCCEEDED, and stays so
    assert 'scored' in job['detail']
    assert '🎨 Medium could not run' in job['detail']
    assert 'button re-runs it' in job['detail']


def test_a_stopped_score_does_not_chain(app, client, tmp_path, monkeypatch):
    """Stop means stop. Chaining a second phase onto a pass the user interrupted
    would be the app carrying on after being told not to."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.png': flat()})
    with app.app_context():
        bank = db.session.get(banks.ImageBank, bank_id)
        rows = BankImage.query.filter_by(bank_id=bank_id).all()
        _fake_prototypes(monkeypatch)
        _fake_embeddings(monkeypatch, bank, rows)

    def cancelled_driver(job, python, script, payload, cache_path, rx, window):
        return {'ok': True, 'cancelled': True, 'cached': 0, 'remaining': 1}, deque(), 0

    monkeypatch.setattr(banks, '_drive_infer_subprocess', cancelled_driver)
    job = _fresh_job('score')
    with app.app_context():
        banks._score_job(bank_id)(job)
    assert '🎨 Medium' not in (job['detail'] or '')
    with app.app_context():
        assert all(r.medium is None for r in
                   BankImage.query.filter_by(bank_id=bank_id).all())


def test_the_chain_finishes_the_job_and_never_re_judges(app, client, tmp_path,
                                                        monkeypatch):
    """The chain classifies what has NO verdict; re-judging an existing one is
    the manual button's ``rescan``, and stays a deliberate act."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.png': flat(), 'b.png': flat()})
    with app.app_context():
        bank = db.session.get(banks.ImageBank, bank_id)
        rows = BankImage.query.filter_by(bank_id=bank_id).order_by(BankImage.id).all()
        rows[0].medium = 'anime'          # a verdict the user already has
        db.session.commit()
        _fake_prototypes(monkeypatch, winner='photo')
        _fake_embeddings(monkeypatch, bank, rows)
    job = _run_score(app, bank_id, monkeypatch)
    with app.app_context():
        got = {r.relpath: r.medium for r in
               BankImage.query.filter_by(bank_id=bank_id).all()}
    assert got['a.png'] == 'anime'        # untouched by the chain
    assert got['b.png'] == 'photo'
    assert '🎨 Medium: 1 classified' in job['detail']
