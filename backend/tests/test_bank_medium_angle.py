"""🗃️ Bank ↔ 🎨 Medium and ⤢ Angle: the two classifier-backed facets.

What is pinned here is the HONESTY of the two measurements, because both are
weaker than a chip row makes them look:

  * a medium verdict is given only when the CLIP margin clears a cut that was
    MEASURED (a low one for photographs, a six-times-higher one for everything
    else), and 'unsure' is a real answer rather than a placeholder;
  * an image the ✨ Score pass never reached has NO medium at all — "not scored
    yet" and "unsure" are different statements and must not merge;
  * a face with no measured yaw is "not measured", never 'frontal';
  * 'from behind' costs TWO passes and refuses to guess from one;
  * the ⤢ backfill exists, is counted, and never runs on its own.

No model is ever loaded: the prototype matrix is the one thing that needs CLIP,
and it is monkeypatched — exactly like the text-search tests do.
"""
import numpy as np
import pytest

from app.extensions import db
from app.models import BankImage
from app.services import image_bank_service as banks
from test_image_bank import _mkbank, flat


# --- the verdict rule (pure) -------------------------------------------------

def test_photo_and_non_photo_do_not_share_a_bar():
    """The measured asymmetry: a photograph is called on a small margin, an
    anime/3D/illustration verdict needs a much bigger one. Collapsing these into
    one number is what filled the 'anime' pile with cosplay photographs."""
    assert banks.MEDIUM_MARGIN_OTHER > banks.MEDIUM_MARGIN_PHOTO * 5

    # A photograph wins narrowly -> still called.
    got, margin = banks.medium_verdict(
        {'photo': 0.160, 'anime': 0.150, 'render3d': 0.140, 'illustration': 0.130,
         '_text': 0.100, '_screen': 0.090})
    assert got == 'photo'
    assert margin == pytest.approx(0.010, abs=1e-6)

    # The SAME narrow win for anime is refused.
    got, margin = banks.medium_verdict(
        {'photo': 0.150, 'anime': 0.160, 'render3d': 0.140, 'illustration': 0.130,
         '_text': 0.100, '_screen': 0.090})
    assert got == 'unsure'
    assert margin == pytest.approx(0.010, abs=1e-6)

    # A wide win for anime IS called.
    got, _m = banks.medium_verdict(
        {'photo': 0.120, 'anime': 0.200, 'render3d': 0.140, 'illustration': 0.130,
         '_text': 0.100, '_screen': 0.090})
    assert got == 'anime'


def test_a_distractor_winning_is_unsure_not_a_medium():
    """Banners and screenshots are neither photo nor drawing. They have their own
    prototypes so they stop being misfiled into one of the four — and those
    prototypes are never shown to the user as a verdict."""
    got, margin = banks.medium_verdict(
        {'photo': 0.120, 'anime': 0.100, 'render3d': 0.110, 'illustration': 0.090,
         '_text': 0.400, '_screen': 0.130})
    assert got == 'unsure'
    # The margin is still reported: it is what a re-tune would be applied to.
    assert margin == pytest.approx(0.270, abs=1e-6)
    assert not any(k.startswith('_') for k in banks.MEDIUMS)


# --- the pass ----------------------------------------------------------------

def _fake_prototypes(monkeypatch, winner):
    """Replace the ONLY step that needs CLIP. `winner` is the bucket whose row is
    aligned with the (constant) embedding the fake score cache holds."""
    names = list(banks.MEDIUM_PROTOTYPES)
    dim = 8
    base = np.zeros((len(names), dim), dtype='float32')
    for i, n in enumerate(names):
        base[i, 0] = 0.30 if n == winner else 0.10
        base[i, 1 + i] = 0.90
        base[i] /= np.linalg.norm(base[i])
    monkeypatch.setattr(banks, '_medium_prototype_matrix', lambda: (names, base))
    # The prereq probe asks the CLIP environment whether it exists; the patched
    # prototype matrix means we never touch it, so say yes without loading it.
    from app.services import clip_text_encoder
    monkeypatch.setattr(clip_text_encoder, 'unavailable_reason', lambda: None)
    return names


def _fake_embeddings(monkeypatch, bank, rows):
    """{abs path: unit vector along axis 0} for the given rows — as if ✨ Score
    had cached them."""
    import os
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


def test_medium_pass_writes_a_verdict_only_for_scored_images(app, client, tmp_path,
                                                             monkeypatch):
    bank_id, _src = _mkbank(client, tmp_path,
                            {'a.png': flat(), 'b.png': flat(), 'c.png': flat()})
    with app.app_context():
        bank = db.session.get(banks.ImageBank, bank_id)
        rows = BankImage.query.filter_by(bank_id=bank_id).order_by(BankImage.id).all()
        assert len(rows) == 3
        # ✨ Score reached only two of the three.
        for r in rows[:2]:
            r.aesthetic_score = 5.0
        db.session.commit()
        _fake_prototypes(monkeypatch, 'photo')
        _fake_embeddings(monkeypatch, bank, rows[:2])

        r = client.post(f'/api/bank/{bank_id}/medium', json={})
        assert r.status_code in (200, 202), r.get_json()
        # The pass commits from its own session; drop what this one has cached
        # or we would be asserting on the rows as they were before it ran.
        db.session.expire_all()

        got = {b.relpath: b.medium for b in
               BankImage.query.filter_by(bank_id=bank_id).all()}
        assert got['a.png'] == 'photo'
        assert got['b.png'] == 'photo'
        # The image with no embedding stays NULL: "not scored yet" is neither a
        # medium nor an 'unsure'.
        assert got['c.png'] is None
        # …and the margin travels with the verdict, so a badge can show it.
        scored = BankImage.query.filter_by(bank_id=bank_id, relpath='a.png').one()
        assert scored.medium_margin is not None and scored.medium_margin > 0


def test_medium_refuses_before_score_and_says_which_pass(app, client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path, {'a.png': flat()})
    with app.app_context():
        assert 'Score' in (banks.medium_prereq(bank_id) or '')
    r = client.post(f'/api/bank/{bank_id}/medium', json={})
    assert r.status_code == 503
    assert 'Score' in r.get_json()['error']


def test_medium_facet_counts_and_filters(app, client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path,
                            {'a.png': flat(), 'b.png': flat(), 'c.png': flat()})
    with app.app_context():
        rows = BankImage.query.filter_by(bank_id=bank_id).order_by(BankImage.id).all()
        rows[0].medium, rows[0].medium_margin = 'photo', 0.04
        rows[1].medium, rows[1].medium_margin = 'anime', 0.05
        rows[2].medium, rows[2].medium_margin = 'unsure', 0.001
        db.session.commit()

    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['mediums'] == {'photo': 1, 'anime': 1, 'render3d': 0,
                                  'illustration': 0, 'unsure': 1}
    assert payload['counts']['medium_classified'] == 3

    # 'unsure' is SELECTABLE — the only way to work through that pile.
    ids = client.get(f'/api/bank/{bank_id}/images?medium=unsure').get_json()
    assert ids['total'] == 1
    assert ids['images'][0]['medium'] == 'unsure'
    # …and it composes with another facet instead of replacing it.
    both = client.get(f'/api/bank/{bank_id}/images?medium=photo&status=pending')
    assert both.get_json()['total'] == 1
    assert client.get(
        f'/api/bank/{bank_id}/images?medium=photo&status=keep').get_json()['total'] == 0


# --- angles ------------------------------------------------------------------

def test_angle_buckets_use_the_absolute_yaw_and_the_measured_cuts(app, client,
                                                                  tmp_path):
    files = {f'{i}.png': flat() for i in range(6)}
    bank_id, _src = _mkbank(client, tmp_path, files)
    with app.app_context():
        rows = BankImage.query.filter_by(bank_id=bank_id).order_by(BankImage.id).all()
        # Two frontal (one turned the other way), two three-quarter, one profile,
        # one never measured.
        for row, yaw in zip(rows, (0.0, -19.9, 20.0, -59.9, 74.0, None)):
            row.face_yaw = yaw
            row.face_state = 'scorable' if yaw is not None else None
        db.session.commit()

    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['angles'] == {'frontal': 2, 'three_quarter': 2, 'profile': 1,
                                 'behind': 0}
    # The unmeasured row is in NO bucket — never quietly counted as frontal.
    assert sum(payload['angles'].values()) == 5
    assert payload['counts']['angle_measured'] == 5

    page = client.get(f'/api/bank/{bank_id}/images?angle=profile').get_json()
    assert page['total'] == 1
    assert page['images'][0]['face_yaw'] == 74.0


def test_from_behind_needs_both_passes(app, client, tmp_path):
    """No face is ALSO what a landscape with nobody in it looks like. So the
    bucket is the crossing of no-face with the framing pass's 'back' verdict, and
    it stays empty rather than guessing."""
    bank_id, _src = _mkbank(client, tmp_path,
                            {'a.png': flat(), 'b.png': flat(), 'c.png': flat()})
    with app.app_context():
        rows = BankImage.query.filter_by(bank_id=bank_id).order_by(BankImage.id).all()
        rows[0].face_state, rows[0].framing = 'no_face', 'back'    # a back view
        rows[1].face_state, rows[1].framing = 'no_face', None      # framing unrun
        rows[2].face_state, rows[2].framing = 'no_face', 'body'    # someone faceless
        db.session.commit()

    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['angles']['behind'] == 1
    page = client.get(f'/api/bank/{bank_id}/images?angle=behind').get_json()
    assert [i['relpath'] for i in page['images']] == ['a.png']


def test_the_angle_backfill_is_counted_priced_and_opt_in(app, client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path, {'a.png': flat(), 'b.png': flat()})
    with app.app_context():
        rows = BankImage.query.filter_by(bank_id=bank_id).order_by(BankImage.id).all()
        # Both were face-scanned by a build that measured the yaw and dropped it.
        for row in rows:
            row.face_state, row.face_yaw = 'scorable', None
        db.session.commit()

    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['counts']['angle_backfillable'] == 2
    assert payload['counts']['angle_measured'] == 0
    # Priced BEFORE the click, never after.
    assert payload['angle_backfill_minutes'] >= 1

    # Nothing left to backfill -> the action refuses rather than re-running a
    # detector over a whole bank for no reason.
    with app.app_context():
        for row in BankImage.query.filter_by(bank_id=bank_id).all():
            row.face_yaw = 3.0
        db.session.commit()
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['counts']['angle_backfillable'] == 0
    assert payload['angle_backfill_minutes'] is None
    r = client.post(f'/api/bank/{bank_id}/angles', json={})
    assert r.status_code == 400
    assert 'already' in r.get_json()['error']


# --- sorting -----------------------------------------------------------------

def test_both_measures_are_sortable_and_the_unmeasured_sink(app, client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path,
                            {'a.png': flat(), 'b.png': flat(), 'c.png': flat()})
    with app.app_context():
        rows = BankImage.query.filter_by(bank_id=bank_id).order_by(BankImage.id).all()
        rows[0].face_yaw, rows[0].medium_margin = -70.0, 0.001
        rows[1].face_yaw, rows[1].medium_margin = 5.0, 0.400
        rows[2].face_yaw, rows[2].medium_margin = None, None
        db.session.commit()
        names = [r.relpath for r in rows]

    assert 'yaw_desc' in banks.GRID_SORTS and 'medium_conf_asc' in banks.GRID_SORTS
    # Most turned away first — on the ABSOLUTE angle, so -70° outranks +5°.
    page = client.get(f'/api/bank/{bank_id}/images?sort=yaw_desc').get_json()
    assert [i['relpath'] for i in page['images']] == [names[0], names[1], names[2]]
    # …and the unmeasured row sinks to the END in BOTH directions.
    page = client.get(f'/api/bank/{bank_id}/images?sort=yaw_asc').get_json()
    assert [i['relpath'] for i in page['images']] == [names[1], names[0], names[2]]
    # Least sure medium first — the pile worth a human glance.
    page = client.get(f'/api/bank/{bank_id}/images?sort=medium_conf_asc').get_json()
    assert [i['relpath'] for i in page['images']] == [names[0], names[1], names[2]]


# --- the infer-side contract -------------------------------------------------

def test_face_cache_without_yaws_loads_as_not_measured(tmp_path):
    """A cache written before the yaw array existed must keep working and report
    "not measured" — never 0.0, which would read as a perfectly frontal face."""
    import sys
    sys.path.insert(0, str(banks.cfg.BACKEND_DIR / 'infer'))
    import face_embed_infer as fei

    p = tmp_path / 'face_cache.npz'
    np.savez_compressed(
        str(p), paths=np.array(['x.png']), states=np.array(['scorable']),
        dets=np.array([0.9], dtype='float32'),
        bfracs=np.array([0.3], dtype='float32'),
        embs=np.zeros((1, 4), dtype='float32'))
    loaded = fei._load_cache(str(p))
    yaw = loaded['x.png'][4]
    assert yaw != yaw                      # NaN, i.e. "not measured"

    # And it round-trips once re-saved, so nobody pays for the migration twice.
    fei._save_cache(str(p), loaded)
    again = fei._load_cache(str(p))
    assert again['x.png'][4] != again['x.png'][4]
