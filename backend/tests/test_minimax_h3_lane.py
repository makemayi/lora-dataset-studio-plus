"""The MiniMax H3 generation lane — what the fan-out does with a batch.

The wiring of the GRAPH is pinned in test_minimax_h3.py. This file pins the
LANE: the row/job contract every engine here shares, and the one thing that is
specific to H3 and invisible everywhere else — the order jobs are queued in.

That ordering is not a style choice. H3's text encode depends only on the
prompt, so ComfyUI serves it from cache for every copy of a card after the
first: 78 s for a new card, 37 s for another copy of it (measured, RTX 3090).
Emitting a card's copies consecutively is therefore worth ~42% of a batch's wall
clock — 75 minutes instead of 130 for 100 images over 20 cards — and nothing in
the UI would ever show that it had been lost.
"""
import os

import pytest
from PIL import Image


def _h3_dataset(app, svc):
    """A dataset with a reference image, ready for the H3 fan-out. `trigger_word`
    is NOT NULL in the model, so it is set here: omitting it fails with an
    IntegrityError that looks unrelated to this feature."""
    from app.models import FaceDataset, db
    ds = FaceDataset(user_id='local', name='ds', trigger_word='zzt',
                     ref_filename='ref.png', train_type='krea')
    db.session.add(ds)
    db.session.commit()
    path = svc._dataset_path(ds.id)
    os.makedirs(path, exist_ok=True)
    Image.new('RGB', (512, 512), (10, 20, 30)).save(os.path.join(path, 'ref.png'))
    return ds


def _capture(monkeypatch):
    """Bypass installation and record every enqueue call, in order."""
    from app.services import minimax_h3_helper as mh
    seen = []

    def _fake(**kw):
        seen.append(kw)
        return f'job-{len(seen)}'

    monkeypatch.setattr(mh, 'preflight', lambda *a, **k: None)
    monkeypatch.setattr(mh, 'enqueue_minimax_h3', _fake)
    return seen


def test_a_cards_copies_are_queued_consecutively(app, monkeypatch):
    """THE ordering test. Two cards times three copies must leave as AAABBB, not
    ABABAB: every prompt change costs a fresh 40-second text encode, and an
    interleaved queue pays it six times instead of twice."""
    from app.services import face_dataset_service as svc
    seen = _capture(monkeypatch)
    with app.app_context():
        ds = _h3_dataset(app, svc)
        svc.generate_variations_minimax_h3(
            'local', ds.id,
            [{'label': 'a', 'prompt': 'first', 'framing': 'face'},
             {'label': 'b', 'prompt': 'second', 'framing': 'bust'}],
            3)
    prompts = [kw['edit_prompt'] for kw in seen]
    assert len(prompts) == 6
    distinct_runs = [p for i, p in enumerate(prompts) if i == 0 or p != prompts[i - 1]]
    assert len(distinct_runs) == 2, f'prompt changed more than twice: {prompts}'


def test_every_row_is_committed_before_its_job_is_enqueued(app, monkeypatch):
    """The contract the other two lanes keep: a row exists first, so a crash
    between the two leaves a visible failed tile rather than an orphan job."""
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    with app.app_context():
        ds = _h3_dataset(app, svc)
        seen = []

        def _fake(**kw):
            rows = FaceDatasetImage.query.filter_by(dataset_id=ds.id).all()
            seen.append(len(rows))
            return 'job'

        from app.services import minimax_h3_helper as mh
        monkeypatch.setattr(mh, 'preflight', lambda *a, **k: None)
        monkeypatch.setattr(mh, 'enqueue_minimax_h3', _fake)
        ids = svc.generate_variations_minimax_h3(
            'local', ds.id, [{'label': 'a', 'prompt': 'p', 'framing': 'face'}], 2)
    assert seen == [1, 2]
    assert len(ids) == 2


def test_rows_carry_the_engine_id_so_the_grid_can_badge_them(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    _capture(monkeypatch)
    with app.app_context():
        ds = _h3_dataset(app, svc)
        ids = svc.generate_variations_minimax_h3(
            'local', ds.id, [{'label': 'a', 'prompt': 'p', 'framing': 'face'}], 1)
        row = FaceDatasetImage.query.get(ids[0])
        assert row.klein_model == svc.MINIMAX_H3_ENGINE == 'minimax_h3'
        # The RAW catalog prompt, never the wrapped one: a regenerate re-applies
        # the CURRENT suffix, and storing the wrapped text would apply it twice.
        assert row.variation_prompt == 'p'


def test_the_prompt_names_the_reference_the_way_h3_addresses_it(app, monkeypatch):
    """`<Picture 1>` is H3's own way of referring to the first reference image.
    A prompt that never mentions it describes a scene with nobody in it."""
    from app.services import face_dataset_service as svc
    from app.services.face_variations import H3_REFERENCE_TOKEN
    seen = _capture(monkeypatch)
    with app.app_context():
        ds = _h3_dataset(app, svc)
        svc.generate_variations_minimax_h3(
            'local', ds.id,
            [{'label': 'a', 'prompt': 'standing by a window', 'framing': 'body'}], 1)
    prompt = seen[0]['edit_prompt']
    assert H3_REFERENCE_TOKEN in prompt
    assert 'standing by a window' in prompt


def test_a_dataset_without_a_reference_is_refused_before_any_row(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.models import FaceDataset, FaceDatasetImage, db
    _capture(monkeypatch)
    with app.app_context():
        ds = FaceDataset(user_id='local', name='no-ref', trigger_word='zzt',
                         train_type='krea')
        db.session.add(ds)
        db.session.commit()
        with pytest.raises(ValueError):
            svc.generate_variations_minimax_h3(
                'local', ds.id, [{'label': 'a', 'prompt': 'p'}], 1)
        assert FaceDatasetImage.query.filter_by(dataset_id=ds.id).count() == 0


def test_the_fanout_ceiling_is_enforced_before_any_row_exists(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import dataset_generation_service as gen
    from app.models import FaceDatasetImage
    _capture(monkeypatch)
    with app.app_context():
        ds = _h3_dataset(app, svc)
        variations = [{'label': f'v{i}', 'prompt': f'p{i}'}
                      for i in range(gen.MAX_FANOUT + 1)]
        with pytest.raises(ValueError):
            svc.generate_variations_minimax_h3('local', ds.id, variations, 1)
        assert FaceDatasetImage.query.filter_by(dataset_id=ds.id).count() == 0


def test_h3_generates_but_does_not_edit_a_reference(app):
    """A scope line, pinned: H3 is in the generation catalog and NOT in the edit
    one. The edit route has no branch for it, so being offered there would fail
    at enqueue instead of at pick time."""
    from app.services import dataset_generation_service as gen
    assert 'minimax_h3' in gen.LOCAL_ENGINES
    assert 'minimax_h3' not in gen.editable_engines()


# --- the progress badge names the engine that is ACTUALLY running ------------

def test_an_h3_run_is_not_badged_klein(app):
    """Reported 2026-08-09 as "the task is stuck": a face swap had finished, two
    H3 jobs were still queued behind it, and the progress indicator named Klein
    — the engine that had already finished. The old two-way test claimed 'krea'
    only when every row was Krea and said 'klein' for everything else, so H3
    could never be named."""
    from app import db
    from app.models import FaceDataset, FaceDatasetImage
    from app.services import dataset_activity
    from app.services.dataset_generation_service import (
        _sync_generate_activity, KREA_ENGINE, MINIMAX_H3_ENGINE)

    with app.app_context():
        ds = FaceDataset(name='badge', trigger_word='badgeperson')
        db.session.add(ds)
        db.session.commit()

        def row(marker, jid):
            db.session.add(FaceDatasetImage(
                dataset_id=ds.id, status='pending', job_id=jid,
                variation_label='shot', klein_model=marker))
            db.session.commit()

        row(MINIMAX_H3_ENGINE, 'j-h3')
        _sync_generate_activity(ds.id)
        assert dataset_activity.get(ds.id)['engine'] == MINIMAX_H3_ENGINE

        # Mixed queue -> the vague answer, never a confident wrong one.
        row(KREA_ENGINE, 'j-krea')
        _sync_generate_activity(ds.id)
        assert dataset_activity.get(ds.id)['engine'] == 'local'

        # A Klein row carries a MODEL FILENAME (or NULL), not an engine id.
        for r in FaceDatasetImage.query.filter_by(dataset_id=ds.id).all():
            db.session.delete(r)
        db.session.commit()
        row('flux-2-klein-9b-fp8.safetensors', 'j-klein')
        row(None, 'j-legacy-klein')
        _sync_generate_activity(ds.id)
        assert dataset_activity.get(ds.id)['engine'] == 'klein'
