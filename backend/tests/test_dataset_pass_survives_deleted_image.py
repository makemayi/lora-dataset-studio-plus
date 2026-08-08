"""📁 Dataset — deleting an image mid-pass skips that image, never the pass.

The dataset half of the window closed for the 🗃️ bank in b87830f6. Same cause,
same consequence, different file: a pass loads its rows, then walks them for
minutes-to-hours with one model call per image, committing as it goes. With
`expire_on_commit` on, every row it has not reached yet is a lazy re-SELECT —
and the grid stays fully interactive while the pass runs, so deleting a bad tile
during a captioning or watermark run is ordinary use, not a race anyone has to
engineer.

SQLAlchemy's default answer is fatal: an attribute read on an expired row whose
database row is gone raises ObjectDeletedError (verified — including for the
PRIMARY KEY, which is why holding ids taken *before* any commit is the immune
shape), and a commit carrying a write staged on such a row raises too, poisoning
the session for the `finally`. Either one kills the WHOLE pass and discards the
work already done on every other image.

The shape these passes are moved to is the one `analyze_faces` already uses and
that this file treats as the reference: hold plain values, not ORM objects, and
re-read by id immediately before touching a row.

Each test below deletes ONE image from the middle of a pass, through that pass's
own inference hook — i.e. while the pass is inside a model call, which is where
the real click lands — and pins: the pass finished, the survivors were written,
and the count it returns is the count it actually wrote.
"""
import io
import json

import pytest
from PIL import Image


def _png(color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def _dataset_with_files(svc, n=3, kind=None, **row_kw):
    """A dataset with ``n`` kept images that really exist on disk."""
    import os
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.models import FaceDatasetImage
    # A concept dataset REFUSES to exist without a concept_desc — without it this
    # helper raises and the test would go red on the setup, proving nothing.
    ds = (svc.create_dataset(LOCAL_USER, 'Pass', 'passds', kind=kind,
                             concept_desc='a recurring act') if kind
          else svc.create_dataset(LOCAL_USER, 'Pass', 'passds'))
    folder = svc._dataset_path(ds.id)
    os.makedirs(folder, exist_ok=True)
    for i in range(n):
        name = f'img{i}.png'
        with open(os.path.join(folder, name), 'wb') as fh:
            fh.write(_png((10 * i, 90, 160)))
        db.session.add(FaceDatasetImage(dataset_id=ds.id, filename=name,
                                        status='keep', **row_kw))
    db.session.commit()
    return ds


def _ids(dataset_id):
    from app.models import FaceDatasetImage
    return [r.id for r in FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id)
            .order_by(FaceDatasetImage.id.asc()).all()]


def _delete_image_row(image_id):
    """Delete one row the way another request would.

    A bulk DELETE with ``synchronize_session=False`` plus a commit reproduces
    exactly what the pass's session sees when a DIFFERENT session removed the
    row: gone from the database, while the pass still holds the (now expired)
    instance in its identity map."""
    from app.extensions import db
    from app.models import FaceDatasetImage
    FaceDatasetImage.query.filter(FaceDatasetImage.id == image_id).delete(
        synchronize_session=False)
    db.session.commit()


@pytest.fixture()
def ctx(app):
    with app.app_context():
        yield


def test_the_framing_classifier_survives_an_image_deleted_under_it(ctx, monkeypatch):
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama

    ds = _dataset_with_files(svc, 3, source='import')
    ids = _ids(ds.id)
    calls = {'n': 0}

    def fake_describe(image_bytes, *a, **k):
        if calls['n'] == 0:
            _delete_image_row(ids[1])
        calls['n'] += 1
        return '{"framing": "face", "label": "portrait"}'

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    n = svc.classify_images(LOCAL_USER, ds.id)
    assert n == 2, f'the pass reported {n} classified over two surviving images'
    for i in (ids[0], ids[2]):
        assert db.session.get(FaceDatasetImage, i).framing, (
            'the pass continued but stopped writing its classifications')


def test_the_watermark_detector_survives_an_image_deleted_under_it(ctx, monkeypatch):
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama

    ds = _dataset_with_files(svc, 3)
    ids = _ids(ds.id)
    calls = {'n': 0}

    def fake_describe(image_bytes, *a, **k):
        if calls['n'] == 0:
            _delete_image_row(ids[1])
        calls['n'] += 1
        return '{"present": false}'

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    counts = svc.detect_watermarks(LOCAL_USER, ds.id)
    # 'checked' counts the images this pass really examined. The vanished one is
    # NOT reported as a fourth key: `counts` is the route's response shape and is
    # pinned exactly by test_watermarks.py, so the skip is logged instead.
    assert counts['checked'] == 2, (
        f'the pass reported {counts["checked"]} checked over two surviving images')
    assert set(counts) == {'detected', 'none', 'checked'}, (
        'the response shape gained a key — test_watermarks.py pins it exactly')
    for i in (ids[0], ids[2]):
        assert db.session.get(FaceDatasetImage, i).watermark_state == 'none', (
            'the pass continued but stopped writing its verdicts')


def test_the_watermark_cleaner_survives_an_image_deleted_under_its_batch(ctx, monkeypatch):
    """LaMa repaints the whole selection in ONE batch, so the pass holds its rows
    across a call that runs for minutes — the widest window of the four. It also
    has to throw away the staged edit of a row that vanished, rather than promote
    a repainted file over a master no row points at any more."""
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama

    ds = _dataset_with_files(svc, 3, watermark_state='detected',
                             watermark_bbox=json.dumps([0.0, 0.90, 1.0, 1.0]))
    ids = _ids(ds.id)

    def fake_batch(items, device='cpu'):
        _delete_image_row(ids[1])          # deleted while the batch was running
        return {it['image_path']: (True, None) for it in items}

    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    monkeypatch.setattr(watermark_lama, 'inpaint_batch', fake_batch)
    monkeypatch.setattr(watermark_lama, 'inpaint_watermark',
                        lambda p, b, **k: (True, None))
    monkeypatch.setattr(watermark_lama, 'inpaint_watermarks',
                        lambda p, b, **k: (True, None))
    # allow_crop=False forces the inpaint route, which is the batch we want to
    # exercise; a border box would otherwise be cropped in PIL and never queued.
    out, _error = svc.clean_watermarks(LOCAL_USER, ds.id, allow_crop=False)
    assert out['inpainted'] == 2, (
        f'the pass reported {out["inpainted"]} inpainted over two surviving images')
    for i in (ids[0], ids[2]):
        assert db.session.get(FaceDatasetImage, i).watermark_state == 'cleaned', (
            'the pass continued but stopped recording its repaints')


def _force_ollama_backend(monkeypatch, svc):
    """Pin the caption backend so the test exercises the Ollama loop, not
    whatever JoyCaption happens to be installed on the machine running it."""
    orig_get = svc.cfg.get
    monkeypatch.setattr(svc.cfg, 'get',
                        lambda k, *a, **kw: ('ollama' if k == 'captioning.backend'
                                             else orig_get(k, *a, **kw)))


def test_the_caption_pass_survives_an_image_deleted_under_it(ctx, monkeypatch):
    """The longest pass in the app: one VLM call per image, a commit per image,
    over a grid the user keeps working in."""
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama

    ds = _dataset_with_files(svc, 3)
    ids = _ids(ds.id)
    _force_ollama_backend(monkeypatch, svc)
    calls = {'n': 0}

    def fake_describe(image_bytes, *a, **k):
        if calls['n'] == 0:
            _delete_image_row(ids[1])
        calls['n'] += 1
        return 'a caption for this image'

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    n = svc.caption_images(LOCAL_USER, ds.id)
    assert n == 2, f'the pass reported {n} captioned over two surviving images'
    for i in (ids[0], ids[2]):
        assert db.session.get(FaceDatasetImage, i).caption, (
            'the pass continued but stopped storing captions')


def test_the_concept_caption_pass_survives_an_image_deleted_under_it(ctx, monkeypatch):
    """Concept datasets take a SEPARATE caption path (_caption_concept) with its
    own loops — fixing caption_images does nothing for it."""
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama

    ds = _dataset_with_files(svc, 3, kind='concept')
    ids = _ids(ds.id)
    _force_ollama_backend(monkeypatch, svc)
    calls = {'n': 0}

    def fake_describe(image_bytes, *a, **k):
        if calls['n'] == 0:
            _delete_image_row(ids[1])
        calls['n'] += 1
        return 'a concept caption for this image'

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    n = svc.caption_images(LOCAL_USER, ds.id)
    assert n == 2, f'the pass reported {n} captioned over two surviving images'
    for i in (ids[0], ids[2]):
        assert db.session.get(FaceDatasetImage, i).caption, (
            'the concept pass continued but stopped storing captions')


def test_the_short_caption_pass_survives_an_image_deleted_under_it(ctx, monkeypatch):
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.services import face_dataset_service as svc

    ds = _dataset_with_files(svc, 3, caption='a long caption to shorten')
    # The pass is a no-op unless the dataset opted into dual captions — without
    # this the test would take an early return and prove nothing at all.
    ds.train_settings = json.dumps({'dual_captions': True})
    db.session.commit()
    ids = _ids(ds.id)
    calls = {'n': 0}

    def fake_generate(prompt):
        if calls['n'] == 0:
            _delete_image_row(ids[1])
        calls['n'] += 1
        return 'a short caption'

    n = svc.derive_short_captions(LOCAL_USER, ds.id, generate=fake_generate)
    assert n == 2, f'the pass reported {n} shortened over two surviving images'
    for i in (ids[0], ids[2]):
        assert db.session.get(FaceDatasetImage, i).caption_short, (
            'the pass continued but stopped writing its short captions')


def test_generate_variations_survives_stop_deleting_its_row(ctx, monkeypatch):
    """⏹ Stop deletes exactly the rows this function is creating (status='pending'
    AND filename IS NULL), and it can land while the enqueue is in flight."""
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    global _ORIG_CFG_GET
    _ORIG_CFG_GET = svc.cfg.get

    ds = _dataset_with_files(svc, 1)
    ds.ref_filename = 'img0.png'
    db.session.commit()
    created = []

    def fake_enqueue(**kw):
        # The row was committed (and therefore expired) just above; Stop removes
        # it while we are here, inside the enqueue.
        row = (FaceDatasetImage.query.filter_by(dataset_id=ds.id, status='pending')
               .order_by(FaceDatasetImage.id.desc()).first())
        if row is not None and not created:
            created.append(row.id)
            FaceDatasetImage.query.filter(FaceDatasetImage.id == row.id).delete(
                synchronize_session=False)
            db.session.commit()
        return 'job-1'

    # Imported INSIDE the function, so it must be patched at its source module —
    # patching it on face_dataset_service silently does nothing.
    from app.services import klein_edit_helper
    monkeypatch.setattr(klein_edit_helper, 'enqueue_klein_edit', fake_enqueue)
    # The pass preflights the Klein model files before creating any row; without
    # this it refuses long before the window under test and the probe is red on
    # its setup (the third time this campaign — see the concept test).
    monkeypatch.setattr(klein_edit_helper, 'klein_missing_assets', lambda *a, **k: [])
    monkeypatch.setattr(svc.cfg, 'get',
                        lambda k, *a, **kw: ('C:/comfy' if k == 'comfyui.base_dir'
                                             else _ORIG_CFG_GET(k, *a, **kw)))
    ids = svc.generate_variations(LOCAL_USER, ds.id,
                                  [{'prompt': 'p', 'label': 'l'}], 1)
    assert ids == [], f'a row deleted by Stop should not be reported as queued: {ids}'


def test_stop_does_not_report_cancellations_its_own_rollback_discarded(ctx, monkeypatch):
    """⏹ Stop stages `db.session.delete(img)` per row and commits ONCE at the end.

    The queue helper it calls per row can roll back internally. A rollback
    discards the deletes staged by EARLIER iterations — but the counter has
    already counted them, so Stop reports cancellations that did not happen and
    the tiles are still there when the grid refreshes. This pins the count
    against reality rather than against itself.
    """
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    ds = _dataset_with_files(svc, 0)
    for i in range(3):
        db.session.add(FaceDatasetImage(dataset_id=ds.id, status='pending',
                                        filename=None, job_id=f'job-{i}'))
    db.session.commit()

    calls = {'n': 0}

    def fake_outcome(job_id, user_id, kind):
        calls['n'] += 1
        if calls['n'] == 2:
            # What the real helper does on one of its failure paths.
            db.session.rollback()
        return 'cancelled'

    from app.job_queue import queue_manager
    monkeypatch.setattr(queue_manager, 'cancel_job_outcome', fake_outcome)
    out = svc.cancel_pending(LOCAL_USER, ds.id)
    left = FaceDatasetImage.query.filter_by(dataset_id=ds.id).count()
    assert out['cancelled'] == 3 - left, (
        f"Stop reported {out['cancelled']} cancelled but {left} row(s) are still "
        f'there — an internal rollback discarded deletes the counter had already '
        f'counted')


def test_improve_survives_stop_deleting_its_candidate(ctx, monkeypatch):
    """✨ Improve creates a pending candidate, commits, then enqueues — and ⏹ Stop
    deletes exactly that shape (pending, no filename), same as the variation
    paths."""
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    ds = _dataset_with_files(svc, 1)
    image_id = _ids(ds.id)[0]

    def fake_enqueue(engine, **kw):
        # Stop removes the freshly created candidate while we are enqueuing.
        cand = (FaceDatasetImage.query
                .filter_by(dataset_id=ds.id, parent_image_id=image_id).first())
        if cand is not None:
            FaceDatasetImage.query.filter(FaceDatasetImage.id == cand.id).delete(
                synchronize_session=False)
            db.session.commit()
        return 'job-improve-1'

    monkeypatch.setattr(svc, '_enqueue_improve', fake_enqueue)
    monkeypatch.setattr(svc, 'resolve_improve_engine', lambda requested=None: 'klein')
    monkeypatch.setattr(svc, '_improve_preflight', lambda engine: None)
    out = svc.improve_existing_image(LOCAL_USER, image_id)
    assert out is None or out.get('candidate_id') is None, (
        f'improve reported a candidate Stop had already deleted: {out}')
