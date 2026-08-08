"""🗃️ Image bank — an inference pass holds NO database transaction while it waits.

Three shapes of wait live here: a child process (face, score), a long series of
Ollama calls (watermark, framing), and one batch that repaints a whole selection
(the inpaint level). The invariant is the same for all of them, and so is the
failure it prevents.

The framing half is the one that was found the expensive way — see its own
section below. It is also the reason this file is worth more than a latency
test: a pass that locks the database does not merely make the app slow, it can
starve the writer that authorises its OWN next model call, and then it stops
producing anything while the progress bar keeps moving.

The face and score passes read their rows, hand a path list to a child process,
and only write results back when the child returns. That wait is not short: the
scoring extra ships CPU-only torch on purpose and a big bank measures near an
hour. Keeping the session's transaction open across it is a loaded gun — WAL
gives concurrent readers, never concurrent WRITERS, so one write joining that
transaction ahead of the child takes the single write lock for the whole pass
and every other writer dies on `database is locked` past the 5 s busy_timeout.
Two paid cloud runs were already abandoned that way on 2026-07-26 by a holder
that lasted five seconds.

These tests pin the invariant rather than the symptom: while the pass sits in
the subprocess, another connection to the same database must be able to write.
They need a FILE database (the shared `app` fixture's `:memory:` one cannot be
opened twice) and keep the app's real journal mode, so the lock they measure is
the lock a real install has.
"""
import sqlite3
from unittest.mock import patch

import pytest
from PIL import Image
from sqlalchemy import text


@pytest.fixture()
def file_db(tmp_path, monkeypatch):
    """(app, db_path, workdir) — a real app on a file-backed SQLite database."""
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    from app import config as cfg
    monkeypatch.setattr(cfg, 'ENV_PATH', tmp_path / '.env')
    monkeypatch.setattr(cfg, '_cache', None)
    from app import create_app
    from app.extensions import db

    application = create_app({'TESTING': True, 'WTF_CSRF_ENABLED': False})
    with application.app_context():
        mode = db.session.execute(text('PRAGMA journal_mode')).scalar()
        db.session.commit()
    # Recorded, not forced: whatever the app configures is what the test measures.
    # WAL is not an escape — it removes reader/writer contention, not the single
    # write lock these tests are about.
    assert str(mode).lower() in ('wal', 'delete'), f'unexpected journal mode {mode!r}'
    return application, str(tmp_path / 'data' / 'studio.db'), tmp_path


def _bank_with_images(workdir, n=2):
    from app.services import image_bank_service as banks
    src = workdir / 'src'
    src.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new('RGB', (800, 800), (10 * i, 90, 160)).save(str(src / f'a{i}.jpg'))
    bank, _added = banks.create_bank('local', 'Dump', str(src))
    return bank.id


def _concurrent_write(db_path):
    """Try a write from ANOTHER connection, as a live request would. Returns None
    on success, else the SQLite error text ('database is locked' when blocked)."""
    con = sqlite3.connect(db_path, timeout=0.5)
    try:
        con.execute("UPDATE image_bank SET name = 'written while inferring'")
        con.commit()
        return None
    except sqlite3.OperationalError as e:
        return str(e)
    finally:
        con.close()


def _writer_verdict_during_inference(file_db, make_job):
    """Run a bank inference pass whose session already carries an uncommitted
    write, and report what a concurrent writer saw from inside the subprocess."""
    from app.extensions import db
    from app.models import ImageBank
    from app.services import image_bank_service as banks

    app, db_path, workdir = file_db
    seen = {}
    with app.app_context():
        bank_id = _bank_with_images(workdir)
        db.session.commit()
        # A write pending on the session when the pass reaches the child. Nothing
        # on today's nominal path does this; the contract is that it would stay
        # harmless if it did, instead of becoming an hour-long global write lock.
        db.session.get(ImageBank, bank_id).name = 'renamed just before inferring'
        db.session.flush()

        def fake_drive(job, python, script, payload, cache_path, progress_re, window):
            seen['error'] = _concurrent_write(db_path)
            return {'ok': True, 'results': {}, 'clusters': {}}, [], 0

        with patch.object(banks, '_drive_infer_subprocess', fake_drive), \
             patch.object(banks.bank_jobs, 'cancelled', lambda job: False), \
             patch.object(banks.bank_jobs, 'progress', lambda job, **kw: None), \
             patch.object(banks.bank_jobs, 'set_stop_notice', lambda job, **kw: None), \
             patch('app.capabilities.bank_scoring_gpu_available', lambda: False), \
             patch.object(banks, '_resolve_face_device', lambda: ('cpu', False)):
            make_job(banks, bank_id)(object())
        assert seen, 'the pass never reached the inference subprocess'
    return seen['error']


def test_the_score_pass_lets_other_writers_through_while_it_infers(file_db):
    error = _writer_verdict_during_inference(
        file_db, lambda banks, bank_id: banks._score_job(bank_id))
    assert error is None, (
        'the scoring pass held a write lock through the inference — a real pass '
        f'would block every other writer for its whole duration ({error})')


def test_the_face_pass_lets_other_writers_through_while_it_infers(file_db):
    error = _writer_verdict_during_inference(
        file_db, lambda banks, bank_id: banks._faces_job(bank_id))
    assert error is None, (
        f'the face pass held a write lock through the inference ({error})')


def test_the_release_helper_really_ends_the_transaction(file_db):
    """The guard on its own: a flush blocks other writers, and the helper frees
    them. Without the first assertion the two tests above could pass for the
    wrong reason (a flush that never locked anything proves nothing)."""
    from app.extensions import db
    from app.models import ImageBank
    from app.services import image_bank_service as banks

    app, db_path, _workdir = file_db
    with app.app_context():
        db.session.add(ImageBank(user_id='local', name='Dump', source_path='x'))
        db.session.flush()
        assert _concurrent_write(db_path), 'a pending write must lock the file'
        banks._release_db_before_inference()
        assert _concurrent_write(db_path) is None


# --- the watermark pass: the same wait, spread over thousands of Ollama calls -
def _watermark_writer_verdict(file_db, answer, probe_at, images=6):
    """Run the watermark pass over a file-backed bank, asking a concurrent writer
    what it sees at the ``probe_at``-th Ollama call, and return its verdict.

    ``answer(i)`` produces (or raises) what the model replies for image ``i`` —
    that is the whole point: the pass must bound its write lock by the images it
    walked through, not by the ones whose answer happened to parse.
    """
    from app.extensions import db
    from app.services import image_bank_service as banks
    from app.services import vision_ollama, vision_pool

    app, db_path, workdir = file_db
    seen = {}
    with app.app_context():
        bank_id = _bank_with_images(workdir, n=images)
        db.session.commit()
        calls = {'n': 0}

        def fake_describe(image_bytes, *a, **k):
            i = calls['n']
            calls['n'] += 1
            if i == probe_at:
                seen['error'] = _concurrent_write(db_path)
            return answer(i)

        job = {'done': 0, 'total': 0, '_touched': 0.0}
        # One call at a time, so "the probe fires while the rows before it are
        # written" is an ordering the test owns rather than a race it hopes for.
        with patch.object(vision_pool, 'vision_concurrency', lambda *a, **k: 1), \
             patch.object(vision_ollama, 'describe_image_ollama', fake_describe), \
             patch.object(vision_ollama, 'unload_vision_model', lambda *a, **k: True), \
             patch.object(banks.bank_jobs, 'cancelled', lambda j: False), \
             patch.object(banks.bank_jobs, 'progress', lambda j, **kw: None):
            banks._watermark_job(bank_id, False)(job)
    assert 'error' in seen, 'the pass never reached the probed image'
    return seen['error']


def test_the_watermark_pass_lets_other_writers_through_when_files_error(file_db):
    """Every image fails to be read: the pass stamps 'error' on each row and asks
    Ollama about the next one. Those stamps are writes — they must not accumulate
    into one lock held across the whole pass."""
    def always_raises(_i):
        raise RuntimeError('unreadable image')

    error = _watermark_writer_verdict(file_db, always_raises, probe_at=2)
    assert error is None, (
        'the watermark pass held a write lock across its Ollama calls — a pass '
        f'over unreadable files would block every other writer for hours ({error})'
    )


def test_the_watermark_pass_lets_other_writers_through_when_ollama_answers_nothing(file_db):
    """One image answers, then Ollama goes silent. The first answer is a write;
    the silent ones are not, so a commit rhythm counted in PARSED answers never
    ticks again and that write outlives the whole pass."""
    def one_answer_then_silence(i):
        return '{"present": false}' if i == 0 else ''

    error = _watermark_writer_verdict(file_db, one_answer_then_silence, probe_at=3)
    assert error is None, (
        'a single parsed answer followed by an unreachable Ollama left the write '
        f'lock held for the rest of the pass ({error})')


# --- the framing pass: the same wait, and the writer it was starving ---------
# This half is not hypothetical. A 211-image bank reported 4 images classified
# out of the 12 the pass got through, with `vision GPU window renewal failed` /
# `database is locked` on the `system_state` UPDATE repeating every few seconds.
# The starved writer there is not a sort click: it is the renewal of the vision
# GPU window itself, which runs before EVERY vision call and is fail-closed. So
# the pass locked out the one writer that authorises it to work, and every image
# after that was refused by its own fence. It is the only pass whose contention
# eats its own output, which is why the probe below is worth its own file half.
def _framing_writer_verdict(file_db, answer, probe_at, images=6):
    """Run 📐 Framing over a file-backed bank and report what a concurrent writer
    sees at the ``probe_at``-th Ollama call — i.e. while the pass is inside a
    model call, holding whatever its previous images left behind."""
    from app.extensions import db
    from app.services import image_bank_service as banks
    from app.services import vision_ollama, vision_pool

    app, db_path, workdir = file_db
    seen = {}
    with app.app_context():
        bank_id = _bank_with_images(workdir, n=images)
        db.session.commit()
        calls = {'n': 0}

        def fake_describe(image_bytes, *a, **k):
            i = calls['n']
            calls['n'] += 1
            if i == probe_at:
                seen['error'] = _concurrent_write(db_path)
            return answer(i)

        job = {'done': 0, 'total': 0, '_touched': 0.0}
        with patch.object(vision_pool, 'vision_concurrency', lambda *a, **k: 1), \
             patch.object(vision_ollama, 'describe_image_ollama', fake_describe), \
             patch.object(vision_ollama, 'unload_vision_model', lambda *a, **k: True), \
             patch.object(banks.bank_jobs, 'cancelled', lambda j: False), \
             patch.object(banks.bank_jobs, 'progress', lambda j, **kw: None):
            banks._framing_job(bank_id, False)(job)
    assert 'error' in seen, 'the pass never reached the probed image'
    return seen['error']


def test_the_framing_pass_lets_other_writers_through_between_classifications(file_db):
    """Every image classifies, so every image WRITES. Those writes must not
    accumulate into one lock spanning the model calls that follow them."""
    error = _framing_writer_verdict(
        file_db, lambda _i: '{"framing": "face"}', probe_at=3)
    assert error is None, (
        'the framing pass held the write lock across its Ollama calls — that is '
        'the writer the vision GPU window needs to renew itself, so the pass '
        f'fences its own remaining images out of the model ({error})')


def test_the_framing_pass_lets_writers_through_when_ollama_goes_silent(file_db):
    """One classification, then silence: a rhythm counted in CLASSIFIED images
    never ticks again, so that single write outlives the whole pass."""
    def one_answer_then_silence(i):
        return '{"framing": "face"}' if i == 0 else ''

    error = _framing_writer_verdict(file_db, one_answer_then_silence, probe_at=3)
    assert error is None, (
        'one classified image followed by a silent Ollama left the write lock '
        f'held for the rest of the pass ({error})')


def test_the_framing_probe_really_sees_a_held_lock(file_db):
    """Green above means nothing unless the probe is shown catching a real
    holder, from inside a model call, on this exact fixture.

    So the flush is put back INSIDE the model call — where the pass used to be
    sitting with an uncommitted classification behind it — and the probe fires
    right after it. It cannot be staged before the pass instead: entering the
    vision GPU window writes and commits `vision_in_progress` of its own, which
    would release any holder set up beforehand. (The two tests above are the same
    measurement with nothing flushed by hand — which is the point: the pass must
    not manufacture that holder itself.)"""
    from app.extensions import db
    from app.models import ImageBank
    from app.services import image_bank_service as banks
    from app.services import vision_ollama, vision_pool

    app, db_path, workdir = file_db
    seen = {}
    with app.app_context():
        bank_id = _bank_with_images(workdir, n=4)
        db.session.commit()

        def fake_describe(image_bytes, *a, **k):
            # vision_concurrency=1 runs the call inline on the pass's own thread
            # and app context, so this is the pass's session.
            if 'error' not in seen:
                db.session.get(ImageBank, bank_id).name = 'renamed mid-call'
                db.session.flush()      # what the autoflushed SELECT used to do
                seen['error'] = _concurrent_write(db_path)
            return '{"framing": "face"}'

        with patch.object(vision_pool, 'vision_concurrency', lambda *a, **k: 1), \
             patch.object(vision_ollama, 'describe_image_ollama', fake_describe), \
             patch.object(vision_ollama, 'unload_vision_model', lambda *a, **k: True), \
             patch.object(banks.bank_jobs, 'cancelled', lambda j: False), \
             patch.object(banks.bank_jobs, 'progress', lambda j, **kw: None):
            banks._framing_job(bank_id, False)({'done': 0, 'total': 0,
                                                '_touched': 0.0})
    assert seen.get('error') and 'locked' in seen['error'], (
        'the probe saw nothing while a write was provably held — it is not '
        f'measuring the write lock at all (got {seen.get("error")!r})')


# --- the inpaint level: one batch, minutes long, staged row writes behind it --
def test_the_inpaint_level_lets_other_writers_through_during_its_batch(file_db):
    """LaMa repaints the whole selection in ONE call that runs for minutes.

    The rows are staged before it, and a staging FAILURE writes to its row
    (`_discard_clean_blob` clears the clean method) without committing. The next
    row's re-read then flushed that write, took the single write lock, and the
    batch ran with it held — the longest hold of any pass in this file."""
    import json
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks
    from app.services import watermark_lama

    app, db_path, workdir = file_db
    seen = {}
    with app.app_context():
        bank_id = _bank_with_images(workdir, n=4)
        bank = db.session.get(ImageBank, bank_id)
        for row in BankImage.query.filter_by(bank_id=bank_id).all():
            row.watermark_state = 'detected'
            # A row that already carries a cleaned version — re-cleaning one is
            # ordinary, and it is what makes the discard below a real write
            # rather than a no-op assignment of None over None.
            row.watermark_clean_method = 'crop'
            row.watermark_bbox = json.dumps([0.0, 0.9, 1.0, 1.0])
            row.watermark_regions = json.dumps([[0.0, 0.9, 1.0, 1.0]])
            row.watermark_fingerprint = transfer.content_fingerprint_path(
                banks.abs_image_path(bank, row))
        db.session.commit()
        real_stage = banks._stage_clean_copy
        staged = {'n': 0}

        def failing_first_stage(bank_id_, row, src):
            staged['n'] += 1
            if staged['n'] == 1:        # one bad file among four
                raise OSError('cannot stage this one')
            return real_stage(bank_id_, row, src)

        def fake_batch(items, device='cpu'):
            seen['error'] = _concurrent_write(db_path)
            return {it['image_path']: (True, None) for it in items}

        with patch.object(banks, '_stage_clean_copy', failing_first_stage), \
             patch.object(watermark_lama, 'is_available', lambda: True), \
             patch.object(watermark_lama, 'resolve_device', lambda: 'cpu'), \
             patch.object(watermark_lama, 'inpaint_batch', fake_batch), \
             patch.object(banks.bank_jobs, 'cancelled', lambda j: False), \
             patch.object(banks.bank_jobs, 'progress', lambda j, **kw: None), \
             patch.object(banks.bank_jobs, 'bump', lambda j, n=1: None):
            banks._watermark_inpaint_job(bank_id, 'auto')({'done': 0, 'total': 0,
                                                           '_touched': 0.0})
    assert 'error' in seen, 'the pass never reached the inpaint batch'
    assert seen['error'] is None, (
        'the inpaint level held the write lock across its LaMa batch — on a real '
        'selection that is minutes of `database is locked` for every other '
        f'writer in the app ({seen["error"]})')


def test_the_framing_pass_lets_writers_through_when_every_image_changed(file_db):
    """The refused-write branch mutates too.

    When `_prepare_analysis_write` declines (the bytes moved between the call and
    the write-back) it invalidates the lanes it refused — a write — and then the
    loop skips to the next image. That write must not travel into the next model
    call any more than a classification does."""
    from app.services import image_bank_service as banks

    with patch.object(banks, '_prepare_analysis_write',
                      lambda row, path, fingerprint: False):
        error = _framing_writer_verdict(
            file_db, lambda _i: '{"framing": "face"}', probe_at=3)
    assert error is None, (
        'the branch that refuses a stale write kept its own invalidation pending '
        f'across the calls that followed it ({error})')
