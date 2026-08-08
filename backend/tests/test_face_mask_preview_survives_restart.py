"""A face-mask preview must survive a server restart.

The pass it stands for is the expensive one — the detector load alone is a fixed
price paid before image 1 (measured ~4-5 s warm on the reference machine, tens of
seconds cold, plus a ~350 MB download on the very first run), then every kept
image on top. Holding that result in a module dict meant a restart threw it away
and the panel offered a full re-run of work the user had already waited for. The
Stop/Resume bargain had the same hole: the boxes banked so a resume would be
cheap died with the process, so the resume they were banked for never happened.

What makes persisting safe is that the honesty guard was already built: a stored
preview is bound to the FINGERPRINT of the kept set it was computed from, and a
mismatch flags it stale (result) or drops it (bank). Writing it to disk changes
where it lives, not what it is allowed to claim — which is why the staleness
cases are asserted here too, ACROSS the restart, rather than trusted.

`fmp.reset()` is the restart: it drops every byte of module state, exactly what a
new process starts with. The same idiom `video_clip_search.reset_memo` already
uses to prove its own on-disk layer.

One thing deliberately does NOT come back: a job that was RUNNING. Its thread
died with the process, so a resurrected "analyzing image 4 of 153" would be a
progress bar for nothing — a ghost the user could only wait out.
"""
import json
import os

from PIL import Image

from app.extensions import db
from app.models import FaceDatasetImage
from app.services import face_dataset_service as svc
from app.services import face_mask_preview as fmp
from app.config import LOCAL_USER, save_config


def _dataset(app, tmp_path, n=3):
    """A concept dataset with `n` kept images, and its fingerprint."""
    save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'FMR', 'fmr_act', kind='concept',
                                concept_desc='balancing a spoon')
        img_dir = svc._dataset_dir(ds.id)
        for i in range(n):
            fn = f'k{i}.png'
            Image.new('RGB', (64, 64)).save(os.path.join(img_dir, fn))
            db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep', filename=fn))
        db.session.commit()
        dataset_id = ds.id
    fmp.reset()
    return dataset_id


def _fingerprint(dataset_id):
    """The fingerprint of the kept set as the route computes it."""
    with_stamps = []
    imgs = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='keep').all())
    for img in imgs:
        st = os.stat(os.path.join(svc._dataset_dir(dataset_id), img.filename))
        with_stamps.append((img.id, img.filename, st.st_size, st.st_mtime_ns))
    return fmp.fingerprint(with_stamps)


def _sample_result():
    return {'samples': [{'image_id': 1, 'filename': 'k0.png', 'state': 'masked',
                         'boxes': [[0.1, 0.1, 0.4, 0.4]]}],
            'coverage': {'masked': 3, 'total': 3},
            'expand': 1.3}


def test_published_preview_is_still_there_after_a_restart(app, tmp_path):
    """The whole point: the boxes the user waited for outlive the process."""
    dataset_id = _dataset(app, tmp_path)
    with app.app_context():
        fp = _fingerprint(dataset_id)
        fmp.set_result(dataset_id, _sample_result(), fp)

        fmp.reset()                                   # ← the restart

        snap = fmp.snapshot(dataset_id, current_fp=fp, total=3)
    assert snap['result'] is not None, 'the preview did not survive the restart'
    assert snap['result']['stale'] is False
    assert snap['result']['coverage'] == {'masked': 3, 'total': 3}
    assert snap['result']['samples'][0]['boxes'] == [[0.1, 0.1, 0.4, 0.4]]


def test_a_restored_preview_still_admits_the_set_moved_under_it(app, tmp_path):
    """Persisted does not mean trusted. A preview restored from disk is bound to
    the same fingerprint it always was, so a kept set that changed while the
    server was down comes back LABELLED — never silently fresh."""
    dataset_id = _dataset(app, tmp_path)
    with app.app_context():
        fmp.set_result(dataset_id, _sample_result(), _fingerprint(dataset_id))

        fmp.reset()

        snap = fmp.snapshot(dataset_id, current_fp='a-different-kept-set', total=3)
    assert snap['result'] is not None
    assert snap['result']['stale'] is True


def test_the_resume_credit_survives_so_stop_stays_worth_offering(app, tmp_path):
    """Stop banks the faces already found. If the bank dies with the process, the
    resume it exists for cannot happen and Stop becomes a discard."""
    dataset_id = _dataset(app, tmp_path)
    with app.app_context():
        fp = _fingerprint(dataset_id)
        banked = {'/x/k0.png': {'state': 'masked', 'boxes': [[0, 0, 1, 1]]},
                  '/x/k1.png': {'state': 'none', 'boxes': []}}
        fmp.remember_partial(dataset_id, banked, fp)

        fmp.reset()

        assert fmp.partial(dataset_id, fp) == banked
        snap = fmp.snapshot(dataset_id, current_fp=fp, total=3)
    assert snap['resume'] == {'done': 2, 'total': 3}


def test_a_bank_whose_set_moved_is_dropped_across_the_restart(app, tmp_path):
    """The bank's rule is stricter than the result's — a stale bank is worth
    nothing, because resuming from it would mix boxes from images that left the
    set. Reading it back off disk must not soften that."""
    dataset_id = _dataset(app, tmp_path)
    with app.app_context():
        fmp.remember_partial(dataset_id, {'/x/k0.png': {'state': 'masked'}},
                             _fingerprint(dataset_id))

        fmp.reset()

        assert fmp.partial(dataset_id, 'a-different-kept-set') == {}
        # ...and the drop is durable: the next restart does not resurrect it.
        fmp.reset()
        snap = fmp.snapshot(dataset_id, current_fp='a-different-kept-set', total=3)
    assert snap['resume'] is None


def test_a_completed_pass_clears_the_bank_for_good(app, tmp_path):
    """clear_partial is what a finished pass calls to supersede the bank. If it
    only cleared memory, the next restart would offer a resume for a pass that
    already ran to completion."""
    dataset_id = _dataset(app, tmp_path)
    with app.app_context():
        fp = _fingerprint(dataset_id)
        fmp.remember_partial(dataset_id, {'/x/k0.png': {'state': 'masked'}}, fp)
        fmp.set_result(dataset_id, _sample_result(), fp)
        fmp.clear_partial(dataset_id)

        fmp.reset()

        snap = fmp.snapshot(dataset_id, current_fp=fp, total=3)
    assert snap['result'] is not None
    assert snap['resume'] is None


def test_a_running_job_does_not_come_back_as_a_ghost(app, tmp_path):
    """The thread died with the process. A restored "analyzing image 4 of 153"
    would be a bar nothing is driving, and the panel would refuse to start the
    pass that actually needs to run."""
    dataset_id = _dataset(app, tmp_path)
    with app.app_context():
        fp = _fingerprint(dataset_id)
        started = []
        fmp.start(app, dataset_id, lambda job: started.append(job), total=3, fp=fp)
        assert started, 'the inline TESTING path did not run the work'

        fmp.reset()

        assert fmp.get(dataset_id) is None
        snap = fmp.snapshot(dataset_id, current_fp=fp, total=3)
    assert snap['job'] is None


def test_an_unreadable_store_degrades_to_no_preview(app, tmp_path):
    """A truncated or hand-edited file must cost the user one recomputation, not
    a broken training panel."""
    dataset_id = _dataset(app, tmp_path)
    with app.app_context():
        fp = _fingerprint(dataset_id)
        fmp.set_result(dataset_id, _sample_result(), fp)
        path = fmp._store_path(dataset_id)
        assert os.path.isfile(path), 'nothing was written to disk'
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('{not json at all')

        fmp.reset()

        snap = fmp.snapshot(dataset_id, current_fp=fp, total=3)
    assert snap['result'] is None
    assert snap['resume'] is None


def test_the_store_is_a_dot_file_inside_the_dataset_folder(app, tmp_path):
    """Where it lives matters: beside the images it describes, so deleting the
    dataset takes it with it, and hidden so it never shows up as an image. Same
    shape as the dataset's existing .bank-analysis-cache sidecar."""
    dataset_id = _dataset(app, tmp_path)
    with app.app_context():
        fmp.set_result(dataset_id, _sample_result(), _fingerprint(dataset_id))
        path = fmp._store_path(dataset_id)
        assert os.path.dirname(path) == svc._dataset_dir(dataset_id)
        assert os.path.basename(path).startswith('.')
        payload = json.loads(open(path, encoding='utf-8').read())
    assert 'job' not in payload, 'a live job must never be written down'
