""""Single person here" — folder-level person assertions on an image bank.

The assertion itself is pure DB work (no model, no subprocess), so most of this
is hermetic by construction. The two places that DO touch the embeddings child
(the full face pass and the ~15-image sample check) drive it through
``_drive_infer_subprocess``, monkeypatched here exactly as the other bank pass
tests do. Background jobs run inline under TESTING (see bank_jobs.start)."""
import hashlib
import json
import os
from collections import deque
from pathlib import Path

from PIL import Image

from app.config import LOCAL_USER
from app.services import folder_person


# --- factories --------------------------------------------------------------
def _save(path, im):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, 'JPEG', quality=92) if path.lower().endswith('.jpg') else im.save(path)


def _flat(value=128, size=64):
    return Image.new('RGB', (size, size), (value, value, value))


def _fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mkbank(client, tmp_path, files, name='B'):
    src = tmp_path / 'src'
    for rel, im in files.items():
        _save(str(src / rel), im)
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _rows(app, bank_id):
    from app.models import BankImage
    with app.app_context():
        return {r.relpath.replace('\\', '/'): (r.face_cluster, r.face_cluster_origin)
                for r in BankImage.query.filter_by(bank_id=bank_id).all()}


def _fresh_job(kind):
    return {'kind': kind, 'done': 0, 'total': 0, 'error': None, 'cancelled': False,
            'finished': False, 'detail': None, '_touched': 0, '_cancel_hook': None,
            'pipeline': None}


_TREE = {os.path.join('anna', 'a1.jpg'): _flat(10),
         os.path.join('anna', 'a2.jpg'): _flat(20),
         os.path.join('anna', 'a3.jpg'): _flat(30),
         os.path.join('bob', 'b1.jpg'): _flat(40),
         'loose.jpg': _flat(50)}


# --- the assertion ----------------------------------------------------------
def test_assert_covers_the_whole_folder_with_no_inference(client, tmp_path, app,
                                                          monkeypatch):
    """One click = every image of the folder in one person group, immediately.
    The embeddings child is booby-trapped: if anything infers, the test fails."""
    bank_id, _src = _mkbank(client, tmp_path, _TREE)
    from app.services import image_bank_service as banks

    def _boom(*a, **k):     # noqa: ANN001 — any call at all is the failure
        raise AssertionError('the assertion must not run a single inference')

    monkeypatch.setattr(banks, '_drive_infer_subprocess', _boom)
    r = client.post(f'/api/bank/{bank_id}/folder-person', json={'subfolder': 'anna'})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['images'] == 3
    rows = _rows(app, bank_id)
    assert rows['anna/a1.jpg'] == rows['anna/a2.jpg'] == rows['anna/a3.jpg']
    assert rows['anna/a1.jpg'][1] == 'asserted'
    # Neither the sibling folder nor the root file is touched.
    assert rows['bob/b1.jpg'] == (None, None)
    assert rows['loose.jpg'] == (None, None)


def test_root_is_an_assertable_folder_of_its_own(client, tmp_path, app):
    """'' is a real subfolder (the bank root) everywhere else in the bank, so it
    is one here too — and it must NOT swallow the nested files."""
    bank_id, _src = _mkbank(client, tmp_path, _TREE)
    r = client.post(f'/api/bank/{bank_id}/folder-person', json={'subfolder': ''})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['images'] == 1
    rows = _rows(app, bank_id)
    assert rows['loose.jpg'][1] == 'asserted'
    assert rows['anna/a1.jpg'] == (None, None)


def test_assertion_survives_a_rescan_and_adopts_new_files(client, tmp_path, app):
    """The rule, not the stamp: a file dropped in the folder later joins the
    group the moment the folder sync sees it — no pass, no second click."""
    bank_id, src = _mkbank(client, tmp_path, _TREE)
    client.post(f'/api/bank/{bank_id}/folder-person', json={'subfolder': 'anna'})
    cid = _rows(app, bank_id)['anna/a1.jpg'][0]
    _save(str(src / 'anna' / 'a4.jpg'), _flat(77))
    _save(str(src / 'bob' / 'b2.jpg'), _flat(88))
    from app.services import image_bank_service as banks
    with app.app_context():
        out = banks.refresh_bank(LOCAL_USER, bank_id, force=True)
    assert out['added'] == 2
    rows = _rows(app, bank_id)
    assert rows['anna/a4.jpg'] == (cid, 'asserted')
    assert rows['bob/b2.jpg'] == (None, None)      # the batch is not stamped whole


def test_revoke_dissolves_the_group_but_spares_computed_ids(client, tmp_path, app):
    bank_id, _src = _mkbank(client, tmp_path, _TREE)
    client.post(f'/api/bank/{bank_id}/folder-person', json={'subfolder': 'anna'})
    # A row of the same folder whose cluster a real pass had computed earlier.
    from app.extensions import db
    from app.models import BankImage
    with app.app_context():
        row = (BankImage.query.filter_by(bank_id=bank_id)
               .filter(BankImage.relpath.contains('a3')).one())
        row.face_cluster, row.face_cluster_origin = 9, None
        db.session.commit()
    r = client.delete(f'/api/bank/{bank_id}/folder-person', json={'subfolder': 'anna'})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['cleared'] == 2
    rows = _rows(app, bank_id)
    assert rows['anna/a1.jpg'] == (None, None)
    assert rows['anna/a3.jpg'] == (9, None)        # not ours to clear
    assert client.get(f'/api/bank/{bank_id}/folder-persons').get_json()['assertions'] == []


def test_revoke_reads_the_subfolder_from_the_query_string_too(client, tmp_path, app):
    """The browser's DELETE carries no body (the shared del() helper sends none),
    so the query-string path is the PRODUCTION one — and '' has to survive it:
    `?subfolder=` means the bank root, not "no subfolder given"."""
    bank_id, _src = _mkbank(client, tmp_path, _TREE)
    client.post(f'/api/bank/{bank_id}/folder-person', json={'subfolder': ''})
    r = client.delete(f'/api/bank/{bank_id}/folder-person?subfolder=')
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['cleared'] == 1
    assert _rows(app, bank_id)['loose.jpg'] == (None, None)
    # And a request that names nothing at all is refused, never guessed.
    assert client.delete(f'/api/bank/{bank_id}/folder-person').status_code == 400


def test_revoking_a_folder_that_was_never_asserted_is_a_400(client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path, _TREE)
    r = client.delete(f'/api/bank/{bank_id}/folder-person', json={'subfolder': 'bob'})
    assert r.status_code == 400
    assert 'not asserted' in r.get_json()['error']


def test_asserting_an_empty_subfolder_is_refused(client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path, _TREE)
    r = client.post(f'/api/bank/{bank_id}/folder-person',
                    json={'subfolder': 'nobody-here'})
    assert r.status_code == 400
    assert 'no images' in r.get_json()['error']


def test_deleting_the_bank_takes_its_assertions(client, tmp_path, app):
    bank_id, _src = _mkbank(client, tmp_path, _TREE)
    client.post(f'/api/bank/{bank_id}/folder-person', json={'subfolder': 'anna'})
    assert client.delete(f'/api/bank/{bank_id}').status_code == 200
    from app.models import BankFolderPerson
    with app.app_context():
        assert BankFolderPerson.query.filter_by(bank_id=bank_id).count() == 0


# --- coexistence with the embeddings pass -----------------------------------
def test_face_pass_skips_asserted_images_and_never_reuses_their_id(
        client, tmp_path, app, monkeypatch):
    """THE saving, and the id-space contract in one test: the pass is not asked
    to embed the asserted folder at all, and the clusters it does compute are
    pushed above the asserted id instead of colliding with it."""
    bank_id, _src = _mkbank(client, tmp_path, _TREE)
    client.post(f'/api/bank/{bank_id}/folder-person', json={'subfolder': 'anna'})
    asserted_id = _rows(app, bank_id)['anna/a1.jpg'][0]
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
    seen = {}

    def fake_driver(job, python, script, payload, cache_path, rx, window):
        imgs = json.loads(payload)['images']
        seen['images'] = imgs
        return ({'ok': True,
                 'results': {p: {'state': 'scorable', 'det': 0.9,
                                 'fingerprint': _fingerprint(p)} for p in imgs},
                 'clusters': {p: 1 for p in imgs}}, deque(), 0)

    monkeypatch.setattr(banks, '_drive_infer_subprocess', fake_driver)
    job = _fresh_job('faces')
    with app.app_context():
        banks._faces_job(bank_id)(job)
    # Not one of the three asserted files was handed to the child.
    assert seen['images'] and not any('anna' in p for p in seen['images'])
    assert len(seen['images']) == 2                # bob + the root file
    rows = _rows(app, bank_id)
    assert rows['anna/a1.jpg'] == (asserted_id, 'asserted')   # untouched
    assert rows['bob/b1.jpg'][0] == asserted_id + 1           # offset, no collision
    assert rows['bob/b1.jpg'][1] is None
    assert 'skipped (subfolder asserted as one person)' in (job['detail'] or '')


def test_face_pass_offsets_past_transferred_asserted_rows_without_a_rule(
        client, tmp_path, app, monkeypatch):
    """A transferred assertion still owns its id after its folder rule is gone."""
    bank_id, _src = _mkbank(client, tmp_path, _TREE)
    from app.extensions import db
    from app.models import BankFolderPerson, BankImage
    from app.services import image_bank_service as banks

    with app.app_context():
        asserted = (BankImage.query.filter_by(bank_id=bank_id)
                    .filter(BankImage.relpath.contains('a1')).one())
        asserted.face_cluster = 37
        asserted.face_cluster_origin = 'asserted'
        db.session.commit()
        asserted_id = asserted.id
        assert BankFolderPerson.query.filter_by(bank_id=bank_id).count() == 0
        assert folder_person.asserted_offset(bank_id) == 37

    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))

    def fake_driver(job, python, script, payload, cache_path, rx, window):
        imgs = json.loads(payload)['images']
        return ({'ok': True,
                 'results': {p: {'state': 'scorable', 'det': 0.9,
                                 'fingerprint': _fingerprint(p)} for p in imgs},
                 'clusters': {p: 1 for p in imgs}}, deque(), 0)

    monkeypatch.setattr(banks, '_drive_infer_subprocess', fake_driver)
    with app.app_context():
        banks._faces_job(bank_id)(_fresh_job('faces'))
        db.session.expire_all()
        rows = BankImage.query.filter_by(bank_id=bank_id).all()
        assert next(r for r in rows if r.id == asserted_id).face_cluster == 37
        computed = [r.face_cluster for r in rows
                    if r.face_cluster_origin != 'asserted']
        assert set(computed) == {38}


def test_face_pass_total_promises_only_what_it_will_embed(client, tmp_path, app,
                                                          monkeypatch):
    bank_id, _src = _mkbank(client, tmp_path, _TREE)
    client.post(f'/api/bank/{bank_id}/folder-person', json={'subfolder': 'anna'})
    from app.services import face_similarity, image_bank_service as banks
    monkeypatch.setattr(face_similarity, 'is_available', lambda: True)
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        lambda *a, **k: ({'ok': True, 'results': {}, 'clusters': {}},
                                         deque(), 0))
    with app.app_context():
        job = banks.start_faces(app, LOCAL_USER, bank_id)
    assert job['total'] == 2       # 5 images, 3 of them asserted away


def test_face_partition_is_discarded_when_one_effective_input_is_unresolved(
        client, tmp_path, app, monkeypatch):
    bank_id, _src = _mkbank(client, tmp_path, _TREE)
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
    monkeypatch.setattr(folder_person, 'probe_after_faces', lambda *_a, **_k: '')
    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        rows = BankImage.query.filter_by(bank_id=bank_id).all()
        for row in rows:
            row.face_cluster = 9
        broken = rows[0]
        broken.watermark_fingerprint = transfer.content_fingerprint_path(
            banks.abs_image_path(bank, broken))
        broken.watermark_clean_method = 'crop'
        assert not banks.clean_image_path(bank_id, broken.id).exists()
        db.session.commit()

        def fake_driver(job, python, script, payload, cache_path, rx, window):
            paths = json.loads(payload)['images']
            return ({'ok': True,
                     'results': {p: {'state': 'scorable', 'det': 0.9,
                                     'fingerprint': _fingerprint(p)} for p in paths},
                     'clusters': {p: 1 for p in paths}}, deque(), 0)

        monkeypatch.setattr(banks, '_drive_infer_subprocess', fake_driver)
        job = _fresh_job('faces')
        banks._faces_job(bank_id)(job)
        db.session.expire_all()
        assert {row.face_cluster for row in
                BankImage.query.filter_by(bank_id=bank_id).all()} == {None}
        assert 'grouping discarded' in job['detail']


# --- the sample check -------------------------------------------------------
def _run_check(app, bank_id, subfolder, clusters_of, monkeypatch, states=None):
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
    seen = {}

    def fake_driver(job, python, script, payload, cache_path, rx, window):
        imgs = json.loads(payload)['images']
        seen['images'] = imgs
        seen['threshold'] = json.loads(payload)['threshold']
        res = {p: {'state': (states or {}).get(os.path.basename(p), 'scorable'),
                   'det': 0.9, 'fingerprint': _fingerprint(p)} for p in imgs}
        return ({'ok': True, 'results': res,
                 'clusters': clusters_of(imgs)}, deque(), 0)

    monkeypatch.setattr(banks, '_drive_infer_subprocess', fake_driver)
    job = _fresh_job('folder-check')
    with app.app_context():
        folder_person._sample_job(bank_id, subfolder)(job)
    return job, seen


def test_sample_check_says_consistent_and_costs_a_sample(client, tmp_path, app,
                                                         monkeypatch):
    files = {os.path.join('anna', f'a{i:03d}.jpg'): _flat(i) for i in range(60)}
    bank_id, _src = _mkbank(client, tmp_path, files)
    client.post(f'/api/bank/{bank_id}/folder-person', json={'subfolder': 'anna'})
    job, seen = _run_check(app, bank_id, 'anna',
                           lambda imgs: {p: 1 for p in imgs}, monkeypatch)
    # 60 images in the folder, 15 embedded — that ratio IS the feature.
    assert len(seen['images']) == folder_person.SAMPLE_SIZE
    assert job['detail'] == 'sample consistent (15/15 same person)'
    data = client.get(f'/api/bank/{bank_id}/folder-persons').get_json()
    sample = data['assertions'][0]['sample']
    assert sample['verdict'] == 'consistent'
    assert sample['faces'] == 1 and sample['sample'] == 15


def test_sample_check_measures_cleaned_rotated_effective_bytes(
        client, tmp_path, app, monkeypatch):
    """Folder Face evidence follows the same strict displayed-byte identity as
    the full pass, while the user's asserted person id remains untouched."""
    files = {os.path.join('anna', f'a{i:03d}.jpg'): _flat(i) for i in range(15)}
    bank_id, _src = _mkbank(client, tmp_path, files)
    client.post(f'/api/bank/{bank_id}/folder-person', json={'subfolder': 'anna'})

    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks
    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        row = (BankImage.query.filter_by(bank_id=bank_id)
               .order_by(BankImage.id.asc()).first())
        asserted = (row.face_cluster, row.face_cluster_origin)
        raw = banks.abs_image_path(bank, row)
        row.watermark_fingerprint = transfer.content_fingerprint_path(raw)
        row.watermark_state = 'detected'
        row.watermark_clean_method = 'crop'
        row.rotation = 90
        clean = banks.clean_image_path(bank_id, row.id)
        clean.parent.mkdir(parents=True, exist_ok=True)
        _flat(213, size=80).resize((80, 40)).save(clean, 'WEBP')
        db.session.commit()
        effective = banks.analysis_image_path(bank, row, refresh_rotation=True)
        expected_fingerprint = transfer.content_fingerprint_path(effective)
        row_id = row.id

    _job, seen = _run_check(
        app, bank_id, 'anna', lambda imgs: {p: 1 for p in imgs}, monkeypatch)
    assert effective in seen['images']
    with app.app_context():
        row = db.session.get(BankImage, row_id)
        assert row.analysis_fingerprint == expected_fingerprint
        assert row.face_state == 'scorable'
        assert (row.face_cluster, row.face_cluster_origin) == asserted
        # Binding the freshly measured fingerprint must not delete the rotated
        # derivative and thereby stale the hash-bearing child cache.
        assert Path(effective).is_file()
    sample = client.get(
        f'/api/bank/{bank_id}/folder-persons').get_json()['assertions'][0]['sample']
    assert sample['verdict'] == 'consistent' and sample['stale'] is False

    with app.app_context():
        banks.rotate_images(LOCAL_USER, bank_id, [row_id], 90)
    stale = client.get(
        f'/api/bank/{bank_id}/folder-persons').get_json()['assertions'][0]['sample']
    assert stale['verdict'] == 'consistent' and stale['stale'] is True


def test_sample_check_discards_report_when_effective_source_switches_midflight(
        client, tmp_path, app, monkeypatch):
    files = {os.path.join('anna', f'a{i:03d}.jpg'): _flat(i) for i in range(15)}
    bank_id, _src = _mkbank(client, tmp_path, files)
    client.post(f'/api/bank/{bank_id}/folder-person', json={'subfolder': 'anna'})
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
    changed = {}

    def fake_driver(job, python, script, payload, cache_path, rx, window):
        paths = json.loads(payload)['images']
        results = {p: {'state': 'scorable', 'det': 0.9,
                       'fingerprint': _fingerprint(p)} for p in paths}
        target_path = paths[0]
        bank = db.session.get(ImageBank, bank_id)
        row = next(r for r in BankImage.query.filter_by(bank_id=bank_id).all()
                   if banks.abs_image_path(bank, r) == target_path)
        clean = banks.clean_image_path(bank_id, row.id)
        clean.parent.mkdir(parents=True, exist_ok=True)
        _flat(244, size=48).save(clean, 'WEBP')
        row.watermark_fingerprint = transfer.content_fingerprint_path(target_path)
        row.watermark_clean_method = 'crop'
        changed['id'] = row.id
        db.session.commit()
        return ({'ok': True, 'results': results,
                 'clusters': {p: 1 for p in paths}}, deque(), 0)

    monkeypatch.setattr(banks, '_drive_infer_subprocess', fake_driver)
    job = _fresh_job('folder-check')
    with app.app_context():
        folder_person._sample_job(bank_id, 'anna')(job)
        row = db.session.get(BankImage, changed['id'])
        assert row.face_state is None
        assert row.face_cluster_origin == 'asserted'
    assertion = client.get(
        f'/api/bank/{bank_id}/folder-persons').get_json()['assertions'][0]
    assert assertion['sample'] is None
    assert 'no report saved' in job['detail']


def test_sample_check_reuses_the_clustering_threshold(client, tmp_path, app,
                                                      monkeypatch):
    """One truth about "same person" in this app: the check must compare at the
    bank's own face_threshold, never at a second number of its own."""
    bank_id, _src = _mkbank(client, tmp_path, _TREE)
    client.post(f'/api/bank/{bank_id}/folder-person', json={'subfolder': 'anna'})
    _job, seen = _run_check(app, bank_id, 'anna',
                            lambda imgs: {p: 1 for p in imgs}, monkeypatch)
    from app.services import image_bank_service as banks
    with app.app_context():
        assert seen['threshold'] == banks.thresholds()['face_threshold']


def test_sample_check_warns_on_two_faces_without_touching_the_assertion(
        client, tmp_path, app, monkeypatch):
    bank_id, _src = _mkbank(client, tmp_path, _TREE)
    client.post(f'/api/bank/{bank_id}/folder-person', json={'subfolder': 'anna'})
    before = _rows(app, bank_id)['anna/a1.jpg']
    job, _seen = _run_check(
        app, bank_id, 'anna',
        lambda imgs: {p: (1 if i else 2) for i, p in enumerate(imgs)}, monkeypatch)
    assert job['detail'] == '2 different faces in the sample — check this folder'
    # The warning INFORMS. The user's folder, the user's call.
    assert _rows(app, bank_id)['anna/a1.jpg'] == before
    data = client.get(f'/api/bank/{bank_id}/folder-persons').get_json()
    assert data['assertions'][0]['sample']['verdict'] == 'mixed'


def test_sample_check_is_honest_when_it_saw_no_face(client, tmp_path, app,
                                                    monkeypatch):
    bank_id, _src = _mkbank(client, tmp_path, _TREE)
    client.post(f'/api/bank/{bank_id}/folder-person', json={'subfolder': 'anna'})
    job, _seen = _run_check(app, bank_id, 'anna', lambda imgs: {}, monkeypatch,
                            states={'a1.jpg': 'no_face', 'a2.jpg': 'no_face',
                                    'a3.jpg': 'unreadable'})
    # It reports the FOLDER, not a broken check — and it does not send the user
    # to a full pass that reads the very same images with the very same detector.
    assert 'no readable face in 3 images tried' in job['detail']
    assert 'the same way' in job['detail']
    data = client.get(f'/api/bank/{bank_id}/folder-persons').get_json()
    entry = data['assertions'][0]
    assert entry['sample']['verdict'] == 'inconclusive'
    # Guard-rail: what the machinery could not read is LISTED, never dropped.
    assert {t['state'] for t in entry['to_check']} == {'no_face', 'unreadable'}
    assert len(entry['to_check']) == 3


def test_sample_check_needs_an_assertion(client, tmp_path, app):
    bank_id, _src = _mkbank(client, tmp_path, _TREE)
    r = client.post(f'/api/bank/{bank_id}/folder-person/check',
                    json={'subfolder': 'anna'})
    assert r.status_code == 400
    assert 'not asserted' in r.get_json()['error']


def test_the_saving_is_counted_in_inferences_not_claimed(client, tmp_path, app,
                                                          monkeypatch):
    """The feature's whole promise is a NUMBER, so it is measured here rather
    than asserted in a comment: on a bank shaped like a real scrape (six folders
    of one person each, plus a mixed one), the embeddings child is handed only
    the images no assertion covers."""
    files = {}
    for f in range(6):
        for i in range(50):
            files[os.path.join(f'person{f}', f'{i:03d}.jpg')] = _flat(i)
    for i in range(40):
        files[os.path.join('mixed', f'{i:03d}.jpg')] = _flat(i)
    bank_id, _src = _mkbank(client, tmp_path, files)       # 340 images
    for f in range(6):
        r = client.post(f'/api/bank/{bank_id}/folder-person',
                        json={'subfolder': f'person{f}'})
        assert r.status_code == 200, r.get_json()
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
    seen = {}

    def fake_driver(job, python, script, payload, cache_path, rx, window):
        req = json.loads(payload)
        imgs = req['images']
        # EVERY call is recorded, not the last one: the pass now chains a folder
        # probe behind itself, and a fake that only remembers the newest call
        # would silently measure the probe and call it the pass.
        seen.setdefault('calls', []).append(len(imgs))
        return ({'ok': True,
                 'results': {p: {'state': 'scorable', 'det': 0.9,
                                 'fingerprint': _fingerprint(p)} for p in imgs},
                 'clusters': {p: 1 for p in imgs},
                 'group_clusters': {g['name']: {p: 1 for p in g['images']}
                                    for g in (req.get('groups') or [])}},
                deque(), 0)

    monkeypatch.setattr(banks, '_drive_infer_subprocess', fake_driver)
    with app.app_context():
        banks._faces_job(bank_id)(_fresh_job('faces'))
    assert seen['calls'][0] == 40          # 340 images, 300 asserted away
    # And every one of those 300 still has a person id — the group is real, it
    # was simply not paid for.
    from app.models import BankImage
    with app.app_context():
        grouped = (BankImage.query.filter_by(bank_id=bank_id)
                   .filter(BankImage.face_cluster_origin == 'asserted').count())
    assert grouped == 300
    # Six declared folders = six distinct person ids, not one merged blob.
    rows = _rows(app, bank_id)
    ids = {rows[f'person{f}/000.jpg'][0] for f in range(6)}
    assert len(ids) == 6


# --- automatic suggestion (probe) -------------------------------------------
def _big_tree(folders=3, per=20):
    files = {}
    for f in range(folders):
        for i in range(per):
            files[os.path.join(f'model{f}', f'{i:03d}.jpg')] = _flat(i)
    return files


def _probe_driver(seen, clusters_of):
    def fake_driver(job, python, script, payload, cache_path, rx, window):
        req = json.loads(payload)
        imgs = req['images']
        seen.setdefault('calls', []).append(req)
        return ({'ok': True,
                 'results': {p: {'state': 'scorable', 'det': 0.9,
                                 'fingerprint': _fingerprint(p)} for p in imgs},
                 'group_clusters': {g['name']: clusters_of(g['name'], g['images'])
                                    for g in (req.get('groups') or [])}},
                deque(), 0)
    return fake_driver


def test_the_probe_suggests_but_never_groups_a_single_image(client, tmp_path, app,
                                                            monkeypatch):
    """The whole safety property in one test: a folder the app believes is one
    person is OFFERED, and until the user clicks, not one image has moved."""
    bank_id, _src = _mkbank(client, tmp_path, _big_tree())
    from app.services import face_similarity, image_bank_service as banks
    monkeypatch.setattr(face_similarity, 'is_available', lambda: True)
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
    seen = {}
    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        _probe_driver(seen, lambda n, imgs: {p: 1 for p in imgs}))
    r = client.post(f'/api/bank/{bank_id}/folder-scan')
    assert r.status_code == 202, r.get_json()

    data = client.get(f'/api/bank/{bank_id}/folder-persons').get_json()
    assert data['assertions'] == []                    # nothing was declared
    assert {s['subfolder'] for s in data['suggestions']} == {'model0', 'model1', 'model2'}
    assert all(s['verdict'] == 'consistent' for s in data['suggestions'])
    # NOT ONE image was grouped. This is the line that must never go green by
    # accident: a suggestion that grouped anything would be an assertion.
    assert all(v == (None, None) for v in _rows(app, bank_id).values())


def test_one_child_call_covers_every_folder_and_clusters_them_apart(
        client, tmp_path, app, monkeypatch):
    """Forty folders must not be forty subprocesses — and the folders must be
    clustered SEPARATELY, or two folders merging into one cluster would read as
    'consistent' when it means the opposite."""
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=4))
    from app.services import face_similarity, image_bank_service as banks
    monkeypatch.setattr(face_similarity, 'is_available', lambda: True)
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
    seen = {}
    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        _probe_driver(seen, lambda n, imgs: {p: 1 for p in imgs}))
    client.post(f'/api/bank/{bank_id}/folder-scan')
    assert len(seen['calls']) == 1                     # ONE subprocess
    req = seen['calls'][0]
    assert {g['name'] for g in req['groups']} == {'model0', 'model1', 'model2', 'model3'}
    assert all(len(g['images']) == folder_person.SAMPLE_SIZE for g in req['groups'])
    # And the child is asked to cluster per group, at the clustering threshold.
    with app.app_context():
        assert req['threshold'] == banks.thresholds()['face_threshold']


def test_a_mixed_folder_is_reported_as_such_and_never_suggested_as_one(
        client, tmp_path, app, monkeypatch):
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=2))
    from app.services import face_similarity, image_bank_service as banks
    monkeypatch.setattr(face_similarity, 'is_available', lambda: True)
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))

    def clusters_of(name, imgs):
        if name == 'model1':      # two people in this one
            return {p: (1 if i % 2 else 2) for i, p in enumerate(imgs)}
        return {p: 1 for p in imgs}

    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        _probe_driver({}, clusters_of))
    client.post(f'/api/bank/{bank_id}/folder-scan')
    got = {s['subfolder']: s['verdict'] for s in
           client.get(f'/api/bank/{bank_id}/folder-persons').get_json()['suggestions']}
    assert got == {'model0': 'consistent', 'model1': 'mixed'}


def test_a_probe_expires_when_the_folder_changes(client, tmp_path, app, monkeypatch):
    """A verdict describes the folder it sampled. Add images and it stops being
    an answer about the folder in front of you — so it is marked stale rather
    than quietly kept."""
    bank_id, src = _mkbank(client, tmp_path, _big_tree(folders=1))
    from app.services import face_similarity, image_bank_service as banks
    monkeypatch.setattr(face_similarity, 'is_available', lambda: True)
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        _probe_driver({}, lambda n, imgs: {p: 1 for p in imgs}))
    client.post(f'/api/bank/{bank_id}/folder-scan')
    assert client.get(f'/api/bank/{bank_id}/folder-persons') \
        .get_json()['suggestions'][0]['stale'] is False
    _save(str(src / 'model0' / 'new.jpg'), _flat(9))
    with app.app_context():
        banks.refresh_bank(LOCAL_USER, bank_id, force=True)
    fresh = client.get(f'/api/bank/{bank_id}/folder-persons').get_json()
    assert fresh['suggestions'][0]['stale'] is True
    # And a stale folder is offered to the scan again rather than left behind.
    with app.app_context():
        assert 'model0' in [n for n, _c in folder_person.scan_candidates(bank_id)]


def test_confirming_a_suggestion_removes_it(client, tmp_path, app, monkeypatch):
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=2))
    from app.services import face_similarity, image_bank_service as banks
    monkeypatch.setattr(face_similarity, 'is_available', lambda: True)
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        _probe_driver({}, lambda n, imgs: {p: 1 for p in imgs}))
    client.post(f'/api/bank/{bank_id}/folder-scan')
    client.post(f'/api/bank/{bank_id}/folder-person', json={'subfolder': 'model0'})
    data = client.get(f'/api/bank/{bank_id}/folder-persons').get_json()
    assert [s['subfolder'] for s in data['suggestions']] == ['model1']
    assert [a['subfolder'] for a in data['assertions']] == ['model0']


def test_tiny_and_asserted_folders_are_not_probed(client, tmp_path, app, monkeypatch):
    """Two folders a suggestion would only add noise to: one already declared,
    one too small for fifteen images to mean anything."""
    files = _big_tree(folders=1)
    files[os.path.join('scraps', 'a.jpg')] = _flat(1)
    files[os.path.join('scraps', 'b.jpg')] = _flat(2)
    bank_id, _src = _mkbank(client, tmp_path, files)
    client.post(f'/api/bank/{bank_id}/folder-person', json={'subfolder': 'model0'})
    with app.app_context():
        assert folder_person.scan_candidates(bank_id) == []


def test_the_scan_states_what_it_did_not_reach(client, tmp_path, app, monkeypatch):
    """A ceiling that says nothing would read as 'the rest are not one person'."""
    from app.services import folder_person as fp
    monkeypatch.setattr(fp, 'MAX_SCAN_FOLDERS', 2)
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=5))
    from app.services import face_similarity, image_bank_service as banks
    monkeypatch.setattr(face_similarity, 'is_available', lambda: True)
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        _probe_driver({}, lambda n, imgs: {p: 1 for p in imgs}))
    job = _fresh_job('folder-scan')
    with app.app_context():
        fp._folder_scan_job(bank_id)(job)
    assert '2 folder(s) sampled' in job['detail']
    assert '3 folder(s) not reached' in job['detail']


def test_the_face_pass_probes_for_free_and_says_what_it_found(
        client, tmp_path, app, monkeypatch):
    """The automatic trigger. It runs inside the face pass because that is the
    one moment the embeddings are already cached."""
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=2))
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
    seen = {}
    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        _probe_driver(seen, lambda n, imgs: {p: 1 for p in imgs}))
    job = _fresh_job('faces')
    with app.app_context():
        banks._faces_job(bank_id)(job)
    assert '2 folder(s) look like a single person' in job['detail']
    # Two calls: the pass, then the probe — and the probe reuses the pass's OWN
    # cache file, which is what makes it free.
    assert len(seen['calls']) == 2
    assert seen['calls'][1]['cache'] == seen['calls'][0]['cache']
    data = client.get(f'/api/bank/{bank_id}/folder-persons').get_json()
    assert len(data['suggestions']) == 2
    assert all(v[1] is None for v in _rows(app, bank_id).values())   # still nothing asserted


def test_a_failing_probe_never_turns_a_finished_face_pass_red(
        client, tmp_path, app, monkeypatch):
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=1))
    from app.services import folder_person as fp, image_bank_service as banks
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))

    def fake_driver(job, python, script, payload, cache_path, rx, window):
        imgs = json.loads(payload)['images']
        return ({'ok': True,
                 'results': {p: {'state': 'scorable', 'det': 0.9,
                                 'fingerprint': _fingerprint(p)} for p in imgs},
                 'clusters': {p: 1 for p in imgs}}, deque(), 0)

    monkeypatch.setattr(banks, '_drive_infer_subprocess', fake_driver)
    monkeypatch.setattr(fp, '_run_probe',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    job = _fresh_job('faces')
    with app.app_context():
        banks._faces_job(bank_id)(job)
    assert job['error'] is None
    assert 'person cluster' in job['detail']       # the pass still reports itself
    assert 'boom' not in (job['detail'] or '')


def test_the_angle_lane_never_probes(client, tmp_path, app, monkeypatch):
    """⤢ Angles re-runs on rows that already have a face_state. Probing there
    would re-suggest folders on every backfill, for nothing new."""
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=1))
    from app.services import image_bank_service as banks
    from app.extensions import db as _db
    from app.models import BankImage
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
    with app.app_context():
        for r in BankImage.query.filter_by(bank_id=bank_id).all():
            r.face_state = 'scorable'
        _db.session.commit()
    seen = {}
    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        _probe_driver(seen, lambda n, imgs: {p: 1 for p in imgs}))
    with app.app_context():
        banks._faces_job(bank_id, angles_only=True)(_fresh_job('angles'))
    assert len(seen['calls']) == 1
    assert not seen['calls'][0].get('groups')


def test_a_stratified_sample_spans_the_whole_folder():
    """The first 15 files of a scraped folder are one shoot; a second person
    appearing halfway through must still be reachable."""
    picked = folder_person._stratified(list(range(100)), k=10)
    assert len(picked) == 10
    assert picked[0] == 0 and picked[-1] >= 80      # reaches the far end
    assert len(set(picked)) == 10                   # no image sampled twice
    assert folder_person._stratified([1, 2, 3], k=10) == [1, 2, 3]


# --- replacing a draw that cannot be read ------------------------------------
# THE HOLE, as a real bank showed it: four folders of six ended the preflight
# with NO verdict — "only 0 of 15 sampled images had a usable face" — and 3 546
# images went to the full pass behind it. Fifteen embeddings bought nothing and
# the pass they were meant to avoid ran anyway. A draw that cannot be read is now
# REPLACED, up to a budget that is the whole design.
def test_the_draw_budget_is_the_smaller_of_a_ceiling_and_a_quarter_of_the_folder():
    """Read the rule off the numbers, not off the comment: small folders keep the
    single draw they have today, big ones may re-draw up to the ceiling, and no
    folder ever gives up more than a quarter of itself to a suggestion."""
    b = folder_person.draw_budget
    size = folder_person.SAMPLE_SIZE
    assert b(0) == 0
    for n in (5, 15, 20, 60):
        assert b(n) == min(n, size)          # unchanged from the single draw
    assert b(100) == 25 and b(200) == 50     # a quarter of the folder
    assert b(240) == b(3546) == folder_person.PROBE_MAX_DRAWS   # the ceiling
    for n in (5, 61, 240, 466, 3546):
        assert b(n) <= folder_person.PROBE_MAX_DRAWS
        assert b(n) <= max(size, -(-n * 25 // 100))
    # The ceiling is 15 usable faces at a hit rate of one in four — the worst
    # rate still worth chasing. Below it the folder is face-poor, and more draws
    # would buy price, not answers.
    assert folder_person.PROBE_MAX_DRAWS == 4 * size


def test_the_scan_summary_counts_the_thin_verdicts_and_names_the_faceless():
    """The one-line report of a scan. A partial folder WILL be pre-ticked in the
    dialog, so it has to be counted here too — announcing 3 and pre-ticking 5
    would send the user hunting for the two that went missing."""
    d = folder_person._probe_detail
    line = d({'consistent': 3, 'partial': 2, 'mixed': 1, 'inconclusive': 1}, 7, 0)
    assert '5 folder(s) look like one person' in line
    assert '2 of them on thin evidence' in line
    assert '1 holds several' in line
    # Not "too few faces to tell", which invited the full pass to try harder on
    # images it reads through the very same detector.
    assert '1 has almost no readable face' in line
    assert 'too few faces to tell' not in line
    many = d({'mixed': 2, 'inconclusive': 3}, 5, 4)
    assert '2 hold several' in many and '3 have almost no readable face' in many
    assert '4 folder(s) not reached' in many


def _face_probe_driver(seen, faces, person=lambda name, path: 1):
    """A child that reads a face only in the images ``faces`` accepts (by their
    numeric basename), the way a folder of crops and backs behaves."""
    def has_face(p):
        return int(os.path.splitext(os.path.basename(p))[0]) in faces

    def fake_driver(job, python, script, payload, cache_path, rx, window):
        req = json.loads(payload)
        seen.setdefault('calls', []).append(req)
        return ({'ok': True,
                 'results': {p: {'state': 'scorable' if has_face(p) else 'no_face',
                                 'det': 0.9 if has_face(p) else 0.0,
                                 'fingerprint': _fingerprint(p)}
                             for p in req['images']},
                 'group_clusters': {
                     g['name']: {p: person(g['name'], p)
                                 for p in g['images'] if has_face(p)}
                     for g in (req.get('groups') or [])}},
                deque(), 0)
    return fake_driver


def _drawn(seen, folder):
    """Every path the probe ever handed to the child for one folder."""
    out = []
    for req in seen['calls']:
        for g in (req.get('groups') or []):
            if g['name'] == folder:
                out = list(g['images'])     # each round carries the whole set
    return out


def test_a_folder_with_no_readable_face_reports_the_folder_and_stops_at_its_budget(
        client, tmp_path, app, monkeypatch):
    """The exact case from the bank: nothing readable anywhere. The probe replaces
    its unusable draws until the budget is spent, then says what it learnt ABOUT
    THE FOLDER — it does not point at a full pass that would read the same images
    through the same detector."""
    banks = _preflight_ready(monkeypatch)
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=1, per=240))
    seen = {}
    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        _face_probe_driver(seen, faces=set()))
    client.post(f'/api/bank/{bank_id}/person-preflight')

    known = client.get(f'/api/bank/{bank_id}/person-preflight').get_json()['known']
    assert [k['verdict'] for k in known] == ['inconclusive']
    assert known[0]['scorable'] == 0
    # It tried FOUR times what it used to, and not one image more than the budget.
    assert known[0]['sample'] == folder_person.draw_budget(240) == 60
    assert 'no readable face in 60 images tried' in known[0]['note']
    assert 'the same way' in known[0]['note']
    assert 'nothing to compare' not in known[0]['note']
    drawn = _drawn(seen, 'model0')
    assert len(drawn) == 60 and len(set(drawn)) == 60      # never twice
    assert len(seen['calls']) == 2                         # bounded rounds
    # …and still not one image grouped. The safety rule survives the re-draw.
    assert all(v == (None, None) for v in _rows(app, bank_id).values())


def test_a_hard_folder_reaches_its_target_by_replacing_the_unreadable_draws(
        client, tmp_path, app, monkeypatch):
    """One image in three has a face — the case the old single draw answered with
    "only 5 of 15 had a usable face" and gave up on. Re-drawing reaches the
    fifteen usable faces the verdict is supposed to rest on."""
    import random
    banks = _preflight_ready(monkeypatch)
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=1, per=240))
    faces = set(random.Random(11).sample(range(240), 80))
    seen = {}
    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        _face_probe_driver(seen, faces))
    client.post(f'/api/bank/{bank_id}/person-preflight')

    known = client.get(f'/api/bank/{bank_id}/person-preflight').get_json()['known']
    assert known[0]['verdict'] == 'consistent'
    assert known[0]['scorable'] >= folder_person.PROBE_TARGET_FACES
    drawn = _drawn(seen, 'model0')
    assert len(set(drawn)) == len(drawn)                   # no image drawn twice
    assert len(drawn) <= folder_person.draw_budget(240)    # and inside the budget
    assert len(drawn) > folder_person.SAMPLE_SIZE          # it did re-draw
    # The re-draw stays SPREAD: replacing the unreadable images must not turn the
    # sample into a clump at the top of the folder, or a second person appearing
    # later would stop being reachable.
    idx = sorted(int(os.path.splitext(os.path.basename(p))[0]) for p in drawn)
    assert idx[0] < 24 and idx[-1] > 216


def test_a_budget_spent_short_of_the_target_still_gives_a_verdict_that_says_so(
        client, tmp_path, app, monkeypatch):
    """One image in ten. Fifteen usable faces are out of reach inside the budget,
    and a weak verdict that states its own weakness beats the nothing this used
    to produce — so it is offered, and the sentence carries the numbers."""
    import random
    banks = _preflight_ready(monkeypatch)
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=1, per=240))
    faces = set(random.Random(5).sample(range(240), 24))
    seen = {}
    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        _face_probe_driver(seen, faces))
    client.post(f'/api/bank/{bank_id}/person-preflight')

    known = client.get(f'/api/bank/{bank_id}/person-preflight').get_json()['known']
    assert known[0]['verdict'] == 'partial'
    assert 2 <= known[0]['scorable'] < folder_person.PROBE_TARGET_FACES
    assert known[0]['sample'] == folder_person.draw_budget(240)
    assert 'on thin evidence' in known[0]['note']
    assert f"in {known[0]['sample']} images tried" in known[0]['note']
    # A partial verdict is an OFFER like any other — accepting it writes the same
    # ordinary, revocable assertion, and nothing was grouped before that click.
    assert all(v == (None, None) for v in _rows(app, bank_id).values())
    r = client.post(f'/api/bank/{bank_id}/folder-persons/accept',
                    json={'subfolders': ['model0']})
    assert r.status_code == 200 and r.get_json()['images'] == 240


def test_a_folder_that_answers_first_time_is_left_exactly_as_it_was(
        client, tmp_path, app, monkeypatch):
    """The regression guard. On ordinary folders — the overwhelming majority —
    the re-draw must be invisible: one child call, fifteen images, same verdict,
    same cost. A fix for the hard case that taxed the easy one would be a bad
    trade made silently."""
    banks = _preflight_ready(monkeypatch)
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=2, per=240))
    seen = {}
    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        _face_probe_driver(seen, faces=set(range(240))))
    client.post(f'/api/bank/{bank_id}/person-preflight')

    assert len(seen['calls']) == 1                     # ONE round, as before
    groups = seen['calls'][0]['groups']
    assert len(groups) == 2
    assert all(len(g['images']) == folder_person.SAMPLE_SIZE for g in groups)
    known = client.get(f'/api/bank/{bank_id}/person-preflight').get_json()['known']
    assert {k['verdict'] for k in known} == {'consistent'}
    assert all(k['sample'] == folder_person.SAMPLE_SIZE for k in known)


def test_the_preflight_announces_the_ceiling_of_its_re_draw_before_it_is_paid(
        client, tmp_path, monkeypatch):
    """The preflight is sold as "a few seconds against a whole pass". A mechanism
    that can multiply its bill has to be visible in the estimate, not discovered
    in the progress bar."""
    _preflight_ready(monkeypatch)
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=2, per=240))
    plan = client.get(f'/api/bank/{bank_id}/person-preflight').get_json()
    assert plan['sample_cost'] == 2 * folder_person.SAMPLE_SIZE          # 30
    assert plan['sample_cost_max'] == 2 * folder_person.draw_budget(240)  # 120
    assert plan['sample_max'] == folder_person.PROBE_MAX_DRAWS
    # Both stay far under the pass they stand in front of — that ratio IS the
    # justification, and the quarter-of-the-folder cap is what guarantees it.
    assert plan['sample_cost_max'] * 4 <= plan['full_cost']


def test_a_folder_already_read_as_unusable_is_not_drawn_from_twice(
        client, tmp_path, app, monkeypatch):
    """An image the detector has already read as faceless cannot become readable:
    same script, same gates, and its embedding is cached. Re-drawing it would burn
    budget for an answer we hold — so the pool skips it, which is what makes the
    probe after a full pass land on faces instead of on crops."""
    banks = _preflight_ready(monkeypatch)
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=1, per=240))
    from app.extensions import db as _db
    from app.models import BankImage
    with app.app_context():
        for r in BankImage.query.filter_by(bank_id=bank_id).all():
            n = int(os.path.splitext(os.path.basename(r.relpath))[0])
            r.face_state = 'scorable' if n % 8 == 0 else 'no_face'
        _db.session.commit()
    seen = {}
    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        _face_probe_driver(seen, faces=set(range(0, 240, 8))))
    client.post(f'/api/bank/{bank_id}/person-preflight')

    drawn = _drawn(seen, 'model0')
    assert drawn and all(int(os.path.splitext(os.path.basename(p))[0]) % 8 == 0
                         for p in drawn)
    # One round is enough once the known-blind images are out of the way.
    assert len(seen['calls']) == 1
    known = client.get(f'/api/bank/{bank_id}/person-preflight').get_json()['known']
    assert known[0]['verdict'] == 'consistent' and known[0]['scorable'] == 15


# --- the preflight: the same probe, moved IN FRONT of the pass ---------------
# The critique that produced it: "the first thing a user does is Launch all, so
# they never go through the folder scan". Everything below is about the DEFAULT
# path — a saving reachable only from a side panel is not a saving.
def _preflight_ready(monkeypatch):
    from app.services import face_similarity, image_bank_service as banks
    monkeypatch.setattr(face_similarity, 'is_available', lambda: True)
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
    return banks


def test_the_preflight_states_its_cost_against_the_pass_it_replaces(
        client, tmp_path, monkeypatch):
    """The number that makes the offer worth reading is the COMPARISON, so the
    plan carries both: what the sample costs and what the pass would."""
    _preflight_ready(monkeypatch)
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=3, per=20))
    plan = client.get(f'/api/bank/{bank_id}/person-preflight').get_json()
    assert plan['available'] is True
    assert plan['candidates'] == 3 and plan['covered'] == 3 and plan['left'] == 0
    assert plan['sample_cost'] == 3 * folder_person.SAMPLE_SIZE   # 45
    assert plan['full_cost'] == 60                                # the whole bank
    assert plan['known'] == []                                    # nothing probed yet


def test_a_bank_with_nothing_to_check_asks_no_question(client, tmp_path, monkeypatch):
    """No subfolder big enough to sample = no dialog. The pass must not gain a
    detour on the banks the feature cannot help."""
    _preflight_ready(monkeypatch)
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _flat(1), 'b.jpg': _flat(2)})
    plan = client.get(f'/api/bank/{bank_id}/person-preflight').get_json()
    assert plan['candidates'] == 0 and plan['known'] == []


def test_the_preflight_offers_and_still_groups_nothing_by_itself(
        client, tmp_path, app, monkeypatch):
    """The module's safety rule, re-proved on the new path: running the preflight
    produces OFFERS. Not one image moves until the user answers."""
    banks = _preflight_ready(monkeypatch)
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=3, per=20))

    def clusters_of(name, imgs):
        if name == 'model2':          # two people in this one
            return {p: (1 if i % 2 else 2) for i, p in enumerate(imgs)}
        return {p: 1 for p in imgs}

    monkeypatch.setattr(banks, '_drive_infer_subprocess', _probe_driver({}, clusters_of))
    r = client.post(f'/api/bank/{bank_id}/person-preflight')
    assert r.status_code == 202, r.get_json()

    plan = client.get(f'/api/bank/{bank_id}/person-preflight').get_json()
    got = {k['subfolder']: k['verdict'] for k in plan['known']}
    assert got == {'model0': 'consistent', 'model1': 'consistent', 'model2': 'mixed'}
    # Each offer carries what accepting it would spare the pass.
    assert all(k['images'] == 20 for k in plan['known'])
    # Nothing left to sample, and NOTHING grouped.
    assert plan['candidates'] == 0
    assert all(v == (None, None) for v in _rows(app, bank_id).values())


def test_accepting_writes_ordinary_assertions_and_the_pass_skips_them(
        client, tmp_path, app, monkeypatch):
    """The end-to-end of the default path: preflight → accept the pre-ticked
    folders → the pass only embeds what is left. And what 'accept' wrote is an
    ORDINARY assertion — same origin, same revoke — not a second state."""
    banks = _preflight_ready(monkeypatch)
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=3, per=20))
    seen = {}
    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        _probe_driver(seen, lambda n, imgs: {p: 1 for p in imgs}))
    client.post(f'/api/bank/{bank_id}/person-preflight')

    r = client.post(f'/api/bank/{bank_id}/folder-persons/accept',
                    json={'subfolders': ['model0', 'model1']})
    assert r.status_code == 200, r.get_json()
    out = r.get_json()
    assert out['accepted'] == ['model0', 'model1'] and out['images'] == 40
    assert out['failed'] == []

    rows = _rows(app, bank_id)
    assert all(rows[f'model0/{i:03d}.jpg'][1] == 'asserted' for i in range(20))
    # Two folders, two DISTINCT people — not one merged blob.
    assert rows['model0/000.jpg'][0] != rows['model1/000.jpg'][0]
    # …and revoking works on them exactly as on a hand-made assertion.
    assert client.delete(f'/api/bank/{bank_id}/folder-person',
                         json={'subfolder': 'model0'}).status_code == 200

    # Re-accept, then run the pass: it embeds model2 only.
    client.post(f'/api/bank/{bank_id}/folder-persons/accept',
                json={'subfolders': ['model0']})
    seen['calls'] = []
    with app.app_context():
        job = _fresh_job('faces')
        banks._faces_job(bank_id)(job)
    assert len(seen['calls'][0]['images']) == 20        # 60 images, 40 asserted away
    assert '40 image(s) skipped' in job['detail']


def test_analyze_everything_anyway_leaves_every_folder_to_the_pass(
        client, tmp_path, app, monkeypatch):
    """The escape hatch has to be real: answering the preflight with nothing
    ticked must leave the bank exactly as the pass would have found it."""
    banks = _preflight_ready(monkeypatch)
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=3, per=20))
    seen = {}
    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        _probe_driver(seen, lambda n, imgs: {p: 1 for p in imgs}))
    client.post(f'/api/bank/{bank_id}/person-preflight')
    r = client.post(f'/api/bank/{bank_id}/folder-persons/accept',
                    json={'subfolders': []})
    assert r.status_code == 200 and r.get_json()['accepted'] == []
    seen['calls'] = []
    with app.app_context():
        banks._faces_job(bank_id)(_fresh_job('faces'))
    assert len(seen['calls'][0]['images']) == 60       # every image, nothing skipped
    with app.app_context():
        assert folder_person.asserted_subfolders(bank_id) == set()


def test_the_preflight_says_what_its_ceiling_did_not_reach(
        client, tmp_path, monkeypatch):
    """A ceiling that stayed quiet would read as 'the rest are not one person'."""
    from app.services import folder_person as fp
    monkeypatch.setattr(fp, 'MAX_PREFLIGHT_FOLDERS', 2)
    banks = _preflight_ready(monkeypatch)
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=5, per=20))
    plan = client.get(f'/api/bank/{bank_id}/person-preflight').get_json()
    assert plan['candidates'] == 5 and plan['covered'] == 2 and plan['left'] == 3
    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        _probe_driver({}, lambda n, imgs: {p: 1 for p in imgs}))
    client.post(f'/api/bank/{bank_id}/person-preflight')
    after = client.get(f'/api/bank/{bank_id}/person-preflight').get_json()
    assert len(after['known']) == 2 and after['candidates'] == 3


def test_accept_reports_the_folders_it_could_not_group(client, tmp_path, monkeypatch):
    """'11 of the 12 you ticked' has to be sayable — a swallowed failure would
    leave the user believing in a skip that will not happen."""
    _preflight_ready(monkeypatch)
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=1, per=20))
    r = client.post(f'/api/bank/{bank_id}/folder-persons/accept',
                    json={'subfolders': ['model0', 'ghost']})
    out = r.get_json()
    assert out['accepted'] == ['model0']
    assert [f['subfolder'] for f in out['failed']] == ['ghost']


def test_the_manual_scan_keeps_its_own_smaller_ceiling(client, tmp_path, monkeypatch):
    """The preflight is generous because it stands in front of a full pass; the
    standalone button is not, because there it is the whole cost. One code path,
    two ceilings — and neither may inherit the other's."""
    assert folder_person.MAX_PREFLIGHT_FOLDERS > folder_person.MAX_SCAN_FOLDERS
    banks = _preflight_ready(monkeypatch)
    bank_id, _src = _mkbank(client, tmp_path, _big_tree(folders=25, per=6))
    seen = {}
    monkeypatch.setattr(banks, '_drive_infer_subprocess',
                        _probe_driver(seen, lambda n, imgs: {p: 1 for p in imgs}))
    client.post(f'/api/bank/{bank_id}/folder-scan')
    assert len(seen['calls'][0]['groups']) == folder_person.MAX_SCAN_FOLDERS
    seen['calls'] = []
    client.post(f'/api/bank/{bank_id}/person-preflight')
    assert len(seen['calls'][0]['groups']) == 5        # the five the scan left
