"""A bank and a dataset must never share bytes on disk — only transit by COPY.

Nothing used to stop `POST /api/bank/create` from being handed a dataset's own
storage folder: the resulting bank listed the dataset's LIVE files, and its
🗑 Delete rejected then deleted images out of the dataset. `overlapping_banks`
never saw it, because it only ever compared banks against banks.

These tests pin the door shut on every spelling of the same folder — the folder
itself, a parent of it, a child of it, another case, other separators, through a
symlink/junction — and pin the two transit paths as COPIES, so the invariant is
"they never share", not "they usually don't".
"""
import os

import pytest


# On Windows normcase folds case; on POSIX it does not, and two spellings that
# differ in case really ARE two folders. The guarantee has to be stated per
# platform rather than assumed.
CASE_INSENSITIVE = os.path.normcase('A') == 'a'


@pytest.fixture()
def ds(app):
    """A dataset with one image on disk, in its real storage folder."""
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDataset, FaceDatasetImage
        from app.services.dataset_storage import ensure_dataset_dir
        d = FaceDataset(user_id='local', name='Lola', trigger_word='Lola69382')
        db.session.add(d)
        db.session.commit()
        folder = ensure_dataset_dir(d.id)
        with open(os.path.join(folder, 'a.png'), 'wb') as fh:
            fh.write(b'\x89PNG\x01')
        db.session.add(FaceDatasetImage(dataset_id=d.id, filename='a.png',
                                        status='keep'))
        db.session.commit()
        yield {'id': d.id, 'folder': str(folder)}


def _create(client, folder, name='B'):
    return client.post('/api/bank/create', json={'name': name, 'folder': folder})


# --- the door that was open --------------------------------------------------

def test_create_on_a_dataset_folder_is_refused_and_says_what_to_do(client, ds):
    r = _create(client, ds['folder'])
    assert r.status_code == 400
    msg = r.get_json()['error']
    assert 'dataset' in msg.lower()
    # A refusal that does not name the alternative turns a trap into a wall.
    assert 'Import to bank' in msg
    assert 'copies' in msg.lower()


def test_the_datasets_root_itself_is_refused(client, ds, app):
    from app import config as cfg
    with app.app_context():
        root = str(cfg.dataset_images_root())
    assert _create(client, root).status_code == 400


def test_a_parent_of_the_datasets_root_is_refused(client, ds, app):
    """A bank walks recursively: a bank on the parent inventories every dataset."""
    from app import config as cfg
    with app.app_context():
        parent = os.path.dirname(str(cfg.dataset_images_root()))
    assert _create(client, parent).status_code == 400


def test_a_subfolder_of_a_dataset_folder_is_refused(client, ds):
    sub = os.path.join(ds['folder'], 'nested')
    os.makedirs(sub, exist_ok=True)
    assert _create(client, sub).status_code == 400


def test_other_separators_spell_the_same_folder(client, ds):
    """`C:/x/y` and `C:\\x\\y` are one folder on Windows; on POSIX only the
    native spelling exists, so that is the one that must be caught."""
    spelled = ds['folder'].replace('\\', '/') if os.sep == '\\' else ds['folder']
    assert _create(client, spelled).status_code == 400


@pytest.mark.skipif(not CASE_INSENSITIVE,
                    reason='POSIX: a differently-cased path is a different folder')
def test_another_case_spells_the_same_folder(client, ds):
    assert _create(client, ds['folder'].upper()).status_code == 400


def test_through_a_symlink_or_junction(client, ds, tmp_path):
    """A link is the sharpest version of the trap: the path looks unrelated."""
    link = tmp_path / 'looks-innocent'
    try:
        if os.name == 'nt':
            import _winapi
            _winapi.CreateJunction(ds['folder'], str(link))
        else:
            os.symlink(ds['folder'], str(link), target_is_directory=True)
    except (OSError, AttributeError, NotImplementedError) as e:
        pytest.skip(f'cannot create a link here: {e}')
    assert _create(client, str(link)).status_code == 400


# --- what must keep working --------------------------------------------------

def test_a_legitimate_folder_is_never_refused(client, ds, tmp_path):
    folder = tmp_path / 'my-scrape-dump'
    folder.mkdir()
    (folder / 'x.png').write_bytes(b'\x89PNG\x02')
    r = _create(client, str(folder))
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['added'] == 1


def test_a_sibling_of_the_datasets_root_is_never_refused(client, ds, app, tmp_path):
    """Only containment counts. A folder NEXT TO the datasets root shares nothing."""
    from app import config as cfg
    with app.app_context():
        sibling = os.path.join(
            os.path.dirname(str(cfg.dataset_images_root())), 'not-datasets')
    os.makedirs(sibling, exist_ok=True)
    assert _create(client, sibling).status_code == 200


# --- relocation is the same door, later --------------------------------------

def test_relocating_a_bank_onto_a_dataset_folder_is_refused(client, ds, tmp_path):
    folder = tmp_path / 'dump'
    folder.mkdir()
    (folder / 'x.png').write_bytes(b'\x89PNG\x03')
    bank_id = _create(client, str(folder)).get_json()['id']
    r = client.post(f'/api/bank/{bank_id}/relocate',
                    json={'folder': ds['folder'], 'confirm': True})
    assert r.status_code == 400
    assert 'Import to bank' in r.get_json()['error']
    # And the dry run refuses too — the dialog must not even offer the move.
    r2 = client.post(f'/api/bank/{bank_id}/relocate', json={'folder': ds['folder']})
    assert r2.status_code == 400


# --- installs that already carry the trap ------------------------------------

def _legacy_bank_on(app, folder, name='Legacy'):
    """A bank pointing at a dataset folder, written straight to the DB — exactly
    what an install created before the guard existed still holds."""
    with app.app_context():
        from app.extensions import db
        from app.models import ImageBank, BankImage
        b = ImageBank(user_id='local', name=name, source_path=os.path.realpath(folder))
        db.session.add(b)
        db.session.flush()
        db.session.add(BankImage(bank_id=b.id, relpath='a.png', status='reject'))
        db.session.commit()
        return b.id


def test_an_existing_bank_on_a_dataset_folder_announces_itself(client, ds, app):
    bank_id = _legacy_bank_on(app, ds['folder'])
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    conflict = payload.get('dataset_conflict')
    assert conflict, 'the workspace must be able to say the bank sits on a dataset'
    assert conflict['dataset_id'] == ds['id']
    assert 'Import to bank' in conflict['message']


def test_delete_rejected_is_refused_on_such_a_bank_and_the_file_survives(
        client, ds, app):
    bank_id = _legacy_bank_on(app, ds['folder'])
    r = client.post(f'/api/bank/{bank_id}/delete-rejected')
    assert r.status_code == 400
    assert 'Import to bank' in r.get_json()['error']
    # The whole point: the dataset's image is still there.
    assert os.path.isfile(os.path.join(ds['folder'], 'a.png'))


def test_such_a_bank_stays_readable(client, ds, app):
    """Triage must keep working — only the destructive door closes."""
    bank_id = _legacy_bank_on(app, ds['folder'])
    assert client.get(f'/api/bank/{bank_id}').status_code == 200


# --- "only transit": both directions COPY ------------------------------------

def test_import_to_bank_copies_so_the_dataset_survives_a_bank_delete(app, ds):
    """dataset -> bank. Deleting the bank's files must not touch the dataset."""
    from unittest.mock import patch
    from app.services import image_bank_service as banks

    def fake_start(_app, bank_id, kind, fn, total=0):
        fn(object())
        return None

    with app.app_context():
        from app.models import ImageBank
        from app.extensions import db
        with patch.object(banks.bank_jobs, 'start', fake_start), \
             patch.object(banks.bank_jobs, 'cancelled', lambda job: False), \
             patch.object(banks.bank_jobs, 'bump', lambda job, n=1: None), \
             patch.object(banks.bank_jobs, 'progress', lambda job, **kw: None):
            bank_id = banks.start_dataset_import(app, 'local', ds['id'], 'Copy')
        bank = db.session.get(ImageBank, bank_id)
        copied = os.path.join(bank.source_path, 'a.png')
        assert os.path.isfile(copied)
        assert os.path.realpath(bank.source_path) != os.path.realpath(ds['folder'])
        os.remove(copied)
        assert os.path.isfile(os.path.join(ds['folder'], 'a.png'))


def test_promote_copies_so_the_bank_survives_a_dataset_delete(app, tmp_path, ds):
    """bank -> dataset. Removing the dataset's copy must not touch the source."""
    from app.services import face_dataset_service as fds
    src = tmp_path / 'pool'
    src.mkdir()
    from PIL import Image
    Image.new('RGB', (800, 800), 'red').save(src / 'p.png')
    with app.app_context():
        blob = (src / 'p.png').read_bytes()
        ids, _bad = fds.import_images('local', ds['id'], [blob])
        assert ids
        from app.models import FaceDatasetImage
        from app.extensions import db
        row = db.session.get(FaceDatasetImage, ids[0])
        landed = os.path.join(ds['folder'], row.filename)
        assert os.path.isfile(landed)
        os.remove(landed)
        assert os.path.isfile(src / 'p.png')
