"""⬇ Getting the pictures OFF the board — one image, or a whole gallery as a ZIP.

The board is the only screen in the app that knows an image's whole ancestry:
dataset → run → checkpoint → seed.  A file that lands in Downloads called
`00042_.png` has lost all of it, and a week later nobody can say which
checkpoint it came from.  There is no sidecar (a settings file per image was
deliberately ruled out), so **the file name is the only carrier of that
lineage** — which makes the naming scheme the load-bearing part of this feature
and the reason most of this file is about strings.

What is asserted here is what a careless implementation gets wrong:

  * the name survives Windows AND Linux — no `<>:"/\\|?*`, no control byte, no
    trailing dot or space, no reserved device name, bounded length;
  * it sorts NATURALLY — step 500 before step 2500, which plain lexicographic
    ordering of un-padded numbers gets backwards;
  * two images of the same step and the same seed still get two names;
  * a file that has vanished from disk is REPORTED, never silently dropped out
    of a ZIP that then looks complete;
  * the cap is real and is reported, so the UI can say it before the click.
"""
import os
import zipfile

import pytest


def _create(client, name='Nova', trigger='nova'):
    return client.post('/api/dataset/create',
                       json={'name': name, 'trigger_word': trigger}).get_json()['id']


def _record(db, dataset_id, family='krea'):
    from app.models import TrainingRunRecord
    rec = TrainingRunRecord(dataset_id=dataset_id, family=family, source='local',
                            fingerprint='f', version=1)
    db.session.add(rec)
    db.session.commit()
    return rec


def _image(db, dataset_id, on_disk=True, **kw):
    """One finished test image, with its file actually written unless asked."""
    from app.models import LoraTestImage
    from app.services.dataset_storage import ensure_dataset_dir
    img = LoraTestImage(dataset_id=dataset_id, checkpoint='a.safetensors',
                        strength=1.0, status=kw.pop('status', 'done'),
                        filename=kw.pop('filename', 'x.png'), **kw)
    db.session.add(img)
    db.session.commit()
    if on_disk and img.filename:
        path = os.path.join(ensure_dataset_dir(dataset_id), img.filename)
        with open(path, 'wb') as fh:
            fh.write(b'\x89PNG\r\n\x1a\n' + img.filename.encode())
    return img


# --- the naming scheme -------------------------------------------------------

WINDOWS_FORBIDDEN = set('<>:"/\\|?*')


def test_the_name_carries_the_whole_lineage(client, app):
    from app.extensions import db
    from app.services import gallery_download as gd
    with app.app_context():
        ds = _create(client, name='Nova Style')
        rec = _record(db, ds)
        img = _image(db, ds, record_id=rec.id, step=2500, seed=208607443,
                     filename='out_00042_.png')
        name = gd.image_download_name(img, 'Nova Style')
        # dataset, run, checkpoint and seed — everything the sidecar would have
        # said, in the only place that survives a copy to another folder.
        assert 'Nova-Style' in name
        assert f'run{rec.id}' in name
        assert 'step002500' in name
        assert 'seed208607443' in name
        assert name.endswith('.png')
        assert str(img.id) in name


def test_the_name_is_legal_on_windows_whatever_the_dataset_is_called(client, app):
    from app.extensions import db
    from app.services import gallery_download as gd
    with app.app_context():
        ds = _create(client, name='ok')
        rec = _record(db, ds)
        img = _image(db, ds, record_id=rec.id, step=100, seed=1)
        for hostile in ('a/b\\c:d*e?f"g<h>i|j', '   ', '....', 'CON', 'aux',
                        '日本語', 'name.with.dots.', 'tab\there'):
            name = gd.image_download_name(img, hostile)
            assert not (set(name) & WINDOWS_FORBIDDEN), name
            assert all(ord(c) >= 32 for c in name), name
            assert not name.endswith(('.', ' ')), name
            assert not name.startswith(('.', ' ', '-')), name
            stem = name.rsplit('.', 1)[0]
            assert stem.upper() not in gd.RESERVED_DEVICE_NAMES, name
            assert len(name) <= gd.MAX_NAME_LEN, name


def test_a_very_long_dataset_name_cannot_blow_the_path_limit(client, app):
    from app.extensions import db
    from app.services import gallery_download as gd
    with app.app_context():
        ds = _create(client, name='ok')
        rec = _record(db, ds)
        img = _image(db, ds, record_id=rec.id, step=1, seed=2)
        name = gd.image_download_name(img, 'x' * 400)
        assert len(name) <= gd.MAX_NAME_LEN
        # …and still says which run and step, which is the whole point.
        assert f'run{rec.id}' in name and 'step000001' in name


def test_names_sort_naturally_by_step_and_the_unknown_step_lands_last(client, app):
    from app.extensions import db
    from app.services import gallery_download as gd
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        rows = [_image(db, ds, record_id=rec.id, step=s, seed=1,
                       filename=f'{s}.png')
                for s in (500, 2500, 10000, 60)]
        rows.append(_image(db, ds, record_id=rec.id, step=None, seed=1,
                           filename='none.png'))
        names = sorted(gd.image_download_name(r, 'Nova') for r in rows)
        # Plain string sort must give training order — that is what the zero
        # padding buys. Un-padded, '2500' sorts before '500'.
        assert [n.split('_step')[1].split('_')[0] for n in names] == [
            '000060', '000500', '002500', '010000', 'unknown']
        assert 'stepunknown' in names[-1]


def test_two_images_of_the_same_step_and_seed_still_get_two_names(client, app):
    from app.extensions import db
    from app.services import gallery_download as gd
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        a = _image(db, ds, record_id=rec.id, step=500, seed=7, filename='a.png')
        b = _image(db, ds, record_id=rec.id, step=500, seed=7, filename='b.png')
        assert gd.image_download_name(a, 'Nova') != gd.image_download_name(b, 'Nova')


def test_a_seedless_image_simply_has_no_seed_segment(client, app):
    from app.extensions import db
    from app.services import gallery_download as gd
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        img = _image(db, ds, record_id=rec.id, step=500, seed=None)
        assert '_seed' not in gallery_name(gd, img)


def gallery_name(gd, img):
    return gd.image_download_name(img, 'Nova')


def test_the_extension_follows_the_file_and_never_smuggles_anything(client, app):
    from app.extensions import db
    from app.services import gallery_download as gd
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        jpg = _image(db, ds, record_id=rec.id, step=1, filename='a.JPEG')
        assert gd.image_download_name(jpg, 'Nova').endswith('.jpeg')
        weird = _image(db, ds, record_id=rec.id, step=2, filename='a.p n g')
        assert gd.image_download_name(weird, 'Nova').endswith('.png')


# --- the plan: what a ZIP would contain, before it is built ------------------

def test_the_plan_covers_the_whole_run_and_counts_what_it_will_hold(client, app):
    from app.extensions import db
    from app.services import gallery_download as gd
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        other = _record(db, ds)
        for s in (500, 500, 1000):
            _image(db, ds, record_id=rec.id, step=s, filename=f'{s}-{id(s)}.png')
        _image(db, ds, record_id=other.id, step=500, filename='other.png')
        plan = gd.gallery_download_plan(rec.id, None)
        assert plan['total'] == 3 and plan['included'] == 3
        assert plan['missing'] == 0 and plan['truncated'] is False
        assert plan['filename'].endswith('.zip')


def test_the_plan_narrows_to_one_checkpoint_when_a_step_is_given(client, app):
    from app.extensions import db
    from app.services import gallery_download as gd
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        _image(db, ds, record_id=rec.id, step=500, filename='a.png')
        _image(db, ds, record_id=rec.id, step=1000, filename='b.png')
        assert gd.gallery_download_plan(rec.id, 500)['total'] == 1


def test_the_plan_honours_an_explicit_selection_and_refuses_foreign_ids(client, app):
    from app.extensions import db
    from app.services import gallery_download as gd
    with app.app_context():
        ds = _create(client)
        rec, other = _record(db, ds), _record(db, ds)
        a = _image(db, ds, record_id=rec.id, step=500, filename='a.png')
        _image(db, ds, record_id=rec.id, step=500, filename='b.png')
        stranger = _image(db, ds, record_id=other.id, step=500, filename='c.png')
        plan = gd.gallery_download_plan(rec.id, None,
                                        image_ids=[a.id, stranger.id, 999999])
        # Scoped exactly like the delete is: an id from another run is not ours.
        assert plan['included'] == 1
        assert [e['id'] for e in plan['entries']] == [a.id]


def test_a_file_gone_from_disk_is_reported_not_silently_dropped(client, app):
    from app.extensions import db
    from app.services import gallery_download as gd
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        _image(db, ds, record_id=rec.id, step=500, filename='here.png')
        _image(db, ds, record_id=rec.id, step=500, filename='gone.png',
               on_disk=False)
        plan = gd.gallery_download_plan(rec.id, None)
        assert plan['total'] == 2
        assert plan['included'] == 1
        assert plan['missing'] == 1
        assert 'no longer on disk' in plan['note']


def test_a_gallery_whose_every_file_is_gone_refuses_instead_of_zipping_nothing(client, app):
    from app.extensions import db
    from app.services import gallery_download as gd
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        _image(db, ds, record_id=rec.id, step=500, filename='gone.png', on_disk=False)
        plan = gd.gallery_download_plan(rec.id, None)
        assert plan['included'] == 0
        assert plan['ok'] is False
        assert 'no longer on disk' in plan['note']


def test_the_cap_is_real_and_is_announced(client, app):
    from app.extensions import db
    from app.services import gallery_download as gd
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        for i in range(7):
            _image(db, ds, record_id=rec.id, step=500, filename=f'i{i}.png')
        plan = gd.gallery_download_plan(rec.id, None, cap=3)
        assert plan['total'] == 7
        assert plan['included'] == 3
        assert plan['truncated'] is True
        assert plan['cap'] == 3
        assert 'newest 3 of 7' in plan['note']
        # The newest are the ones kept — the same rule the panel already uses.
        assert len(plan['entries']) == 3


# --- the ZIP itself ----------------------------------------------------------

def test_the_zip_holds_the_files_under_their_lineage_names(client, app, tmp_path):
    from app.extensions import db
    from app.services import gallery_download as gd
    with app.app_context():
        ds = _create(client, name='Nova Style')
        rec = _record(db, ds)
        _image(db, ds, record_id=rec.id, step=500, seed=11, filename='a.png')
        _image(db, ds, record_id=rec.id, step=2500, seed=22, filename='b.png')
        plan = gd.gallery_download_plan(rec.id, None)
        out = tmp_path / 'g.zip'
        with open(out, 'wb') as fh:
            gd.write_gallery_zip(plan['entries'], fh)
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
            assert len(names) == 2
            assert names == sorted(names)          # step order, naturally
            assert all('/' not in n for n in names)  # flat, no folders
            assert 'step000500' in names[0] and 'step002500' in names[1]
            assert z.read(names[0]).endswith(b'a.png')


def test_two_rows_pointing_at_one_file_do_not_collide_inside_the_zip(client, app, tmp_path):
    from app.extensions import db
    from app.services import gallery_download as gd
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        # A preview re-uses an existing cell's file: two rows, one picture.
        _image(db, ds, record_id=rec.id, step=500, seed=1, filename='same.png')
        _image(db, ds, record_id=rec.id, step=500, seed=1, filename='same.png')
        plan = gd.gallery_download_plan(rec.id, None)
        out = tmp_path / 'g.zip'
        with open(out, 'wb') as fh:
            gd.write_gallery_zip(plan['entries'], fh)
        with zipfile.ZipFile(out) as z:
            assert len(set(z.namelist())) == len(z.namelist()) == 2


# --- the routes --------------------------------------------------------------

def test_downloading_one_image_answers_it_under_its_lineage_name(client, app):
    from app.extensions import db
    with app.app_context():
        ds = _create(client, name='Nova Style')
        rec = _record(db, ds)
        img = _image(db, ds, record_id=rec.id, step=2500, seed=7, filename='a.png')
        image_id = img.id
    r = client.get(f'/api/train/image/{image_id}/download')
    assert r.status_code == 200
    assert 'Nova-Style' in r.headers['Content-Disposition']
    assert 'step002500' in r.headers['Content-Disposition']
    assert r.data.endswith(b'a.png')


def test_downloading_an_image_whose_file_is_gone_says_so(client, app):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        img = _image(db, ds, record_id=rec.id, step=1, filename='gone.png',
                     on_disk=False)
        image_id = img.id
    r = client.get(f'/api/train/image/{image_id}/download')
    assert r.status_code == 404
    assert 'no longer on disk' in r.get_json()['error']


def test_the_run_zip_route_streams_a_real_archive(client, app):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        _image(db, ds, record_id=rec.id, step=500, filename='a.png')
        _image(db, ds, record_id=rec.id, step=1000, filename='b.png')
        rid = rec.id
    r = client.get(f'/api/train/run/{rid}/images/zip')
    assert r.status_code == 200
    assert r.mimetype == 'application/zip'
    import io
    with zipfile.ZipFile(io.BytesIO(r.data)) as z:
        assert len(z.namelist()) == 2


def test_the_zip_plan_route_answers_before_any_byte_is_built(client, app):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        _image(db, ds, record_id=rec.id, step=500, filename='a.png')
        _image(db, ds, record_id=rec.id, step=500, filename='gone.png',
               on_disk=False)
        rid = rec.id
    d = client.get(f'/api/train/run/{rid}/images/zip/plan').get_json()
    assert d['total'] == 2 and d['included'] == 1 and d['missing'] == 1
    assert d['ok'] is True and 'no longer on disk' in d['note']


def test_a_run_with_nothing_downloadable_refuses_with_a_reason(client, app):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        rid = rec.id
    r = client.get(f'/api/train/run/{rid}/images/zip')
    assert r.status_code == 404
    assert 'nothing' in r.get_json()['error'].lower()


def test_the_checkpoint_zip_route_takes_only_that_step(client, app):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        _image(db, ds, record_id=rec.id, step=500, filename='a.png')
        _image(db, ds, record_id=rec.id, step=1000, filename='b.png')
        rid = rec.id
    import io
    r = client.get(f'/api/train/checkpoint/{rid}/500/images/zip')
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.data)) as z:
        assert len(z.namelist()) == 1
        assert 'step000500' in z.namelist()[0]


def test_the_zip_route_honours_an_explicit_selection(client, app):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        a = _image(db, ds, record_id=rec.id, step=500, filename='a.png')
        _image(db, ds, record_id=rec.id, step=1000, filename='b.png')
        rid, aid = rec.id, a.id
    import io
    r = client.get(f'/api/train/run/{rid}/images/zip?ids={aid}')
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.data)) as z:
        assert len(z.namelist()) == 1
    # …and the name says it was a selection, not the whole gallery.
    assert 'selection' in r.headers['Content-Disposition']


@pytest.mark.parametrize('bad', ('abc', '1,,2', ''))
def test_a_malformed_ids_argument_is_ignored_rather_than_crashing(client, app, bad):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        _image(db, ds, record_id=rec.id, step=500, filename='a.png')
        rid = rec.id
    r = client.get(f'/api/train/run/{rid}/images/zip/plan?ids={bad}')
    assert r.status_code == 200
