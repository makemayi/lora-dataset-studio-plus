"""The workspace rail's "recent variations" strip.

It is the only place in the app that shows output ACROSS datasets — everything
else is scoped to the one that is open — so what it must never do is show a
picture that is not one: an import, a pending row with no file, or something
already rejected.
"""
import io
import os

from PIL import Image


def _png(color=(0, 128, 255)):
    buf = io.BytesIO(); Image.new('RGB', (32, 32), color).save(buf, 'PNG')
    return buf.getvalue()


def _image(svc, ds, name, **kw):
    d = svc._dataset_dir(ds.id)
    os.makedirs(d, exist_ok=True)
    if name:
        with open(os.path.join(d, name), 'wb') as fh:
            fh.write(_png())
    row = svc.FaceDatasetImage(dataset_id=ds.id, filename=name,
                               source=kw.pop('source', 'generated'),
                               status=kw.pop('status', 'keep'), **kw)
    svc.db.session.add(row)
    svc.db.session.commit()
    return row


def test_the_newest_variations_come_back_first_with_their_dataset(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        a = svc.create_dataset(LOCAL_USER, 'Ana', 'ana')
        b = svc.create_dataset(LOCAL_USER, 'Bea', 'bea')
        _image(svc, a, 'a1.png', variation_label='Bust')
        _image(svc, b, 'b1.png', variation_label='Face')
        out = svc.recent_generated_images(LOCAL_USER)
        assert [i['filename'] for i in out] == ['b1.png', 'a1.png']
        assert out[0]['dataset_id'] == b.id and out[0]['dataset_name'] == 'Bea'
        assert out[0]['label'] == 'Face'


def test_only_real_generated_pictures_are_offered(app):
    """An import is not a variation, a pending row has no picture, and a strip
    of images you already threw away is not a memory aid."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Mixed', 'mx')
        _image(svc, ds, 'kept.png')
        _image(svc, ds, 'imported.png', source='import')
        _image(svc, ds, None, status='pending')
        _image(svc, ds, 'rejected.png', status='reject')
        assert [i['filename'] for i in svc.recent_generated_images(LOCAL_USER)] == ['kept.png']


def test_the_limit_is_bounded_and_junk_tolerant(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Many', 'mny')
        for i in range(30):
            _image(svc, ds, f'v{i}.png')
        assert len(svc.recent_generated_images(LOCAL_USER, 3)) == 3
        assert len(svc.recent_generated_images(LOCAL_USER, 999)) == svc.RECENT_VARIATIONS_MAX
        assert len(svc.recent_generated_images(LOCAL_USER, 'lots')) == 12


def test_the_route_answers_the_strip(app, client):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Ana', 'ana')
        _image(svc, ds, 'a1.png')
        ds_id = ds.id
    body = client.get('/api/dataset/recent-images?limit=5').get_json()
    assert body['images'][0]['dataset_id'] == ds_id
    assert body['images'][0]['filename'] == 'a1.png'


def test_a_face_swap_moves_its_tile_to_the_FRONT(app, monkeypatch, tmp_path):
    """A swap REUSES its row — same id, new file — so ordering by id (or by
    created_at) buries a swap from a minute ago behind rows that merely happened
    to be created later. The strip is about pictures, not about row numbers."""
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.services import face_swap_helper
    from app.config import LOCAL_USER
    with app.app_context():
        base = tmp_path / 'comfyui'
        (base / 'input').mkdir(parents=True); (base / 'output').mkdir(parents=True)
        (base / 'main.py').write_text('# fake', encoding='utf-8')
        cfg.save_config({'comfyui': {'base_dir': str(base)}})
        ds = svc.create_dataset(LOCAL_USER, 'Ana', 'ana')
        d = svc._dataset_dir(ds.id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'ref.png'), 'wb') as fh:
            fh.write(_png((200, 10, 10)))
        ds.ref_filename = 'ref.png'
        old_row = _image(svc, ds, 'old.png')          # generated first
        _image(svc, ds, 'newer.png')                  # ...and this one after it
        assert [i['filename'] for i in svc.recent_generated_images(LOCAL_USER)] \
            == ['newer.png', 'old.png']

        monkeypatch.setattr(face_swap_helper, 'enqueue_face_swap',
                            lambda **_kw: 'swap-job')
        job_id = svc.face_swap_image(LOCAL_USER, old_row.id)
        out_dir = svc._comfy_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'swapped.png'), 'wb') as fh:
            fh.write(_png((5, 5, 200)))
        svc.link_completed_dataset_image(job_id, 'swapped.png')

        out = svc.recent_generated_images(LOCAL_USER)
        assert [i['filename'] for i in out] == ['swapped.png', 'newer.png']


def test_rows_that_predate_the_column_still_sort(app):
    """Every existing row has content_changed_at NULL; without the coalesce they
    would all sort together at one end and the strip would look random."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Old', 'old')
        for name in ('a.png', 'b.png', 'c.png'):
            _image(svc, ds, name)
        got = [i['filename'] for i in svc.recent_generated_images(LOCAL_USER)]
        assert got == ['c.png', 'b.png', 'a.png']
