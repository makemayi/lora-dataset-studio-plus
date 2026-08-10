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
