"""Per-image delete guard: locking an image blocks single delete, batch delete
and purge — nothing else. Reject/regenerate/face-swap/crop/mirror are untouched
(no test needed here; the guard lives only in the delete-shaped code paths)."""
import io
import os

from PIL import Image

from app.config import LOCAL_USER


def _png(color=(255, 0, 0)):
    buf = io.BytesIO(); Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def _seed_images(svc, ds_id, n=3, status='keep'):
    """N committed image rows with real files, returns their ids."""
    from app.models import FaceDatasetImage
    d = svc._dataset_dir(ds_id); os.makedirs(d, exist_ok=True)
    ids = []
    for i in range(n):
        fn = f'img{i}.webp'
        open(os.path.join(d, fn), 'wb').write(_png((i * 40, 0, 0)))
        img = FaceDatasetImage(dataset_id=ds_id, filename=fn, status=status, framing='face')
        svc.db.session.add(img); svc.db.session.flush(); ids.append(img.id)
    svc.db.session.commit()
    return ids


def test_is_locked_defaults_to_false(app):
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Lk', 'lk')
        [img_id] = _seed_images(svc, ds.id, n=1)
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert payload['images'][0]['is_locked'] is False


def test_set_image_locked_toggles_and_persists(app):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Lk2', 'lk2')
        [img_id] = _seed_images(svc, ds.id, n=1)
        assert svc.set_image_locked(LOCAL_USER, img_id, True) is True
        assert svc.db.session.get(FaceDatasetImage, img_id).is_locked is True
        assert svc.set_image_locked(LOCAL_USER, img_id, False) is True
        assert svc.db.session.get(FaceDatasetImage, img_id).is_locked is False


def test_set_image_locked_returns_false_for_unknown_or_foreign_image(app):
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds1 = svc.create_dataset(LOCAL_USER, 'Own', 'own')
        ds2 = svc.create_dataset(LOCAL_USER, 'Other', 'other')
        [foreign_id] = _seed_images(svc, ds2.id, n=1)
        assert svc.set_image_locked('someone-else', foreign_id, True) is False
        assert svc.set_image_locked(LOCAL_USER, 999999, True) is False


def test_single_delete_refuses_a_locked_image_and_leaves_it_intact(app):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Ld', 'ld')
        [img_id] = _seed_images(svc, ds.id, n=1)
        svc.set_image_locked(LOCAL_USER, img_id, True)
        try:
            svc.delete_image(LOCAL_USER, img_id)
            assert False, 'delete_image should have refused a locked image'
        except RuntimeError as e:
            assert 'locked' in str(e).lower()
        row = svc.db.session.get(FaceDatasetImage, img_id)
        assert row is not None
        assert os.path.exists(os.path.join(svc._dataset_dir(ds.id), row.filename))


def test_single_delete_still_works_once_unlocked(app):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Lu', 'lu')
        [img_id] = _seed_images(svc, ds.id, n=1)
        svc.set_image_locked(LOCAL_USER, img_id, True)
        svc.set_image_locked(LOCAL_USER, img_id, False)
        assert svc.delete_image(LOCAL_USER, img_id) is True
        assert svc.db.session.get(FaceDatasetImage, img_id) is None


def test_batch_delete_skips_locked_rows_and_reports_the_count(app):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Bl', 'bl')
        ids = _seed_images(svc, ds.id, n=3)
        svc.set_image_locked(LOCAL_USER, ids[0], True)
        n, skipped_locked = svc.batch_image_action(LOCAL_USER, ds.id, ids, 'delete')
        assert n == 2
        assert skipped_locked == 1
        # the locked row survives, the other two are gone
        remaining = FaceDatasetImage.query.filter_by(dataset_id=ds.id).all()
        assert [r.id for r in remaining] == [ids[0]]


def test_batch_delete_with_no_locked_rows_reports_zero_skipped(app):
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Bl2', 'bl2')
        ids = _seed_images(svc, ds.id, n=2)
        n, skipped_locked = svc.batch_image_action(LOCAL_USER, ds.id, ids, 'delete')
        assert (n, skipped_locked) == (2, 0)


def test_purge_unused_skips_locked_rows(app):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Pu', 'pu')
        ids = _seed_images(svc, ds.id, n=2, status='reject')
        svc.set_image_locked(LOCAL_USER, ids[0], True)
        purged = svc.purge_unused(LOCAL_USER, ds.id)
        assert purged == 1
        remaining = FaceDatasetImage.query.filter_by(dataset_id=ds.id).all()
        assert [r.id for r in remaining] == [ids[0]]


def test_dataset_delete_refuses_whole_dataset_with_any_locked_image(app):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Dd', 'dd')
        ids = _seed_images(svc, ds.id, n=3)
        svc.set_image_locked(LOCAL_USER, ids[1], True)
        try:
            svc.delete_dataset(LOCAL_USER, ds.id)
            assert False, 'delete_dataset should have refused with a locked image present'
        except RuntimeError as e:
            assert 'locked' in str(e).lower()
        # nothing was touched — all three rows and the dataset itself survive
        assert FaceDatasetImage.query.filter_by(dataset_id=ds.id).count() == 3
        assert svc.get_dataset(LOCAL_USER, ds.id) is not None


def test_dataset_delete_still_works_once_every_image_is_unlocked(app):
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Du', 'du')
        [img_id] = _seed_images(svc, ds.id, n=1)
        svc.set_image_locked(LOCAL_USER, img_id, True)
        svc.set_image_locked(LOCAL_USER, img_id, False)
        assert svc.delete_dataset(LOCAL_USER, ds.id) is True
        assert svc.get_dataset(LOCAL_USER, ds.id) is None


# --- HTTP layer: the lock route + the delete routes' 409 mapping --------------

def test_lock_route_toggles_and_reflects_in_the_payload(client, app):
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Route', 'route')
        ds_id = ds.id
        [img_id] = _seed_images(svc, ds.id, n=1)

    res = client.post(f'/api/dataset/image/{img_id}/lock', json={'locked': True})
    assert res.status_code == 200 and res.get_json()['ok'] is True

    payload = client.get(f'/api/dataset/{ds_id}').get_json()
    assert payload['images'][0]['is_locked'] is True

    res = client.post(f'/api/dataset/image/{img_id}/lock', json={'locked': False})
    assert res.status_code == 200
    payload = client.get(f'/api/dataset/{ds_id}').get_json()
    assert payload['images'][0]['is_locked'] is False


def test_lock_route_404s_on_an_unknown_image(client):
    res = client.post('/api/dataset/image/999999/lock', json={'locked': True})
    assert res.status_code == 404


def test_single_delete_route_returns_409_for_a_locked_image(client, app):
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'RouteD', 'routed')
        [img_id] = _seed_images(svc, ds.id, n=1)
    client.post(f'/api/dataset/image/{img_id}/lock', json={'locked': True})

    res = client.post(f'/api/dataset/image/{img_id}/delete')
    assert res.status_code == 409
    assert 'locked' in res.get_json()['error'].lower()


def test_batch_delete_route_surfaces_skipped_locked_count(client, app):
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'RouteB', 'routeb')
        ds_id = ds.id
        ids = _seed_images(svc, ds.id, n=2)
    client.post(f'/api/dataset/image/{ids[0]}/lock', json={'locked': True})

    res = client.post(f'/api/dataset/{ds_id}/images/batch',
                      json={'ids': ids, 'action': 'delete'})
    body = res.get_json()
    assert res.status_code == 200
    assert body['affected'] == 1
    assert body['skipped_locked'] == 1


def test_dataset_delete_route_returns_409_with_any_locked_image(client, app):
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'RouteDD', 'routedd')
        ds_id = ds.id
        [img_id] = _seed_images(svc, ds.id, n=1)
    client.post(f'/api/dataset/image/{img_id}/lock', json={'locked': True})

    res = client.post(f'/api/dataset/{ds_id}/delete')
    assert res.status_code == 409
    assert 'locked' in res.get_json()['error'].lower()
