"""train_type is chosen at CREATION (drives caption style + menu grouping from the
start), normalized, exposed in the list, and stays settable later so the grouped menu
re-sorts and the SDXL->booru / else->prose caption default follows."""
import pytest

from app.services import face_dataset_service as svc


def test_create_dataset_persists_train_type(app):
    with app.app_context():
        ds = svc.create_dataset('local', 'Emma', 'zchar_emma', train_type='sdxl')
        assert ds.train_type == 'sdxl'


def test_create_dataset_defaults_and_normalizes(app):
    with app.app_context():
        assert svc.create_dataset('local', 'A', 'a').train_type == 'zimage'                    # default
        assert svc.create_dataset('local', 'B', 'b', train_type='SDXL').train_type == 'sdxl'   # case-fold
        assert svc.create_dataset('local', 'C', 'c', train_type='bogus').train_type == 'zimage'  # unknown


def test_set_train_type_updates_and_normalizes(app):
    with app.app_context():
        ds = svc.create_dataset('local', 'Emma', 'zchar_emma')
        assert svc.set_train_type('local', ds.id, 'krea') is True
        assert svc.get_dataset('local', ds.id).train_type == 'krea'
        assert svc.set_train_type('local', ds.id, 'nope') is True
        assert svc.get_dataset('local', ds.id).train_type == 'zimage'          # unknown -> zimage
        assert svc.set_train_type('local', 999999, 'sdxl') is False            # absent dataset


def test_set_train_type_refuses_non_krea_while_dense(app):
    with app.app_context():
        ds = svc.create_dataset('local', 'Dense', 'dense', train_type='krea')
        ds.training_mode = 'full_transformer'
        ds.train_variant = 'base'
        svc.db.session.commit()

        with pytest.raises(ValueError, match='Switch the training mode to LoRA'):
            svc.set_train_type('local', ds.id, 'sdxl')
        svc.db.session.refresh(ds)
        assert ds.training_mode == 'full_transformer'
        assert ds.train_type == 'krea'
        # Re-selecting Krea remains a harmless legacy-tab refresh.
        assert svc.set_train_type('local', ds.id, 'krea') is True


def test_create_route_forwards_train_type_and_list_exposes_it(client):
    did = client.post('/api/dataset/create',
                      json={'name': 'Zoe', 'trigger_word': 'zchar_zoe', 'train_type': 'krea'}).get_json()['id']
    rows = client.get('/api/dataset/list').get_json()['datasets']
    assert next(r for r in rows if r['id'] == did)['train_type'] == 'krea'


def test_train_type_route_updates(client):
    did = client.post('/api/dataset/create',
                      json={'name': 'Ivy', 'trigger_word': 'zchar_ivy'}).get_json()['id']
    assert client.post(f'/api/dataset/{did}/train-type', json={'train_type': 'sdxl'}).status_code == 200
    rows = client.get('/api/dataset/list').get_json()['datasets']
    assert next(r for r in rows if r['id'] == did)['train_type'] == 'sdxl'


def test_train_type_route_refuses_non_krea_while_dense(app, client):
    from app.extensions import db
    from app.models import FaceDataset

    did = client.post(
        '/api/dataset/create',
        json={'name': 'DenseRoute', 'trigger_word': 'dense_route',
              'train_type': 'krea'}).get_json()['id']
    with app.app_context():
        ds = db.session.get(FaceDataset, did)
        ds.training_mode = 'full_transformer'
        ds.train_variant = 'base'
        db.session.commit()

    response = client.post(
        f'/api/dataset/{did}/train-type', json={'train_type': 'sdxl'})
    assert response.status_code == 400
    assert 'Switch the training mode to LoRA' in response.get_json()['error']
    rows = client.get('/api/dataset/list').get_json()['datasets']
    row = next(item for item in rows if item['id'] == did)
    assert row['train_type'] == 'krea'
    with app.app_context():
        ds = db.session.get(FaceDataset, did)
        assert ds.training_mode == 'full_transformer'


def test_modern_settings_can_atomically_leave_dense_and_change_family(
        app, client, monkeypatch):
    from app import capabilities
    from app.extensions import db
    from app.models import FaceDataset

    monkeypatch.setattr(capabilities, 'probe', lambda: {
        'aitoolkit': {'valid': True}, 'cloud_training': False})
    did = client.post(
        '/api/dataset/create',
        json={'name': 'DenseAtomic', 'trigger_word': 'dense_atomic',
              'train_type': 'krea'}).get_json()['id']
    with app.app_context():
        ds = db.session.get(FaceDataset, did)
        ds.training_mode = 'full_transformer'
        ds.train_variant = 'base'
        db.session.commit()

    response = client.post(
        f'/api/dataset/{did}/train/settings',
        json={'training_mode': 'lora', 'train_type': 'sdxl',
              'base_model': '', 'variant': 'base'})
    assert response.status_code == 200
    body = response.get_json()
    assert (body['training_mode'], body['train_type'], body['base_model'],
            body['variant']) == ('lora', 'sdxl', '', 'base')
    with app.app_context():
        ds = db.session.get(FaceDataset, did)
        assert (ds.training_mode, ds.train_type, ds.train_base_model,
                ds.train_variant) == ('lora', 'sdxl', None, 'base')


def test_train_type_route_unknown_dataset_404(client):
    assert client.post('/api/dataset/999999/train-type', json={'train_type': 'sdxl'}).status_code == 404
