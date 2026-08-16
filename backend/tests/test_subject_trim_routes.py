"""Subject trim — the routes.

The service has its own tests; these pin the HTTP contract only: which status
code each refusal maps to, and that the apply route passes IDS and never
coordinates.
"""


def test_preview_route_answers_409_when_masking_is_not_set_up(client, monkeypatch):
    from app.services import subject_trim_batch as batch

    def _refuse(*a, **kw):
        raise batch.auto_mask.AutoMaskUnavailable('SAM3 environment missing')

    monkeypatch.setattr(batch, 'start_preview', _refuse)
    r = client.post('/api/dataset/1/trim-preview', json={'image_ids': [1]})
    assert r.status_code == 409
    body = r.get_json()
    assert body['ok'] is False
    assert 'Setup' in body['error']


def test_apply_route_passes_the_ids_through(client, monkeypatch):
    from app.services import subject_trim_batch as batch
    seen = {}

    def _apply(user_id, dataset_id, image_ids):
        seen['args'] = (dataset_id, list(image_ids))
        return {'trimmed': 2, 'refused': 0, 'failed': 0}

    monkeypatch.setattr(batch, 'apply_preview', _apply)
    r = client.post('/api/dataset/4/trim-apply', json={'image_ids': [7, 8]})
    assert r.status_code == 200
    assert r.get_json()['trimmed'] == 2
    assert seen['args'] == (4, [7, 8])


def test_preview_route_returns_null_when_no_preview_is_pending(client, monkeypatch):
    from app.services import subject_trim_batch as batch
    monkeypatch.setattr(batch, 'preview_report', lambda _id: None)
    monkeypatch.setattr(batch, 'trim_report', lambda _id: None)
    r = client.get('/api/dataset/4/trim-preview')
    assert r.status_code == 200
    assert r.get_json() == {'preview': None, 'undo': None}


def test_apply_route_ignores_coordinates_in_the_body(client, monkeypatch):
    """The apply route must forward only ids to the service — coordinates in
    the body (however they got there) are never allowed to reach it."""
    from app.services import subject_trim_batch as batch
    seen = {}

    def _apply(user_id, dataset_id, image_ids):
        seen['args'] = (dataset_id, list(image_ids))
        return {'trimmed': 1, 'refused': 0, 'failed': 0}

    monkeypatch.setattr(batch, 'apply_preview', _apply)
    r = client.post('/api/dataset/4/trim-apply',
                    json={'image_ids': [9],
                          'frame': [10, 10, 200, 200],
                          'coordinates': {'x': 1, 'y': 2}})
    assert r.status_code == 200
    # Only ids reached the service — the extra keys were never read.
    assert seen['args'] == (4, [9])


def test_undo_route_returns_the_restore_counts(client, monkeypatch):
    from app.services import subject_trim_batch as batch

    def _restore(user_id, dataset_id):
        return {'restored': 3, 'gone': 1, 'failed': 0}

    monkeypatch.setattr(batch, 'restore_trim_batch', _restore)
    r = client.post('/api/dataset/4/trim-undo')
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True
    assert body['restored'] == 3
    assert body['gone'] == 1
    assert body['failed'] == 0
