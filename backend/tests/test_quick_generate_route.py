"""POST /dataset/<id>/quick-generate/compose (pure composition, no queuing)
and GET/PUT /dataset/quick-generate/components (view the effective pool,
save custom additions) — same validation/response conventions as the
existing shot-catalog and generate routes."""
import pytest


def _make_ds(client):
    return client.post('/api/dataset/create',
                       json={'name': 'QG', 'trigger_word': 'qg'}).get_json()['id']


def test_compose_returns_exactly_total_variations_with_no_debug_fields(client):
    ds_id = _make_ds(client)
    resp = client.post(f'/api/dataset/{ds_id}/quick-generate/compose', json={
        'total': 6, 'framing_ratios': {'face': 100, 'bust': 0, 'body': 0},
        'angle_ratios': {'face': {'front': 100}},
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True
    assert len(body['variations']) == 6
    for v in body['variations']:
        assert set(v) == {'id', 'framing', 'label', 'prompt'}
        assert v['framing'] == 'face'


def test_compose_404s_for_an_unknown_dataset(client):
    resp = client.post('/api/dataset/999999/quick-generate/compose', json={
        'total': 1, 'framing_ratios': {'face': 100}, 'angle_ratios': {'face': {'front': 100}}})
    assert resp.status_code == 404


def test_compose_400s_on_a_bad_ratio_shape(client):
    ds_id = _make_ds(client)
    resp = client.post(f'/api/dataset/{ds_id}/quick-generate/compose', json={
        'total': 5, 'framing_ratios': {'face': 60, 'bust': 30}, 'angle_ratios': {}})
    assert resp.status_code == 400
    assert 'sum to 100' in resp.get_json()['error']


def test_compose_400s_over_the_total_cap(client):
    ds_id = _make_ds(client)
    resp = client.post(f'/api/dataset/{ds_id}/quick-generate/compose', json={
        'total': 500, 'framing_ratios': {'face': 100}, 'angle_ratios': {'face': {'front': 100}}})
    assert resp.status_code == 400


def test_components_get_returns_shipped_and_custom(client, app):
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'quick_generate': {'custom_components': {
            'human': {'face': {'expression': [{'id': 'wink', 'phrase': 'a playful wink'}]}}}}})
    resp = client.get('/api/dataset/quick-generate/components?subject_type=human')
    assert resp.status_code == 200
    body = resp.get_json()
    face_expr_ids = {e['id'] for e in body['pools']['face']['expression']}
    assert 'neutral' in face_expr_ids and 'wink' in face_expr_ids


def test_components_put_saves_and_rejects_a_shadowing_id(client):
    resp = client.put('/api/dataset/quick-generate/components', json={
        'subject_type': 'human',
        'custom_components': {'face': {'expression': [
            {'id': 'my_new_expr', 'phrase': 'a new mild expression'},
            {'id': 'neutral', 'phrase': 'trying to shadow a built-in'},
        ]}}})
    assert resp.status_code == 200
    body = resp.get_json()
    saved_ids = {e['id'] for e in body['saved']['face']['expression']}
    assert saved_ids == {'my_new_expr'}
    assert body['dropped'] == 1
