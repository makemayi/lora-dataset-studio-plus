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


def test_compose_400s_instead_of_500_on_non_dict_ratio_shapes(client):
    ds_id = _make_ds(client)
    resp = client.post(f'/api/dataset/{ds_id}/quick-generate/compose', json={
        'total': 5, 'framing_ratios': [1, 2, 3], 'angle_ratios': {'face': {'front': 100}}})
    assert resp.status_code == 400
    body = resp.get_json()
    assert body is not None
    assert body.get('ok') is False
    assert 'error' in body


def test_components_put_400s_instead_of_500_on_a_non_string_subject_type(client):
    resp = client.put('/api/dataset/quick-generate/components', json={
        'subject_type': 5, 'custom_components': {}})
    assert resp.status_code == 400
    body = resp.get_json()
    assert body is not None
    assert 'error' in body


def test_components_get_includes_the_users_own_custom_entries_for_round_tripping(client):
    # PUT a custom entry for 'human', then GET the same subject: the response
    # must carry a 'custom' key with exactly what was saved, so an editor can
    # reopen and show the user their prior entries instead of starting blank
    # (a blind '{}' start would silently wipe them on the very next save,
    # since PUT replaces a subject's whole custom set each call).
    put_resp = client.put('/api/dataset/quick-generate/components', json={
        'subject_type': 'human',
        'custom_components': {'face': {'expression': [
            {'id': 'my_custom_expr', 'phrase': 'a custom expression'},
        ]}}})
    assert put_resp.status_code == 200

    resp = client.get('/api/dataset/quick-generate/components?subject_type=human')
    assert resp.status_code == 200
    body = resp.get_json()
    assert 'custom' in body
    custom_ids = {e['id'] for e in body['custom']['face']['expression']}
    assert custom_ids == {'my_custom_expr'}
    # The shipped built-in must appear only in 'pools', never leak into 'custom'.
    assert 'neutral' not in custom_ids


def test_components_put_reports_a_drop_for_a_structurally_malformed_framing(client):
    # 'face' should map to a dict of axes (e.g. {'expression': [...]}), not a
    # bare list. Everything under it is discarded by sanitization, but the
    # user still typed something — the response must say so via 'dropped',
    # not silently report a clean save.
    resp = client.put('/api/dataset/quick-generate/components', json={
        'subject_type': 'human',
        'custom_components': {'face': ['not', 'a', 'dict']}})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['dropped'] >= 1


def test_compose_accepts_nsfw_ratio_and_tags_nsfw_slots(client):
    ds_id = _make_ds(client)
    resp = client.post(f'/api/dataset/{ds_id}/quick-generate/compose', json={
        'total': 10, 'framing_ratios': {'face': 100, 'bust': 0, 'body': 0},
        'angle_ratios': {'face': {'front': 100}}, 'nsfw_ratio': 100,
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True
    assert len(body['variations']) == 10
    for v in body['variations']:
        assert v.get('nsfw') is True


def test_compose_defaults_nsfw_ratio_to_zero_when_omitted(client):
    ds_id = _make_ds(client)
    resp = client.post(f'/api/dataset/{ds_id}/quick-generate/compose', json={
        'total': 10, 'framing_ratios': {'face': 100, 'bust': 0, 'body': 0},
        'angle_ratios': {'face': {'front': 100}},
    })
    assert resp.status_code == 200
    body = resp.get_json()
    for v in body['variations']:
        assert not v.get('nsfw')


def test_compose_400s_on_an_out_of_range_nsfw_ratio(client):
    ds_id = _make_ds(client)
    resp = client.post(f'/api/dataset/{ds_id}/quick-generate/compose', json={
        'total': 5, 'framing_ratios': {'face': 100, 'bust': 0, 'body': 0},
        'angle_ratios': {'face': {'front': 100}}, 'nsfw_ratio': 150,
    })
    assert resp.status_code == 400


def test_nsfw_components_get_returns_shipped_and_custom(client, app):
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'quick_generate': {'custom_nsfw': {
            'human': {'bust': [{'id': 'my_bust_x', 'label': 'My Bust X', 'prompt': 'a prompt'}]}}}})
    resp = client.get('/api/dataset/quick-generate/nsfw-components?subject_type=human')
    assert resp.status_code == 200
    body = resp.get_json()
    ids = {e['id'] for e in body['pool']['bust']}
    assert 'my_bust_x' in ids
    assert 'nsfw_bust_lingerie' in ids  # shipped default still present


def test_nsfw_components_put_saves_and_rejects_a_shadowing_id(client):
    resp = client.put('/api/dataset/quick-generate/nsfw-components', json={
        'subject_type': 'human',
        'custom_nsfw': {'bust': [
            {'id': 'my_new_nsfw', 'label': 'My New', 'prompt': 'a prompt'},
            {'id': 'nsfw_bust_lingerie', 'label': 'trying to shadow', 'prompt': 'x'},
        ]}})
    assert resp.status_code == 200
    body = resp.get_json()
    saved_ids = {e['id'] for e in body['saved']['bust']}
    assert saved_ids == {'my_new_nsfw'}
    assert body['dropped'] == 1
