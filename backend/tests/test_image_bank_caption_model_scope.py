"""🗃️ Image bank — the caption pass gets its two per-run dials: WHICH engine/model
writes the captions, and WHICH pile gets captioned.

Both follow the contract the vocabulary/length options already set: the choice rides
per call, and a call that omits it is byte-identical to the pass that existed before
the option did. The scope moves on the values stored in the column ('keep'/'pending');
the UI is the only place that says "Kept" and "Undecided".

Every scope test asserts WHICH ROWS came back captioned, never which arguments were
forwarded — a pass that receives statuses=['keep'] and captions the whole bank passes
the second check and fails the users. The fixtures therefore always hold all three
statuses, because a bank where everything is 'pending' cannot tell the scopes apart.
"""
import os

from PIL import Image


# --- factories (mirror test_image_bank_captions) -----------------------------
def _save(path, im):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)


def _flat(size=64, value=128):
    return Image.new('RGB', (size, size), (value, value, value))


def _mkbank(client, tmp_path, files, name='B'):
    src = tmp_path / 'src'
    for rel, im in files.items():
        _save(str(src / rel), im)
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _by_name(client, bank_id):
    # No `status` filter = every pile, rejected included: the scope tests have to
    # be able to SEE that the bin stayed empty of captions.
    url = f'/api/bank/{bank_id}/images?limit=200'
    return {i['name']: i for i in client.get(url).get_json()['images']}


def _use_ollama_backend(app, backend='ollama'):
    with app.app_context():
        import app.config as cfg
        cfg.save_config({'captioning': {'backend': backend}})


def _mock_vision(monkeypatch):
    """Mock the Ollama vision seam and record the MODEL each call ran with, so the
    per-run override can be checked where it actually lands."""
    from app.services import vision_ollama
    seen = []

    def fake_describe(image_bytes, *a, **k):
        seen.append(k.get('model'))
        return 'described'

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    return seen


def _mixed_bank(client, tmp_path, app, monkeypatch):
    """A bank holding one KEPT, one UNDECIDED and one REJECTED image — the only
    shape that can tell three scopes apart."""
    bank_id, _ = _mkbank(client, tmp_path, {
        'kept.png': _flat(value=10), 'undecided.png': _flat(value=20),
        'rejected.png': _flat(value=30)})
    by = _by_name(client, bank_id)
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [by['kept.png']['id']], 'status': 'keep'})
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [by['rejected.png']['id']], 'status': 'reject'})
    return bank_id


def _captioned(client, bank_id):
    """The NAMES that actually carry a caption now — the property under test."""
    return {n for n, i in _by_name(client, bank_id).items() if (i.get('caption') or '')}


def _run(client, bank_id, **body):
    r = client.post(f'/api/bank/{bank_id}/caption', json=body)
    assert r.status_code == 202, r.get_json()
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['activity']['error'] is None, payload['activity']
    return payload


# --- (B) scope: which pile actually gets captioned ---------------------------
def test_no_scope_captions_every_non_rejected_image(client, tmp_path, app, monkeypatch):
    """NON-REGRESSION. A body with none of the new keys captions exactly what the
    pass captioned before they existed: the kept AND the undecided, never the bin."""
    _use_ollama_backend(app)
    bank_id = _mixed_bank(client, tmp_path, app, monkeypatch)
    _mock_vision(monkeypatch)
    _run(client, bank_id)
    assert _captioned(client, bank_id) == {'kept.png', 'undecided.png'}


def test_scope_keep_only_captions_the_kept_pile(client, tmp_path, app, monkeypatch):
    _use_ollama_backend(app)
    bank_id = _mixed_bank(client, tmp_path, app, monkeypatch)
    _mock_vision(monkeypatch)
    _run(client, bank_id, statuses=['keep'])
    assert _captioned(client, bank_id) == {'kept.png'}


def test_scope_pending_only_captions_the_undecided_pile(client, tmp_path, app, monkeypatch):
    _use_ollama_backend(app)
    bank_id = _mixed_bank(client, tmp_path, app, monkeypatch)
    _mock_vision(monkeypatch)
    _run(client, bank_id, statuses=['pending'])
    assert _captioned(client, bank_id) == {'undecided.png'}


def test_scope_both_equals_the_default_set(client, tmp_path, app, monkeypatch):
    """The third option the owner asked for is today's default made EXPLICIT — so
    it must land on the same rows, not merely be accepted."""
    _use_ollama_backend(app)
    bank_id = _mixed_bank(client, tmp_path, app, monkeypatch)
    _mock_vision(monkeypatch)
    _run(client, bank_id, statuses=['keep', 'pending'])
    assert _captioned(client, bank_id) == {'kept.png', 'undecided.png'}


def test_scope_order_does_not_matter(client, tmp_path, app, monkeypatch):
    _use_ollama_backend(app)
    bank_id = _mixed_bank(client, tmp_path, app, monkeypatch)
    _mock_vision(monkeypatch)
    _run(client, bank_id, statuses=['pending', 'keep', 'keep'])
    assert _captioned(client, bank_id) == {'kept.png', 'undecided.png'}


def test_rejected_is_never_captioned_by_any_scope(client, tmp_path, app, monkeypatch):
    """The bin is out of reach whatever is asked — including by an explicit id."""
    _use_ollama_backend(app)
    bank_id = _mixed_bank(client, tmp_path, app, monkeypatch)
    _mock_vision(monkeypatch)
    rejected_id = _by_name(client, bank_id)['rejected.png']['id']
    for body in ({'image_ids': [rejected_id]},
                 {'image_ids': [rejected_id], 'statuses': ['keep', 'pending']},
                 {'statuses': ['keep']}, {'statuses': ['pending']},
                 {'statuses': ['keep', 'pending']}, {}):
        _run(client, bank_id, **body)
        assert 'rejected.png' not in _captioned(client, bank_id), body


def test_scope_intersects_a_selection_it_never_widens_it(client, tmp_path, app, monkeypatch):
    """Selection AND scope both bind. Selecting the undecided image while asking for
    the kept pile captions NOTHING — the narrower of the two wins."""
    _use_ollama_backend(app)
    bank_id = _mixed_bank(client, tmp_path, app, monkeypatch)
    _mock_vision(monkeypatch)
    by = _by_name(client, bank_id)
    _run(client, bank_id, image_ids=[by['undecided.png']['id']], statuses=['keep'])
    assert _captioned(client, bank_id) == set()
    _run(client, bank_id, image_ids=[by['undecided.png']['id']], statuses=['pending'])
    assert _captioned(client, bank_id) == {'undecided.png'}


def test_scope_still_skips_already_captioned_rows_unless_force(client, tmp_path, app,
                                                               monkeypatch):
    _use_ollama_backend(app)
    bank_id = _mixed_bank(client, tmp_path, app, monkeypatch)
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        row = BankImage.query.filter(BankImage.relpath.like('%kept.png')).first()
        row.caption = 'written by hand'
        db.session.commit()
    _mock_vision(monkeypatch)
    _run(client, bank_id, statuses=['keep'])
    assert _by_name(client, bank_id)['kept.png']['caption'] == 'written by hand'
    _run(client, bank_id, statuses=['keep'], force=True)
    assert _by_name(client, bank_id)['kept.png']['caption'] == 'described'


def test_invalid_scope_is_400(client, tmp_path, app):
    """A status that is not a status is still a 400.

    ⚠️ 'reject' LEFT THIS LIST on purpose. The bin used to be refused outright;
    the launch dialogs now offer it as an explicit choice, so it is a VALID scope
    (see test_the_bin_is_reachable_only_by_asking_for_it). Everything that is not
    one of the three column values is still rejected the same way it always was."""
    _use_ollama_backend(app)
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    for bad in (['deleted'], 'garbage', [None], [3], 5, {'keep': True}):
        r = client.post(f'/api/bank/{bank_id}/caption', json={'statuses': bad})
        assert r.status_code == 400, (bad, r.get_json())
        assert 'status' in r.get_json()['error']


# --- the count the UI quotes before the click --------------------------------
def test_payload_prices_each_scope_with_what_it_would_really_caption(
        client, tmp_path, app, monkeypatch):
    """The number the button shows must be the number the run moves. It counts the
    UNCAPTIONED rows of each status — an already-captioned image is in the pile but
    not in the run, and quoting the pile would advertise work that never happens."""
    _use_ollama_backend(app)
    bank_id = _mixed_bank(client, tmp_path, app, monkeypatch)
    counts = client.get(f'/api/bank/{bank_id}').get_json()['counts']
    assert (counts['caption_todo_keep'], counts['caption_todo_pending']) == (1, 1)

    _mock_vision(monkeypatch)
    _run(client, bank_id, statuses=['keep'])
    counts = client.get(f'/api/bank/{bank_id}').get_json()['counts']
    # The kept pile is unchanged in size; what shrank is the work left in it.
    assert counts['keep'] == 1 and counts['caption_todo_keep'] == 0
    assert counts['caption_todo_pending'] == 1


def test_scope_count_never_includes_the_rejected(client, tmp_path, app, monkeypatch):
    _use_ollama_backend(app)
    bank_id = _mixed_bank(client, tmp_path, app, monkeypatch)
    counts = client.get(f'/api/bank/{bank_id}').get_json()['counts']
    assert counts['reject'] == 1
    assert counts['caption_todo_keep'] + counts['caption_todo_pending'] == 2


# --- (A) engine + model, per run ---------------------------------------------
def _capture_caption_paths(monkeypatch):
    """Record the engine/model caption_paths is handed, and still run it."""
    from app.services import face_dataset_service as fds
    seen = {}
    real = fds.caption_paths

    def spy(paths, **kw):
        seen.update(kw)
        seen['paths'] = list(paths)
        return real(paths, **kw)

    monkeypatch.setattr(fds, 'caption_paths', spy)
    return seen


def test_no_model_option_forwards_nothing_and_uses_the_global_setting(
        client, tmp_path, app, monkeypatch):
    """NON-REGRESSION on the model half: with neither key, both overrides are None,
    i.e. caption_paths resolves the global settings exactly as it always did."""
    _use_ollama_backend(app)
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    seen = _capture_caption_paths(monkeypatch)
    _mock_vision(monkeypatch)
    _run(client, bank_id)
    assert seen['backend'] is None
    assert seen['ollama_model'] is None


def test_per_run_engine_and_model_reach_the_engine(client, tmp_path, app, monkeypatch):
    _use_ollama_backend(app, backend='auto')
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    seen = _capture_caption_paths(monkeypatch)
    models = _mock_vision(monkeypatch)
    _run(client, bank_id, backend='ollama', ollama_model='someorg/some-vl:8b')
    assert seen['backend'] == 'ollama'
    assert seen['ollama_model'] == 'someorg/some-vl:8b'
    # …and it is the model the vision call actually ran with, not just a forwarded
    # argument.
    assert models == ['someorg/some-vl:8b']


def test_per_run_choice_never_writes_the_global_settings(client, tmp_path, app,
                                                         monkeypatch):
    """The settings stay the default. A run is a run, not a preference change."""
    _use_ollama_backend(app, backend='auto')
    with app.app_context():
        import app.config as cfg
        cfg.save_config({'ollama': {'vision_model': 'global/model:1b'}})
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    _mock_vision(monkeypatch)
    _run(client, bank_id, backend='ollama', ollama_model='someorg/some-vl:8b')
    with app.app_context():
        import app.config as cfg
        assert cfg.get('captioning.backend') == 'auto'
        assert cfg.get('ollama.vision_model') == 'global/model:1b'


def test_per_run_engine_rescues_an_install_whose_global_backend_is_none(
        client, tmp_path, app, monkeypatch):
    """The 400 gate reads the RESOLVED engine, so choosing one for the run is
    enough — otherwise the picker would be unreachable on the very install that
    most needs it."""
    _use_ollama_backend(app, backend='none')
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    assert client.post(f'/api/bank/{bank_id}/caption', json={}).status_code == 400
    _mock_vision(monkeypatch)
    _run(client, bank_id, backend='ollama')
    assert _captioned(client, bank_id) == {'a.png'}


def test_per_run_engine_none_is_refused(client, tmp_path, app):
    _use_ollama_backend(app)
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    r = client.post(f'/api/bank/{bank_id}/caption', json={'backend': 'none'})
    assert r.status_code == 400
    assert 'backend' in r.get_json()['error']


def test_invalid_engine_is_400(client, tmp_path, app):
    _use_ollama_backend(app)
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    r = client.post(f'/api/bank/{bank_id}/caption', json={'backend': 'nonsense'})
    assert r.status_code == 400
    assert 'backend' in r.get_json()['error']


def test_invalid_ollama_model_is_400(client, tmp_path, app):
    """Same charset contract as the Caption Lab — a name is a JSON field to the
    local server, never a shell word, and a line break is refused before stripping."""
    _use_ollama_backend(app)
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    for bad in (123, True, ['a'], {'a': 1}, 'bad model!', '/leading',
                'valid:tag\n', 'valid:tag\r\nsecond:tag', 'a' * 201):
        r = client.post(f'/api/bank/{bank_id}/caption', json={'ollama_model': bad})
        assert r.status_code == 400, (bad, r.get_json())
        assert 'ollama_model' in r.get_json()['error']


def test_auto_still_chains_joycaption_then_ollama(client, tmp_path, app, monkeypatch):
    """The 'auto' engine is not "one of the two": JoyCaption drafts and Ollama picks
    up whatever it missed. Choosing it per run must keep BOTH halves — a picker that
    quietly turned auto into a single engine would change every caption it wrote."""
    _use_ollama_backend(app, backend='ollama')       # global says otherwise
    bank_id, _ = _mkbank(client, tmp_path, {
        'jc.png': _flat(value=10), 'oll.png': _flat(value=20)})
    from app.services import joycaption
    monkeypatch.setattr(joycaption, 'is_available', lambda: True)

    def fake_jc(paths, **kw):
        # JoyCaption covers the first file only; the rest must fall through.
        return {p: ('from joycaption' if p.endswith('jc.png') else '') for p in paths}

    monkeypatch.setattr(joycaption, 'caption_images_joycaption', fake_jc)
    _mock_vision(monkeypatch)
    _run(client, bank_id, backend='auto')
    by = _by_name(client, bank_id)
    assert by['jc.png']['caption'] == 'from joycaption'
    assert by['oll.png']['caption'] == 'described'    # the Ollama half still ran


def test_model_and_scope_compose(client, tmp_path, app, monkeypatch):
    """The two dials are independent: a per-run model on a narrowed scope writes the
    narrowed rows, with that model."""
    _use_ollama_backend(app, backend='auto')
    bank_id = _mixed_bank(client, tmp_path, app, monkeypatch)
    models = _mock_vision(monkeypatch)
    _run(client, bank_id, backend='ollama', ollama_model='someorg/some-vl:8b',
         statuses=['pending'])
    assert _captioned(client, bank_id) == {'undecided.png'}
    assert models == ['someorg/some-vl:8b']
