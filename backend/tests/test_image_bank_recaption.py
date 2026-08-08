"""🔄 Image bank — RE-CAPTION: the forced pass, and the number the button quotes for it.

The pass normally skips rows that already carry a caption, so on a finished bank the
🏷️ Caption button reaches zero and goes inert — taking the engine/model/pile selects
beside it down with it. ``force`` drops that filter and rewrites the pile. It is the one
DESTRUCTIVE thing this row can do, and the whole point of these tests is that the figure
the UI shows before the click is the figure the run acts on.

Every test here asserts WHICH ROWS carry which caption afterwards, and how many vision
calls really happened. A test that checked "force=True was forwarded" would pass on a
pass that forwarded it and then captioned nothing — which is the failure being guarded
against. The fixtures therefore always hold all three statuses, half of them
pre-captioned, because no smaller bank can tell the scopes apart.
"""
import os

from PIL import Image


# --- factories (mirror test_image_bank_caption_model_scope) ------------------
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
    # No `status` filter = every pile, rejected included: a re-caption test has to be
    # able to SEE that the bin was left alone.
    url = f'/api/bank/{bank_id}/images?limit=200'
    return {i['name']: i for i in client.get(url).get_json()['images']}


def _captions(client, bank_id):
    """name -> caption, for every image in the bank. The property under test."""
    return {n: (i.get('caption') or '') for n, i in _by_name(client, bank_id).items()}


def _use_ollama_backend(app, backend='ollama'):
    with app.app_context():
        import app.config as cfg
        cfg.save_config({'captioning': {'backend': backend}})


def _mock_vision(monkeypatch, text='fresh'):
    """Mock the Ollama vision seam and RECORD ONE ENTRY PER CALL, so "how many images
    did this run really process" is measured rather than inferred from the diff (a row
    re-captioned with the same text it already had would be invisible otherwise)."""
    from app.services import vision_ollama
    calls = []

    def fake_describe(image_bytes, *a, **k):
        calls.append(k.get('model'))
        return text

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    return calls


def _run(client, bank_id, **body):
    r = client.post(f'/api/bank/{bank_id}/caption', json=body)
    assert r.status_code == 202, r.get_json()
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['activity']['error'] is None, payload['activity']
    return payload


def _six_image_bank(client, tmp_path, app):
    """Two images per status, one of each pair already captioned by hand.

    kept_old / pending_old / reject_old carry 'by hand'; the three *_new carry nothing.
    That shape answers all three questions at once: does force rewrite the captioned
    rows, does it also fill the empty ones, and does it stay out of the bin."""
    bank_id, _ = _mkbank(client, tmp_path, {
        'kept_old.png': _flat(value=10), 'kept_new.png': _flat(value=20),
        'pending_old.png': _flat(value=30), 'pending_new.png': _flat(value=40),
        'reject_old.png': _flat(value=50), 'reject_new.png': _flat(value=60)})
    by = _by_name(client, bank_id)
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [by['kept_old.png']['id'], by['kept_new.png']['id']],
                      'status': 'keep'})
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [by['reject_old.png']['id'], by['reject_new.png']['id']],
                      'status': 'reject'})
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        for name in ('kept_old.png', 'pending_old.png', 'reject_old.png'):
            row = db.session.get(BankImage, by[name]['id'])
            row.caption = 'by hand'
        db.session.commit()
    return bank_id


# --- what a forced run really rewrites ---------------------------------------
def test_force_rewrites_the_captioned_rows_of_the_kept_pile_only(
        client, tmp_path, app, monkeypatch):
    """The demand, stated as rows: the kept pile is redone WHOLE — the row that already
    had a caption included — and nothing outside it moves."""
    _use_ollama_backend(app)
    bank_id = _six_image_bank(client, tmp_path, app)
    calls = _mock_vision(monkeypatch)
    _run(client, bank_id, statuses=['keep'], force=True)
    assert _captions(client, bank_id) == {
        'kept_old.png': 'fresh',        # OVERWRITTEN — the destructive half
        'kept_new.png': 'fresh',        # …and the empty one filled, same run
        'pending_old.png': 'by hand',   # untouched: another pile
        'pending_new.png': '',
        'reject_old.png': 'by hand',    # the bin is never captioned, force or not
        'reject_new.png': '',
    }
    assert len(calls) == 2              # two images processed, no more, no fewer


def test_force_rewrites_the_captioned_rows_of_the_undecided_pile_only(
        client, tmp_path, app, monkeypatch):
    _use_ollama_backend(app)
    bank_id = _six_image_bank(client, tmp_path, app)
    calls = _mock_vision(monkeypatch)
    _run(client, bank_id, statuses=['pending'], force=True)
    assert _captions(client, bank_id) == {
        'kept_old.png': 'by hand', 'kept_new.png': '',
        'pending_old.png': 'fresh', 'pending_new.png': 'fresh',
        'reject_old.png': 'by hand', 'reject_new.png': '',
    }
    assert len(calls) == 2


def test_force_without_a_scope_rewrites_both_piles_and_spares_the_bin(
        client, tmp_path, app, monkeypatch):
    """The default scope is the server's ``status != 'reject'`` set. Forced, it is the
    widest thing this button can do — and it still stops at the bin."""
    _use_ollama_backend(app)
    bank_id = _six_image_bank(client, tmp_path, app)
    calls = _mock_vision(monkeypatch)
    _run(client, bank_id, force=True)
    assert _captions(client, bank_id) == {
        'kept_old.png': 'fresh', 'kept_new.png': 'fresh',
        'pending_old.png': 'fresh', 'pending_new.png': 'fresh',
        'reject_old.png': 'by hand', 'reject_new.png': '',
    }
    assert len(calls) == 4


def test_force_with_both_statuses_equals_the_default_scope(
        client, tmp_path, app, monkeypatch):
    """The explicit "Kept + undecided" choice must land on the same rows as the default,
    not merely be accepted — the scope select offers it as the same thing."""
    _use_ollama_backend(app)
    bank_id = _six_image_bank(client, tmp_path, app)
    calls = _mock_vision(monkeypatch)
    _run(client, bank_id, statuses=['keep', 'pending'], force=True)
    assert _captions(client, bank_id) == {
        'kept_old.png': 'fresh', 'kept_new.png': 'fresh',
        'pending_old.png': 'fresh', 'pending_new.png': 'fresh',
        'reject_old.png': 'by hand', 'reject_new.png': '',
    }
    assert len(calls) == 4


def test_force_never_reaches_the_bin_even_by_explicit_id(client, tmp_path, app,
                                                         monkeypatch):
    """Naming a rejected image outright does not get it captioned either — the scope
    filter runs before the selection, and force does not lift it."""
    _use_ollama_backend(app)
    bank_id = _six_image_bank(client, tmp_path, app)
    by = _by_name(client, bank_id)
    calls = _mock_vision(monkeypatch)
    for body in ({'image_ids': [by['reject_old.png']['id']]},
                 {'image_ids': [by['reject_old.png']['id']],
                  'statuses': ['keep', 'pending']},
                 {'statuses': ['keep']}, {'statuses': ['pending']}, {}):
        _run(client, bank_id, force=True, **body)
        caps = _captions(client, bank_id)
        assert caps['reject_old.png'] == 'by hand', body
        assert caps['reject_new.png'] == '', body
    assert 'fresh' not in (
        _captions(client, bank_id)['reject_old.png'],)
    assert calls  # the runs did happen; they just never touched the bin


def test_force_carries_the_per_run_model_onto_the_rows_it_overwrites(
        client, tmp_path, app, monkeypatch):
    """The reason the button exists: redoing a finished bank with a BETTER model. The
    override must reach the vision call for every row it rewrites, not just the run."""
    _use_ollama_backend(app, backend='auto')
    bank_id = _six_image_bank(client, tmp_path, app)
    calls = _mock_vision(monkeypatch)
    _run(client, bank_id, statuses=['keep'], force=True,
         backend='ollama', ollama_model='someorg/some-vl:8b')
    assert calls == ['someorg/some-vl:8b', 'someorg/some-vl:8b']
    assert _captions(client, bank_id)['kept_old.png'] == 'fresh'


# --- the number the button quotes == the number the run moves ----------------
def _announced(counts, scope):
    """The two figures the UI derives from the payload, mirroring bankCaptionScope.js.

    pile  = what a FORCED run walks (captionForcePileSize)
    over  = how many EXISTING captions it destroys (captionOverwriteCount)"""
    if scope == ['keep']:
        pile, todo = counts['keep'], counts['caption_todo_keep']
    elif scope == ['pending']:
        pile, todo = counts['pending'], counts['caption_todo_pending']
    else:
        pile = counts['keep'] + counts['pending']
        todo = counts['caption_todo_keep'] + counts['caption_todo_pending']
    return pile, pile - todo


def test_the_announced_pile_is_exactly_what_a_forced_run_processes(
        client, tmp_path, app, monkeypatch):
    """THE CONTRACT. For each of the three scopes: the number the button shows equals
    the number of images the forced pass really hands to the captioner. Measured on the
    vision calls, so a run that walked more rows and skipped them would still fail."""
    _use_ollama_backend(app)
    for scope in (['keep'], ['pending'], None):
        bank_id = _six_image_bank(client, tmp_path / str(scope), app)
        counts = client.get(f'/api/bank/{bank_id}').get_json()['counts']
        pile, over = _announced(counts, scope)
        calls = _mock_vision(monkeypatch)
        body = {'statuses': scope} if scope else {}
        _run(client, bank_id, force=True, **body)
        assert len(calls) == pile, (scope, len(calls), pile)
        # …and the destructive half of the promise: exactly `over` rows held a caption
        # before the run, so exactly `over` captions were destroyed.
        assert over == (1 if scope else 2), (scope, over)


def test_the_announced_overwrite_count_is_the_rows_that_lose_a_caption(
        client, tmp_path, app, monkeypatch):
    """The amber line's number, checked against the rows themselves: `over` counts the
    images whose caption text is REPLACED, never the ones merely filled in."""
    _use_ollama_backend(app)
    bank_id = _six_image_bank(client, tmp_path, app)
    counts = client.get(f'/api/bank/{bank_id}').get_json()['counts']
    before = _captions(client, bank_id)
    pile, over = _announced(counts, None)
    _mock_vision(monkeypatch)
    _run(client, bank_id, force=True)
    after = _captions(client, bank_id)
    lost = {n for n, c in before.items() if c and after[n] != c}
    filled = {n for n, c in before.items() if not c and after[n]}
    assert len(lost) == over == 2
    assert lost == {'kept_old.png', 'pending_old.png'}
    assert len(lost) + len(filled) == pile == 4


def test_the_three_piles_partition_the_bank(client, tmp_path, app):
    """The default scope's announced size is keep + pending, while the server filters on
    ``status != 'reject'``. The two agree only because the three statuses are the whole
    bank — pinned here so a fourth status value could never make the button understate a
    destructive run in silence."""
    bank_id = _six_image_bank(client, tmp_path, app)
    counts = client.get(f'/api/bank/{bank_id}').get_json()['counts']
    assert counts['keep'] + counts['pending'] + counts['reject'] == counts['total']


def test_the_pile_and_the_todo_count_are_different_numbers(client, tmp_path, app):
    """Guards the reason there are two figures at all: on this bank the forced run walks
    4 images while 🏷️ Caption would walk 2. A UI quoting one for the other would either
    understate the destruction or advertise work that never happens."""
    bank_id = _six_image_bank(client, tmp_path, app)
    counts = client.get(f'/api/bank/{bank_id}').get_json()['counts']
    assert (counts['keep'], counts['caption_todo_keep']) == (2, 1)
    assert (counts['pending'], counts['caption_todo_pending']) == (2, 1)


# --- and the pass WITHOUT force is untouched ---------------------------------
def test_without_force_the_pass_is_exactly_what_it_was(client, tmp_path, app,
                                                       monkeypatch):
    """NON-REGRESSION. The same bank, the same scopes, no force: the pre-captioned rows
    keep their text and only the empty ones are written — one vision call per pile."""
    _use_ollama_backend(app)
    for scope, expected in ((['keep'], {'kept_new.png'}),
                            (['pending'], {'pending_new.png'}),
                            (None, {'kept_new.png', 'pending_new.png'})):
        bank_id = _six_image_bank(client, tmp_path / f'nf{scope}', app)
        calls = _mock_vision(monkeypatch)
        body = {'statuses': scope} if scope else {}
        _run(client, bank_id, **body)
        caps = _captions(client, bank_id)
        assert {n for n, c in caps.items() if c == 'fresh'} == expected, scope
        assert caps['kept_old.png'] == 'by hand' and caps['pending_old.png'] == 'by hand'
        assert len(calls) == len(expected), scope


def test_force_false_is_the_same_body_as_no_force_at_all(client, tmp_path, app,
                                                         monkeypatch):
    """The UI sends `force` only on the re-caption path; an explicit false must behave
    like the key being absent, not like a third mode."""
    _use_ollama_backend(app)
    bank_id = _six_image_bank(client, tmp_path, app)
    _mock_vision(monkeypatch)
    _run(client, bank_id, force=False)
    caps = _captions(client, bank_id)
    assert caps['kept_old.png'] == 'by hand' and caps['pending_old.png'] == 'by hand'
    assert caps['kept_new.png'] == 'fresh' and caps['pending_new.png'] == 'fresh'


# --- (B2) a non-string option is a bad REQUEST, not a broken server ----------
def test_non_string_caption_options_are_400_not_500(client, tmp_path, app):
    """`{'backend': 5}` used to reach ``.strip()`` and answer 500 — a rejected request
    rendered as a crash, while `statuses` next door already answered 400 for the same
    mistake. Every free-text option of this endpoint now agrees."""
    _use_ollama_backend(app)
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    for key in ('backend', 'vocabulary', 'length'):
        for bad in (5, 3.5, True, ['ollama'], {'x': 1}):
            r = client.post(f'/api/bank/{bank_id}/caption', json={key: bad})
            assert r.status_code == 400, (key, bad, r.status_code, r.get_json())
            assert key in r.get_json()['error'], (key, bad, r.get_json())


def test_valid_options_still_pass_and_empty_ones_still_mean_default(
        client, tmp_path, app, monkeypatch):
    """The guard rejects types, never values: the real names and the empty string (which
    the route turns into None) go through exactly as before."""
    _use_ollama_backend(app)
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    _mock_vision(monkeypatch)
    _run(client, bank_id, backend='ollama', vocabulary='explicit', length='concise')
    _run(client, bank_id, force=True, backend='', vocabulary='', length='')
