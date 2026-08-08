"""The `else: klein` fallthroughs — one test per place a third local engine fell
through a two-engine assumption.

WHY THIS FILE EXISTS
--------------------
Klein was the only local engine once, so several dispatches were written as
"API engines, else Klein". Krea got explicit branches. MiniMax H3 shipped
without them, and every one of the resulting bugs was SILENT — the app answered
confidently with the wrong engine rather than failing:

  * a tile made by H3 wore a "Klein" badge;
  * pressing 🔄 on that tile re-rendered it on KLEIN, so the retry answered with
    a different engine than the one being retried.

Both were found by looking, not by a test, which is why each fallthrough now has
one. The rule these encode: a local engine's identity must be DERIVED from
LOCAL_ENGINES, never from a chain of `==` against the two engines that happened
to exist when the code was written.
"""
import os

import pytest
from PIL import Image


def _dataset(app, svc, engine_tag):
    from app.models import FaceDataset, FaceDatasetImage, db
    ds = FaceDataset(user_id='local', name='ds', trigger_word='zzt',
                     ref_filename='ref.png', train_type='krea')
    db.session.add(ds)
    db.session.commit()
    path = svc._dataset_path(ds.id)
    os.makedirs(path, exist_ok=True)
    Image.new('RGB', (512, 512), (10, 20, 30)).save(os.path.join(path, 'ref.png'))
    img = FaceDatasetImage(dataset_id=ds.id, source='generated', status='kept',
                           variation_label='a', framing='face',
                           variation_prompt='p', klein_model=engine_tag,
                           filename='shot.png')
    db.session.add(img)
    db.session.commit()
    return ds, img


def test_a_row_made_by_h3_is_not_badged_as_klein(app):
    """`_image_engine` used to end in `return 'klein'` for anything that was not
    an API engine or Krea. An engine id is not a model filename, and a tile that
    names the wrong engine is worse than a tile that names none."""
    from app.services import face_dataset_service as svc
    with app.app_context():
        _ds, img = _dataset(app, svc, 'minimax_h3')
        assert svc._image_engine(img) == 'minimax_h3'


def test_every_local_engine_id_survives_the_badge_resolver(app):
    """Derived, not enumerated: the next local engine must not need this file
    edited to be badged correctly."""
    from app.services import face_dataset_service as svc
    from app.services import dataset_generation_service as gen

    class _Row:
        pass

    with app.app_context():
        for engine in gen.LOCAL_ENGINES:
            row = _Row()
            row.klein_model = engine
            assert svc._image_engine(row) == engine, engine
        # A real model FILENAME still reads as Klein — that is the fallthrough's
        # actual job, and it must not be broken by the fix above.
        row = _Row()
        row.klein_model = 'flux-2-klein-9b.safetensors'
        assert svc._image_engine(row) == 'klein'
        # Nothing recorded stays "unknown": a guess would mislabel old rows.
        row = _Row()
        row.klein_model = ''
        assert svc._image_engine(row) is None


def test_regenerating_an_h3_tile_uses_h3(app, monkeypatch):
    """THE red one. Retry omits the engine, so the row's own tag decides — and
    the tag reached an `else` that meant Klein."""
    from app.services import face_dataset_service as svc
    from app.services import minimax_h3_helper as mh
    from app.services import klein_edit_helper as keh
    seen = {}
    monkeypatch.setattr(mh, 'preflight', lambda *a, **k: None)
    monkeypatch.setattr(mh, 'enqueue_minimax_h3',
                        lambda **kw: (seen.setdefault('h3', kw), 'job-h3')[1])
    monkeypatch.setattr(keh, 'enqueue_klein_edit',
                        lambda **kw: (seen.setdefault('klein', kw), 'job-klein')[1])
    with app.app_context():
        ds, img = _dataset(app, svc, 'minimax_h3')
        svc.regenerate_image('local', ds.id, img.id)
    assert 'h3' in seen, 'the H3 row was regenerated on another engine'
    assert 'klein' not in seen


def test_an_explicit_engine_override_still_wins_over_the_row(app, monkeypatch):
    """The branch must not become "H3 rows are always H3": the workspace can
    still ask for a different engine on purpose."""
    from app.services import face_dataset_service as svc
    from app.services import minimax_h3_helper as mh
    from app.services import krea_edit_helper as keh2
    seen = {}
    monkeypatch.setattr(mh, 'preflight', lambda *a, **k: None)
    monkeypatch.setattr(mh, 'enqueue_minimax_h3',
                        lambda **kw: (seen.setdefault('h3', kw), 'j')[1])
    monkeypatch.setattr(keh2, 'preflight', lambda *a, **k: None)
    monkeypatch.setattr(keh2, 'enqueue_krea_edit',
                        lambda **kw: (seen.setdefault('krea', kw), 'j')[1])
    with app.app_context():
        ds, img = _dataset(app, svc, 'minimax_h3')
        svc.regenerate_image('local', ds.id, img.id, engine='krea')
    assert 'krea' in seen and 'h3' not in seen


def test_an_engine_selection_saved_before_h3_existed_does_not_reroute_it(
        app, monkeypatch):
    """The SECOND way 🔄 answered with Krea 2 on an H3 tile — reported after the
    first was fixed, and not the same bug.

    `engines.enabled` is what a user ticked the last time they looked at
    Settings. Anyone who narrowed it before H3 shipped has a list without H3 in
    it, and regenerate used to rewrite the target to the default engine on that
    basis. The row already ran on H3; the list is about what a NEW batch may
    use. (`_merge_new_engines` also re-offers an engine that appeared since the
    save, so this is belt AND braces — the two answer different questions: that
    one restores the CHECKBOX, this one stops a retry from being rerouted at
    all, including for an engine the user really did untick.)"""
    from app.services import face_dataset_service as svc
    from app.services import minimax_h3_helper as mh
    from app.services import krea_edit_helper as keh2
    from app import config as cfg
    seen = {}
    monkeypatch.setattr(mh, 'preflight', lambda *a, **k: None)
    monkeypatch.setattr(mh, 'enqueue_minimax_h3',
                        lambda **kw: (seen.setdefault('h3', kw), 'job-h3')[1])
    monkeypatch.setattr(keh2, 'preflight', lambda *a, **k: None)
    monkeypatch.setattr(keh2, 'enqueue_krea_edit',
                        lambda **kw: (seen.setdefault('krea', kw), 'j')[1])
    with app.app_context():
        cfg.save_config({'engines': {'enabled': ['krea'], 'default': 'krea'}})
        ds, img = _dataset(app, svc, 'minimax_h3')
        svc.regenerate_image('local', ds.id, img.id)
    assert 'h3' in seen, 'the H3 row was rerouted by a Settings list'
    assert 'krea' not in seen


def test_the_settings_toggle_list_offers_every_engine_the_backend_ships(app):
    """An engine absent from ENGINE_OPTIONS has no checkbox, so it can never be
    disabled — and, on an install that dropped it from engines.enabled, never
    re-enabled either. Read from the JS the same crude way the engine-list
    contract does, so a moved declaration fails loudly."""
    import re
    from pathlib import Path
    from app.services import dataset_generation_service as gen

    js = (Path(__file__).resolve().parents[2] / 'frontend' / 'src' / 'components'
          / 'settings' / 'EnginesSection.jsx')
    if not js.exists():
        pytest.skip('frontend source not present')
    src = js.read_text(encoding='utf-8')
    m = re.search(r'const ENGINE_OPTIONS\s*=\s*\[(.*?)\n\]', src, re.S)
    assert m, 'ENGINE_OPTIONS declaration not found in EnginesSection.jsx'
    ids = set(re.findall(r"id:\s*'([^']+)'", m.group(1)))
    for engine in gen.KNOWN_ENGINES:
        assert engine in ids, f'{engine} has no Enabled-engines checkbox'
