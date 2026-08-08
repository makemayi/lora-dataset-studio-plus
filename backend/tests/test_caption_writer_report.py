"""WHO wrote the captions of a pass.

The captioning backend defaults to 'auto', which is not a choice between two engines
but a CHAIN: JoyCaption captions what it can in one batch, the Ollama vision model
covers the rest — and on a Concept dataset Ollama rewrites JoyCaption's drafts. The
two engines write in visibly different styles, and until now a pass reported a bare
count, so someone whose captions suddenly read differently had no way to learn why.

These tests pin the PROPERTY: the counts describe who actually wrote the text that
was STORED, image by image — not which backend was configured, which is the proxy
that would go green while the run did something else entirely.

The vision seam is imported locally by the pipeline, so it is patched at
app.services.vision_ollama.*; JoyCaption likewise at app.services.joycaption.*.
"""
import os

from PIL import Image

from app.extensions import db
from app.models import FaceDatasetImage
from app.services import face_dataset_service as svc
from app.config import LOCAL_USER, save_config


def _kept_image(ds, fn):
    img_dir = svc._dataset_dir(ds.id)
    os.makedirs(img_dir, exist_ok=True)
    Image.new('RGB', (32, 32)).save(os.path.join(img_dir, fn))
    row = FaceDatasetImage(dataset_id=ds.id, status='keep', filename=fn)
    db.session.add(row)
    db.session.commit()
    return row


def _dataset(n=3, name='Emma', trigger='zchar_emma'):
    ds = svc.create_dataset(LOCAL_USER, name, trigger)
    rows = [_kept_image(ds, f'k{i}.png') for i in range(n)]
    return ds, [os.path.join(svc._dataset_dir(ds.id), r.filename) for r in rows]


def _ollama(monkeypatch, text='an ollama caption of a person in a room'):
    from app.services import vision_ollama
    monkeypatch.setattr(vision_ollama, 'describe_image_ollama',
                        lambda *a, **k: text)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)


def _joycaption(monkeypatch, covered):
    """JoyCaption available, captioning exactly the paths in `covered`."""
    import app.services.joycaption as jc
    monkeypatch.setattr(jc, 'is_available', lambda: True)
    monkeypatch.setattr(jc, 'availability', lambda: {'ok': True, 'detail': ''})
    monkeypatch.setattr(
        jc, 'caption_images_joycaption',
        lambda paths, **k: {p: 'a joycaption draft of a person in a room'
                            for p in paths if p in covered})


def test_ollama_only_pass_reports_ollama(app, monkeypatch):
    _ollama(monkeypatch)
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})
        ds, _ = _dataset(2)
        report = {}
        n = svc.caption_images(LOCAL_USER, ds.id, report=report)
        assert n == 2
        assert report == {'ollama': 2}


def test_joycaption_only_pass_reports_joycaption(app, monkeypatch):
    with app.app_context():
        save_config({'captioning': {'backend': 'joycaption'}})
        ds, paths = _dataset(2)
        _joycaption(monkeypatch, set(paths))
        report = {}
        n = svc.caption_images(LOCAL_USER, ds.id, report=report)
        assert n == 2
        assert report == {'joycaption': 2}


def test_auto_splits_the_batch_and_says_so(app, monkeypatch):
    """THE case this exists for. 'auto' silently hands the images JoyCaption missed
    to Ollama; both wrote, and both must be counted — a single "3 captioned" hides
    that two of them are in a completely different voice."""
    _ollama(monkeypatch)
    with app.app_context():
        save_config({'captioning': {'backend': 'auto'}})
        ds, paths = _dataset(3)
        _joycaption(monkeypatch, {paths[0]})     # JoyCaption covers ONE of the three
        report = {}
        n = svc.caption_images(LOCAL_USER, ds.id, report=report)
        assert n == 3
        assert report == {'joycaption': 1, 'ollama': 2}


def test_the_count_follows_what_was_STORED_not_what_was_configured(app, monkeypatch):
    """The property, not a proxy: JoyCaption is declared available and is asked, but
    it returns nothing usable for any image. The backend setting still says 'auto'
    with JoyCaption preferred — and the honest report names Ollama alone, because
    Ollama wrote every stored caption."""
    _ollama(monkeypatch)
    with app.app_context():
        save_config({'captioning': {'backend': 'auto'}})
        ds, _ = _dataset(2)
        _joycaption(monkeypatch, set())          # asked, answered nothing
        report = {}
        assert svc.caption_images(LOCAL_USER, ds.id, report=report) == 2
        assert report == {'ollama': 2}
        assert 'joycaption' not in report


def test_an_image_that_yields_no_caption_is_counted_by_nobody(app, monkeypatch):
    """An empty answer stores nothing, so no engine may claim it: the per-engine
    numbers must always add up to the reported total."""
    from app.services import vision_ollama
    answers = iter(['a real caption of a person', '', '   '])
    monkeypatch.setattr(vision_ollama, 'describe_image_ollama',
                        lambda *a, **k: next(answers))
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})
        ds, _ = _dataset(3)
        report = {}
        n = svc.caption_images(LOCAL_USER, ds.id, report=report)
        assert n == 1
        assert sum(report.values()) == n
        assert report == {'ollama': 1}


def test_report_is_optional_and_the_return_contract_is_unchanged(app, monkeypatch):
    """Every existing caller passes no report and still gets the plain count."""
    _ollama(monkeypatch)
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})
        ds, _ = _dataset(2)
        assert svc.caption_images(LOCAL_USER, ds.id) == 2


def test_concept_auto_reports_the_joycaption_draft_rewritten_by_ollama(app, monkeypatch):
    """A Concept dataset in 'auto' chains the engines on the SAME image: JoyCaption
    drafts, Ollama rewrites. That is one writer of its own — calling it either engine
    alone would be false, so it gets its own key."""
    _ollama(monkeypatch, text='a rewritten caption of a person in a room, seen from the side')
    with app.app_context():
        save_config({'captioning': {'backend': 'auto'}})
        ds = svc.create_dataset(LOCAL_USER, 'Kneel', 'zconcept_kneel', kind='concept',
                                concept_desc='kneeling on the floor')
        rows = [_kept_image(ds, f'c{i}.png') for i in range(2)]
        paths = [os.path.join(svc._dataset_dir(ds.id), r.filename) for r in rows]
        _joycaption(monkeypatch, set(paths))
        report = {}
        n = svc.caption_images(LOCAL_USER, ds.id, report=report)
        assert n == 2
        assert report == {'joycaption_refined': 2}


def test_the_caption_route_hands_the_engines_to_the_UI(app, client, monkeypatch):
    """End of the wire: the screen that shows the result can only name the engine if
    the response carries it."""
    _ollama(monkeypatch)
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})
        ds, _ = _dataset(2)
        ds_id = ds.id
    r = client.post(f'/api/dataset/{ds_id}/caption', json={'force': True})
    assert r.status_code == 200
    body = r.get_json()
    assert body['captioned'] == 2
    assert body['engines'] == {'ollama': 2}
