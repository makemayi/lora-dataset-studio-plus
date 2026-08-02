"""◉ LoRA Canvas index — the cheap list the canvas draws its dataset filter from.

The canvas puts EVERY dataset on one surface, so the first thing it needs is
"which datasets are worth drawing". That answer must be cheap: it is computed
from the run table alone, never by assembling the genealogies (each of those
scans the run's saves on disk, so a thirty-dataset library would stare at a
spinner before anything appeared). These tests pin exactly that contract —
datasets with runs only, correct counts and families, another user's datasets
excluded, and an empty library answering 200 with an empty list rather than
an error page.
"""


def _dataset(name, user_id='local'):
    from app.extensions import db
    from app.models import FaceDataset
    d = FaceDataset(user_id=user_id, name=name, trigger_word=name.lower())
    db.session.add(d)
    db.session.commit()
    return d


def _rec(dataset_id, family='zimage', steps=1000, created_at=None):
    from app.extensions import db
    from app.models import TrainingRunRecord
    r = TrainingRunRecord(dataset_id=dataset_id, family=family, source='local',
                          base_model='', variant='turbo', steps=steps, version=1,
                          fingerprint='fp', manifest='[]')
    if created_at is not None:
        r.created_at = created_at
    db.session.add(r)
    db.session.commit()
    return r


def test_index_lists_only_datasets_that_have_runs(app):
    from app.services import cloud_training as ct
    with app.app_context():
        trained = _dataset('Trained')
        _dataset('Never trained')
        _rec(trained.id)
        out = ct.canvas_dataset_index('local')
        assert [d['name'] for d in out['datasets']] == ['Trained']
        assert out['datasets'][0]['runs'] == 1


def test_index_counts_runs_and_collects_families(app):
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _dataset('Two families')
        _rec(ds.id, family='zimage')
        _rec(ds.id, family='zimage')
        _rec(ds.id, family='krea')
        row = ct.canvas_dataset_index('local')['datasets'][0]
        assert row['runs'] == 3
        assert row['families'] == ['krea', 'zimage']
        assert row['last_run_at'] is not None


def test_index_is_ordered_newest_run_first(app):
    from datetime import datetime
    from app.services import cloud_training as ct
    with app.app_context():
        old = _dataset('Old')
        recent = _dataset('Recent')
        _rec(old.id, created_at=datetime(2024, 1, 1))
        _rec(recent.id, created_at=datetime(2026, 1, 1))
        names = [d['name'] for d in ct.canvas_dataset_index('local')['datasets']]
        assert names == ['Recent', 'Old']


def test_index_never_leaks_another_users_datasets(app):
    from app.services import cloud_training as ct
    with app.app_context():
        mine = _dataset('Mine')
        theirs = _dataset('Theirs', user_id='someone-else')
        _rec(mine.id)
        _rec(theirs.id)
        assert [d['name'] for d in ct.canvas_dataset_index('local')['datasets']] == ['Mine']


def test_index_empty_library_is_safe(app):
    from app.services import cloud_training as ct
    with app.app_context():
        assert ct.canvas_dataset_index('local') == {'datasets': []}


def test_index_endpoint_answers_200(client, app):
    with app.app_context():
        ds = _dataset('Endpoint')
        _rec(ds.id)
        ds_id = ds.id
    r = client.get('/api/train/canvas/datasets')
    assert r.status_code == 200
    body = r.get_json()
    assert [d['id'] for d in body['datasets']] == [ds_id]
    # The index carries NO checkpoints: that payload is what makes it cheap, and
    # the canvas fetches genealogies per dataset from the lineage endpoint.
    assert 'checkpoints' not in body['datasets'][0]
    assert 'nodes' not in body['datasets'][0]


def test_index_carries_the_star_pinned_loras(app):
    """The ★ pin travels with the index so a delete FROM THE BOARD can warn that
    it is about to break the saved winning combo. Stored per family → a list.
    Without this the canvas confirmed the destructive action with the plain
    wording and never showed the ⚠ line the dataset panel shows."""
    import json
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _dataset('Pinned')
        _rec(ds.id, family='zimage')
        _rec(ds.id, family='sdxl')
        ds.best_settings = json.dumps({
            'zimage': {'lora_filename': 'zit/win_002000.safetensors', 'strength': 1.0},
            'sdxl': {'lora_filename': 'sdxl/win_001000.safetensors', 'strength': 0.9},
        })
        db.session.commit()
        row = ct.canvas_dataset_index('local')['datasets'][0]
        assert row['best_settings_loras'] == ['zit/win_002000.safetensors',
                                              'sdxl/win_001000.safetensors']


def test_index_pin_reads_the_legacy_flat_best_settings(app):
    """A database written before best settings went per family stores ONE flat
    object. It must still produce the warning — a legacy install losing its
    guard-rail is exactly the failure this fix is about."""
    import json
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _dataset('Legacy')
        _rec(ds.id)
        ds.best_settings = json.dumps({'lora_filename': 'old/win_001000.safetensors',
                                       'strength': 1.0})
        db.session.commit()
        row = ct.canvas_dataset_index('local')['datasets'][0]
        assert row['best_settings_loras'] == ['old/win_001000.safetensors']


def test_index_pin_is_empty_and_never_raises_without_one(app):
    import json
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _dataset('Unpinned')
        _rec(ds.id)
        assert ct.canvas_dataset_index('local')['datasets'][0]['best_settings_loras'] == []
        ds.best_settings = 'not json at all'
        db.session.commit()
        assert ct.canvas_dataset_index('local')['datasets'][0]['best_settings_loras'] == []
        ds.best_settings = json.dumps({'zimage': {'strength': 1.0}})   # pinned, no file
        db.session.commit()
        assert ct.canvas_dataset_index('local')['datasets'][0]['best_settings_loras'] == []


def test_index_carries_the_dataset_reference_face_and_its_kind(app):
    """🪪 The canvas draws the reference beside each lane's name — a board full
    of renders of a person with the person nowhere on it made every likeness
    check a memory test. Both fields come off the dataset row the index already
    holds, so the "no extra query, no disk" invariant of this endpoint still
    holds. `kind` travels because only a CHARACTER dataset has a reference face;
    a concept or a style must not be shown an empty frame."""
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        person = _dataset('Person')
        person.ref_filename = 'local_datasetref_abcd1234.webp'
        concept = _dataset('A concept')
        concept.kind = 'concept'
        db.session.commit()
        _rec(person.id)
        _rec(concept.id)

        rows = {d['name']: d for d in ct.canvas_dataset_index('local')['datasets']}
        assert rows['Person']['ref_filename'] == 'local_datasetref_abcd1234.webp'
        assert rows['Person']['kind'] == 'character'
        assert rows['A concept']['kind'] == 'concept'
        assert rows['A concept']['ref_filename'] is None
